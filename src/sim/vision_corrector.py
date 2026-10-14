"""
src/sim/vision_corrector.py

Phase 5 Closed-Loop Vision Corrector.

The corrector may use ONLY:
  - The TopDownVisionSensor image (via CoppeliaController.get_vision_sensor_image)
  - The target stroke list (from IR JSON)
  - The nominal robot model (FK/IK via src.kinematics and src.path_planner)

It must NOT use:
  - The true tip position (no get_tip_position calls)
  - Telemetry / logged execution data
  - Disturbance parameters (no DisturbanceConfig access)

Architecture:
  1. PerStrokeObserver: after each stroke, diff current vs previous camera image,
     extract ink pixels, convert to workspace coordinates.
  2. StrokeMatcherICP: match observed pixels to target stroke via nearest-point
     iteration (ICP-style) → estimate 2D translation offset (and optionally affine).
  3. RunningCorrector: smooth per-stroke estimates with exponential moving average,
     apply inverse correction to remaining planned waypoints before IK.
  4. CalibrationPrePass (optional): draw a small cross first, fit the plant transform,
     apply correction globally before the real drawing begins.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import KDTree

from src.config import CameraConfig, DEFAULT_CAMERA_CONFIG


def camera_pixel_to_arm(
    u: float,
    v: float,
    camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
) -> Tuple[float, float]:
    """Convert camera pixel (u, v) to robot arm workspace coordinates (x, y) in cm."""
    center_u = camera_config.resolution_x / 2.0
    center_v = camera_config.resolution_y / 2.0
    scale_cm_per_px = camera_config.view_size_cm / camera_config.resolution_x
    x_cm = camera_config.center_x_cm + (u - center_u) * scale_cm_per_px
    y_cm = camera_config.center_y_cm - (v - center_v) * scale_cm_per_px
    return x_cm, y_cm


def extract_ink_mask_from_camera_image(
    image: np.ndarray,
    is_bgr: bool = True,
) -> np.ndarray:
    """Extract visible blue ink trace binary mask from a TopDownVisionSensor capture."""
    img = np.asarray(image, dtype=np.uint8)
    if is_bgr:
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    else:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    ink_mask = (b > 100) & (r < 70) & (g < 100)
    return ink_mask.astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CorrectorConfig:
    """
    Configuration for the closed-loop vision corrector.

    Attributes:
        enabled: Master toggle for correction.
        max_offset_cm: Maximum allowed correction magnitude per stroke in cm.
        ema_alpha: Exponential moving average weight for new estimate (0=ignore, 1=take immediately).
        min_observed_pixels: Minimum ink pixels required to accept a stroke observation.
        icp_iterations: Number of ICP nearest-point refinement iterations.
        icp_inlier_fraction: Fraction of closest pairs kept as inliers (to reject outliers).
        use_affine: If True, fit full 2D affine transform (scale+rotation+translation);
                    if False, estimate translation offset only (safer, more conservative).
        correction_start_stroke: Apply correction starting from this stroke index.
        calib_prepass: If True, run calibration pre-pass before main drawing.
        calib_pattern: 'cross' or 'grid' for calibration pattern type.
    """
    enabled: bool = True
    max_offset_cm: float = 0.50          # Cap at 5.0 mm per step
    ema_alpha: float = 0.40              # 40% new estimate, 60% previous smoothed
    min_observed_pixels: int = 30        # Minimum ink pixels to accept observation
    icp_iterations: int = 20            # ICP refinement iterations
    icp_inlier_fraction: float = 0.70   # Keep top 70% closest pairs as inliers
    use_affine: bool = False             # Conservative: translation only by default
    correction_start_stroke: int = 1    # Start correcting from 2nd stroke onward
    calib_prepass: bool = False          # Off by default (per-stroke variant)
    calib_pattern: str = "cross"        # 'cross' or 'grid'


# ---------------------------------------------------------------------------
# 1. Per-Stroke Ink Observation
# ---------------------------------------------------------------------------

class PerStrokeObserver:
    """
    Extracts per-stroke ink pixels from the camera by differencing consecutive images.

    After each stroke is drawn, captures the camera image and subtracts the previous
    capture to isolate ink deposited in that stroke only.
    """

    def __init__(
        self,
        camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
        min_pixels: int = 30,
    ) -> None:
        self.camera_config = camera_config
        self.min_pixels = min_pixels
        self._prev_ink_mask: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset the previous ink mask (call before starting a new drawing)."""
        self._prev_ink_mask = None

    def capture_baseline(self, camera_image_bgr: np.ndarray) -> None:
        """
        Capture the initial (empty) camera state before any drawing begins.
        Must be called once before the first stroke.
        """
        self._prev_ink_mask = extract_ink_mask_from_camera_image(camera_image_bgr, is_bgr=True)

    def observe_stroke(
        self,
        camera_image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract newly deposited ink pixels by differencing current vs previous capture.

        Returns:
            (stroke_mask_px, stroke_pts_cm):
                stroke_mask_px: Binary mask (H, W, uint8) of new ink pixels.
                stroke_pts_cm: Array (N, 2) of workspace coordinates in cm of those pixels.
        """
        cur_mask = extract_ink_mask_from_camera_image(camera_image_bgr, is_bgr=True)

        if self._prev_ink_mask is None:
            # No previous baseline – treat all current ink as this stroke
            new_ink = cur_mask
        else:
            # Isolate only newly deposited ink
            new_ink = np.zeros_like(cur_mask)
            new_ink[(cur_mask > 0) & (self._prev_ink_mask == 0)] = 255

        self._prev_ink_mask = cur_mask

        # Convert non-zero pixels → workspace coordinates
        ys, xs = np.where(new_ink > 0)
        stroke_pts_cm = np.array(
            [camera_pixel_to_arm(float(x), float(y), self.camera_config)
             for x, y in zip(xs, ys)],
            dtype=float,
        )
        if len(stroke_pts_cm) == 0:
            stroke_pts_cm = np.empty((0, 2), dtype=float)

        return new_ink, stroke_pts_cm


# ---------------------------------------------------------------------------
# 2. Stroke Matcher (ICP-style nearest-point)
# ---------------------------------------------------------------------------

def _densify_stroke(stroke: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Resample stroke vertices uniformly along polyline arc length."""
    pts = np.asarray(stroke, dtype=float)
    if len(pts) < 2:
        return pts
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    total_len = float(np.sum(seg_lens))
    if total_len < 1e-6:
        return pts
    cum_lens = np.insert(np.cumsum(seg_lens), 0, 0.0)
    target_dists = np.linspace(0.0, total_len, max(num_points, len(pts)))
    densified = np.empty((len(target_dists), 2), dtype=float)
    for i, d in enumerate(target_dists):
        idx = int(np.searchsorted(cum_lens, d, side="right") - 1)
        idx = min(idx, len(seg_lens) - 1)
        seg_d = d - cum_lens[idx]
        t = (seg_d / seg_lens[idx]) if seg_lens[idx] > 1e-6 else 0.0
        densified[i] = pts[idx] + t * diffs[idx]
    return densified


class StrokeMatcher:
    """
    Matches observed ink points to target stroke via iterative nearest-point (ICP).

    Estimates a 2D correction (translation or affine) that maps observed points
    to best align with the target stroke.
    """

    def __init__(self, config: CorrectorConfig) -> None:
        self.config = config

    def estimate_translation(
        self,
        observed_pts_cm: np.ndarray,
        target_stroke: np.ndarray,
        initial_offset: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Estimate 2D translation offset that minimises mean nearest-point distance
        from observed to target stroke.

        Parameters:
            observed_pts_cm: (N, 2) observed ink workspace coordinates in cm.
            target_stroke: (M, 2) target stroke workspace coordinates in cm.
            initial_offset: Optional (2,) starting offset for ICP initialisation.

        Returns:
            (2,) offset estimate [dx, dy] in cm such that:
                corrected_observed ≈ target_stroke
                equivalently: apply -offset to remaining planned waypoints.
            Returns None if too few points or fit fails.
        """
        if len(observed_pts_cm) < self.config.min_observed_pixels:
            return None
        if len(target_stroke) < 2:
            return None

        obs = np.asarray(observed_pts_cm, dtype=float)
        dense_tgt = _densify_stroke(target_stroke, num_points=max(100, len(obs)))

        # Robust centroid offset initialization
        q25, q75 = np.percentile(obs, [25, 75], axis=0)
        iqr = q75 - q25
        in_box = (obs >= q25 - 1.5 * iqr) & (obs <= q75 + 1.5 * iqr)
        valid_obs = obs[in_box.all(axis=1)]
        if len(valid_obs) < 2:
            valid_obs = obs
        centroid_offset = np.mean(valid_obs, axis=0) - np.mean(dense_tgt, axis=0)

        offset = np.asarray(initial_offset, dtype=float) if initial_offset is not None else centroid_offset

        # Build KD-tree on densified target stroke
        tree = KDTree(dense_tgt)

        for _ in range(self.config.icp_iterations):
            shifted_obs = obs - offset         # apply current running correction to observed

            # Find nearest target points for each observed point
            dists, idxs = tree.query(shifted_obs)

            # Keep only inlier fraction (discard outlier-distance pairs)
            n_inliers = max(2, int(len(dists) * self.config.icp_inlier_fraction))
            inlier_mask = dists <= np.sort(dists)[n_inliers - 1]

            inlier_obs = shifted_obs[inlier_mask]
            inlier_tgt = dense_tgt[idxs[inlier_mask]]

            if len(inlier_obs) < 2:
                break

            delta_offset = np.mean(inlier_obs - inlier_tgt, axis=0)
            offset = offset + delta_offset

            if np.linalg.norm(delta_offset) < 1e-5:
                break

        return offset  # shape (2,) in cm: observed ≈ target + offset

    def estimate_affine(
        self,
        observed_pts_cm: np.ndarray,
        target_stroke: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Estimate full 2D affine transform (translation + scale + rotation) via ICP.

        Returns:
            (2, 3) affine matrix A such that: observed ≈ A @ [target; 1]
            Returns None if fit fails.
        """
        if len(observed_pts_cm) < 6 or len(target_stroke) < 6:
            return None

        obs = np.asarray(observed_pts_cm, dtype=float)
        tgt = np.asarray(target_stroke, dtype=float)
        tree = KDTree(tgt)

        # First get translation estimate
        trans_offset = self.estimate_translation(obs, tgt)
        if trans_offset is None:
            return None

        # Use translation to initialize: obs - trans_offset ≈ tgt
        shifted_obs = obs - trans_offset
        dists, idxs = tree.query(shifted_obs)

        n_inliers = max(6, int(len(dists) * self.config.icp_inlier_fraction))
        inlier_mask = dists <= np.sort(dists)[min(n_inliers - 1, len(dists) - 1)]

        inlier_obs = obs[inlier_mask]
        inlier_tgt = tgt[idxs[inlier_mask]]

        if len(inlier_obs) < 6:
            return None

        # Fit affine with RANSAC-style cv2.estimateAffine2D
        M, _ = cv2.estimateAffine2D(
            inlier_tgt.astype(np.float32),
            inlier_obs.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=0.05,
        )
        return M  # (2, 3) or None


# ---------------------------------------------------------------------------
# 3. Running Corrector (EMA-smoothed, capped)
# ---------------------------------------------------------------------------

class RunningCorrector:
    """
    Maintains a smoothed estimate of the plant's systematic offset (or affine transform)
    and applies its inverse to remaining planned waypoints before IK.

    Uses exponential moving average to smooth per-stroke estimates, preventing
    over-reaction to noisy single observations.
    """

    def __init__(self, config: CorrectorConfig) -> None:
        self.config = config
        self._offset: np.ndarray = np.zeros(2)   # [dx, dy] in cm: observed = target + offset
        self._n_observations: int = 0
        self._history: List[np.ndarray] = []

    def reset(self) -> None:
        """Reset correction state."""
        self._offset = np.zeros(2)
        self._n_observations = 0
        self._history = []

    @property
    def current_offset(self) -> np.ndarray:
        """Current smoothed plant offset estimate in cm."""
        return self._offset.copy()

    @property
    def n_observations(self) -> int:
        return self._n_observations

    def update(self, new_offset_estimate: np.ndarray) -> np.ndarray:
        """
        Ingest a new per-stroke offset estimate and update the smoothed estimate.

        Parameters:
            new_offset_estimate: (2,) raw offset estimate for this stroke in cm.

        Returns:
            Updated smoothed offset in cm.
        """
        raw = np.asarray(new_offset_estimate, dtype=float)

        # EMA update: α * new + (1-α) * prev
        if self._n_observations == 0:
            self._offset = raw.copy()
        else:
            self._offset = self.config.ema_alpha * raw + (1 - self.config.ema_alpha) * self._offset

        # Cap magnitude
        magnitude = np.linalg.norm(self._offset)
        if magnitude > self.config.max_offset_cm:
            self._offset = self._offset * (self.config.max_offset_cm / magnitude)

        self._n_observations += 1
        self._history.append(self._offset.copy())
        return self._offset.copy()

    def correct_waypoint(self, x_cm: float, y_cm: float) -> Tuple[float, float]:
        """
        Apply the inverse correction to a planned waypoint before sending to IK.

        The plant offset means: observed_tip ≈ planned + offset
        So to make observed_tip land at the planned target, we command:
            corrected_command = planned - offset

        Returns:
            (corrected_x_cm, corrected_y_cm)
        """
        if not self.config.enabled or self._n_observations == 0:
            return x_cm, y_cm
        cx = x_cm - self._offset[0]
        cy = y_cm - self._offset[1]
        return cx, cy


# ---------------------------------------------------------------------------
# 4. Calibration Pre-Pass
# ---------------------------------------------------------------------------

class CalibrationPrePass:
    """
    Optional pre-pass: draw a known calibration pattern (cross or grid),
    observe it with the camera, estimate the plant kinematic offset globally,
    and return the correction to apply before the main drawing.

    This gives a single global calibration estimate, as opposed to the
    per-stroke running corrector.
    """

    # Calibration cross pattern (5 points in workspace cm)
    CROSS_PATTERN_CM: List[Tuple[float, float]] = [
        (7.00, 9.50),   # Centre
        (7.00, 8.50),   # South arm
        (7.00, 10.50),  # North arm
        (6.00, 9.50),   # West arm
        (8.00, 9.50),   # East arm
    ]

    # 3x3 grid pattern (9 points)
    GRID_PATTERN_CM: List[Tuple[float, float]] = [
        (5.50, 8.25), (7.00, 8.25), (8.50, 8.25),
        (5.50, 9.50), (7.00, 9.50), (8.50, 9.50),
        (5.50, 10.75), (7.00, 10.75), (8.50, 10.75),
    ]

    def __init__(
        self,
        config: CorrectorConfig,
        camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
    ) -> None:
        self.config = config
        self.camera_config = camera_config
        self.matcher = StrokeMatcher(config)
        self._calib_offset: Optional[np.ndarray] = None

    def get_pattern_strokes(self) -> List[np.ndarray]:
        """Return the calibration pattern as a list of 2-point line strokes."""
        if self.config.calib_pattern == "cross":
            pts = self.CROSS_PATTERN_CM
        else:
            pts = self.GRID_PATTERN_CM

        # Build as a single connected stroke: centre → each arm
        center = np.array(pts[0], dtype=float)
        strokes = []
        for pt in pts[1:]:
            strokes.append(np.array([center, pt], dtype=float))
        return strokes

    def estimate_offset_from_image(
        self,
        camera_image_bgr: np.ndarray,
        baseline_image_bgr: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Given the camera image after drawing the calibration pattern,
        extract ink pixels and estimate the global plant offset.

        Returns:
            (2,) offset estimate in cm, or None if not enough ink observed.
        """
        observer = PerStrokeObserver(self.camera_config, self.config.min_observed_pixels)
        if baseline_image_bgr is not None:
            observer.capture_baseline(baseline_image_bgr)

        _, obs_pts_cm = observer.observe_stroke(camera_image_bgr)

        # Use all calibration points as one target cloud
        pattern_strokes = self.get_pattern_strokes()
        all_tgt_pts = np.vstack(pattern_strokes)

        offset = self.matcher.estimate_translation(obs_pts_cm, all_tgt_pts)
        self._calib_offset = offset
        return offset

    @property
    def calibrated_offset(self) -> Optional[np.ndarray]:
        return self._calib_offset
