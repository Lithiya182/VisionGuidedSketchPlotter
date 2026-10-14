"""
image_processor.py

Computer Vision extraction and Intermediate Representation (IR) module.
Extracts pixel-space strokes on a standardized 480x480 canvas (top-left origin,
X right, Y down). Outputs versioned contour JSON with full scaling, padding,
and simplification deviation metadata.
"""

from dataclasses import dataclass, field, asdict
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import VisionConfig, DEFAULT_VISION_CONFIG


@dataclass
class SimplificationStats:
    """Quantitative fidelity metrics for Douglas-Peucker polygon simplification."""
    raw_points_count: int = 0
    simplified_points_count: int = 0
    compression_ratio: float = 0.0
    max_deviation_px: float = 0.0
    mean_deviation_px: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContourMetadata:
    """Metadata detailing source image dimensions, letterbox transform, and extraction mode."""
    version: str = "1.0"
    source_image: str = ""
    original_shape: Tuple[int, int] = (0, 0)   # (height, width)
    canvas_size: Tuple[int, int] = (480, 480)  # (width, height)
    scale: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0
    mode: str = "canny"                        # "boundary", "canny", or "skeleton"
    simplification_epsilon: float = 1.5
    simplification_stats: SimplificationStats = field(default_factory=SimplificationStats)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["original_shape"] = list(self.original_shape)
        d["canvas_size"] = list(self.canvas_size)
        return d


@dataclass
class StrokeIR:
    """
    Intermediate Representation (IR) containing pixel-space strokes
    and versioned metadata on the standardized canvas.
    """
    metadata: ContourMetadata
    strokes: List[List[Tuple[float, float]]]  # List of strokes; each stroke is [(x, y), ...]

    @property
    def version(self) -> str:
        return self.metadata.version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.metadata.version,
            "metadata": self.metadata.to_dict(),
            "strokes": [
                [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in stroke]
                for stroke in self.strokes
            ],
        }

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        if filepath is not None:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrokeIR":
        meta_dict = data.get("metadata", {})
        stats_dict = meta_dict.get("simplification_stats", {})
        stats = SimplificationStats(
            raw_points_count=stats_dict.get("raw_points_count", 0),
            simplified_points_count=stats_dict.get("simplified_points_count", 0),
            compression_ratio=stats_dict.get("compression_ratio", 0.0),
            max_deviation_px=stats_dict.get("max_deviation_px", 0.0),
            mean_deviation_px=stats_dict.get("mean_deviation_px", 0.0),
        )
        metadata = ContourMetadata(
            version=data.get("version", "1.0"),
            source_image=meta_dict.get("source_image", ""),
            original_shape=tuple(meta_dict.get("original_shape", (0, 0))),
            canvas_size=tuple(meta_dict.get("canvas_size", (480, 480))),
            scale=meta_dict.get("scale", 1.0),
            pad_x=meta_dict.get("pad_x", 0.0),
            pad_y=meta_dict.get("pad_y", 0.0),
            mode=meta_dict.get("mode", "canny"),
            simplification_epsilon=meta_dict.get("simplification_epsilon", 1.5),
            simplification_stats=stats,
        )
        strokes = [
            [(float(p[0]), float(p[1])) for p in stroke]
            for stroke in data.get("strokes", [])
        ]
        return cls(metadata=metadata, strokes=strokes)

    @classmethod
    def from_json(cls, source: str) -> "StrokeIR":
        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(source)
        return cls.from_dict(data)


def point_to_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Calculate the Euclidean distance from point P to line segment AB."""
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def compute_simplification_deviation(
    raw_points: np.ndarray,
    simplified_points: np.ndarray,
    is_closed: bool = False,
) -> Tuple[float, float]:
    """
    Compute maximum and mean perpendicular deviation from raw points to simplified polyline.

    Parameters:
        raw_points: (N, 2) array of original curve coordinates.
        simplified_points: (M, 2) array of simplified polygon coordinates.
        is_closed: If True, include the closing segment between the last and first vertices.

    Returns:
        (max_deviation_px, mean_deviation_px)
    """
    if len(raw_points) == 0:
        return 0.0, 0.0
    if len(simplified_points) < 2:
        if len(simplified_points) == 1:
            diffs = raw_points - simplified_points[0]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            return float(np.max(dists)), float(np.mean(dists))
        return 0.0, 0.0

    poly = simplified_points
    if is_closed and not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    segs = [
        (poly[i][0], poly[i][1],
         poly[i + 1][0], poly[i + 1][1])
        for i in range(len(poly) - 1)
    ]

    total_deviation = 0.0
    max_deviation = 0.0

    for pt in raw_points:
        px, py = float(pt[0]), float(pt[1])
        min_d = min(
            point_to_segment_distance(px, py, seg[0], seg[1], seg[2], seg[3])
            for seg in segs
        )
        if min_d > max_deviation:
            max_deviation = min_d
        total_deviation += min_d

    mean_deviation = total_deviation / len(raw_points)
    return float(max_deviation), float(mean_deviation)


class ImageProcessor:
    """
    Extracts vectorized drawing strokes from images using configurable CV pipelines.
    Supports Boundary contours, Canny edges, and Morphological Skeletonization modes.
    """

    def __init__(self, config: VisionConfig = DEFAULT_VISION_CONFIG):
        self.config = config

    def resize_and_pad(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Scale image to fit within target canvas without aspect ratio distortion,
        padding borders with 255 (white background).

        Returns:
            (canvas_image, scale, pad_x, pad_y)
        """
        orig_h, orig_w = image.shape[:2]
        target_w = self.config.canvas_width
        target_h = self.config.canvas_height

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

        pad_x = (target_w - new_w) / 2.0
        pad_y = (target_h - new_h) / 2.0

        top = int(math.floor(pad_y))
        bottom = target_h - new_h - top
        left = int(math.floor(pad_x))
        right = target_w - new_w - left

        # Border color white (255) for both grayscale and color
        border_val = [255, 255, 255] if image.ndim == 3 else 255
        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=border_val
        )

        return canvas, scale, float(left), float(top)

    def preprocess_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply Gaussian smoothing."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        k = self.config.blur_kernel_size
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), self.config.blur_sigma)
        return gray

    def extract_boundary_contours(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Extract outline/silhouette contours of dark foreground shapes.
        Uses Otsu thresholding followed by hierarchy-aware contour extraction.
        """
        # Invert so drawing strokes/dark ink are foreground (255)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        valid_contours = []
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            arc_len = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if (arc_len >= self.config.min_contour_length_px and
                    area >= self.config.min_contour_area_px):
                # Ensure closed loop by repeating the first point if necessary
                if len(pts) > 2 and not np.array_equal(pts[0], pts[-1]):
                    pts = np.vstack([pts, pts[0]])
                valid_contours.append(pts)

        return valid_contours

    def extract_canny_contours(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Extract boundary edge contours using Canny edge detection.
        """
        edges = cv2.Canny(
            gray,
            self.config.canny_threshold_low,
            self.config.canny_threshold_high,
        )

        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        valid_contours = []
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            arc_len = cv2.arcLength(cnt, False)
            if arc_len >= self.config.min_contour_length_px and len(pts) >= 3:
                valid_contours.append(pts)

        return valid_contours

    def extract_skeleton_contours(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Extract 1-pixel wide centerline strokes using morphological skeletonization.
        Prevents double-contour outlines for line-art and sketch drawings.
        """
        # Foreground ink is 255
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological skeletonization
        skeleton = np.zeros(binary.shape, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = binary.copy()

        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break

        # Extract strokes from skeleton using 8-connected graph traversal
        return self._trace_skeleton_paths(skeleton)

    def _trace_skeleton_paths(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """
        Trace 1-pixel wide skeleton branches into ordered pixel paths.
        Identifies endpoints and junctions to prevent cyclic branch doubling.
        """
        h, w = skeleton.shape
        binary = (skeleton > 0).astype(np.uint8)

        # 8-connectivity kernel to count neighbors
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(binary, -1, kernel) * binary

        visited = np.zeros((h, w), dtype=bool)
        strokes = []

        # 8-neighbor coordinate offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

        # Priority 1: Trace paths starting from endpoints (neighbor_count == 1)
        endpoint_ys, endpoint_xs = np.where((neighbor_count == 1) & binary)
        for ey, ex in zip(endpoint_ys, endpoint_xs):
            if visited[ey, ex]:
                continue
            path = [(ex, ey)]
            visited[ey, ex] = True
            curr_x, curr_y = ex, ey

            while True:
                next_pt = None
                for dy, dx in offsets:
                    ny, nx = curr_y + dy, curr_x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if binary[ny, nx] and not visited[ny, nx]:
                            next_pt = (nx, ny)
                            break
                if next_pt is None:
                    break
                curr_x, curr_y = next_pt
                path.append((curr_x, curr_y))
                visited[curr_y, curr_x] = True

            if len(path) >= 3:
                pts = np.array(path, dtype=np.float32)
                arc_len = float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
                if arc_len >= self.config.min_contour_length_px:
                    strokes.append(pts)

        # Priority 2: Trace remaining closed loops (pixels with neighbor_count >= 2)
        rem_ys, rem_xs = np.where(binary & ~visited)
        for ry, rx in zip(rem_ys, rem_xs):
            if visited[ry, rx]:
                continue
            path = [(rx, ry)]
            visited[ry, rx] = True
            curr_x, curr_y = rx, ry

            while True:
                next_pt = None
                for dy, dx in offsets:
                    ny, nx = curr_y + dy, curr_x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if binary[ny, nx] and not visited[ny, nx]:
                            next_pt = (nx, ny)
                            break
                if next_pt is None:
                    break
                curr_x, curr_y = next_pt
                path.append((curr_x, curr_y))
                visited[curr_y, curr_x] = True

            if len(path) >= 3:
                pts = np.array(path, dtype=np.float32)
                arc_len = float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
                if arc_len >= self.config.min_contour_length_px:
                    strokes.append(pts)

        return strokes

    def simplify_strokes(
        self, raw_strokes: List[np.ndarray], epsilon: float
    ) -> Tuple[List[List[Tuple[float, float]]], SimplificationStats]:
        """
        Simplify raw pixel contours using the Douglas-Peucker algorithm
        and compute aggregate deviation statistics.
        """
        simplified_strokes: List[List[Tuple[float, float]]] = []
        total_raw_points = 0
        total_simplified_points = 0
        max_dev_overall = 0.0
        weighted_dev_sum = 0.0

        for raw in raw_strokes:
            raw_pts = raw.astype(np.float32)
            n_raw = len(raw_pts)
            total_raw_points += n_raw

            is_closed = np.array_equal(raw_pts[0], raw_pts[-1]) and n_raw > 2

            cnt_cv = raw_pts.reshape((-1, 1, 2))
            poly = cv2.approxPolyDP(cnt_cv, epsilon, is_closed).reshape((-1, 2))

            # If contour was closed, ensure polyline also explicitly forms a closed loop
            if is_closed and len(poly) > 2 and not np.array_equal(poly[0], poly[-1]):
                poly = np.vstack([poly, poly[0]])

            n_sim = len(poly)
            total_simplified_points += n_sim

            max_d, mean_d = compute_simplification_deviation(raw_pts, poly, is_closed=is_closed)
            if max_d > max_dev_overall:
                max_dev_overall = max_d
            weighted_dev_sum += mean_d * n_raw

            simplified_strokes.append(
                [(float(pt[0]), float(pt[1])) for pt in poly]
            )

        mean_dev_overall = (
            weighted_dev_sum / total_raw_points if total_raw_points > 0 else 0.0
        )
        compression = (
            1.0 - (total_simplified_points / total_raw_points)
            if total_raw_points > 0 else 0.0
        )

        stats = SimplificationStats(
            raw_points_count=total_raw_points,
            simplified_points_count=total_simplified_points,
            compression_ratio=round(compression, 4),
            max_deviation_px=round(max_dev_overall, 3),
            mean_deviation_px=round(mean_dev_overall, 3),
        )

        return simplified_strokes, stats

    def process_image(
        self,
        image_or_path: Union[str, np.ndarray],
        mode: Optional[str] = None,
        epsilon: Optional[float] = None,
    ) -> StrokeIR:
        """
        Process an input image and generate the versioned StrokeIR.

        Parameters:
            image_or_path: Filepath or NumPy image array.
            mode: Extraction mode ("boundary", "canny", "skeleton").
            epsilon: Douglas-Peucker simplification tolerance in pixels.

        Returns:
            StrokeIR dataclass with canvas-pixel coordinates and full metadata.
        """
        target_mode = mode if mode is not None else self.config.default_mode
        if target_mode not in ("boundary", "canny", "skeleton"):
            raise ValueError(
                f"Invalid mode '{target_mode}'. Must be 'boundary', 'canny', or 'skeleton'."
            )

        target_eps = (
            epsilon if epsilon is not None
            else self.config.simplification_epsilon_px
        )

        source_name = ""
        if isinstance(image_or_path, str):
            source_name = os.path.basename(image_or_path)
            if not os.path.exists(image_or_path):
                raise FileNotFoundError(f"Image file not found: {image_or_path}")
            img = cv2.imread(image_or_path)
            if img is None:
                raise ValueError(f"Failed to decode image from: {image_or_path}")
        elif isinstance(image_or_path, np.ndarray):
            img = image_or_path.copy()
            source_name = "in_memory_array"
        else:
            raise TypeError("image_or_path must be a file path string or numpy ndarray.")

        orig_shape = (img.shape[0], img.shape[1])

        # 1. Resize and letterbox/pad to canvas (480x480)
        canvas_img, scale, pad_x, pad_y = self.resize_and_pad(img)

        # 2. Preprocess grayscale and smoothing
        gray = self.preprocess_grayscale(canvas_img)

        # 3. Extract raw contours according to selected mode
        if target_mode == "boundary":
            raw_contours = self.extract_boundary_contours(gray)
        elif target_mode == "canny":
            raw_contours = self.extract_canny_contours(gray)
        elif target_mode == "skeleton":
            raw_contours = self.extract_skeleton_contours(gray)

        # 4. Polygon simplification and deviation tracking
        simplified_strokes, stats = self.simplify_strokes(raw_contours, target_eps)

        # 5. Assemble versioned metadata and StrokeIR
        metadata = ContourMetadata(
            version="1.0",
            source_image=source_name,
            original_shape=orig_shape,
            canvas_size=(self.config.canvas_width, self.config.canvas_height),
            scale=round(scale, 5),
            pad_x=round(pad_x, 2),
            pad_y=round(pad_y, 2),
            mode=target_mode,
            simplification_epsilon=round(target_eps, 2),
            simplification_stats=stats,
        )

        return StrokeIR(metadata=metadata, strokes=simplified_strokes)
