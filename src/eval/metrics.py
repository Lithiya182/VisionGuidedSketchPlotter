"""
src/eval/metrics.py

Comprehensive evaluation and verification metrics for the Vision-Guided Sketch Plotter:
1. Trajectory tracking errors: pointwise Euclidean error, RMSE, mean, max, std.
2. Endpoint errors: stroke boundary discrepancies.
3. Point-set Chamfer distance and Hausdorff distance (cm and mm).
4. Trajectory smoothness: RMS jerk, acceleration variance, and dimensionless jerk.
5. Pen travel metrics: pen-down distance, pen-up distance, and travel efficiency ratio.
6. Error decomposition: Planning Error (Target vs Plan) and Execution Error (Plan vs Measured Tip).
7. Top-down vision sensor metrics: raster IoU, Dice, Precision, Recall, and distance-transform Chamfer/Contour Error.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial import KDTree

from src.config import CameraConfig, DEFAULT_CAMERA_CONFIG, DEFAULT_EVAL_CONFIG, EvalConfig


# ==============================================================================
# 1. Pointwise & Trajectory Error Analysis
# ==============================================================================

def trajectory_errors(target_points: np.ndarray, actual_points: np.ndarray) -> np.ndarray:
    """
    Calculate pointwise Euclidean errors between corresponding points.

    Both arrays must contain corresponding (x, y) coordinates in centimetres
    and have matching shape (N, 2).

    Returns:
        1D NumPy array of shape (N,) containing Euclidean distances in cm.
    """
    target = np.asarray(target_points, dtype=float)
    actual = np.asarray(actual_points, dtype=float)

    if target.size == 0 or actual.size == 0:
        raise ValueError("At least one point is required.")

    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("Target points must have shape (N, 2).")

    if actual.shape != target.shape:
        raise ValueError("Actual and target arrays must have identical shapes.")

    if not np.isfinite(target).all() or not np.isfinite(actual).all():
        raise ValueError("Coordinates must be finite.")

    return np.linalg.norm(actual - target, axis=1)


pointwise_tracking_errors = trajectory_errors


def calculate_trajectory_metrics(
    target_points: np.ndarray,
    actual_points: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate trajectory error metrics (RMSE, mean, max, std) in centimetres and millimetres.

    Returns:
        Dictionary containing tracking error statistics.
    """
    errors_cm = trajectory_errors(target_points, actual_points)

    rmse_cm = float(np.sqrt(np.mean(errors_cm ** 2)))
    mean_cm = float(np.mean(errors_cm))
    max_cm = float(np.max(errors_cm))
    std_cm = float(np.std(errors_cm))

    return {
        "rmse_cm": rmse_cm,
        "mean_error_cm": mean_cm,
        "max_error_cm": max_cm,
        "std_error_cm": std_cm,
        "rmse_mm": rmse_cm * 10.0,
        "mean_error_mm": mean_cm * 10.0,
        "max_error_mm": max_cm * 10.0,
        "std_error_mm": std_cm * 10.0,
        "point_count": int(len(errors_cm)),
    }


def calculate_reachability_percentage(requested_count: int, reached_count: int) -> float:
    """Calculate the percentage of requested trajectory points successfully reached."""
    if (
        isinstance(requested_count, bool)
        or isinstance(reached_count, bool)
        or not isinstance(requested_count, int)
        or not isinstance(reached_count, int)
    ):
        raise ValueError("Point counts must be integers.")

    if requested_count <= 0:
        raise ValueError("Requested count must be positive.")

    if not 0 <= reached_count <= requested_count:
        raise ValueError("Reached count must be between zero and requested count.")

    return 100.0 * reached_count / requested_count


def calculate_joint_angle_rmse(target_angles: np.ndarray, actual_angles: np.ndarray) -> float:
    """
    Calculate joint-angle RMSE across trajectory in degrees.

    Arrays must have matching shape (N, 2) in degrees.
    """
    target = np.asarray(target_angles, dtype=float)
    actual = np.asarray(actual_angles, dtype=float)

    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("Target angles must have shape (N, 2).")

    if actual.shape != target.shape or len(target) == 0:
        raise ValueError("Angle arrays must have matching, non-empty shapes.")

    if not np.isfinite(target).all() or not np.isfinite(actual).all():
        raise ValueError("Angles must be finite.")

    errors = actual - target
    return float(math.sqrt(np.mean(errors ** 2)))


# ==============================================================================
# 2. Stroke Endpoint Errors
# ==============================================================================

def endpoint_errors(
    target_strokes: List[np.ndarray],
    actual_strokes: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute start-point and end-point displacement errors across strokes.

    Parameters:
        target_strokes: List of (N_i, 2) arrays in cm for each stroke.
        actual_strokes: List of (M_i, 2) arrays in cm for corresponding strokes.

    Returns:
        Dictionary with mean and max endpoint errors in mm and cm.
    """
    if len(target_strokes) == 0 or len(actual_strokes) == 0:
        raise ValueError("Stroke lists must not be empty.")

    if len(target_strokes) != len(actual_strokes):
        raise ValueError("Target and actual stroke lists must have identical lengths.")

    start_errors_cm = []
    end_errors_cm = []

    for tgt, act in zip(target_strokes, actual_strokes):
        tgt_arr = np.asarray(tgt, dtype=float)
        act_arr = np.asarray(act, dtype=float)

        if len(tgt_arr) == 0 or len(act_arr) == 0:
            raise ValueError("Stroke points cannot be empty.")

        err_start = float(np.linalg.norm(act_arr[0] - tgt_arr[0]))
        err_end = float(np.linalg.norm(act_arr[-1] - tgt_arr[-1]))

        start_errors_cm.append(err_start)
        end_errors_cm.append(err_end)

    all_endpoint_errors_cm = start_errors_cm + end_errors_cm

    return {
        "mean_start_error_cm": float(np.mean(start_errors_cm)),
        "max_start_error_cm": float(np.max(start_errors_cm)),
        "mean_end_error_cm": float(np.mean(end_errors_cm)),
        "max_end_error_cm": float(np.max(end_errors_cm)),
        "mean_endpoint_error_cm": float(np.mean(all_endpoint_errors_cm)),
        "max_endpoint_error_cm": float(np.max(all_endpoint_errors_cm)),
        "mean_endpoint_error_mm": float(np.mean(all_endpoint_errors_cm) * 10.0),
        "max_endpoint_error_mm": float(np.max(all_endpoint_errors_cm) * 10.0),
        "stroke_count": len(target_strokes),
    }


# ==============================================================================
# 3. Point-Set Chamfer Distance (Metric cm / mm)
# ==============================================================================

def point_set_chamfer_distance(
    set_a: np.ndarray,
    set_b: np.ndarray,
) -> Dict[str, float]:
    """
    Compute symmetric Chamfer distance and Hausdorff distance between two 2D point clouds.

    d_CD(A, B) = 0.5 * [ (1/|A|) sum_{a in A} min_{b in B} ||a - b|| + (1/|B|) sum_{b in B} min_{a in A} ||b - a|| ]

    Parameters:
        set_a: (N, 2) array of coordinates in cm.
        set_b: (M, 2) array of coordinates in cm.

    Returns:
        Dictionary containing forward, backward, symmetric Chamfer, and Hausdorff distances in cm and mm.
    """
    pts_a = np.asarray(set_a, dtype=float)
    pts_b = np.asarray(set_b, dtype=float)

    if pts_a.size == 0 or pts_b.size == 0:
        raise ValueError("Point sets must contain at least one point.")

    if pts_a.ndim != 2 or pts_a.shape[1] != 2 or pts_b.ndim != 2 or pts_b.shape[1] != 2:
        raise ValueError("Point sets must have shape (N, 2).")

    tree_b = KDTree(pts_b)
    dist_a_to_b, _ = tree_b.query(pts_a)

    tree_a = KDTree(pts_a)
    dist_b_to_a, _ = tree_a.query(pts_b)

    fwd_chamfer_cm = float(np.mean(dist_a_to_b))
    bwd_chamfer_cm = float(np.mean(dist_b_to_a))
    sym_chamfer_cm = 0.5 * (fwd_chamfer_cm + bwd_chamfer_cm)
    hausdorff_cm = float(max(np.max(dist_a_to_b), np.max(dist_b_to_a)))

    return {
        "forward_chamfer_cm": fwd_chamfer_cm,
        "backward_chamfer_cm": bwd_chamfer_cm,
        "symmetric_chamfer_cm": sym_chamfer_cm,
        "hausdorff_cm": hausdorff_cm,
        "forward_chamfer_mm": fwd_chamfer_cm * 10.0,
        "backward_chamfer_mm": bwd_chamfer_cm * 10.0,
        "symmetric_chamfer_mm": sym_chamfer_cm * 10.0,
        "hausdorff_mm": hausdorff_cm * 10.0,
    }


def point_to_strokes_distance(
    points: np.ndarray,
    strokes: List[np.ndarray],
) -> np.ndarray:
    """
    Calculate minimum Euclidean distance from each 2D point to the nearest line segment
    of a collection of polyline strokes.

    Parameters:
        points: (N, 2) array of query coordinates.
        strokes: List of (M_i, 2) arrays of stroke vertex coordinates.

    Returns:
        1D array of length N containing distances to the closest continuous stroke segment.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty(0, dtype=float)

    min_dists = np.full(len(pts), np.inf)

    for stroke in strokes:
        s = np.asarray(stroke, dtype=float)
        if len(s) == 0:
            continue
        if len(s) == 1:
            d = np.linalg.norm(pts - s[0], axis=1)
            min_dists = np.minimum(min_dists, d)
            continue

        for i in range(len(s) - 1):
            a, b = s[i], s[i + 1]
            ab = b - a
            l2 = float(np.sum(ab ** 2))
            if l2 == 0.0:
                d = np.linalg.norm(pts - a, axis=1)
            else:
                t = np.clip(np.sum((pts - a) * ab, axis=1) / l2, 0.0, 1.0)
                proj = a + t[:, None] * ab
                d = np.linalg.norm(pts - proj, axis=1)
            min_dists = np.minimum(min_dists, d)

    return min_dists


def stroke_contour_chamfer_distance(
    target_strokes: List[np.ndarray],
    path_points: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate continuous curve-aware Chamfer distance between target stroke polylines and path waypoints.

    - Forward distance (path -> target strokes): orthogonal distance from each path point
      to the nearest target stroke segment.
    - Backward distance (target vertices -> path): Euclidean distance from each target
      stroke vertex to the nearest path point.
    - Symmetric Chamfer: 0.5 * (mean(forward) + mean(backward)).

    Parameters:
        target_strokes: List of polyline strokes in cm.
        path_points: (N, 2) array of path waypoints in cm.

    Returns:
        Dictionary with forward, backward, symmetric Chamfer and Hausdorff distances in cm and mm.
    """
    pts = np.asarray(path_points, dtype=float)
    if pts.size == 0 or len(target_strokes) == 0:
        raise ValueError("Inputs cannot be empty.")

    # Forward distance: each path point to the target polyline segments
    fwd_dists_cm = point_to_strokes_distance(pts, target_strokes)

    # Backward distance: each target vertex to the path points
    all_target_vertices = np.vstack(target_strokes)
    tree_path = KDTree(pts)
    bwd_dists_cm, _ = tree_path.query(all_target_vertices)

    fwd_chamfer_cm = float(np.mean(fwd_dists_cm))
    bwd_chamfer_cm = float(np.mean(bwd_dists_cm))
    sym_chamfer_cm = 0.5 * (fwd_chamfer_cm + bwd_chamfer_cm)
    hausdorff_cm = float(max(np.max(fwd_dists_cm), np.max(bwd_dists_cm)))

    return {
        "forward_chamfer_cm": fwd_chamfer_cm,
        "backward_chamfer_cm": bwd_chamfer_cm,
        "symmetric_chamfer_cm": sym_chamfer_cm,
        "hausdorff_cm": hausdorff_cm,
        "forward_chamfer_mm": fwd_chamfer_cm * 10.0,
        "backward_chamfer_mm": bwd_chamfer_cm * 10.0,
        "symmetric_chamfer_mm": sym_chamfer_cm * 10.0,
        "hausdorff_mm": hausdorff_cm * 10.0,
    }



# ==============================================================================
# 4. Trajectory Smoothness & Dynamics Metrics
# ==============================================================================

def trajectory_smoothness(
    trajectory_points: np.ndarray,
    dt: float = 0.05,
) -> Dict[str, float]:
    """
    Compute kinematic smoothness indicators: jerk, acceleration variance, and dimensionless jerk.

    Parameters:
        trajectory_points: (N, 2) array of trajectory positions in cm.
        dt: Time interval between consecutive steps in seconds.

    Returns:
        Dictionary with RMS jerk, mean jerk, acceleration variance, and dimensionless jerk.
    """
    pts = np.asarray(trajectory_points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Trajectory points must have shape (N, 2).")

    if len(pts) < 4:
        raise ValueError("Smoothness evaluation requires at least 4 trajectory points.")

    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")

    # Numerical differentiation
    vel = np.diff(pts, axis=0) / dt                 # (N-1, 2) in cm/s
    acc = np.diff(vel, axis=0) / dt                 # (N-2, 2) in cm/s^2
    jerk = np.diff(acc, axis=0) / dt                # (N-3, 2) in cm/s^3

    acc_norms = np.linalg.norm(acc, axis=1)
    jerk_norms = np.linalg.norm(jerk, axis=1)

    rms_jerk = float(np.sqrt(np.mean(jerk_norms ** 2)))
    mean_jerk = float(np.mean(jerk_norms))
    max_jerk = float(np.max(jerk_norms))

    rms_acc = float(np.sqrt(np.mean(acc_norms ** 2)))
    acc_variance = float(np.var(acc_norms))
    mean_acc = float(np.mean(acc_norms))

    # Total duration and path length
    total_time = len(pts) * dt
    path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    # Dimensionless jerk: (integral ||j||^2 dt) * (T^5 / L^2)
    # Scaled to prevent division by zero on static paths
    if path_length > 1e-6:
        integrated_squared_jerk = float(np.sum(jerk_norms ** 2) * dt)
        dimensionless_jerk = integrated_squared_jerk * (total_time ** 5) / (path_length ** 2)
    else:
        dimensionless_jerk = 0.0

    return {
        "rms_jerk_cm_s3": rms_jerk,
        "mean_jerk_cm_s3": mean_jerk,
        "max_jerk_cm_s3": max_jerk,
        "rms_acceleration_cm_s2": rms_acc,
        "acceleration_variance_cm_s2": acc_variance,
        "mean_acceleration_cm_s2": mean_acc,
        "dimensionless_jerk": float(dimensionless_jerk),
        "total_time_s": total_time,
        "path_length_cm": path_length,
    }


# ==============================================================================
# 5. Pen Travel Distance & Efficiency
# ==============================================================================

def pen_travel_metrics(
    positions: np.ndarray,
    pen_states: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate pen-down drawing distance, pen-up repositioning distance, and travel efficiency ratio.

    Parameters:
        positions: (N, 2) array of positions in cm.
        pen_states: 1D array of length N (1 for pen-down, 0 for pen-up).

    Returns:
        Dictionary with pen-down distance, pen-up distance, total distance, and efficiency ratio (%).
    """
    pts = np.asarray(positions, dtype=float)
    states = np.asarray(pen_states, dtype=int)

    if pts.shape[0] != states.shape[0]:
        raise ValueError("Positions and pen states must have matching lengths.")

    if len(pts) < 2:
        return {
            "pen_down_distance_cm": 0.0,
            "pen_up_distance_cm": 0.0,
            "total_distance_cm": 0.0,
            "pen_travel_efficiency_pct": 0.0,
        }

    pen_down_dist = 0.0
    pen_up_dist = 0.0

    for i in range(1, len(pts)):
        seg_dist = float(np.linalg.norm(pts[i] - pts[i - 1]))
        if states[i] == 1:
            pen_down_dist += seg_dist
        else:
            pen_up_dist += seg_dist

    total_dist = pen_down_dist + pen_up_dist
    efficiency = (pen_down_dist / total_dist * 100.0) if total_dist > 0.0 else 0.0

    return {
        "pen_down_distance_cm": pen_down_dist,
        "pen_up_distance_cm": pen_up_dist,
        "total_distance_cm": total_dist,
        "pen_travel_efficiency_pct": efficiency,
    }


# ==============================================================================
# 6. Decomposed Error Breakdown (Planning vs Execution vs Total)
# ==============================================================================

def decompose_tracking_errors(
    target_strokes_arm: List[np.ndarray],
    plan_positions_cm: np.ndarray,
    measured_positions_cm: np.ndarray,
    pen_states: np.ndarray,
    stroke_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Decompose end-to-end tracing error into Planning Error (Target vs Plan) and Execution Error (Plan vs Measured).

    Parameters:
        target_strokes_arm: List of target strokes in cm (from contour JSON mapped to arm space).
        plan_positions_cm: (N, 2) planned waypoints across trajectory in cm.
        measured_positions_cm: (N, 2) measured arm tip positions in cm.
        pen_states: (N,) binary pen states (1=down, 0=up).
        stroke_ids: (N,) stroke index corresponding to each trajectory step.

    Returns:
        Structured dictionary separating planning, execution, and total tracking errors.
    """
    plan = np.asarray(plan_positions_cm, dtype=float)
    meas = np.asarray(measured_positions_cm, dtype=float)
    states = np.asarray(pen_states, dtype=int)

    # Filter to pen-down segments
    down_mask = (states == 1)
    plan_down = plan[down_mask]
    meas_down = meas[down_mask]

    # Combine target stroke points
    all_target_pts = np.vstack(target_strokes_arm)

    # 1. Execution Error (Plan vs Measured at corresponding steps)
    exec_metrics = calculate_trajectory_metrics(plan_down, meas_down)
    exec_chamfer = point_set_chamfer_distance(plan_down, meas_down)

    # 2. Planning Error (Target Strokes vs Planned Pen-Down Waypoints)
    plan_contour_chamfer = stroke_contour_chamfer_distance(target_strokes_arm, plan_down)
    plan_pointset_chamfer = point_set_chamfer_distance(all_target_pts, plan_down)

    # Target stroke endpoint error vs planned stroke endpoints
    # Extract planned strokes from trajectory by stroke_id if available
    plan_stroke_endpoints_tgt = []
    plan_stroke_endpoints_act = []
    meas_stroke_endpoints_act = []

    if stroke_ids is not None:
        all_s_ids = np.asarray(stroke_ids)
        if len(all_s_ids) == len(plan):
            unique_strokes = sorted(list(set(all_s_ids)))
            for s_idx in unique_strokes:
                mask_s = (all_s_ids == s_idx)
                pts_s_plan = plan[mask_s]
                pts_s_meas = meas[mask_s]

                if s_idx < len(target_strokes_arm) and len(pts_s_plan) >= 2:
                    plan_stroke_endpoints_tgt.append(target_strokes_arm[s_idx])
                    plan_stroke_endpoints_act.append(pts_s_plan)
                    meas_stroke_endpoints_act.append(pts_s_meas)
        else:
            s_ids = all_s_ids[down_mask] if len(all_s_ids) == len(states) else all_s_ids
            unique_strokes = sorted(list(set(s_ids)))
            for s_idx in unique_strokes:
                mask_s = (s_ids == s_idx)
                pts_s_plan = plan_down[mask_s]
                pts_s_meas = meas_down[mask_s]

                if s_idx < len(target_strokes_arm) and len(pts_s_plan) >= 2:
                    plan_stroke_endpoints_tgt.append(target_strokes_arm[s_idx])
                    plan_stroke_endpoints_act.append(pts_s_plan)
                    meas_stroke_endpoints_act.append(pts_s_meas)

    plan_endpoint_err = (
        endpoint_errors(plan_stroke_endpoints_tgt, plan_stroke_endpoints_act)
        if plan_stroke_endpoints_tgt else {}
    )
    exec_endpoint_err = (
        endpoint_errors(plan_stroke_endpoints_act, meas_stroke_endpoints_act)
        if plan_stroke_endpoints_act else {}
    )
    total_endpoint_err = (
        endpoint_errors(plan_stroke_endpoints_tgt, meas_stroke_endpoints_act)
        if plan_stroke_endpoints_tgt else {}
    )

    # 3. Total Tracking Error (Target Strokes vs Measured Pen-Down Waypoints)
    total_contour_chamfer = stroke_contour_chamfer_distance(target_strokes_arm, meas_down)
    total_pointset_chamfer = point_set_chamfer_distance(all_target_pts, meas_down)

    return {
        "planning_error": {
            "chamfer_symmetric_mm": plan_contour_chamfer["symmetric_chamfer_mm"],
            "chamfer_forward_mm": plan_contour_chamfer["forward_chamfer_mm"],
            "chamfer_backward_mm": plan_contour_chamfer["backward_chamfer_mm"],
            "pointset_chamfer_symmetric_mm": plan_pointset_chamfer["symmetric_chamfer_mm"],
            "hausdorff_mm": plan_contour_chamfer["hausdorff_mm"],
            "mean_endpoint_error_mm": plan_endpoint_err.get("mean_endpoint_error_mm", 0.0),
            "max_endpoint_error_mm": plan_endpoint_err.get("max_endpoint_error_mm", 0.0),
        },
        "execution_error": {
            "rmse_mm": exec_metrics["rmse_mm"],
            "mean_error_mm": exec_metrics["mean_error_mm"],
            "max_error_mm": exec_metrics["max_error_mm"],
            "std_error_mm": exec_metrics["std_error_mm"],
            "chamfer_symmetric_mm": exec_chamfer["symmetric_chamfer_mm"],
            "mean_endpoint_error_mm": exec_endpoint_err.get("mean_endpoint_error_mm", 0.0),
            "max_endpoint_error_mm": exec_endpoint_err.get("max_endpoint_error_mm", 0.0),
        },
        "total_error": {
            "chamfer_symmetric_mm": total_contour_chamfer["symmetric_chamfer_mm"],
            "chamfer_forward_mm": total_contour_chamfer["forward_chamfer_mm"],
            "chamfer_backward_mm": total_contour_chamfer["backward_chamfer_mm"],
            "pointset_chamfer_symmetric_mm": total_pointset_chamfer["symmetric_chamfer_mm"],
            "hausdorff_mm": total_contour_chamfer["hausdorff_mm"],
            "mean_endpoint_error_mm": total_endpoint_err.get("mean_endpoint_error_mm", 0.0),
            "max_endpoint_error_mm": total_endpoint_err.get("max_endpoint_error_mm", 0.0),
        },
    }


# ==============================================================================
# 7. Vision Sensor Image & Raster Metrics
# ==============================================================================

def arm_to_camera_pixel(
    x_cm: float,
    y_cm: float,
    camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
) -> Tuple[int, int]:
    """
    Project workspace coordinate (x_cm, y_cm) to camera image pixel (u, v).

    With camera orientation [0.0, pi, 0.0]:
    - u increases with world +X (right in image)
    - v decreases with world +Y (up in image, toward row 0)
    """
    scale_px_per_cm = camera_config.resolution_x / camera_config.view_size_cm
    center_u = camera_config.resolution_x / 2.0
    center_v = camera_config.resolution_y / 2.0

    u = center_u + (x_cm - camera_config.center_x_cm) * scale_px_per_cm
    v = center_v - (y_cm - camera_config.center_y_cm) * scale_px_per_cm

    return int(round(u)), int(round(v))


def camera_pixel_to_arm(
    u: float,
    v: float,
    camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
) -> Tuple[float, float]:
    """Convert camera pixel (u, v) back to robot workspace coordinate (x_cm, y_cm)."""
    scale_cm_per_px = camera_config.view_size_cm / camera_config.resolution_x
    center_u = camera_config.resolution_x / 2.0
    center_v = camera_config.resolution_y / 2.0

    x_cm = camera_config.center_x_cm + (u - center_u) * scale_cm_per_px
    y_cm = camera_config.center_y_cm - (v - center_v) * scale_cm_per_px

    return x_cm, y_cm


def rasterize_strokes_to_camera_mask(
    strokes_cm: List[np.ndarray],
    camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
    line_thickness_px: int = 2,
) -> np.ndarray:
    """
    Rasterize workspace strokes into a binary image matching the top-down camera sensor.

    Returns:
        Binary mask (uint8) of shape (H, W) where ink pixels have value 255.
    """
    mask = np.zeros((camera_config.resolution_y, camera_config.resolution_x), dtype=np.uint8)

    for stroke in strokes_cm:
        pts = np.asarray(stroke, dtype=float)
        if len(pts) < 2:
            continue

        cam_pts = np.array(
            [arm_to_camera_pixel(p[0], p[1], camera_config) for p in pts],
            dtype=np.int32,
        )
        cv2.polylines(mask, [cam_pts], isClosed=False, color=255, thickness=line_thickness_px)

    return mask


def extract_ink_mask_from_camera_image(
    image: np.ndarray,
    is_bgr: bool = True,
) -> np.ndarray:
    """
    Extract the visible ink trace binary mask from a TopDownVisionSensor capture.

    The ink is rendered in dark blue (RGB ~ [0.05, 0.15, 0.70]) on white paper.

    Returns:
        Binary mask (uint8) of shape (H, W) where ink pixels have value 255.
    """
    img = np.asarray(image, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must have shape (H, W, 3).")

    if is_bgr:
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    else:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Blue ink criterion: high blue relative to red/green, and red/green significantly lower
    ink_mask = (b > 100) & (r < 70) & (g < 100)

    return (ink_mask.astype(np.uint8)) * 255


def raster_iou(
    mask_target: np.ndarray,
    mask_observed: np.ndarray,
    tolerance_px: int = 2,
) -> Dict[str, float]:
    """
    Calculate raster Intersection-over-Union (IoU), Dice coefficient, Precision, and Recall.

    Supports tolerance-matched IoU by dilating masks with a circular structuring element
    of radius `tolerance_px` (default 2 pixels ~ 0.47 mm).

    Parameters:
        mask_target: Binary mask of target strokes (nonzero = foreground).
        mask_observed: Binary mask of observed drawn strokes.
        tolerance_px: Morphological dilation radius in pixels for tolerance matching.

    Returns:
        Dictionary with headline tolerance IoU, secondary plain IoU, Dice, Precision, and Recall.
    """
    m_tgt = np.asarray(mask_target) > 0
    m_obs = np.asarray(mask_observed) > 0

    if m_tgt.shape != m_obs.shape:
        raise ValueError("Target and observed masks must have matching dimensions.")

    # 1. Plain (exact pixel) overlap
    intersection = int(np.logical_and(m_tgt, m_obs).sum())
    union = int(np.logical_or(m_tgt, m_obs).sum())
    tgt_count = int(m_tgt.sum())
    obs_count = int(m_obs.sum())

    plain_iou = float(intersection / union) if union > 0 else 1.0 if (tgt_count == 0 and obs_count == 0) else 0.0
    plain_dice = float(2.0 * intersection / (tgt_count + obs_count)) if (tgt_count + obs_count) > 0 else 1.0
    plain_precision = float(intersection / obs_count) if obs_count > 0 else 0.0
    plain_recall = float(intersection / tgt_count) if tgt_count > 0 else 0.0

    # 2. Tolerance-matched overlap (masks dilated by ~tolerance_px)
    if tolerance_px > 0 and (tgt_count > 0 or obs_count > 0):
        k_size = 2 * tolerance_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        d_tgt = cv2.dilate(m_tgt.astype(np.uint8), kernel) > 0
        d_obs = cv2.dilate(m_obs.astype(np.uint8), kernel) > 0

        tol_intersection = int(np.logical_and(d_tgt, d_obs).sum())
        tol_union = int(np.logical_or(d_tgt, d_obs).sum())
        tol_tgt_count = int(d_tgt.sum())
        tol_obs_count = int(d_obs.sum())

        tol_iou = float(tol_intersection / tol_union) if tol_union > 0 else 1.0
        tol_dice = float(2.0 * tol_intersection / (tol_tgt_count + tol_obs_count)) if (tol_tgt_count + tol_obs_count) > 0 else 1.0
        tol_precision = float(tol_intersection / tol_obs_count) if tol_obs_count > 0 else 0.0
        tol_recall = float(tol_intersection / tol_tgt_count) if tol_tgt_count > 0 else 0.0
    else:
        tol_iou = plain_iou
        tol_dice = plain_dice
        tol_precision = plain_precision
        tol_recall = plain_recall
        tol_intersection = intersection
        tol_union = union

    return {
        # Headline tolerance-matched metrics
        "tolerance_iou": tol_iou,
        "tolerance_iou_pct": tol_iou * 100.0,
        "tolerance_dice": tol_dice,
        "tolerance_precision": tol_precision,
        "tolerance_recall": tol_recall,
        "tolerance_px": tolerance_px,
        # Secondary plain metrics
        "plain_iou": plain_iou,
        "plain_iou_pct": plain_iou * 100.0,
        "plain_dice": plain_dice,
        "plain_precision": plain_precision,
        "plain_recall": plain_recall,
        # Backward-compatible aliases
        "iou": plain_iou,
        "iou_pct": plain_iou * 100.0,
        "dice": plain_dice,
        "precision": plain_precision,
        "recall": plain_recall,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "target_pixels": tgt_count,
        "observed_pixels": obs_count,
        "tolerance_intersection_pixels": tol_intersection,
        "tolerance_union_pixels": tol_union,
    }


def raster_chamfer_and_contour_error(
    mask_target: np.ndarray,
    mask_observed: np.ndarray,
    pixel_size_mm: float = DEFAULT_EVAL_CONFIG.pixel_size_mm,
) -> Dict[str, float]:
    """
    Compute raster Chamfer distance and contour tracking error using Euclidean distance transforms.

    Parameters:
        mask_target: Binary mask of target shape (255=foreground, 0=background).
        mask_observed: Binary mask of observed drawn trace.
        pixel_size_mm: Millimetres per pixel for metric conversion.

    Returns:
        Dictionary containing forward, backward, symmetric Chamfer and contour error statistics in mm.
    """
    m_tgt = (np.asarray(mask_target) > 0).astype(np.uint8) * 255
    m_obs = (np.asarray(mask_observed) > 0).astype(np.uint8) * 255

    if m_tgt.shape != m_obs.shape:
        raise ValueError("Mask shapes must match.")

    obs_pixels = np.where(m_obs > 0)
    tgt_pixels = np.where(m_tgt > 0)

    if len(obs_pixels[0]) == 0 or len(tgt_pixels[0]) == 0:
        raise ValueError("Both masks must contain at least one foreground pixel.")

    # Distance to target mask for each observed pixel (exact Euclidean distance)
    dist_to_tgt = cv2.distanceTransform(255 - m_tgt, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    fwd_errors_px = dist_to_tgt[obs_pixels]
    fwd_errors_mm = fwd_errors_px * pixel_size_mm

    # Distance to observed mask for each target pixel (exact Euclidean distance)
    dist_to_obs = cv2.distanceTransform(255 - m_obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    bwd_errors_px = dist_to_obs[tgt_pixels]
    bwd_errors_mm = bwd_errors_px * pixel_size_mm

    mean_fwd_mm = float(np.mean(fwd_errors_mm))
    rmse_fwd_mm = float(np.sqrt(np.mean(fwd_errors_mm ** 2)))
    max_fwd_mm = float(np.max(fwd_errors_mm))
    p95_fwd_mm = float(np.percentile(fwd_errors_mm, 95))

    mean_bwd_mm = float(np.mean(bwd_errors_mm))
    rmse_bwd_mm = float(np.sqrt(np.mean(bwd_errors_mm ** 2)))
    max_bwd_mm = float(np.max(bwd_errors_mm))

    symmetric_chamfer_mm = 0.5 * (mean_fwd_mm + mean_bwd_mm)

    return {
        "contour_error_mean_mm": mean_fwd_mm,
        "contour_error_rmse_mm": rmse_fwd_mm,
        "contour_error_max_mm": max_fwd_mm,
        "contour_error_p95_mm": p95_fwd_mm,
        "backward_chamfer_mean_mm": mean_bwd_mm,
        "backward_chamfer_rmse_mm": rmse_bwd_mm,
        "symmetric_chamfer_mm": symmetric_chamfer_mm,
        "pixel_size_mm": pixel_size_mm,
    }


# ==============================================================================
# 8. Unified Full-Run Evaluation
# ==============================================================================

def evaluate_full_run(
    target_strokes_arm: List[np.ndarray],
    plan_positions_cm: np.ndarray,
    measured_positions_cm: np.ndarray,
    pen_states: np.ndarray,
    stroke_ids: np.ndarray,
    camera_image_bgr: Optional[np.ndarray] = None,
    dt_seconds: float = DEFAULT_EVAL_CONFIG.dt_seconds,
    camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
) -> Dict[str, Any]:
    """
    Perform a complete, multi-tiered evaluation of a drawing execution run.

    Computes:
    - Decomposed errors: planning error vs execution error vs total error
    - Trajectory smoothness: RMS jerk, acceleration variance
    - Pen travel: pen-down distance, pen-up distance, efficiency
    - Vision sensor metrics (if camera image is provided): IoU, Dice, Chamfer, contour error.

    Returns:
        Unified results dictionary.
    """
    decomposed = decompose_tracking_errors(
        target_strokes_arm=target_strokes_arm,
        plan_positions_cm=plan_positions_cm,
        measured_positions_cm=measured_positions_cm,
        pen_states=pen_states,
        stroke_ids=stroke_ids,
    )

    travel = pen_travel_metrics(positions=measured_positions_cm, pen_states=pen_states)
    smoothness = trajectory_smoothness(trajectory_points=measured_positions_cm, dt=dt_seconds)

    vision_metrics: Dict[str, Any] = {}
    if camera_image_bgr is not None:
        target_mask = rasterize_strokes_to_camera_mask(
            strokes_cm=target_strokes_arm,
            camera_config=camera_config,
            line_thickness_px=camera_config.line_thickness_px,
        )
        obs_mask = extract_ink_mask_from_camera_image(camera_image_bgr, is_bgr=True)
        iou_res = raster_iou(target_mask, obs_mask)
        chamfer_res = raster_chamfer_and_contour_error(
            target_mask, obs_mask, pixel_size_mm=camera_config.view_size_cm * 10.0 / camera_config.resolution_x
        )
        vision_metrics = {**iou_res, **chamfer_res}

    return {
        "decomposition": decomposed,
        "travel": travel,
        "smoothness": smoothness,
        "vision": vision_metrics,
    }
