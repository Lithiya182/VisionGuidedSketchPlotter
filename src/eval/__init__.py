"""
src/eval

Evaluation and metrics module for VisionGuidedSketchPlotter.
"""

from src.eval.metrics import (
    pointwise_tracking_errors,
    calculate_trajectory_metrics,
    calculate_reachability_percentage,
    calculate_joint_angle_rmse,
    endpoint_errors,
    point_set_chamfer_distance,
    trajectory_smoothness,
    pen_travel_metrics,
    decompose_tracking_errors,
    rasterize_strokes_to_camera_mask,
    extract_ink_mask_from_camera_image,
    raster_iou,
    raster_chamfer_and_contour_error,
    evaluate_full_run,
)

__all__ = [
    "pointwise_tracking_errors",
    "calculate_trajectory_metrics",
    "calculate_reachability_percentage",
    "calculate_joint_angle_rmse",
    "endpoint_errors",
    "point_set_chamfer_distance",
    "trajectory_smoothness",
    "pen_travel_metrics",
    "decompose_tracking_errors",
    "rasterize_strokes_to_camera_mask",
    "extract_ink_mask_from_camera_image",
    "raster_iou",
    "raster_chamfer_and_contour_error",
    "evaluate_full_run",
]
