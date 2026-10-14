"""
trajectory_metrics.py

Evaluation metrics for the Vision-Guided Sketch Plotter.
Maintained for backward compatibility; canonical metrics reside in src.eval.metrics.
"""

from src.eval.metrics import (
    trajectory_errors,
    calculate_trajectory_metrics,
    calculate_reachability_percentage,
    calculate_joint_angle_rmse,
    pointwise_tracking_errors,
    endpoint_errors,
    point_set_chamfer_distance,
    trajectory_smoothness,
    pen_travel_metrics,
    decompose_tracking_errors,
    raster_iou,
    raster_chamfer_and_contour_error,
)

__all__ = [
    "trajectory_errors",
    "calculate_trajectory_metrics",
    "calculate_reachability_percentage",
    "calculate_joint_angle_rmse",
    "pointwise_tracking_errors",
    "endpoint_errors",
    "point_set_chamfer_distance",
    "trajectory_smoothness",
    "pen_travel_metrics",
    "decompose_tracking_errors",
    "raster_iou",
    "raster_chamfer_and_contour_error",
]
