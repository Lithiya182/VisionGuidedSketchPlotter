
"""
trajectory_metrics.py

Evaluation metrics for the Vision-Guided Sketch Plotter.
"""

import math
import numpy as np



def trajectory_errors(target_points, actual_points):
    """
    Calculate pointwise Euclidean errors.

    Both arrays must contain corresponding (x, y) coordinates
    in centimetres and have shape (N, 2).
    """
    target = np.asarray(target_points, dtype=float)
    actual = np.asarray(actual_points, dtype=float)

    # Check empty input before validating array dimensions.
    if target.size == 0 or actual.size == 0:
        raise ValueError("At least one point is required.")

    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("Target points must have shape (N, 2).")

    if actual.shape != target.shape:
        raise ValueError(
            "Actual and target arrays must have identical shapes."
        )

    if not np.isfinite(target).all() or not np.isfinite(actual).all():
        raise ValueError("Coordinates must be finite.")

    return np.linalg.norm(actual - target, axis=1)



def calculate_trajectory_metrics(target_points, actual_points):
    """
    Return RMSE, mean error, and maximum error in centimetres.
    """
    errors = trajectory_errors(target_points, actual_points)

    return {
        "rmse_cm": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error_cm": float(np.mean(errors)),
        "max_error_cm": float(np.max(errors)),
        "point_count": int(len(errors)),
    }


def calculate_reachability_percentage(requested_count, reached_count):
    """Calculate the percentage of requested points reached."""
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


def calculate_joint_angle_rmse(target_angles, actual_angles):
    """
    Calculate joint-angle RMSE in degrees.

    Arrays must contain corresponding [theta1, theta2] values
    in degrees. Angles should be normalized consistently first.
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
