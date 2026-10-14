
"""
path_planner.py

Plans drawing strokes for a 2-DOF planar robotic arm.
Preserves contour boundaries and models pen-up / pen-down states.
"""

from dataclasses import dataclass
import math

import numpy as np

from src.kinematics import inverse_kinematics
from src.coordinate_transform import (
    validate_workspace_point,
    THETA1_LIMITS_DEG,
    THETA2_LIMITS_DEG,
)


@dataclass
class MotionPoint:
    """
    One planned robot configuration.

    pen_down specifies the pen state for the movement
    arriving at this point from the preceding point.
    """
    x: float
    y: float
    theta1: float
    theta2: float
    pen_down: bool
    stroke_id: int


def solve_target(x, y, branch="positive"):
    """Solve one target using the configured joint limits."""
    return inverse_kinematics(
        x,
        y,
        branch=branch,
        theta1_limits_deg=THETA1_LIMITS_DEG,
        theta2_limits_deg=THETA2_LIMITS_DEG,
    )


def interpolate_points(start, end, spacing_cm=0.5):
    """
    Linearly interpolate between two Cartesian points.

    Returns points including start and end.
    """
    if not math.isfinite(spacing_cm) or spacing_cm <= 0:
        raise ValueError("Spacing must be positive and finite.")

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("Start and end must each contain x and y.")

    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError("Points must contain finite coordinates.")

    distance = float(np.linalg.norm(end - start))
    segments = max(1, math.ceil(distance / spacing_cm))

    return np.linspace(start, end, segments + 1)


def plan_strokes(
    contours,
    branch="positive",
    spacing_cm=0.5,
    skip_invalid_strokes=False,
):
    """
    Convert independent contours into a motion sequence.

    Each contour must be an ordered iterable of (x, y) arm-space points.

    The first point of each stroke is approached with the pen up.
    The contour is then drawn with the pen down.
    Pen-up repositioning is represented explicitly by the first
    point of each new stroke.

    Returns:
        list[MotionPoint]

    If skip_invalid_strokes=True, invalid strokes are skipped entirely.
    Otherwise, an invalid point raises ValueError.
    """
    if branch not in ("positive", "negative"):
        raise ValueError("branch must be 'positive' or 'negative'.")

    if not math.isfinite(spacing_cm) or spacing_cm <= 0:
        raise ValueError("Spacing must be positive and finite.")

    motion = []

    for stroke_id, contour in enumerate(contours):
        points = np.asarray(contour, dtype=float)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"Stroke {stroke_id} must have shape (N, 2)."
            )

        if len(points) < 2:
            raise ValueError(
                f"Stroke {stroke_id} must contain at least two points."
            )

        if not np.isfinite(points).all():
            raise ValueError(
                f"Stroke {stroke_id} contains non-finite coordinates."
            )

        # Validate the entire stroke before adding any of its motion.
        stroke_motion = []

        try:
            for x, y in points:
                validate_workspace_point(float(x), float(y))
                solve_target(float(x), float(y), branch)

            # Approach the first point with pen up.
            x0, y0 = map(float, points[0])
            theta1, theta2 = solve_target(x0, y0, branch)

            stroke_motion.append(
                MotionPoint(x0, y0, theta1, theta2, False, stroke_id)
            )

            # Interpolate and draw each segment with pen down.
            for start, end in zip(points[:-1], points[1:]):
                samples = interpolate_points(start, end, spacing_cm)

                # The start is already represented by the previous segment.
                for point in samples[1:]:
                    x, y = map(float, point)

                    validate_workspace_point(x, y)
                    theta1, theta2 = solve_target(x, y, branch)

                    stroke_motion.append(
                        MotionPoint(x, y, theta1, theta2, True, stroke_id)
                    )

        except ValueError as exc:
            if skip_invalid_strokes:
                continue
            raise ValueError(
                f"Cannot plan stroke {stroke_id}: {exc}"
            ) from exc

        motion.extend(stroke_motion)

    return motion
