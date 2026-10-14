
"""
coordinate_transform.py

Maps image pixel coordinates to the planar robot workspace
and validates targets against geometric and joint constraints.
"""

import math
import numpy as np

from src.kinematics import inverse_kinematics, L1, L2


# Image dimensions (pixels)
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 480

# Drawing area in robot coordinates (cm).
# Origin is the TOP-LEFT corner of the drawing rectangle.
DRAW_ORIGIN_X = 4.0
DRAW_ORIGIN_Y = 12.5
DRAW_WIDTH_CM = 6.0
DRAW_HEIGHT_CM = 6.0

# Safety margins from singular configurations (cm).
MIN_RADIUS_CM = 2.0
MAX_RADIUS_MARGIN_CM = 2.0

# Simulated joint limits (degrees).
# These are provisional simulation constraints, not hardware specifications.
THETA1_LIMITS_DEG = (-90.0, 90.0)
THETA2_LIMITS_DEG = (-150.0, 150.0)


def image_to_arm_space(
    points_px,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    origin_x=DRAW_ORIGIN_X,
    origin_y=DRAW_ORIGIN_Y,
    draw_width=DRAW_WIDTH_CM,
    draw_height=DRAW_HEIGHT_CM,
):
    """
    Convert an ordered sequence of image (x, y) pixel coordinates
    into robot workspace coordinates in centimetres.

    Image y increases downward; robot y increases upward.

    Returns:
        NumPy array with shape (N, 2).
    """
    points = np.asarray(points_px, dtype=float)

    if points.size == 0:
        raise ValueError("At least one coordinate is required.")

    if points.ndim == 1:
        if points.shape[0] != 2:
            raise ValueError("A single point must contain x and y.")
        points = points.reshape(1, 2)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Coordinates must have exactly two columns.")

    if not np.isfinite(points).all():
        raise ValueError("Coordinates must be finite.")

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if draw_width <= 0 or draw_height <= 0:
        raise ValueError("Drawing dimensions must be positive.")

    if np.any(points[:, 0] < 0) or np.any(points[:, 0] > image_width):
        raise ValueError("An x-coordinate lies outside the image.")

    if np.any(points[:, 1] < 0) or np.any(points[:, 1] > image_height):
        raise ValueError("A y-coordinate lies outside the image.")

    scale_x = draw_width / image_width
    scale_y = draw_height / image_height

    arm_x = origin_x + points[:, 0] * scale_x
    arm_y = origin_y - points[:, 1] * scale_y

    return np.column_stack((arm_x, arm_y))


def validate_workspace_point(
    x,
    y,
    min_radius=MIN_RADIUS_CM,
    max_radius_margin=MAX_RADIUS_MARGIN_CM,
    l1=L1,
    l2=L2,
):
    """
    Validate radial reachability and avoid the fully folded
    and fully extended singular configurations.
    """
    if not all(math.isfinite(v) for v in (x, y, min_radius, max_radius_margin)):
        raise ValueError("Workspace values must be finite.")

    if min_radius < 0 or max_radius_margin < 0:
        raise ValueError("Safety margins cannot be negative.")

    radius = math.hypot(x, y)

    minimum = max(abs(l1 - l2), min_radius)
    maximum = l1 + l2 - max_radius_margin

    if radius < minimum or radius > maximum:
        raise ValueError(
            f"Point ({x:.2f}, {y:.2f}) is outside the safe radial "
            f"workspace [{minimum:.2f}, {maximum:.2f}] cm."
        )

    return True


def validate_joint_configuration(x, y):
    """
    Check whether at least one of the configured elbow branches
    can reach the target within the simulated joint limits.
    """
    for branch in ("positive", "negative"):
        try:
            inverse_kinematics(
                x,
                y,
                branch=branch,
                theta1_limits_deg=THETA1_LIMITS_DEG,
                theta2_limits_deg=THETA2_LIMITS_DEG,
            )
            return True
        except ValueError:
            continue

    raise ValueError(
        f"No configured joint configuration can reach ({x:.2f}, {y:.2f})."
    )


def validate_drawing_path(points_arm):
    """
    Validate each point in an arm-space path.

    Returns:
        (valid_points, rejected_points)

    Note:
        This function validates points individually. It does not yet
        guarantee safe continuous motion between consecutive points.
    """
    valid = []
    rejected = []

    for point in points_arm:
        x, y = map(float, point)

        try:
            validate_workspace_point(x, y)
            validate_joint_configuration(x, y)
            valid.append((x, y))
        except ValueError as exc:
            rejected.append((x, y, str(exc)))

    return np.asarray(valid, dtype=float).reshape(-1, 2), rejected
