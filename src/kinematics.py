
"""
kinematics.py

Forward and inverse kinematics for a 2-DOF planar robotic arm.
All internal joint angles are represented in radians.
Joint-limit configuration is supplied in degrees.
"""

import math


# Default simulated robot geometry (centimeters)
L1 = 10.0
L2 = 10.0

# Numerical tolerance for floating-point calculations
TOLERANCE = 1e-9


def forward_kinematics(theta1, theta2, l1=L1, l2=L2):
    """
    Calculate the elbow and end-effector positions.

    Parameters:
        theta1: Base joint angle in radians.
        theta2: Relative elbow joint angle in radians.
        l1: First link length in cm.
        l2: Second link length in cm.

    Returns:
        (x, y, (elbow_x, elbow_y))
    """
    if l1 <= 0 or l2 <= 0:
        raise ValueError("Link lengths must be positive.")

    elbow_x = l1 * math.cos(theta1)
    elbow_y = l1 * math.sin(theta1)

    x = elbow_x + l2 * math.cos(theta1 + theta2)
    y = elbow_y + l2 * math.sin(theta1 + theta2)

    return x, y, (elbow_x, elbow_y)


def inverse_kinematics(
    x,
    y,
    l1=L1,
    l2=L2,
    branch="positive",
    theta1_limits_deg=(-180.0, 180.0),
    theta2_limits_deg=(-180.0, 180.0),
):
    """
    Calculate joint angles for a target Cartesian position.

    branch:
        "positive" -> positive theta2 solution
        "negative" -> negative theta2 solution

    Raises ValueError if the target is unreachable or the calculated
    angles violate the configured joint limits.
    """
    if l1 <= 0 or l2 <= 0:
        raise ValueError("Link lengths must be positive.")

    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Target coordinates must be finite.")

    if branch not in ("positive", "negative"):
        raise ValueError(
            "branch must be 'positive' or 'negative'."
        )

    distance_squared = x * x + y * y

    cos_theta2 = (
        distance_squared - l1 * l1 - l2 * l2
    ) / (2.0 * l1 * l2)

    # Allow tiny numerical errors at the workspace boundary.
    if cos_theta2 < -1.0 - TOLERANCE or cos_theta2 > 1.0 + TOLERANCE:
        raise ValueError(
            f"Target ({x:.3f}, {y:.3f}) is unreachable. "
            f"Reachable radial range: "
            f"{abs(l1 - l2):.3f} to {l1 + l2:.3f} cm."
        )

    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    sign = 1.0 if branch == "positive" else -1.0

    sin_magnitude = math.sqrt(
        max(0.0, 1.0 - cos_theta2 * cos_theta2)
    )

    theta2 = math.atan2(
        sign * sin_magnitude,
        cos_theta2,
    )

    theta1 = math.atan2(y, x) - math.atan2(
        l2 * math.sin(theta2),
        l1 + l2 * math.cos(theta2),
    )

    # Normalize theta1 to [-pi, pi).
    theta1 = (theta1 + math.pi) % (2.0 * math.pi) - math.pi

    theta1_deg = math.degrees(theta1)
    theta2_deg = math.degrees(theta2)

    lower1, upper1 = theta1_limits_deg
    lower2, upper2 = theta2_limits_deg

    if not lower1 <= theta1_deg <= upper1:
        raise ValueError(
            f"Joint 1 limit violated: {theta1_deg:.2f} degrees."
        )

    if not lower2 <= theta2_deg <= upper2:
        raise ValueError(
            f"Joint 2 limit violated: {theta2_deg:.2f} degrees."
        )

    return theta1, theta2
