
import math
import pytest

from src.kinematics import (
    forward_kinematics,
    inverse_kinematics,
)


def test_fully_extended_arm():
    x, y, elbow = forward_kinematics(0.0, 0.0)

    assert x == pytest.approx(20.0)
    assert y == pytest.approx(0.0)
    assert elbow == pytest.approx((10.0, 0.0))


@pytest.mark.parametrize(
    "theta1, theta2, branch",
    [
        (0.3, 0.7, "positive"),
        (-0.4, 1.0, "positive"),
        (0.5, -0.8, "negative"),
        (-0.7, -1.1, "negative"),
    ],
)
def test_forward_inverse_round_trip(theta1, theta2, branch):
    x, y, _ = forward_kinematics(theta1, theta2)

    solved_theta1, solved_theta2 = inverse_kinematics(
        x, y, branch=branch
    )

    reconstructed_x, reconstructed_y, _ = forward_kinematics(
        solved_theta1, solved_theta2
    )

    assert reconstructed_x == pytest.approx(x, abs=1e-8)
    assert reconstructed_y == pytest.approx(y, abs=1e-8)
    assert solved_theta1 == pytest.approx(theta1, abs=1e-8)
    assert solved_theta2 == pytest.approx(theta2, abs=1e-8)


def test_positive_and_negative_branches():
    x, y = 10.0, 10.0

    _, positive_theta2 = inverse_kinematics(
        x, y, branch="positive"
    )
    _, negative_theta2 = inverse_kinematics(
        x, y, branch="negative"
    )

    assert positive_theta2 > 0
    assert negative_theta2 < 0


def test_unreachable_target_is_rejected():
    with pytest.raises(ValueError, match="unreachable"):
        inverse_kinematics(25.0, 0.0)


def test_joint_limit_violation_is_rejected():
    with pytest.raises(ValueError, match="Joint 2 limit violated"):
        inverse_kinematics(
            10.0,
            10.0,
            branch="positive",
            theta2_limits_deg=(-20.0, 20.0),
        )


def test_invalid_branch_is_rejected():
    with pytest.raises(ValueError, match="branch"):
        inverse_kinematics(10.0, 10.0, branch="up")


def test_invalid_link_length_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        forward_kinematics(0.0, 0.0, l1=0.0)


def test_non_finite_coordinates_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        inverse_kinematics(math.nan, 5.0)
