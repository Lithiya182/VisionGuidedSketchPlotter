
import numpy as np
import pytest

from src.trajectory_metrics import (
    trajectory_errors,
    calculate_trajectory_metrics,
    calculate_reachability_percentage,
    calculate_joint_angle_rmse,
)


def test_identical_paths_have_zero_error():
    points = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert np.allclose(trajectory_errors(points, points), 0.0)


def test_trajectory_rmse():
    target = np.array([[0.0, 0.0], [0.0, 0.0]])
    actual = np.array([[3.0, 4.0], [0.0, 0.0]])

    metrics = calculate_trajectory_metrics(target, actual)

    assert metrics["rmse_cm"] == pytest.approx(
        np.sqrt(12.5)
    )
    assert metrics["mean_error_cm"] == pytest.approx(2.5)
    assert metrics["max_error_cm"] == pytest.approx(5.0)
    assert metrics["point_count"] == 2


def test_mismatched_trajectory_shapes_are_rejected():
    with pytest.raises(ValueError, match="identical shapes"):
        trajectory_errors([[0, 0]], [[0, 0], [1, 1]])


def test_empty_trajectory_is_rejected():
    with pytest.raises(ValueError, match="(?i)at least one"):
        calculate_trajectory_metrics([], [])


def test_reachability_percentage():
    assert calculate_reachability_percentage(20, 18) == pytest.approx(90.0)


def test_invalid_reachability_counts_are_rejected():
    with pytest.raises(ValueError):
        calculate_reachability_percentage(10, 11)


def test_joint_angle_rmse():
    target = np.array([[0.0, 0.0], [0.0, 0.0]])
    actual = np.array([[3.0, 4.0], [0.0, 0.0]])

    assert calculate_joint_angle_rmse(target, actual) == pytest.approx(
        np.sqrt(6.25)
    )


def test_non_finite_trajectory_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        trajectory_errors([[0, 0]], [[np.nan, 0]])
