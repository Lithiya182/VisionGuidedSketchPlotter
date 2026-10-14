
import numpy as np
import pytest

from src.path_planner import (
    interpolate_points,
    plan_strokes,
)


def test_interpolation_includes_start_and_end():
    points = interpolate_points(
        (0.0, 0.0), (1.0, 0.0), spacing_cm=0.25
    )

    assert points[0] == pytest.approx([0.0, 0.0])
    assert points[-1] == pytest.approx([1.0, 0.0])
    assert len(points) == 5


def test_interpolation_rejects_invalid_spacing():
    with pytest.raises(ValueError, match="Spacing"):
        interpolate_points((0, 0), (1, 1), spacing_cm=0)


def test_invalid_branch_is_rejected():
    with pytest.raises(ValueError, match="branch"):
        plan_strokes([], branch="up")


def test_stroke_starts_with_pen_up():
    contours = [
        [(10.0, 10.0), (11.0, 10.0)]
    ]

    motion = plan_strokes(contours)

    assert motion[0].pen_down is False
    assert motion[0].stroke_id == 0


def test_drawing_points_have_pen_down():
    contours = [
        [(10.0, 10.0), (11.0, 10.0)]
    ]

    motion = plan_strokes(contours)

    assert all(point.pen_down for point in motion[1:])


def test_contours_remain_separate():
    contours = [
        [(10.0, 10.0), (11.0, 10.0)],
        [(10.0, 12.0), (11.0, 12.0)],
    ]

    motion = plan_strokes(contours)

    starts = [point for point in motion if not point.pen_down]

    assert len(starts) == 2
    assert starts[0].stroke_id == 0
    assert starts[1].stroke_id == 1


def test_invalid_stroke_is_rejected():
    contours = [
        [(10.0, 10.0), (25.0, 0.0)]
    ]

    with pytest.raises(ValueError, match="stroke 0"):
        plan_strokes(contours)


def test_invalid_stroke_can_be_skipped():
    contours = [
        [(10.0, 10.0), (25.0, 0.0)],
        [(10.0, 10.0), (11.0, 10.0)],
    ]

    motion = plan_strokes(
        contours,
        skip_invalid_strokes=True,
    )

    assert motion
    assert all(point.stroke_id == 1 for point in motion)


def test_empty_contour_is_rejected():
    with pytest.raises(ValueError, match="Stroke 0"):
        plan_strokes([[]])
