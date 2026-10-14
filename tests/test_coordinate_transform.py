
import numpy as np
import pytest

from src.coordinate_transform import (
    image_to_arm_space,
    validate_workspace_point,
    validate_joint_configuration,
    validate_drawing_path,
)


def test_image_top_left_maps_to_drawing_origin():
    result = image_to_arm_space([[0, 0]])
    assert result[0] == pytest.approx([4.0, 12.5])


def test_image_bottom_right_maps_correctly():
    result = image_to_arm_space([[480, 480]])
    assert result[0] == pytest.approx([10.0, 6.5])


def test_image_y_axis_is_flipped():
    result = image_to_arm_space([[240, 0], [240, 480]])
    assert result[0, 1] > result[1, 1]

def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="(?i)at least one"):
        image_to_arm_space([])


def test_invalid_coordinate_shape_is_rejected():
    with pytest.raises(ValueError, match="two"):
        image_to_arm_space([[1, 2, 3]])


def test_out_of_image_coordinate_is_rejected():
    with pytest.raises(ValueError, match="outside the image"):
        image_to_arm_space([[500, 20]])


def test_near_base_point_is_outside_safe_workspace():
    with pytest.raises(ValueError, match="safe radial workspace"):
        validate_workspace_point(0.5, 0.0)


def test_safe_radius_point_is_reachable_with_joint_limits():
    assert validate_workspace_point(10.0, 10.0)
    assert validate_joint_configuration(10.0, 10.0)


def test_drawing_path_separates_valid_and_rejected_points():
    points = np.array([[10.0, 10.0], [0.5, 0.0]])
    valid, rejected = validate_drawing_path(points)

    assert len(valid) == 1
    assert len(rejected) == 1
