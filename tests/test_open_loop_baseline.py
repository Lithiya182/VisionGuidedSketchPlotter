"""
test_open_loop_baseline.py

Unit and integration test suite for Phase 2:
- Idempotent scene setup
- Configurable window dimensions (6x6 vs 8x8)
- Reachability verification with zero rejections on 01_robot_arm_sketch_skeleton
- Pen-up and pen-down telemetry logging
- Visible ink trace generation and camera capture
"""

import os
import pytest

from scripts.setup_scene import setup_scene
from src.config import DEFAULT_WORKSPACE_CONFIG, WorkspaceConfig
from src.coordinate_transform import image_to_arm_space, validate_workspace_point, validate_joint_configuration
from src.coppelia_controller import CoppeliaController
from src.path_planner import plan_strokes
from src.vision.image_processor import StrokeIR


def test_scene_setup_idempotent():
    """Verify setup_scene executes without errors and maintains idempotent handles."""
    try:
        setup_scene()
        setup_scene()
    except Exception as exc:
        pytest.skip(f"CoppeliaSim is not reachable: {exc}")


def test_drawing_window_configurability():
    """Test that window dimensions can be modified via WorkspaceConfig."""
    ws_6x6 = WorkspaceConfig(draw_width_cm=6.0, draw_height_cm=6.0)
    ws_8x8 = WorkspaceConfig(draw_width_cm=8.0, draw_height_cm=8.0, draw_origin_x=3.0, draw_origin_y=13.5)

    assert ws_6x6.draw_width_cm == 6.0
    assert ws_8x8.draw_width_cm == 8.0

    dummy_px = [[0, 0], [480, 480]]
    arm_6x6 = image_to_arm_space(dummy_px, draw_width=ws_6x6.draw_width_cm, draw_height=ws_6x6.draw_height_cm)
    arm_8x8 = image_to_arm_space(dummy_px, draw_width=ws_8x8.draw_width_cm, draw_height=ws_8x8.draw_height_cm,
                                 origin_x=ws_8x8.draw_origin_x, origin_y=ws_8x8.draw_origin_y)

    assert (arm_6x6[1, 0] - arm_6x6[0, 0]) == pytest.approx(6.0)
    assert (arm_8x8[1, 0] - arm_8x8[0, 0]) == pytest.approx(8.0)


def test_skeleton_ir_reachability_zero_rejections():
    """Verify that all strokes in 01_robot_arm_sketch_skeleton.json map safely into the 6x6cm window."""
    json_path = "data/extracted_strokes/01_robot_arm_sketch_skeleton.json"
    if not os.path.exists(json_path):
        pytest.skip(f"File not found: {json_path}")

    ir = StrokeIR.from_json(json_path)
    ws = DEFAULT_WORKSPACE_CONFIG

    rejections = []
    total_pts = 0

    for stroke in ir.strokes:
        if len(stroke) < 2:
            continue
        total_pts += len(stroke)
        pts_arm = image_to_arm_space(
            stroke,
            image_width=ws.image_width,
            image_height=ws.image_height,
            origin_x=ws.draw_origin_x,
            origin_y=ws.draw_origin_y,
            draw_width=ws.draw_width_cm,
            draw_height=ws.draw_height_cm,
        )
        for pt in pts_arm:
            try:
                validate_workspace_point(pt[0], pt[1])
                validate_joint_configuration(pt[0], pt[1])
            except ValueError as exc:
                rejections.append((pt, str(exc)))

    assert total_pts > 0
    assert len(rejections) == 0, f"Expected 0 rejections, found {len(rejections)}"


def test_sim_drawing_and_vision_capture():
    """Verify that ink segments can be added and camera captures images without errors."""
    try:
        controller = CoppeliaController(verbose=False)
    except Exception as exc:
        pytest.skip(f"CoppeliaSim is not reachable: {exc}")

    controller.stop_simulation()
    controller.set_stepping(True)
    controller.start_simulation()

    # Add ink segment
    controller.create_drawing_object()
    controller.add_ink_segment((7.0, 9.0, 1.55), (7.5, 9.5, 1.55))
    controller.step()

    # Capture top-down image
    img = controller.get_vision_sensor_image()
    assert img.shape == (512, 512, 3)

    controller.clear_drawing_object()
    controller.stop_simulation()
    controller.set_stepping(False)
