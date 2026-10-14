"""
tests/test_metrics.py

Comprehensive unit tests for the evaluation metrics module (src.eval.metrics).
Tests known ground-truth cases:
1. Identical paths & masks -> zero error, Chamfer=0, IoU=1.0.
2. Fixed translation shift -> exact known shift in RMSE, Chamfer, and analytical IoU.
3. Mirrored shape -> high Chamfer distance and zero/near-zero IoU.
4. Smooth vs oscillating trajectories -> jerk and acceleration variance contrast.
5. Pen travel distance and efficiency calculations.
6. Error decomposition: Planning vs Execution vs Total error separation.
7. Camera coordinate projection round-trip and ink extraction.
"""

import math
import numpy as np
import pytest

from src.config import CameraConfig, EvalConfig
from src.eval.metrics import (
    pointwise_tracking_errors,
    calculate_trajectory_metrics,
    calculate_reachability_percentage,
    calculate_joint_angle_rmse,
    endpoint_errors,
    point_set_chamfer_distance,
    trajectory_smoothness,
    pen_travel_metrics,
    decompose_tracking_errors,
    arm_to_camera_pixel,
    camera_pixel_to_arm,
    rasterize_strokes_to_camera_mask,
    extract_ink_mask_from_camera_image,
    raster_iou,
    raster_chamfer_and_contour_error,
)


# ==============================================================================
# 1. Pointwise & Trajectory Metrics Tests
# ==============================================================================

def test_identical_paths_pointwise_zero():
    pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    errs = pointwise_tracking_errors(pts, pts)
    assert np.allclose(errs, 0.0)

    metrics = calculate_trajectory_metrics(pts, pts)
    assert metrics["rmse_cm"] == pytest.approx(0.0)
    assert metrics["mean_error_cm"] == pytest.approx(0.0)
    assert metrics["max_error_cm"] == pytest.approx(0.0)
    assert metrics["rmse_mm"] == pytest.approx(0.0)


def test_fixed_translation_shift_metrics():
    # Target points
    target = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    # Shifted by dx = 0.3 cm, dy = 0.4 cm -> Euclidean shift = 0.5 cm = 5.0 mm
    shift = np.array([0.3, 0.4])
    actual = target + shift

    errs = pointwise_tracking_errors(target, actual)
    assert np.allclose(errs, 0.5)

    metrics = calculate_trajectory_metrics(target, actual)
    assert metrics["rmse_cm"] == pytest.approx(0.5)
    assert metrics["mean_error_cm"] == pytest.approx(0.5)
    assert metrics["max_error_cm"] == pytest.approx(0.5)
    assert metrics["rmse_mm"] == pytest.approx(5.0)
    assert metrics["mean_error_mm"] == pytest.approx(5.0)


def test_trajectory_metrics_error_handling():
    with pytest.raises(ValueError, match="At least one point"):
        calculate_trajectory_metrics(np.empty((0, 2)), np.empty((0, 2)))

    with pytest.raises(ValueError, match="identical shapes"):
        calculate_trajectory_metrics(np.zeros((2, 2)), np.zeros((3, 2)))

    with pytest.raises(ValueError, match="finite"):
        calculate_trajectory_metrics(np.array([[np.nan, 0]]), np.array([[0, 0]]))


# ==============================================================================
# 2. Endpoint Errors Tests
# ==============================================================================

def test_endpoint_errors_identical():
    stroke1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    stroke2 = np.array([[2.0, 0.0], [2.0, 1.0]])
    strokes = [stroke1, stroke2]

    res = endpoint_errors(strokes, strokes)
    assert res["mean_endpoint_error_cm"] == pytest.approx(0.0)
    assert res["max_endpoint_error_cm"] == pytest.approx(0.0)
    assert res["mean_endpoint_error_mm"] == pytest.approx(0.0)
    assert res["stroke_count"] == 2


def test_endpoint_errors_known_shift():
    stroke = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    # Start shifted by (0.3, 0.4) -> 0.5 cm; end shifted by (0.6, 0.8) -> 1.0 cm
    shifted_stroke = np.array([[1.3, 1.4], [2.0, 2.0], [3.6, 3.8]])

    res = endpoint_errors([stroke], [shifted_stroke])
    assert res["mean_start_error_cm"] == pytest.approx(0.5)
    assert res["mean_end_error_cm"] == pytest.approx(1.0)
    assert res["mean_endpoint_error_cm"] == pytest.approx(0.75)
    assert res["max_endpoint_error_cm"] == pytest.approx(1.0)
    assert res["mean_endpoint_error_mm"] == pytest.approx(7.5)


# ==============================================================================
# 3. Point-Set Chamfer Distance Tests
# ==============================================================================

def test_chamfer_identical_sets():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    res = point_set_chamfer_distance(pts, pts)
    assert res["forward_chamfer_cm"] == pytest.approx(0.0)
    assert res["backward_chamfer_cm"] == pytest.approx(0.0)
    assert res["symmetric_chamfer_cm"] == pytest.approx(0.0)
    assert res["hausdorff_cm"] == pytest.approx(0.0)


def test_chamfer_known_shift():
    pts_a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Shift all points by (0.3, 0.4) cm -> distance is exactly 0.5 cm = 5.0 mm
    pts_b = pts_a + np.array([0.3, 0.4])

    res = point_set_chamfer_distance(pts_a, pts_b)
    assert res["forward_chamfer_cm"] == pytest.approx(0.5)
    assert res["backward_chamfer_cm"] == pytest.approx(0.5)
    assert res["symmetric_chamfer_cm"] == pytest.approx(0.5)
    assert res["hausdorff_cm"] == pytest.approx(0.5)
    assert res["symmetric_chamfer_mm"] == pytest.approx(5.0)


def test_chamfer_mirrored_shape_high():
    # Asymmetric shape on the left: X in [1.0, 2.0], Y in [1.0, 5.0]
    shape_left = np.array([
        [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
        [2.0, 1.0]
    ])
    # Mirrored across X=5 to the right: X' = 10 - X -> X' in [8.0, 9.0]
    shape_right = np.copy(shape_left)
    shape_right[:, 0] = 10.0 - shape_right[:, 0]

    res = point_set_chamfer_distance(shape_left, shape_right)
    # The minimum horizontal distance between any left point (X<=2) and right point (X>=8) is >= 6.0 cm = 60.0 mm
    assert res["symmetric_chamfer_cm"] >= 6.0
    assert res["symmetric_chamfer_mm"] >= 60.0
    assert res["hausdorff_mm"] >= 80.0


# ==============================================================================
# 4. Trajectory Smoothness Tests
# ==============================================================================

def test_smoothness_constant_velocity_zero_jerk():
    # Straight line with constant velocity: zero acceleration, zero jerk
    t = np.linspace(0, 1, 20)
    pts = np.column_stack((2.0 * t, 3.0 * t))
    res = trajectory_smoothness(pts, dt=0.05)
    assert res["rms_jerk_cm_s3"] == pytest.approx(0.0, abs=1e-6)
    assert res["acceleration_variance_cm_s2"] == pytest.approx(0.0, abs=1e-6)


def test_smoothness_contrast_smooth_vs_oscillating():
    # Smooth sine wave vs high-frequency oscillating trajectory
    t = np.linspace(0, 2 * np.pi, 50)
    smooth_path = np.column_stack((t, np.sin(t)))

    # Oscillating zig-zag
    noisy_path = np.copy(smooth_path)
    noisy_path[:, 1] += 0.2 * np.sin(20 * t)

    res_smooth = trajectory_smoothness(smooth_path, dt=0.05)
    res_noisy = trajectory_smoothness(noisy_path, dt=0.05)

    # Jerk and acceleration variance should be significantly higher for the oscillating path
    assert res_noisy["rms_jerk_cm_s3"] > 10.0 * res_smooth["rms_jerk_cm_s3"]
    assert res_noisy["acceleration_variance_cm_s2"] > 5.0 * res_smooth["acceleration_variance_cm_s2"]


# ==============================================================================
# 5. Pen Travel Distance & Efficiency Tests
# ==============================================================================

def test_pen_travel_metrics_known():
    # 4 points:
    # Segment 1: (0,0) -> (3,0) (dist=3.0, pen_down=1)
    # Segment 2: (3,0) -> (3,4) (dist=4.0, pen_down=1)
    # Segment 3: (3,4) -> (3,9) (dist=5.0, pen_down=0, pen-up repositioning)
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [3.0, 9.0]])
    pen_states = np.array([1, 1, 1, 0])

    res = pen_travel_metrics(pts, pen_states)
    assert res["pen_down_distance_cm"] == pytest.approx(7.0)
    assert res["pen_up_distance_cm"] == pytest.approx(5.0)
    assert res["total_distance_cm"] == pytest.approx(12.0)
    assert res["pen_travel_efficiency_pct"] == pytest.approx(7.0 / 12.0 * 100.0)


# ==============================================================================
# 6. Raster IoU and Distance Transform Chamfer Tests
# ==============================================================================

def test_raster_iou_identical():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255  # 40x40 square

    res = raster_iou(mask, mask)
    assert res["iou"] == pytest.approx(1.0)
    assert res["iou_pct"] == pytest.approx(100.0)
    assert res["dice"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)

    chamfer_res = raster_chamfer_and_contour_error(mask, mask, pixel_size_mm=1.0)
    assert chamfer_res["contour_error_mean_mm"] == pytest.approx(0.0)
    assert chamfer_res["contour_error_rmse_mm"] == pytest.approx(0.0)
    assert chamfer_res["symmetric_chamfer_mm"] == pytest.approx(0.0)


def test_raster_iou_and_chamfer_known_shift():
    # Solid square 50x50 at [20:70, 20:70]
    mask_a = np.zeros((120, 120), dtype=np.uint8)
    mask_a[20:70, 20:70] = 255

    # Shifted horizontally by 10 pixels to [20:70, 30:80]
    mask_b = np.zeros((120, 120), dtype=np.uint8)
    mask_b[20:70, 30:80] = 255

    # Analytical IoU:
    # Intersection: 50 * 40 = 2000 px
    # Union: 2 * 2500 - 2000 = 3000 px
    # IoU = 2000 / 3000 = 2/3 ~ 0.6667
    res = raster_iou(mask_a, mask_b)
    assert res["intersection_pixels"] == 2000
    assert res["union_pixels"] == 3000
    assert res["iou"] == pytest.approx(2.0 / 3.0)
    assert res["iou_pct"] == pytest.approx(66.6667, rel=1e-3)

    # Chamfer distance with pixel_size_mm = 0.5
    chamfer_res = raster_chamfer_and_contour_error(mask_a, mask_b, pixel_size_mm=0.5)
    # The non-overlapping strip is shifted by 10 px -> max error is 10 px * 0.5 mm = 5.0 mm
    assert chamfer_res["contour_error_max_mm"] == pytest.approx(5.0)


def test_raster_iou_and_chamfer_mirrored_shape():
    # Asymmetric shape on the left side: [20:80, 10:30]
    mask_left = np.zeros((100, 100), dtype=np.uint8)
    mask_left[20:80, 10:30] = 255

    # Mirrored to right side: [20:80, 70:90]
    mask_right = np.zeros((100, 100), dtype=np.uint8)
    mask_right[20:80, 70:90] = 255

    # Masks are completely disjoint -> IoU = 0.0
    res = raster_iou(mask_left, mask_right)
    assert res["iou"] == pytest.approx(0.0)
    assert res["intersection_pixels"] == 0

    # Distance between left strip (max col 29) and right strip (min col 70) is >= 40 pixels
    chamfer_res = raster_chamfer_and_contour_error(mask_left, mask_right, pixel_size_mm=1.0)
    assert chamfer_res["contour_error_mean_mm"] >= 40.0
    assert chamfer_res["symmetric_chamfer_mm"] >= 40.0


# ==============================================================================
# 7. Error Decomposition Tests (Planning vs Execution vs Total)
# ==============================================================================

def test_decomposed_error_separation():
    # Target: two straight strokes
    tgt_stroke1 = np.array([[5.0, 8.0], [5.0, 10.0]])
    tgt_stroke2 = np.array([[6.0, 8.0], [6.0, 10.0]])
    target_strokes = [tgt_stroke1, tgt_stroke2]

    # Plan: perfectly discretizes target strokes with pen-up move between them
    plan_pts = np.array([
        [5.0, 8.0], [5.0, 9.0], [5.0, 10.0],  # Stroke 0 (pen down)
        [6.0, 8.0],                            # Repositioning (pen up)
        [6.0, 8.0], [6.0, 9.0], [6.0, 10.0],  # Stroke 1 (pen down)
    ])
    pen_states = np.array([1, 1, 1, 0, 1, 1, 1])
    stroke_ids = np.array([0, 0, 0, 1, 1, 1, 1])

    # Measured: has a fixed execution tracking offset of +0.5 mm = +0.05 cm in X
    meas_pts = np.copy(plan_pts)
    meas_pts[:, 0] += 0.05

    decomp = decompose_tracking_errors(
        target_strokes_arm=target_strokes,
        plan_positions_cm=plan_pts,
        measured_positions_cm=meas_pts,
        pen_states=pen_states,
        stroke_ids=stroke_ids,
    )

    # 1. Planning error: plan points lie exactly on target line segments -> planning Chamfer ~ 0
    assert decomp["planning_error"]["chamfer_symmetric_mm"] == pytest.approx(0.0, abs=1e-4)

    # 2. Execution error: 0.05 cm offset = 0.5 mm error
    assert decomp["execution_error"]["rmse_mm"] == pytest.approx(0.5, abs=1e-3)
    assert decomp["execution_error"]["mean_error_mm"] == pytest.approx(0.5, abs=1e-3)

    # 3. Total error: target vs measured -> ~0.5 mm
    assert decomp["total_error"]["chamfer_symmetric_mm"] == pytest.approx(0.5, abs=1e-2)


# ==============================================================================
# 8. Camera Projection & Ink Extraction Tests
# ==============================================================================

def test_camera_projection_round_trip():
    cfg = CameraConfig(
        resolution_x=512,
        resolution_y=512,
        view_size_cm=12.0,
        center_x_cm=7.0,
        center_y_cm=9.5,
    )

    # Test center
    u, v = arm_to_camera_pixel(7.0, 9.5, cfg)
    assert (u, v) == (256, 256)
    x_rec, y_rec = camera_pixel_to_arm(u, v, cfg)
    assert x_rec == pytest.approx(7.0)
    assert y_rec == pytest.approx(9.5)

    # Test arbitrary workspace point inside drawing window
    orig_x, orig_y = 5.25, 11.75
    u, v = arm_to_camera_pixel(orig_x, orig_y, cfg)
    x_rec, y_rec = camera_pixel_to_arm(u, v, cfg)
    assert x_rec == pytest.approx(orig_x, abs=0.03)  # within sub-pixel quantization
    assert y_rec == pytest.approx(orig_y, abs=0.03)


def test_extract_ink_mask():
    # Synthetic RGB image: white paper with a blue ink square
    img = np.full((100, 100, 3), 250, dtype=np.uint8)  # White paper (BGR)
    # Draw blue ink patch in BGR: high blue (180), low red (20), low green (40)
    img[40:60, 40:60] = [180, 40, 20]

    mask = extract_ink_mask_from_camera_image(img, is_bgr=True)
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 255
    assert mask[10, 10] == 0
    assert np.sum(mask > 0) == 20 * 20


def test_evaluate_full_run_synthetic():
    from src.eval.metrics import evaluate_full_run

    tgt_stroke = np.array([[5.0, 8.0], [5.0, 10.0]])
    plan_pts = np.array([
        [5.0, 8.0], [5.0, 8.5], [5.0, 9.0], [5.0, 9.5], [5.0, 10.0]
    ])
    meas_pts = np.copy(plan_pts)
    pen_states = np.array([1, 1, 1, 1, 1])
    stroke_ids = np.array([0, 0, 0, 0, 0])

    res = evaluate_full_run(
        target_strokes_arm=[tgt_stroke],
        plan_positions_cm=plan_pts,
        measured_positions_cm=meas_pts,
        pen_states=pen_states,
        stroke_ids=stroke_ids,
        camera_image_bgr=None,
    )
    assert "decomposition" in res
    assert "travel" in res
    assert "smoothness" in res
    assert res["decomposition"]["execution_error"]["rmse_mm"] == pytest.approx(0.0)
    assert res["travel"]["pen_down_distance_cm"] == pytest.approx(2.0)
    assert res["travel"]["pen_travel_efficiency_pct"] == pytest.approx(100.0)

