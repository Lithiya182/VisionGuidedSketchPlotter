"""
tests/test_phase6_optimization.py

Phase 6 Tests:
1. Geometric Invariance: Reordering and reversing strokes preserves identical drawn geometry.
2. Hard Constraints: No point violates workspace bounds, radial limits, or joint angle limits.
3. Monotonic Optimality: 2-Opt pen-up distance is strictly <= baseline pen-up distance.
4. Exact Optimality Benchmark: Exact DP solves small instances and provides a ground truth bound.
5. Weight Sensitivity: Increasing reversal weight reduces or maintains joint direction reversals.
"""

import math
import cv2
import numpy as np
import pytest

from src.config import (
    DEFAULT_WORKSPACE_CONFIG,
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_STROKE_ORDER_CONFIG,
    StrokeOrderConfig,
)
from src.coordinate_transform import (
    image_to_arm_space,
    validate_workspace_point,
    THETA1_LIMITS_DEG,
    THETA2_LIMITS_DEG,
)
from src.kinematics import inverse_kinematics
from src.planning.stroke_optimizer import (
    create_oriented_stroke,
    order_strokes_naive,
    order_strokes_nn,
    order_strokes_2opt,
    order_strokes_exact_dp,
    evaluate_tour_metrics,
    compute_tour_total_cost,
)
from src.vision.image_processor import StrokeIR


@pytest.fixture
def robot_arm_strokes():
    """Load arm-space strokes for 01_robot_arm_sketch_skeleton.json."""
    ir = StrokeIR.from_json("data/extracted_strokes/01_robot_arm_sketch_skeleton.json")
    wc = DEFAULT_WORKSPACE_CONFIG
    strokes = []
    for s in ir.strokes:
        if len(s) >= 2:
            pts = image_to_arm_space(
                s,
                wc.image_width,
                wc.image_height,
                wc.draw_origin_x,
                wc.draw_origin_y,
                wc.draw_width_cm,
                wc.draw_height_cm,
            )
            strokes.append(pts)
    return strokes


class TestPhase6GeometricInvariance:
    """Verifies that altering stroke sequence and directions preserves identical drawn geometry."""

    def test_drawn_geometry_is_identical(self, robot_arm_strokes):
        """Rasterize naive vs optimized strokes into camera masks and verify 100% pixel equality."""
        naive_tour = order_strokes_naive(robot_arm_strokes)
        opt_tour = order_strokes_2opt(robot_arm_strokes, DEFAULT_STROKE_ORDER_CONFIG)

        cam = DEFAULT_CAMERA_CONFIG
        scale = cam.resolution_x / cam.view_size_cm
        cu, cv = cam.resolution_x / 2.0, cam.resolution_y / 2.0

        def rasterize_tour(tour):
            mask = np.zeros((cam.resolution_y, cam.resolution_x), dtype=np.uint8)
            for s in tour:
                pts_px = []
                for p in s.points:
                    u = int(round(cu + (p[0] - cam.center_x_cm) * scale))
                    v = int(round(cv - (p[1] - cam.center_y_cm) * scale))
                    pts_px.append([u, v])
                cv2.polylines(mask, [np.array(pts_px, dtype=np.int32)], isClosed=False, color=255, thickness=2)
            return mask

        mask_naive = rasterize_tour(naive_tour)
        mask_opt = rasterize_tour(opt_tour)

        diff = np.abs(mask_naive.astype(int) - mask_opt.astype(int))
        assert np.max(diff) == 0, "Drawn geometry must be pixel-for-pixel identical between orderings!"


class TestPhase6HardConstraints:
    """Verifies that hard constraints (workspace limits, joint limits) are strictly satisfied."""

    def test_all_tour_points_satisfy_joint_limits(self, robot_arm_strokes):
        opt_tour = order_strokes_2opt(robot_arm_strokes, DEFAULT_STROKE_ORDER_CONFIG)
        for s in opt_tour:
            assert s.is_valid, f"Stroke {s.stroke_id} must be valid!"
            for p in s.points:
                assert validate_workspace_point(float(p[0]), float(p[1]))
                th1, th2 = inverse_kinematics(
                    float(p[0]), float(p[1]),
                    theta1_limits_deg=THETA1_LIMITS_DEG,
                    theta2_limits_deg=THETA2_LIMITS_DEG,
                )
                assert THETA1_LIMITS_DEG[0] <= math.degrees(th1) <= THETA1_LIMITS_DEG[1]
                assert THETA2_LIMITS_DEG[0] <= math.degrees(th2) <= THETA2_LIMITS_DEG[1]

    def test_unreachable_point_receives_infinite_cost(self):
        """Out-of-reach stroke receives infinite transition cost."""
        s_valid = np.array([[7.0, 9.0], [7.0, 10.0]])
        s_unreachable = np.array([[25.0, 25.0], [26.0, 26.0]])  # > 20 cm radius

        os_valid = create_oriented_stroke(s_valid, 0)
        os_unreach = create_oriented_stroke(s_unreachable, 1)

        assert os_valid.is_valid is True
        assert os_unreach.is_valid is False

        from src.planning.stroke_optimizer import compute_transition_cost
        cost = compute_transition_cost(os_valid, os_unreach)
        assert math.isinf(cost), "Transition to unreachable stroke must have infinite cost!"


class TestPhase6MonotonicOptimization:
    """Verifies that optimized tours strictly improve or match baseline pen-up distance."""

    def test_penup_distance_monotonicity(self, robot_arm_strokes):
        naive = order_strokes_naive(robot_arm_strokes)
        nn = order_strokes_nn(robot_arm_strokes, DEFAULT_STROKE_ORDER_CONFIG)
        opt = order_strokes_2opt(robot_arm_strokes, DEFAULT_STROKE_ORDER_CONFIG)

        m_naive = evaluate_tour_metrics(naive)
        m_nn = evaluate_tour_metrics(nn)
        m_opt = evaluate_tour_metrics(opt)

        assert m_nn["pen_up_distance_cm"] <= m_naive["pen_up_distance_cm"], "NN pen-up must be <= Naive"
        assert m_opt["pen_up_distance_cm"] <= m_nn["pen_up_distance_cm"] + 1e-4, "2-Opt pen-up must be <= NN"
        # Check substantial reduction (at least 50% air travel saved)
        assert m_opt["pen_up_distance_cm"] < m_naive["pen_up_distance_cm"] * 0.50

    def test_baseline_reproduction(self, robot_arm_strokes):
        """Check reproducing the Phase 2 baseline pen-up distance of ~28.10 cm (or 26.95 cm)."""
        naive = order_strokes_naive(robot_arm_strokes)
        m = evaluate_tour_metrics(naive)
        # Pen-up repositioning between the 30 strokes without initial move is 28.10 cm
        assert math.isclose(m["pen_up_distance_cm"], 28.10, abs_tol=0.25)
        assert math.isclose(m["pen_travel_efficiency_pct"], 45.0, abs_tol=1.0)


class TestPhase6ExactOptimalityBenchmark:
    """Requirement 3: Check optimality on small instances to measure heuristic gap."""

    def test_exact_dp_optimality_n6(self, robot_arm_strokes):
        sub_strokes = robot_arm_strokes[:6]
        dp_tour, dp_cost = order_strokes_exact_dp(sub_strokes, DEFAULT_STROKE_ORDER_CONFIG)
        opt_tour = order_strokes_2opt(sub_strokes, DEFAULT_STROKE_ORDER_CONFIG)

        cost_opt = compute_tour_total_cost(opt_tour, DEFAULT_STROKE_ORDER_CONFIG)

        assert len(dp_tour) == 6
        assert len(opt_tour) == 6
        # Gap should be <= 10% on small instances
        gap_pct = (cost_opt - dp_cost) / dp_cost * 100.0
        assert gap_pct <= 10.0, f"Optimality gap too large: {gap_pct:.1f}%"


class TestPhase6WeightSensitivity:
    """Verifies that tuning multi-objective weights changes solution characteristics."""

    def test_reversal_weight_sensitivity(self, robot_arm_strokes):
        sub_strokes = robot_arm_strokes[:10]
        # Low reversal penalty
        cfg_low = StrokeOrderConfig(w_cartesian=1.0, w_joint=0.0, w_reversal=0.0, w_singularity=0.0)
        # High reversal penalty
        cfg_high = StrokeOrderConfig(w_cartesian=0.1, w_joint=0.1, w_reversal=2.0, w_singularity=0.0)

        tour_low = order_strokes_2opt(sub_strokes, cfg_low)
        tour_high = order_strokes_2opt(sub_strokes, cfg_high)

        m_low = evaluate_tour_metrics(tour_low)
        m_high = evaluate_tour_metrics(tour_high)

        # High reversal penalty configuration must yield <= reversals
        assert m_high["joint_reversals_count"] <= m_low["joint_reversals_count"]
