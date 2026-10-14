"""
tests/test_phase5_regression.py

Phase 5 Regression and Architectural Enforcement Tests:
1. Regression locking for nominal numbers and settle tolerance (0.25 deg).
2. Architectural enforcement: Vision Corrector uses ONLY camera images, target strokes,
   and nominal IK. It must NOT import or access get_tip_position, disturbance parameters,
   or telemetry records.
3. StrokeMatcher ICP translation recovery on synthetic ground truth.
4. RunningCorrector EMA smoothing and magnitude capping.
5. PerStrokeObserver image differencing and ink coordinate mapping.
"""

import ast
import inspect
import math
import numpy as np
import pytest

from src.config import DEFAULT_CAMERA_CONFIG, DEFAULT_WORKSPACE_CONFIG
from src.eval.metrics import arm_to_camera_pixel, camera_pixel_to_arm
from src.sim.vision_corrector import (
    CorrectorConfig,
    PerStrokeObserver,
    StrokeMatcher,
    RunningCorrector,
    CalibrationPrePass,
)
import src.sim.vision_corrector as vc_module


class TestPhase5NominalRegression:
    """Verifies that nominal baseline parameters and configs are locked."""

    def test_settle_tolerance_is_locked_to_0_25_deg(self):
        """
        Explaining the Phase 4 RMSE discrepancy (0.432 mm vs Phase 3's 0.318 mm):
        Phase 4's script mistakenly used tolerance_deg=0.35 in wait_for_settle().
        At 0.35 deg per joint, link lengths 10 cm + 10 cm can deviate by ~0.35 mm more.
        This test enforces that runner scripts specify tolerance_deg=0.25.
        """
        import inspect
        from scripts import run_disturbed_baseline
        from scripts import run_closed_loop

        src_disturbed = inspect.getsource(run_disturbed_baseline)
        src_closed_loop = inspect.getsource(run_closed_loop)

        assert "tolerance_deg=0.25" in src_disturbed, "run_disturbed_baseline must use tolerance_deg=0.25"
        assert "tolerance_deg=0.25" in src_closed_loop, "run_closed_loop must use tolerance_deg=0.25"

    def test_nominal_disturbance_preset_is_zero(self):
        """Verify the 'none' preset has exactly zero disturbance in all fields."""
        from src.sim.disturbances import get_disturbance_preset
        cfg = get_disturbance_preset("none")
        assert cfg.enabled is False
        assert cfg.delta_L1_cm == 0.0
        assert cfg.delta_L2_cm == 0.0
        assert cfg.delta_theta1_deg == 0.0
        assert cfg.delta_theta2_deg == 0.0
        assert cfg.backlash_theta1_deg == 0.0
        assert cfg.backlash_theta2_deg == 0.0
        assert cfg.noise_std_theta1_deg == 0.0


class TestCorrectorArchitecturalEnforcement:
    """
    Requirement 2:
    The corrector may use ONLY the TopDownVisionSensor image, the target strokes,
    and the nominal model. It must NOT read the true tip position, telemetry,
    or disturbance parameters.
    """

    def test_vision_corrector_ast_has_no_forbidden_imports(self):
        """Check AST of src/sim/vision_corrector.py to enforce zero access to forbidden modules."""
        source_file = inspect.getfile(vc_module)
        with open(source_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        forbidden_names = {
            "disturbances",
            "PlantDisturbanceManager",
            "DisturbanceConfig",
            "get_disturbance_preset",
            "get_tip_position",
            "telemetry",
            "CoppeliaController",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_names:
                        assert forbidden not in alias.name, f"Forbidden import found: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for forbidden in forbidden_names:
                    assert forbidden not in module, f"Forbidden module import: {module}"
                for alias in node.names:
                    for forbidden in forbidden_names:
                        assert forbidden not in alias.name, f"Forbidden import name: {alias.name}"

    def test_vision_corrector_source_has_no_tip_position_call(self):
        """Verify get_tip_position is never called or imported inside src/sim/vision_corrector.py."""
        source_file = inspect.getfile(vc_module)
        with open(source_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id != "get_tip_position", "Direct call to get_tip_position forbidden!"
                elif isinstance(node.func, ast.Attribute):
                    assert node.func.attr != "get_tip_position", "Attribute call to get_tip_position forbidden!"

    def test_corrector_operates_purely_on_images_and_target_strokes(self):
        """
        Verify that RunningCorrector, StrokeMatcher, and PerStrokeObserver
        function purely on synthetic camera images and target strokes.
        """
        config = CorrectorConfig(enabled=True, ema_alpha=0.5, max_offset_cm=0.5, min_observed_pixels=5)
        matcher = StrokeMatcher(config)
        corrector = RunningCorrector(config)

        # Target stroke: 10 points along a line
        tgt = np.array([[7.0, float(y)] for y in np.linspace(8.0, 11.0, 10)])

        # Simulated observation with a known offset (+0.12 cm, -0.08 cm)
        true_offset = np.array([0.12, -0.08])
        obs = tgt + true_offset

        # StrokeMatcher estimates offset
        est_offset = matcher.estimate_translation(obs, tgt)
        assert est_offset is not None
        np.testing.assert_allclose(est_offset, true_offset, atol=1e-3)

        # RunningCorrector updates
        smoothed = corrector.update(est_offset)
        np.testing.assert_allclose(smoothed, true_offset, atol=1e-3)

        # Inverse correction: cmd = target - plant_offset
        cx, cy = corrector.correct_waypoint(7.0, 9.5)
        expected_x = 7.0 - true_offset[0]
        expected_y = 9.5 - true_offset[1]
        assert math.isclose(cx, expected_x, abs_tol=1e-4)
        assert math.isclose(cy, expected_y, abs_tol=1e-4)


class TestStrokeMatcherICP:
    """Verifies ICP translation estimation under various synthetic shifts and noise."""

    def test_icp_recovers_known_shift(self):
        config = CorrectorConfig(icp_iterations=20, icp_inlier_fraction=0.8, min_observed_pixels=10)
        matcher = StrokeMatcher(config)

        tgt = np.array([[6.0 + 0.1 * i, 8.0 + 0.2 * i] for i in range(20)])
        expected_shift = np.array([-0.18, 0.25])
        obs = tgt + expected_shift

        est = matcher.estimate_translation(obs, tgt)
        assert est is not None
        np.testing.assert_allclose(est, expected_shift, atol=1e-3)

    def test_icp_robust_to_outliers(self):
        config = CorrectorConfig(icp_iterations=25, icp_inlier_fraction=0.8, min_observed_pixels=10)
        matcher = StrokeMatcher(config)

        tgt = np.array([[7.0, float(y)] for y in np.linspace(8.0, 11.0, 50)])
        expected_shift = np.array([0.10, -0.15])
        obs = tgt + expected_shift

        # Add 5 outlier points
        outliers = np.array([[9.0, 12.0], [5.0, 7.0], [10.0, 10.0], [6.0, 12.0], [9.5, 8.0]])
        obs_with_outliers = np.vstack([obs, outliers])

        est = matcher.estimate_translation(obs_with_outliers, tgt)
        assert est is not None
        np.testing.assert_allclose(est, expected_shift, atol=0.03)

    def test_icp_returns_none_if_too_few_points(self):
        config = CorrectorConfig(min_observed_pixels=30)
        matcher = StrokeMatcher(config)

        few_pts = np.array([[7.0, 9.0], [7.1, 9.1]])
        tgt = np.array([[7.0, 9.0], [7.1, 9.1]])
        assert matcher.estimate_translation(few_pts, tgt) is None


class TestRunningCorrector:
    """Verifies EMA smoothing and magnitude capping."""

    def test_ema_smoothing(self):
        config = CorrectorConfig(ema_alpha=0.5, max_offset_cm=1.0)
        rc = RunningCorrector(config)

        # 1st observation: takes it directly
        o1 = rc.update(np.array([0.20, 0.20]))
        np.testing.assert_allclose(o1, [0.20, 0.20])

        # 2nd observation: 0.5 * 0.40 + 0.5 * 0.20 = 0.30
        o2 = rc.update(np.array([0.40, 0.40]))
        np.testing.assert_allclose(o2, [0.30, 0.30])

    def test_magnitude_capping(self):
        config = CorrectorConfig(ema_alpha=1.0, max_offset_cm=0.50)
        rc = RunningCorrector(config)

        # Offset of norm 1.0 cm (> max_offset 0.5 cm)
        large_offset = np.array([0.60, 0.80])   # norm = 1.0
        capped = rc.update(large_offset)
        assert math.isclose(np.linalg.norm(capped), 0.50, abs_tol=1e-5)
        np.testing.assert_allclose(capped, [0.30, 0.40])


class TestPerStrokeObserver:
    """Verifies image differencing and camera pixel to workspace conversion."""

    def test_observe_stroke_differencing(self):
        observer = PerStrokeObserver(DEFAULT_CAMERA_CONFIG, min_pixels=5)

        # Blank image
        h, w = DEFAULT_CAMERA_CONFIG.resolution_y, DEFAULT_CAMERA_CONFIG.resolution_x
        img0 = np.full((h, w, 3), 255, dtype=np.uint8)
        observer.capture_baseline(img0)

        # Draw ink pixels (dark blue in BGR: B=180, G=50, R=20)
        img1 = img0.copy()
        img1[250:260, 250:260] = [180, 50, 20]  # 100 ink pixels

        diff_mask, pts_cm = observer.observe_stroke(img1)
        assert np.count_nonzero(diff_mask) == 100
        assert len(pts_cm) == 100

        # All points should be near the center of the workspace (7.0, 9.5)
        for pt in pts_cm:
            assert 6.0 <= pt[0] <= 8.0
            assert 8.5 <= pt[1] <= 10.5
