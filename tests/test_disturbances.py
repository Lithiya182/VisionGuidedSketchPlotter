"""
tests/test_disturbances.py

Unit tests for Phase 4 disturbance injection models:
1. Direction-dependent backlash deadband play operator on synthetic reversals.
2. Seeded Gaussian noise determinism (identical seed -> identical output).
3. Disturbance disabled ('none') preserves exact nominal commands.
4. Monotonic disturbance scaling across presets (none < mild < moderate < severe).
5. CoppeliaSim scene modification and restoration idempotence.
6. Tolerance-matched IoU vs plain IoU behavior.
"""

import math
import numpy as np
import pytest

from src.sim.disturbances import (
    BacklashModel,
    DisturbanceConfig,
    DISTURBANCE_PRESETS,
    get_disturbance_preset,
    PlantDisturbanceManager,
)
from src.eval.metrics import raster_iou


# ==============================================================================
# 1. Backlash Play Operator & Deadband Tests
# ==============================================================================

def test_backlash_reproduces_known_deadband_on_synthetic_reversal():
    """
    Verify that BacklashModel reproduces the exact theoretical deadband B
    upon direction reversal.
    """
    backlash_deg = 1.0  # 1.0 degree total deadband (half_b = 0.5 deg)
    model = BacklashModel(backlash_deg=backlash_deg, initial_position_deg=0.0)

    # 1. Drive forward continuously: 0 -> 2 -> 4 -> 6 -> 8 -> 10 deg
    forward_angles = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    forward_outputs = [math.degrees(model.step(math.radians(a))) for a in forward_angles]

    # After initial engagement, output should follow input with offset -half_b (-0.5 deg)
    assert forward_outputs[-1] == pytest.approx(10.0 - 0.5, abs=1e-5)
    assert model.reversal_count == 0

    # 2. Reverse inside deadband: reverse by 0.4 deg (from 10.0 to 9.6 deg)
    # The reversal change is 0.4 deg < 1.0 deg (deadband width) -> output must NOT change!
    rev_in_deadband_1 = math.degrees(model.step(math.radians(9.6)))
    assert rev_in_deadband_1 == pytest.approx(9.5, abs=1e-5)  # unchanged!

    rev_in_deadband_2 = math.degrees(model.step(math.radians(9.1)))
    # Total reversal is 0.9 deg < 1.0 deg -> output still unchanged!
    assert rev_in_deadband_2 == pytest.approx(9.5, abs=1e-5)

    # 3. Traverse through deadband: reversal reaches 1.0 deg (input = 9.0 deg)
    # At exact boundary, output still 9.5
    boundary_output = math.degrees(model.step(math.radians(9.0)))
    assert boundary_output == pytest.approx(9.5, abs=1e-5)

    # 4. Exceed deadband: reverse further to 8.0 deg (total reversal of 2.0 deg)
    # Output must engage on opposite flank: 8.0 + half_b = 8.5 deg!
    # Output moved by 9.5 - 8.5 = 1.0 deg (which is 2.0 input change - 1.0 deadband)
    exceeded_output = math.degrees(model.step(math.radians(8.0)))
    assert exceeded_output == pytest.approx(8.0 + 0.5, abs=1e-5)
    assert model.reversal_count == 1

    # 5. Reverse back to positive: move from 8.0 to 8.4 deg (reversal of 0.4 deg < 1.0 deg)
    pos_in_deadband = math.degrees(model.step(math.radians(8.4)))
    assert pos_in_deadband == pytest.approx(8.5, abs=1e-5)  # unchanged!

    # Move to 9.5 deg (reversal of 1.5 deg > 1.0 deg)
    pos_engaged = math.degrees(model.step(math.radians(9.5)))
    assert pos_engaged == pytest.approx(9.5 - 0.5, abs=1e-5)
    assert model.reversal_count == 2


def test_zero_backlash_passes_transparently():
    """Zero backlash model should pass all angles with exact 1:1 identity."""
    model = BacklashModel(backlash_deg=0.0)
    for ang in [-45.0, -10.0, 0.0, 30.0, 15.0, 60.0]:
        out = math.degrees(model.step(math.radians(ang)))
        assert out == pytest.approx(ang)


# ==============================================================================
# 2. Seeded Gaussian Noise & Plant Disturbance Determinism
# ==============================================================================

def test_identical_seed_gives_identical_output():
    """
    Two PlantDisturbanceManager instances with identical seeds must produce
    numerically bitwise-identical output trajectories.
    """
    cfg1 = get_disturbance_preset("moderate", seed=12345)
    cfg2 = get_disturbance_preset("moderate", seed=12345)

    mgr1 = PlantDisturbanceManager(cfg1)
    mgr2 = PlantDisturbanceManager(cfg2)

    # Command trajectory with direction changes
    command_sequence = [
        (math.radians(10.0), math.radians(20.0)),
        (math.radians(15.0), math.radians(25.0)),
        (math.radians(12.0), math.radians(22.0)),  # reversal
        (math.radians(8.0),  math.radians(18.0)),  # continue reverse
        (math.radians(14.0), math.radians(24.0)),  # forward reversal
    ]

    outputs1 = [mgr1.transform_plant_command(t1, t2) for t1, t2 in command_sequence]
    outputs2 = [mgr2.transform_plant_command(t1, t2) for t1, t2 in command_sequence]

    for (p1_1, p2_1, telem1), (p1_2, p2_2, telem2) in zip(outputs1, outputs2):
        assert p1_1 == pytest.approx(p1_2, abs=1e-12)
        assert p2_1 == pytest.approx(p2_2, abs=1e-12)
        assert telem1["noise1_deg"] == pytest.approx(telem2["noise1_deg"], abs=1e-12)
        assert telem1["noise2_deg"] == pytest.approx(telem2["noise2_deg"], abs=1e-12)


def test_different_seeds_give_different_outputs():
    """Different seeds must produce different noise realizations."""
    cfg1 = get_disturbance_preset("moderate", seed=100)
    cfg2 = get_disturbance_preset("moderate", seed=200)

    mgr1 = PlantDisturbanceManager(cfg1)
    mgr2 = PlantDisturbanceManager(cfg2)

    out1 = [mgr1.transform_plant_command(0.5, 0.5) for _ in range(10)]
    out2 = [mgr2.transform_plant_command(0.5, 0.5) for _ in range(10)]

    diffs = [abs(o1[0] - o2[0]) for o1, o2 in zip(out1, out2)]
    assert np.mean(diffs) > 1e-4


# ==============================================================================
# 3. Disturbance 'none' Preserves Nominal Commands Exactly
# ==============================================================================

def test_disturbance_none_preserves_nominal_commands():
    """Preset 'none' must not alter nominal joint angles in any way."""
    cfg = get_disturbance_preset("none")
    mgr = PlantDisturbanceManager(cfg)

    test_angles = [(0.1, -0.2), (0.5, 1.2), (-0.8, 0.4), (0.0, 0.0)]
    for t1, t2 in test_angles:
        p1, p2, telem = mgr.transform_plant_command(t1, t2)
        assert p1 == t1
        assert p2 == t2
        assert telem["backlash_error1_deg"] == 0.0
        assert telem["backlash_error2_deg"] == 0.0
        assert telem["noise1_deg"] == 0.0
        assert telem["noise2_deg"] == 0.0


# ==============================================================================
# 4. Preset Monotonic Error Scaling
# ==============================================================================

def test_presets_monotonic_scaling():
    """Disturbance parameters must scale monotonically from none to severe."""
    p_none = DISTURBANCE_PRESETS["none"]
    p_mild = DISTURBANCE_PRESETS["mild"]
    p_mod = DISTURBANCE_PRESETS["moderate"]
    p_sev = DISTURBANCE_PRESETS["severe"]

    assert p_none.delta_L1_cm < p_mild.delta_L1_cm < p_mod.delta_L1_cm < p_sev.delta_L1_cm
    assert p_none.delta_theta1_deg < p_mild.delta_theta1_deg < p_mod.delta_theta1_deg < p_sev.delta_theta1_deg
    assert p_none.backlash_theta1_deg < p_mild.backlash_theta1_deg < p_mod.backlash_theta1_deg < p_sev.backlash_theta1_deg
    assert p_none.noise_std_theta1_deg < p_mild.noise_std_theta1_deg < p_mod.noise_std_theta1_deg < p_sev.noise_std_theta1_deg


# ==============================================================================
# 5. Tolerance-Matched IoU vs Plain IoU
# ==============================================================================

def test_tolerance_matched_iou_properties():
    """
    Tolerance-matched IoU must be >= plain IoU, and equal 1.0 for identical masks.
    """
    mask_a = np.zeros((100, 100), dtype=np.uint8)
    mask_a[30:70, 30:70] = 255

    # Shifted mask by 3 pixels (within tolerance of 2 pixels dilation)
    mask_b = np.zeros((100, 100), dtype=np.uint8)
    mask_b[30:70, 33:73] = 255

    res = raster_iou(mask_a, mask_b, tolerance_px=2)

    # Tolerance IoU must be substantially higher than plain IoU
    assert res["tolerance_iou"] > res["plain_iou"]
    assert res["tolerance_iou_pct"] > res["plain_iou_pct"]
    assert res["tolerance_px"] == 2

    # For identical masks, both should be 1.0
    res_ident = raster_iou(mask_a, mask_a, tolerance_px=2)
    assert res_ident["tolerance_iou"] == pytest.approx(1.0)
    assert res_ident["plain_iou"] == pytest.approx(1.0)


# ==============================================================================
# 6. CoppeliaSim Scene Apply and Restore Test
# ==============================================================================

def test_sim_scene_disturbance_apply_and_restore():
    """
    Test applying scene disturbances to CoppeliaSim and restoring the nominal scene.
    Skips if CoppeliaSim is not running.
    """
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient()
        sim = client.require("sim")
        # Quick check that simulation objects exist
        j1 = sim.getObject("/RobotBase/Joint1")
    except Exception:
        pytest.skip("CoppeliaSim remote API not available.")

    cfg = get_disturbance_preset("severe")
    mgr = PlantDisturbanceManager(cfg)

    link1 = sim.getObject("/RobotBase/Joint1/Link1")
    j2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2")
    tip = sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2/Tip")

    # Record original transforms
    orig_j2_pos = sim.getObjectPosition(j2, j1)
    orig_tip_pos = sim.getObjectPosition(tip, j2)

    # Use active context
    with mgr.active_context(sim):
        dist_j2_pos = sim.getObjectPosition(j2, j1)
        dist_tip_pos = sim.getObjectPosition(tip, j2)
        # Link 1 should be shifted by +0.3 cm = +0.003 m
        assert dist_j2_pos[0] == pytest.approx(orig_j2_pos[0] + 0.003, abs=1e-5)
        # Link 2 should be shifted by -0.25 cm = -0.0025 m
        assert dist_tip_pos[0] == pytest.approx(orig_tip_pos[0] - 0.0025, abs=1e-5)

    # After exit, nominal positions must be restored
    restored_j2_pos = sim.getObjectPosition(j2, j1)
    restored_tip_pos = sim.getObjectPosition(tip, j2)
    assert restored_j2_pos[0] == pytest.approx(orig_j2_pos[0], abs=1e-5)
    assert restored_tip_pos[0] == pytest.approx(orig_tip_pos[0], abs=1e-5)
