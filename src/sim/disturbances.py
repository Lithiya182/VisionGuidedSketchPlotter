"""
src/sim/disturbances.py

Plant disturbance models for the Vision-Guided Sketch Plotter:
1. Calibration offsets (link length delta and joint zero offsets applied in the CoppeliaSim scene).
2. Direction-dependent gear backlash / deadband play operator on joint reversals.
3. Seeded Gaussian joint actuator noise.

All disturbances act exclusively on the SIMULATED ROBOT (the plant).
The planner, forward/inverse kinematics, and nominal model remain completely unaware of them.
"""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np


@dataclass
class DisturbanceConfig:
    """
    Disturbance configuration parameters acting on the physical plant.

    Attributes:
        enabled: Master toggle for disturbance injection.
        name: Profile or preset name (e.g. 'none', 'mild', 'moderate', 'severe').
        delta_L1_cm: Physical Link 1 length error in cm (applied in sim scene).
        delta_L2_cm: Physical Link 2 length error in cm (applied in sim scene).
        delta_theta1_deg: Joint 1 zero calibration angular offset in degrees.
        delta_theta2_deg: Joint 2 zero calibration angular offset in degrees.
        backlash_theta1_deg: Joint 1 backlash deadband width in degrees.
        backlash_theta2_deg: Joint 2 backlash deadband width in degrees.
        noise_std_theta1_deg: Standard deviation of Gaussian actuator noise on Joint 1 in degrees.
        noise_std_theta2_deg: Standard deviation of Gaussian actuator noise on Joint 2 in degrees.
        noise_seed: Deterministic integer seed for Gaussian random number generator.
    """
    enabled: bool = False
    name: str = "none"
    delta_L1_cm: float = 0.0
    delta_L2_cm: float = 0.0
    delta_theta1_deg: float = 0.0
    delta_theta2_deg: float = 0.0
    backlash_theta1_deg: float = 0.0
    backlash_theta2_deg: float = 0.0
    noise_std_theta1_deg: float = 0.0
    noise_std_theta2_deg: float = 0.0
    noise_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisturbanceConfig":
        """Instantiate configuration from dictionary."""
        return cls(**data)


DISTURBANCE_PRESETS: Dict[str, DisturbanceConfig] = {
    "none": DisturbanceConfig(
        enabled=False,
        name="none",
        delta_L1_cm=0.0,
        delta_L2_cm=0.0,
        delta_theta1_deg=0.0,
        delta_theta2_deg=0.0,
        backlash_theta1_deg=0.0,
        backlash_theta2_deg=0.0,
        noise_std_theta1_deg=0.0,
        noise_std_theta2_deg=0.0,
        noise_seed=None,
    ),
    "mild": DisturbanceConfig(
        enabled=True,
        name="mild",
        delta_L1_cm=0.05,            # +0.5 mm
        delta_L2_cm=-0.05,           # -0.5 mm
        delta_theta1_deg=0.30,       # +0.3 deg
        delta_theta2_deg=-0.30,      # -0.3 deg
        backlash_theta1_deg=0.25,    # 0.25 deg deadband
        backlash_theta2_deg=0.25,    # 0.25 deg deadband
        noise_std_theta1_deg=0.05,   # 0.05 deg Gaussian jitter
        noise_std_theta2_deg=0.05,   # 0.05 deg Gaussian jitter
        noise_seed=42,
    ),
    "moderate": DisturbanceConfig(
        enabled=True,
        name="moderate",
        delta_L1_cm=0.15,            # +1.5 mm
        delta_L2_cm=-0.10,           # -1.0 mm
        delta_theta1_deg=0.80,       # +0.8 deg
        delta_theta2_deg=-0.80,      # -0.8 deg
        backlash_theta1_deg=0.50,    # 0.50 deg deadband
        backlash_theta2_deg=0.60,    # 0.60 deg deadband
        noise_std_theta1_deg=0.12,   # 0.12 deg Gaussian jitter
        noise_std_theta2_deg=0.12,   # 0.12 deg Gaussian jitter
        noise_seed=42,
    ),
    "severe": DisturbanceConfig(
        enabled=True,
        name="severe",
        delta_L1_cm=0.30,            # +3.0 mm
        delta_L2_cm=-0.25,           # -2.5 mm
        delta_theta1_deg=1.50,       # +1.5 deg
        delta_theta2_deg=-1.50,      # -1.5 deg
        backlash_theta1_deg=1.00,    # 1.00 deg deadband
        backlash_theta2_deg=1.20,    # 1.20 deg deadband
        noise_std_theta1_deg=0.25,   # 0.25 deg Gaussian jitter
        noise_std_theta2_deg=0.25,   # 0.25 deg Gaussian jitter
        noise_seed=42,
    ),
    "offset_only": DisturbanceConfig(
        enabled=True,
        name="offset_only",
        delta_L1_cm=0.15,            # +1.5 mm
        delta_L2_cm=-0.10,           # -1.0 mm
        delta_theta1_deg=0.80,       # +0.8 deg
        delta_theta2_deg=-0.80,      # -0.8 deg
        backlash_theta1_deg=0.0,
        backlash_theta2_deg=0.0,
        noise_std_theta1_deg=0.0,
        noise_std_theta2_deg=0.0,
        noise_seed=None,
    ),
    "backlash_only": DisturbanceConfig(
        enabled=True,
        name="backlash_only",
        delta_L1_cm=0.0,
        delta_L2_cm=0.0,
        delta_theta1_deg=0.0,
        delta_theta2_deg=0.0,
        backlash_theta1_deg=0.50,    # 0.50 deg deadband
        backlash_theta2_deg=0.60,    # 0.60 deg deadband
        noise_std_theta1_deg=0.0,
        noise_std_theta2_deg=0.0,
        noise_seed=None,
    ),
    "noise_only": DisturbanceConfig(
        enabled=True,
        name="noise_only",
        delta_L1_cm=0.0,
        delta_L2_cm=0.0,
        delta_theta1_deg=0.0,
        delta_theta2_deg=0.0,
        backlash_theta1_deg=0.0,
        backlash_theta2_deg=0.0,
        noise_std_theta1_deg=0.12,   # 0.12 deg Gaussian jitter
        noise_std_theta2_deg=0.12,   # 0.12 deg Gaussian jitter
        noise_seed=42,
    ),
}


def get_disturbance_preset(
    preset_name: str,
    seed: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DisturbanceConfig:
    """
    Retrieve a disturbance preset by name with optional seed and parameter overrides.

    Parameters:
        preset_name: One of 'none', 'mild', 'moderate', 'severe'.
        seed: Optional RNG seed to override the preset default.
        overrides: Optional dictionary of parameter overrides.

    Returns:
        Configured DisturbanceConfig instance.
    """
    key = preset_name.strip().lower()
    if key not in DISTURBANCE_PRESETS:
        valid_keys = list(DISTURBANCE_PRESETS.keys())
        raise ValueError(f"Unknown disturbance preset '{preset_name}'. Valid presets: {valid_keys}")

    base = DISTURBANCE_PRESETS[key]
    cfg_dict = base.to_dict()

    if seed is not None:
        cfg_dict["noise_seed"] = seed

    if overrides:
        cfg_dict.update(overrides)

    return DisturbanceConfig.from_dict(cfg_dict)


class BacklashModel:
    """
    Direction-dependent mechanical gear backlash / deadband play operator for a revolute joint.

    Implements the classical continuous play hysteresis operator:
    When the motor shaft reverses, it traverses an angular deadband of width B
    before re-engaging the gear teeth in the new direction.

    Parameters:
        backlash_deg: Total deadband width B in degrees.
        initial_position_deg: Initial joint position in degrees.
    """
    def __init__(self, backlash_deg: float = 0.0, initial_position_deg: float = 0.0):
        self.backlash_rad = math.radians(max(0.0, backlash_deg))
        self.half_b = self.backlash_rad / 2.0
        self.output_rad = math.radians(initial_position_deg)
        self.reversal_count = 0
        self._last_direction = 0  # +1 positive, -1 negative, 0 initial

    def reset(self, initial_position_deg: float = 0.0) -> None:
        """Reset internal hysteresis state."""
        self.output_rad = math.radians(initial_position_deg)
        self.reversal_count = 0
        self._last_direction = 0

    def step(self, input_rad: float) -> float:
        """
        Compute output angle given new input motor angle in radians.

        If input is inside the deadband [-B/2, +B/2] relative to current output,
        the output angle remains unchanged.
        """
        if self.backlash_rad <= 1e-9:
            self.output_rad = input_rad
            return input_rad

        diff = input_rad - self.output_rad

        if diff > self.half_b:
            if self._last_direction == -1:
                self.reversal_count += 1
            self._last_direction = 1
            self.output_rad = input_rad - self.half_b
        elif diff < -self.half_b:
            if self._last_direction == 1:
                self.reversal_count += 1
            self._last_direction = -1
            self.output_rad = input_rad + self.half_b
        else:
            # Inside deadband: gear teeth disengaged, link remains static
            pass

        return self.output_rad


class PlantDisturbanceManager:
    """
    Manages physical disturbance injection on the simulated CoppeliaSim plant:
    1. Modifies CoppeliaSim scene transforms (link lengths and joint zero orientations).
    2. Modifies actuator commands sent to the plant (backlash play operator and seeded noise).
    3. Guarantees clean restoration of nominal scene transforms after execution.
    """
    # Nominal known baseline transforms
    NOMINAL_J2_POS = [0.10, 0.0, 0.005]  # Link 1 = 10 cm
    NOMINAL_TIP_POS = [0.10, 0.0, 0.0]   # Link 2 = 10 cm
    NOMINAL_J1_ORI = [0.0, 0.0, 0.0]
    NOMINAL_J2_ORI = [0.0, 0.0, 0.0]

    def __init__(self, config: Optional[DisturbanceConfig] = None):
        self.config = config or DISTURBANCE_PRESETS["none"]
        self.backlash1 = BacklashModel(self.config.backlash_theta1_deg)
        self.backlash2 = BacklashModel(self.config.backlash_theta2_deg)

        if self.config.noise_seed is not None:
            self.rng = np.random.RandomState(self.config.noise_seed)
        else:
            self.rng = np.random.RandomState()

        self._scene_modified = False
        self._saved_transforms: Dict[str, Any] = {}

    def reset_state(self, initial_theta1_deg: float = 0.0, initial_theta2_deg: float = 0.0) -> None:
        """Reset internal backlash and RNG states."""
        self.backlash1.reset(initial_theta1_deg)
        self.backlash2.reset(initial_theta2_deg)
        if self.config.noise_seed is not None:
            self.rng = np.random.RandomState(self.config.noise_seed)

    def apply_scene_disturbances(self, sim: Any) -> Dict[str, Any]:
        """
        Apply kinematic calibration offsets (link lengths & joint zero orientations)
        directly to CoppeliaSim scene objects.

        Saves previous transforms so they can be restored exactly.
        """
        if not self.config.enabled:
            return {"applied": False}

        base = sim.getObject("/RobotBase")
        j1 = sim.getObject("/RobotBase/Joint1")
        link1 = sim.getObject("/RobotBase/Joint1/Link1")
        j2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2")
        tip = sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2/Tip")

        # Record pre-disturbance transforms
        cur_j2_pos = sim.getObjectPosition(j2, j1)
        cur_tip_pos = sim.getObjectPosition(tip, j2)
        cur_j1_ori = sim.getObjectOrientation(j1, base)
        cur_j2_ori = sim.getObjectOrientation(j2, link1)

        self._saved_transforms = {
            "j2_pos": cur_j2_pos,
            "tip_pos": cur_tip_pos,
            "j1_ori": cur_j1_ori,
            "j2_ori": cur_j2_ori,
        }

        # Compute disturbed parameters (converting cm to meters)
        d_L1_m = self.config.delta_L1_cm / 100.0
        d_L2_m = self.config.delta_L2_cm / 100.0
        d_theta1_rad = math.radians(self.config.delta_theta1_deg)
        d_theta2_rad = math.radians(self.config.delta_theta2_deg)

        new_j2_pos = [self.NOMINAL_J2_POS[0] + d_L1_m, self.NOMINAL_J2_POS[1], self.NOMINAL_J2_POS[2]]
        new_tip_pos = [self.NOMINAL_TIP_POS[0] + d_L2_m, self.NOMINAL_TIP_POS[1], self.NOMINAL_TIP_POS[2]]
        new_j1_ori = [self.NOMINAL_J1_ORI[0], self.NOMINAL_J1_ORI[1], self.NOMINAL_J1_ORI[2] + d_theta1_rad]
        new_j2_ori = [self.NOMINAL_J2_ORI[0], self.NOMINAL_J2_ORI[1], self.NOMINAL_J2_ORI[2] + d_theta2_rad]

        sim.setObjectPosition(j2, j1, new_j2_pos)
        sim.setObjectPosition(tip, j2, new_tip_pos)
        sim.setObjectOrientation(j1, base, new_j1_ori)
        sim.setObjectOrientation(j2, link1, new_j2_ori)

        self._scene_modified = True
        return {
            "applied": True,
            "delta_L1_cm": self.config.delta_L1_cm,
            "delta_L2_cm": self.config.delta_L2_cm,
            "delta_theta1_deg": self.config.delta_theta1_deg,
            "delta_theta2_deg": self.config.delta_theta2_deg,
        }

    def restore_nominal_scene(self, sim: Any) -> bool:
        """
        Restore CoppeliaSim scene transforms back to their exact nominal baseline.
        """
        if not self._scene_modified:
            return False

        try:
            base = sim.getObject("/RobotBase")
            j1 = sim.getObject("/RobotBase/Joint1")
            link1 = sim.getObject("/RobotBase/Joint1/Link1")
            j2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2")
            tip = sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2/Tip")

            # Fall back to known nominals if saved transforms are missing
            j2_pos = self._saved_transforms.get("j2_pos", self.NOMINAL_J2_POS)
            tip_pos = self._saved_transforms.get("tip_pos", self.NOMINAL_TIP_POS)
            j1_ori = self._saved_transforms.get("j1_ori", self.NOMINAL_J1_ORI)
            j2_ori = self._saved_transforms.get("j2_ori", self.NOMINAL_J2_ORI)

            sim.setObjectPosition(j2, j1, j2_pos)
            sim.setObjectPosition(tip, j2, tip_pos)
            sim.setObjectOrientation(j1, base, j1_ori)
            sim.setObjectOrientation(j2, link1, j2_ori)

            self._scene_modified = False
            return True
        except Exception as exc:
            print(f"[PlantDisturbanceManager] Warning during scene restore: {exc}")
            return False

    def transform_plant_command(
        self,
        theta1_nominal_rad: float,
        theta2_nominal_rad: float,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Transform nominal commanded joint angles into disturbed plant actuator commands.
        Applies:
        1. Backlash play operator on joint reversals.
        2. Seeded Gaussian joint noise.

        Returns:
            (theta1_plant_rad, theta2_plant_rad, disturbance_telemetry_dict)
        """
        if not self.config.enabled:
            return theta1_nominal_rad, theta2_nominal_rad, {
                "backlash_error1_deg": 0.0,
                "backlash_error2_deg": 0.0,
                "noise1_deg": 0.0,
                "noise2_deg": 0.0,
            }

        # 1. Backlash deadband on input motor angle
        theta1_bl = self.backlash1.step(theta1_nominal_rad)
        theta2_bl = self.backlash2.step(theta2_nominal_rad)

        # 2. Additive Gaussian noise
        std1_rad = math.radians(self.config.noise_std_theta1_deg)
        std2_rad = math.radians(self.config.noise_std_theta2_deg)

        noise1_rad = float(self.rng.normal(0.0, std1_rad)) if std1_rad > 0.0 else 0.0
        noise2_rad = float(self.rng.normal(0.0, std2_rad)) if std2_rad > 0.0 else 0.0

        theta1_plant = theta1_bl + noise1_rad
        theta2_plant = theta2_bl + noise2_rad

        telem = {
            "backlash_error1_deg": math.degrees(theta1_bl - theta1_nominal_rad),
            "backlash_error2_deg": math.degrees(theta2_bl - theta2_nominal_rad),
            "noise1_deg": math.degrees(noise1_rad),
            "noise2_deg": math.degrees(noise2_rad),
        }

        return theta1_plant, theta2_plant, telem

    @contextmanager
    def active_context(self, sim: Any) -> Generator["PlantDisturbanceManager", None, None]:
        """
        Context manager that applies scene disturbances upon entry and guarantees
        nominal scene restoration upon exit, even if errors occur.
        """
        try:
            self.apply_scene_disturbances(sim)
            yield self
        finally:
            self.restore_nominal_scene(sim)
