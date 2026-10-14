"""
config.py

Centralized configuration for the Vision-Guided Sketch Plotter.
Contains robot kinematic parameters, workspace bounds, simulation settings,
planning parameters, and computer vision extraction settings.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RobotConfig:
    """Robot link geometry and joint limit constraints."""
    link1_length_cm: float = 10.0
    link2_length_cm: float = 10.0
    theta1_limits_deg: Tuple[float, float] = (-90.0, 90.0)
    theta2_limits_deg: Tuple[float, float] = (-150.0, 150.0)
    tolerance: float = 1e-9


@dataclass(frozen=True)
class WorkspaceConfig:
    """Drawing workspace dimensions and camera mapping parameters."""
    image_width: int = 480
    image_height: int = 480
    draw_origin_x: float = 4.0   # cm (top-left of drawing rectangle in arm space)
    draw_origin_y: float = 12.5  # cm
    draw_width_cm: float = 6.0   # cm
    draw_height_cm: float = 6.0  # cm
    min_radius_cm: float = 2.0   # cm
    max_radius_margin_cm: float = 2.0  # cm


@dataclass(frozen=True)
class SimConfig:
    """CoppeliaSim connection, scene objects, and controller settings."""
    host: str = "localhost"
    port: int = 23000
    base_path: str = "/RobotBase"
    joint1_path: str = "/RobotBase/Joint1"
    link1_path: str = "/RobotBase/Joint1/Link1"
    joint2_path: str = "/RobotBase/Joint1/Link1/Joint2"
    link2_path: str = "/RobotBase/Joint1/Link1/Joint2/Link2"
    tip_path: str = "/RobotBase/Joint1/Link1/Joint2/Link2/Tip"
    pid_p: float = 30.0
    pid_d: float = 1.0
    max_vel: float = 5.0      # rad/s (~286 deg/s)
    max_accel: float = 20.0   # rad/s^2
    default_step_size: float = 0.05
    position_tolerance_deg: float = 0.2
    tip_tolerance_cm: float = 0.05  # 0.5 mm tolerance between FK and simulated tip


@dataclass(frozen=True)
class PlannerConfig:
    """Path planner settings."""
    spacing_cm: float = 0.5
    default_branch: str = "positive"


@dataclass(frozen=True)
class VisionConfig:
    """Computer vision preprocessing, extraction, and simplification settings."""
    canvas_width: int = 480
    canvas_height: int = 480
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    canny_threshold_low: int = 50
    canny_threshold_high: int = 150
    adaptive_thresh_block_size: int = 11
    adaptive_thresh_c: int = 2
    min_contour_length_px: float = 12.0
    min_contour_area_px: float = 10.0
    simplification_epsilon_px: float = 1.5
    default_mode: str = "canny"  # "boundary", "canny", or "skeleton"


@dataclass(frozen=True)
class CameraConfig:
    """Top-down vision sensor parameters in CoppeliaSim."""
    camera_path: str = "/TopDownVisionSensor"
    resolution_x: int = 512
    resolution_y: int = 512
    view_size_cm: float = 12.0
    center_x_cm: float = 7.0
    center_y_cm: float = 9.5
    height_z_cm: float = 35.0
    line_thickness_px: int = 2


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation metrics configuration."""
    dt_seconds: float = 0.05
    pixel_size_mm: float = 120.0 / 512.0  # 0.234375 mm/px for 12cm FOV at 512x512


# Default global instances for convenience
DEFAULT_ROBOT_CONFIG = RobotConfig()
DEFAULT_WORKSPACE_CONFIG = WorkspaceConfig()
DEFAULT_SIM_CONFIG = SimConfig()
DEFAULT_PLANNER_CONFIG = PlannerConfig()
DEFAULT_VISION_CONFIG = VisionConfig()
DEFAULT_CAMERA_CONFIG = CameraConfig()
DEFAULT_EVAL_CONFIG = EvalConfig()

# Re-export disturbance configuration
from src.sim.disturbances import (
    DisturbanceConfig,
    DISTURBANCE_PRESETS,
    get_disturbance_preset,
)

