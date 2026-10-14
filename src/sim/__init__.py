"""
src/sim/__init__.py

Simulation interfaces, plant disturbance injection, and CoppeliaSim bridge.
"""

from src.sim.disturbances import (
    DisturbanceConfig,
    DISTURBANCE_PRESETS,
    get_disturbance_preset,
    BacklashModel,
    PlantDisturbanceManager,
)

from src.sim.vision_corrector import (
    CorrectorConfig,
    PerStrokeObserver,
    StrokeMatcher,
    RunningCorrector,
    CalibrationPrePass,
)

__all__ = [
    "DisturbanceConfig",
    "DISTURBANCE_PRESETS",
    "get_disturbance_preset",
    "BacklashModel",
    "PlantDisturbanceManager",
    "CorrectorConfig",
    "PerStrokeObserver",
    "StrokeMatcher",
    "RunningCorrector",
    "CalibrationPrePass",
]
