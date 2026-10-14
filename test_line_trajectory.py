
import math
import time

from src.path_planner import plan_strokes
from src.coppelia_controller import CoppeliaController

# One horizontal stroke in robot-base coordinates (cm).
contours = [
    [
        (14.0, 2.0),
        (16.0, 2.0),
    ]
]

motion = plan_strokes(contours, spacing_cm=0.5)

print(f"Planned motion points: {len(motion)}")

controller = CoppeliaController()
sim = controller.sim

if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError(
        "Stop the CoppeliaSim simulation before running this test."
    )

sim.startSimulation()

try:
    for index, point in enumerate(motion, start=1):
        controller.set_joint_targets(point.theta1, point.theta2)

        # Allow time for the joint targets to settle in this initial test.
        time.sleep(0.2)

        actual1, actual2 = controller.get_joint_angles()

        print(
            f"{index:02d}/{len(motion)} "
            f"XY=({point.x:.2f}, {point.y:.2f}) cm | "
            f"target=({math.degrees(point.theta1):.2f}°, "
            f"{math.degrees(point.theta2):.2f}°) | "
            f"actual=({math.degrees(actual1):.2f}°, "
            f"{math.degrees(actual2):.2f}°) | "
            f"pen_down={point.pen_down}"
        )

finally:
    sim.stopSimulation()
