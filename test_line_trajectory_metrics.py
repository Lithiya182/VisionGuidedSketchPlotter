
import math
import time

from src.path_planner import plan_strokes
from src.coppelia_controller import CoppeliaController

motion = plan_strokes(
    [[(14.0, 2.0), (16.0, 2.0)]],
    spacing_cm=0.5,
)

controller = CoppeliaController()
sim = controller.sim
link2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2")

if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError("Stop CoppeliaSim before running this test.")

def get_tip_xy_cm():
    matrix = sim.getObjectMatrix(link2, -1)
    x = (matrix[0] * 0.05 + matrix[3]) * 100
    y = (matrix[4] * 0.05 + matrix[7]) * 100
    return x, y

sim.startSimulation()

try:
    for i, point in enumerate(motion, start=1):
        controller.set_joint_targets(point.theta1, point.theta2)
        print("Immediate targets (deg):", tuple(round(math.degrees(sim.getJointTargetPosition(j)), 2)
    for j in (controller.joint1, controller.joint2)
))
        
        time.sleep(1.0)

        actual_theta1, actual_theta2 = controller.get_joint_angles()
        actual_x, actual_y = get_tip_xy_cm()

        print(f"\nPoint {i}/{len(motion)}")
        print(f"Target XY: ({point.x:.2f}, {point.y:.2f}) cm")
        print(
            f"Target angles: "
            f"({math.degrees(point.theta1):.2f}°, "
            f"{math.degrees(point.theta2):.2f}°)"
        )
        print(
            f"Actual angles: "
            f"({math.degrees(actual_theta1):.2f}°, "
            f"{math.degrees(actual_theta2):.2f}°)"
        )
        print(f"Actual tip: ({actual_x:.2f}, {actual_y:.2f}) cm")

finally:
    sim.stopSimulation()
