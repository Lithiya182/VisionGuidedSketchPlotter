
import math
import time

from src.coppelia_controller import CoppeliaController

controller = CoppeliaController()
sim = controller.sim

link2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2")

def get_tip_xy_cm():
    matrix = sim.getObjectMatrix(link2, -1)
    x = (matrix[0] * 0.05 + matrix[3]) * 100
    y = (matrix[4] * 0.05 + matrix[7]) * 100
    return x, y

if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError("Stop CoppeliaSim before running this test.")

tests = [
    (20.0, 0.0),
    (0.0, 30.0),
    (20.0, 30.0),
]

sim.startSimulation()

try:
    for theta1_deg, theta2_deg in tests:
        theta1 = math.radians(theta1_deg)
        theta2 = math.radians(theta2_deg)

        controller.set_joint_targets(theta1, theta2)
        time.sleep(1.5)

        actual_theta1, actual_theta2 = controller.get_joint_angles()
        actual_x, actual_y = get_tip_xy_cm()

        print(f"\nRequested joints: ({theta1_deg:.1f}°, {theta2_deg:.1f}°)")
        print(
            f"Actual joints:    "
            f"({math.degrees(actual_theta1):.1f}°, "
            f"{math.degrees(actual_theta2):.1f}°)"
        )
        print(f"Actual tip:       ({actual_x:.2f}, {actual_y:.2f}) cm")

finally:
    sim.stopSimulation()
