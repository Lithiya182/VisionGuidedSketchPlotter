
import time
from src.coppelia_controller import CoppeliaController
from src.kinematics import forward_kinematics

controller = CoppeliaController()
sim = controller.sim

link2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2")

if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError("Stop the simulation before running this test.")

def get_tip_xy_cm():
    matrix = sim.getObjectMatrix(link2, -1)

    # Far endpoint of Link2: local x = +0.05 m
    x_m = matrix[0] * 0.05 + matrix[3]
    y_m = matrix[4] * 0.05 + matrix[7]

    return x_m * 100, y_m * 100

sim.startSimulation()

try:
    controller.set_joint_targets(0.0, 0.0)
    time.sleep(1.0)

    actual_x, actual_y = get_tip_xy_cm()
    print(f"CoppeliaSim tip: ({actual_x:.2f}, {actual_y:.2f}) cm")

    # Adjust this call only if your function uses different argument names.
    expected = forward_kinematics(0.0, 0.0)
    print(f"Python FK tip:    {expected}")

finally:
    sim.stopSimulation()
