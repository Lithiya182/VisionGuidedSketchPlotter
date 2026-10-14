
import math
import time

from src.kinematics import inverse_kinematics
from src.coppelia_controller import CoppeliaController

# Target position in centimeters, relative to the robot base.
target_x = 15.0
target_y = 5.0

# Match the configured CoppeliaSim joint limits.
theta1, theta2 = inverse_kinematics(
    target_x,
    target_y,
    branch="positive",
    theta1_limits_deg=(-90.0, 90.0),
    theta2_limits_deg=(-150.0, 150.0),
)

print("Cartesian target:", (target_x, target_y), "cm")
print(f"IK Joint 1: {math.degrees(theta1):.2f}°")
print(f"IK Joint 2: {math.degrees(theta2):.2f}°")

controller = CoppeliaController()

if controller.sim.getSimulationState() == controller.sim.simulation_stopped:
    controller.sim.startSimulation()
    time.sleep(0.5)

controller.set_joint_targets(theta1, theta2)

# Allow the joints time to reach their targets.
for sample in range(10):
    time.sleep(0.2)
    actual1, actual2 = controller.get_joint_angles()
    print(
        f"Sample {sample + 1}: "
        f"Joint1={math.degrees(actual1):.2f}°, "
        f"Joint2={math.degrees(actual2):.2f}°"
    )
