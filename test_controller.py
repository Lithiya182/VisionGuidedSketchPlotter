
from src.coppelia_controller import CoppeliaController
import math

controller = CoppeliaController()

theta1, theta2 = controller.get_joint_angles()

print("Connected to CoppeliaSim!")
print(f"Joint 1: {math.degrees(theta1):.2f}°")
print(f"Joint 2: {math.degrees(theta2):.2f}°")
