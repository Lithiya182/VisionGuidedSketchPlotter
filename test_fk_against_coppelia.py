
import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from src.kinematics import forward_kinematics

client = RemoteAPIClient()
sim = client.require("sim")

joint1 = sim.getObject("/RobotBase/Joint1")
joint2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2")
link2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2")

theta1 = sim.getJointPosition(joint1)
theta2 = sim.getJointPosition(joint2)

# Python FK uses centimeters.
fk_x, fk_y, _ = forward_kinematics(theta1, theta2)

# Far endpoint of Link2, expressed in its local frame.
matrix = sim.getObjectMatrix(link2, -1)
tip_x = matrix[0] * 0.05 + matrix[3]
tip_y = matrix[4] * 0.05 + matrix[7]
tip_z = matrix[8] * 0.05 + matrix[11]

print(f"Joint angles: {math.degrees(theta1):.2f}°, "
      f"{math.degrees(theta2):.2f}°")
print(f"Python FK: ({fk_x:.3f}, {fk_y:.3f}) cm")
print(f"CoppeliaSim tip XY: ({tip_x * 100:.3f}, "
      f"{tip_y * 100:.3f}) cm")
print(f"CoppeliaSim tip Z: {tip_z * 100:.3f} cm")
print(f"XY error: {math.hypot(fk_x - tip_x * 100, fk_y - tip_y * 100):.4f} cm")
