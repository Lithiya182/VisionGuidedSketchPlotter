
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math

client = RemoteAPIClient()
sim = client.require("sim")

joint1 = sim.getObject("/RobotBase/Joint1")
joint2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2")

print("Connected to CoppeliaSim!")

# Reset both joint positions to zero
sim.setJointPosition(joint1, 0.0)
sim.setJointPosition(joint2, 0.0)

# Read back the positions
for name, handle in [("Joint1", joint1), ("Joint2", joint2)]:
    angle_rad = sim.getJointPosition(handle)
    angle_deg = math.degrees(angle_rad)
    print(f"{name}: {angle_deg:.2f} degrees")

print("\nJoint reset test completed.")
