
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math

client = RemoteAPIClient()
sim = client.require("sim")

joint1 = sim.getObject("/RobotBase/Joint1")
joint2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2")

for name, joint in [("Joint1", joint1), ("Joint2", joint2)]:
    position = math.degrees(sim.getJointPosition(joint))

    try:
        target = math.degrees(
            sim.getJointTargetPosition(joint)
        )
        target_text = f"{target:.2f}°"
    except Exception as error:
        target_text = f"Unavailable: {error}"

    print(f"{name} current position: {position:.2f}°")
    print(f"{name} target position:  {target_text}")
    print()
