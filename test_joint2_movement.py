
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import time

client = RemoteAPIClient()
sim = client.require("sim")

joint1 = sim.getObject("/RobotBase/Joint1")
joint2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2")

if sim.getSimulationState() == sim.simulation_stopped:
    print("Start the simulation in CoppeliaSim first.")
else:
    # Set the target for Joint 1
    sim.setJointTargetPosition(joint1, math.radians(20))

    # Set the target for Joint 2
    sim.setJointTargetPosition(joint2, math.radians(30))

    print("Joint 1 target: 20 degrees")
    print("Joint 2 target: 30 degrees")

    for _ in range(20):
        time.sleep(0.1)

        angle1 = math.degrees(sim.getJointPosition(joint1))
        angle2 = math.degrees(sim.getJointPosition(joint2))

        print(
            f"Joint1: {angle1:.2f}°, "
            f"Joint2: {angle2:.2f}°"
        )
