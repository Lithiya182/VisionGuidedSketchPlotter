
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import time

client = RemoteAPIClient()
sim = client.require("sim")

joint1 = sim.getObject("/RobotBase/Joint1")
joint2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2")

print("Connected to CoppeliaSim!")

# Check that the simulation is running
if sim.getSimulationState() == sim.simulation_stopped:
    print("Start the simulation in CoppeliaSim, then run this script again.")
else:
    # Command Joint 1 to 20 degrees
    target_rad = math.radians(20)
    sim.setJointTargetPosition(joint1, target_rad)

    print("Joint 1 target set to 20 degrees.")
    print("Joint 2 remains at its current target.")

    # Allow the controller time to respond
    for _ in range(20):
        time.sleep(0.1)
        angle1 = math.degrees(sim.getJointPosition(joint1))
        angle2 = math.degrees(sim.getJointPosition(joint2))
        print(f"Joint1: {angle1:.2f}°, Joint2: {angle2:.2f}°")
