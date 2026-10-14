
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import time

client = RemoteAPIClient()
sim = client.require("sim")

joint1 = sim.getObject("/RobotBase/Joint1")

print("Simulation state:", sim.getSimulationState())
print("Joint1 current angle:",
      math.degrees(sim.getJointPosition(joint1)))

if sim.getSimulationState() == sim.simulation_stopped:
    sim.startSimulation()
    time.sleep(1)

sim.setJointTargetPosition(joint1, math.radians(20))

for i in range(10):
    time.sleep(0.2)
    current = math.degrees(sim.getJointPosition(joint1))
    target = math.degrees(sim.getJointTargetPosition(joint1))
    print(f"Sample {i + 1}: current={current:.2f}°, target={target:.2f}°")

sim.stopSimulation()