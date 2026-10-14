
import math
import time
from src.coppelia_controller import CoppeliaController

controller = CoppeliaController()
sim = controller.sim

joint1 = controller.joint1
joint2 = controller.joint2
link1 = sim.getObject("/RobotBase/Joint1/Link1")
link2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2")

if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError("Stop CoppeliaSim first.")

def get_position(handle):
    m = sim.getObjectMatrix(handle, -1)
    return tuple(round(m[i] * 100, 2) for i in (3, 7, 11))

sim.startSimulation()

try:
    for a, b in [(0, 0), (30, 0), (0, 30), (30, 30)]:
        sim.setJointPosition(joint1, math.radians(a))
        sim.setJointPosition(joint2, math.radians(b))
        time.sleep(0.2)

        print(f"\nSet positions: J1={a}°, J2={b}°")
        print("Read positions:", tuple(
            round(math.degrees(sim.getJointPosition(j)), 2)
            for j in (joint1, joint2)
        ))
        print("Link1 world position (cm):", get_position(link1))
        print("Link2 world position (cm):", get_position(link2))

finally:
    sim.stopSimulation()
