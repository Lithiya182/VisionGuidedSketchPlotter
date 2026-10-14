
import math
import time
from src.coppelia_controller import CoppeliaController

controller = CoppeliaController()
sim = controller.sim

if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError("Stop CoppeliaSim first.")

sim.startSimulation()

try:
    controller.set_joint_targets(math.radians(20), 0.0)

    for _ in range(5):
        time.sleep(0.5)

        print(
            "Actual angles:",
            tuple(round(math.degrees(a), 2)
                  for a in controller.get_joint_angles())
        )

        print(
            "Joint 1 target:",
            math.degrees(sim.getJointTargetPosition(controller.joint1))
        )

finally:
    sim.stopSimulation()
