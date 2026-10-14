
import math
import time
from src.coppelia_controller import CoppeliaController

controller = CoppeliaController()
sim = controller.sim

if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError("Stop CoppeliaSim first.")

sim.startSimulation()

try:
    for target_deg in [0, 10, 20, 10, 0]:
        target = math.radians(target_deg)
        controller.set_joint_targets(target, 0.0)

        time.sleep(2.0)

        actual1, actual2 = controller.get_joint_angles()

        print(
            f"Target J1={target_deg:>2}° | "
            f"Actual J1={math.degrees(actual1):>6.2f}° | "
            f"Actual J2={math.degrees(actual2):>6.2f}°"
        )
finally:
    sim.stopSimulation()
