
import math
import time
from src.coppelia_controller import CoppeliaController

controller = CoppeliaController()
sim = controller.sim

objects = {
    "Joint1": controller.joint1,
    "Link1": sim.getObject("/RobotBase/Joint1/Link1"),
    "Joint2": controller.joint2,
    "Link2": sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2"),
}

if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError("Stop CoppeliaSim first.")

sim.startSimulation()

try:
    for j1_deg, j2_deg in [(0, 0), (20, 0), (0, 30), (20, 30)]:
        controller.set_joint_targets(
            math.radians(j1_deg),
            math.radians(j2_deg),
        )
        time.sleep(0.5)

        print(f"\nRequested: J1={j1_deg}°, J2={j2_deg}°")
        print(
            "Actual angles:",
            tuple(round(math.degrees(a), 2)
                  for a in controller.get_joint_angles()),
        )

        for name, handle in objects.items():
            m = sim.getObjectMatrix(handle, -1)
            print(
                f"{name}: position cm="
                f"({m[3]*100:.2f}, {m[7]*100:.2f}, {m[11]*100:.2f})"
            )
            print(
                f"  X axis=({m[0]:.3f}, {m[4]:.3f}, {m[8]:.3f})"
                f"  Y axis=({m[1]:.3f}, {m[5]:.3f}, {m[9]:.3f})"
                f"  Z axis=({m[2]:.3f}, {m[6]:.3f}, {m[10]:.3f})"
            )
finally:
    sim.stopSimulation()
