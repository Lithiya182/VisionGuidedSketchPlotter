
from src.coppelia_controller import CoppeliaController

controller = CoppeliaController()
sim = controller.sim

for name, handle in [
    ("Joint1", controller.joint1),
    ("Joint2", controller.joint2),
]:
    print(f"\n{name}:")
    print("Joint mode:", sim.getJointMode(handle))
    print("Joint type:", sim.getJointType(handle))
    print("Joint position:", sim.getJointPosition(handle))
