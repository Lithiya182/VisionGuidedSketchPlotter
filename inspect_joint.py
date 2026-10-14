
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require("sim")

for path in [
    "/RobotBase/Joint1",
    "/RobotBase/Joint1/Link1",
    "/RobotBase/Joint1/Link1/Joint2",
    "/RobotBase/Joint1/Link1/Joint2/Link2",
]:
    handle = sim.getObject(path)
    print(f"\nObject: {path}")
    print(f"Handle: {handle}")
    print(f"Joint position (if joint): ", end="")

    try:
        print(sim.getJointPosition(handle))
    except Exception:
        print("Not a readable joint")
