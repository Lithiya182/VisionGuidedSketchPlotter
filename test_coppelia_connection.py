
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

try:
    client = RemoteAPIClient()
    sim = client.require("sim")

    print("Connected to CoppeliaSim successfully!")

    objects = [
        "/RobotBase",
        "/RobotBase/Joint1",
        "/RobotBase/Joint1/Link1",
        "/RobotBase/Joint1/Link1/Joint2",
        "/RobotBase/Joint1/Link1/Joint2/Link2",
    ]

    for path in objects:
        try:
            handle = sim.getObject(path)
            print(f"FOUND: {path} | Handle: {handle}")
        except Exception as error:
            print(f"NOT FOUND: {path} | {error}")

    print("\nConnection test completed.")

except Exception as error:
    print("Connection failed!")
    print(f"Error: {error}")
