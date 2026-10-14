
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class CoppeliaController:
    def __init__(self, host="localhost", port=23000):
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.require("sim")

        self.joint1 = self.sim.getObject("/RobotBase/Joint1")
        self.joint2 = self.sim.getObject(
            "/RobotBase/Joint1/Link1/Joint2"
        )

        print("Joint 1 handle:", self.joint1)
        print("Joint 2 handle:", self.joint2)
        print("Joint 1 path:", self.sim.getObjectAlias(self.joint1, 4))
        print("Joint 2 path:", self.sim.getObjectAlias(self.joint2, 4))

    def get_joint_angles(self):
        return (
            self.sim.getJointPosition(self.joint1),
            self.sim.getJointPosition(self.joint2),
        )

    def set_joint_targets(self, theta1, theta2):
        self.sim.setJointTargetPosition(self.joint1, theta1)
        self.sim.setJointTargetPosition(self.joint2, theta2)
