
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require("sim")

link2 = sim.getObject("/RobotBase/Joint1/Link1/Joint2/Link2")

# Local endpoints of a 10 cm link.
for label, local_point in [
    ("Near end", [-0.05, 0, 0]),
    ("Far end", [0.05, 0, 0]),
]:
    world_point = sim.getObjectPosition(link2, -1)
    world_matrix = sim.getObjectMatrix(link2, -1)

    x, y, z = local_point
    world_x = world_matrix[0] * x + world_matrix[1] * y + world_matrix[2] * z + world_matrix[3]
    world_y = world_matrix[4] * x + world_matrix[5] * y + world_matrix[6] * z + world_matrix[7]
    world_z = world_matrix[8] * x + world_matrix[9] * y + world_matrix[10] * z + world_matrix[11]

    print(f"{label}: ({world_x:.4f}, {world_y:.4f}, {world_z:.4f}) m")
