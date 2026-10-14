"""
setup_scene.py

Idempotent scene configuration script for the VisionGuidedSketchPlotter:
1. Verifies/creates the distal Tip dummy on Link2.
2. Ensures dynamic collision masks between Link1 and Link2 are disjoint (prevents elbow jamming).
3. Verifies/creates a pure white DrawingPaper plane at the pen-down height (z = 0.0145m).
4. Verifies/creates an orthographic TopDownVisionSensor positioned above the drawing window.
5. Prints a complete audit table of all scene objects: name, handle, parent, and position.
"""

import math
import sys

sys.path.insert(0, ".")

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from src.config import DEFAULT_SIM_CONFIG, DEFAULT_WORKSPACE_CONFIG


def get_object_or_none(sim, path: str):
    try:
        return sim.getObject(path)
    except Exception:
        return None


def setup_scene(save_scene: bool = False):
    client = RemoteAPIClient(host=DEFAULT_SIM_CONFIG.host, port=DEFAULT_SIM_CONFIG.port)
    sim = client.require("sim")

    # Stop simulation if running
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        import time
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.02)

    # 1. Base hierarchy objects
    base = sim.getObject(DEFAULT_SIM_CONFIG.base_path)
    joint1 = sim.getObject(DEFAULT_SIM_CONFIG.joint1_path)
    link1 = sim.getObject(DEFAULT_SIM_CONFIG.link1_path)
    joint2 = sim.getObject(DEFAULT_SIM_CONFIG.joint2_path)
    link2 = sim.getObject(DEFAULT_SIM_CONFIG.link2_path)

    # 2. Physics & Dynamic Properties
    # Prevent self-collision between Link1 and Link2
    sim.setObjectInt32Param(link1, sim.shapeintparam_respondable_mask, 1)
    sim.setObjectInt32Param(link2, sim.shapeintparam_respondable_mask, 2)
    for joint in (joint1, joint2):
        sim.setObjectFloatParam(joint, sim.jointfloatparam_pid_p, DEFAULT_SIM_CONFIG.pid_p)
        sim.setObjectFloatParam(joint, sim.jointfloatparam_pid_d, DEFAULT_SIM_CONFIG.pid_d)
        sim.setObjectFloatParam(joint, sim.jointfloatparam_maxvel, DEFAULT_SIM_CONFIG.max_vel)
        sim.setObjectFloatParam(joint, sim.jointfloatparam_maxaccel, DEFAULT_SIM_CONFIG.max_accel)

    # 3. Tip Dummy on Link2 (idempotent)
    tip = get_object_or_none(sim, DEFAULT_SIM_CONFIG.tip_path)
    if tip is None:
        tip = sim.createDummy(0.005)
        sim.setObjectAlias(tip, "Tip")
        sim.setObjectParent(tip, link2, False)
        sim.setObjectPosition(tip, [0.05, 0.0, 0.0], link2)
        sim.setObjectOrientation(tip, [0.0, 0.0, 0.0], link2)
    else:
        sim.setObjectPosition(tip, [0.05, 0.0, 0.0], link2)
        sim.setObjectOrientation(tip, [0.0, 0.0, 0.0], link2)

    # 4. White DrawingPaper Sheet (idempotent)
    paper_path = "/DrawingPaper"
    paper = get_object_or_none(sim, paper_path)

    # Center of drawing window in meters
    ws = DEFAULT_WORKSPACE_CONFIG
    center_x_m = (ws.draw_origin_x + ws.draw_width_cm / 2.0) / 100.0  # 0.070 m
    center_y_m = (ws.draw_origin_y - ws.draw_height_cm / 2.0) / 100.0  # 0.095 m
    paper_width_m = (ws.draw_width_cm + 4.0) / 100.0   # 0.100 m (10 cm x 10 cm paper)
    paper_height_m = (ws.draw_height_cm + 4.0) / 100.0  # 0.100 m
    pen_z_m = 0.0150  # Tip height
    paper_thickness_m = 0.001
    paper_z_m = pen_z_m - (paper_thickness_m / 2.0)  # Top surface exactly at z = 0.015 m

    if paper is None:
        paper = sim.createPrimitiveShape(
            sim.primitiveshape_cuboid,
            [paper_width_m, paper_height_m, paper_thickness_m]
        )
        sim.setObjectAlias(paper, "DrawingPaper")
    else:
        # Re-scale shape if needed
        pass

    sim.setObjectPosition(paper, -1, [center_x_m, center_y_m, paper_z_m])
    sim.setObjectOrientation(paper, -1, [0.0, 0.0, 0.0])
    sim.setShapeColor(paper, None, sim.colorcomponent_ambient_diffuse, [0.98, 0.98, 0.98])
    sim.setObjectInt32Param(paper, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(paper, sim.shapeintparam_respondable, 0)

    # 5. Top-Down Vision Sensor (idempotent, needed for closed-loop Phase 5)
    cam_path = "/TopDownVisionSensor"
    camera = get_object_or_none(sim, cam_path)
    camera_z_m = 0.35  # 35 cm height above table
    ortho_view_size_m = max(paper_width_m, paper_height_m) + 0.02  # Covers the drawing sheet

    if camera is None:
        camera = sim.createVisionSensor(
            0,  # 0 = orthographic projection for metric fidelity without radial distortion
            [512, 512, 0, 0],
            [0.01, 1.0, ortho_view_size_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        sim.setObjectAlias(camera, "TopDownVisionSensor")

    sim.setObjectPosition(camera, -1, [center_x_m, center_y_m, camera_z_m])
    # Optical axis looks down along -Z; up in image is +Y world, right in image is +X world
    sim.setObjectOrientation(camera, -1, [0.0, math.pi, 0.0])

    # 6. Audit Table Output
    objects_to_audit = [
        ("RobotBase", base),
        ("Joint1", joint1),
        ("Link1", link1),
        ("Joint2", joint2),
        ("Link2", link2),
        ("Tip", tip),
        ("DrawingPaper", paper),
        ("TopDownVisionSensor", camera),
    ]

    print("=" * 95)
    print("COPPELIASIM SCENE SETUP & AUDIT TABLE")
    print("=" * 95)
    print(f"{'Object Name':<22} | {'Handle':<8} | {'Parent':<20} | {'World Position (cm)':<28}")
    print("-" * 95)

    for name, handle in objects_to_audit:
        parent_h = sim.getObjectParent(handle)
        if parent_h != -1:
            parent_name = sim.getObjectAlias(parent_h, 3)
            parent_str = f"{parent_name} ({parent_h})"
        else:
            parent_str = "None (World)"
        pos = sim.getObjectPosition(handle, -1)
        pos_cm_str = f"({pos[0]*100:.2f}, {pos[1]*100:.2f}, {pos[2]*100:.2f}) cm"
        print(f"{name:<22} | {handle:<8} | {parent_str:<20} | {pos_cm_str:<28}")

    print("=" * 95)
    print("Scene setup completed idempotently.")


if __name__ == "__main__":
    setup_scene()
