"""
scripts/run_disturbed_baseline.py

Executes a sketch plotting run in CoppeliaSim under configurable plant disturbances.
Applies:
1. Kinematic calibration offsets (link lengths & joint zero orientations) in the sim scene.
2. Direction-dependent gear backlash play operator on joint reversals.
3. Seeded Gaussian joint actuator noise.

All disturbances act exclusively on the simulated plant in CoppeliaSim.
The nominal planner and IK remain completely unaware of them.
Scene transforms are guaranteed to be restored to nominal upon completion.
"""

import argparse
import csv
import math
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, ".")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scripts.setup_scene import setup_scene
from src.config import (
    CameraConfig,
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_WORKSPACE_CONFIG,
    DEFAULT_EVAL_CONFIG,
    WorkspaceConfig,
)
from src.coordinate_transform import (
    image_to_arm_space,
    validate_workspace_point,
    validate_joint_configuration,
)
from src.coppelia_controller import CoppeliaController
from src.eval.metrics import (
    decompose_tracking_errors,
    extract_ink_mask_from_camera_image,
    pen_travel_metrics,
    raster_chamfer_and_contour_error,
    raster_iou,
    rasterize_strokes_to_camera_mask,
    trajectory_smoothness,
)
from src.path_planner import plan_strokes
from src.sim.disturbances import (
    DisturbanceConfig,
    DISTURBANCE_PRESETS,
    get_disturbance_preset,
    PlantDisturbanceManager,
)
from src.vision.image_processor import StrokeIR


def execute_disturbed_run(
    preset_name: str = "none",
    seed: Optional[int] = 42,
    ir_json_path: str = "data/extracted_strokes/01_robot_arm_sketch_skeleton.json",
    output_dir: str = "logs/disturbances",
    workspace_config: WorkspaceConfig = DEFAULT_WORKSPACE_CONFIG,
    camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
    spacing_cm: float = 0.25,
) -> Dict[str, Any]:
    """
    Execute drawing in CoppeliaSim under the specified disturbance preset and seed.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    cfg = get_disturbance_preset(preset_name, seed=seed)
    mgr = PlantDisturbanceManager(cfg)

    run_id = f"{preset_name}_seed{seed if seed is not None else 0}"
    os.makedirs(output_dir, exist_ok=True)
    telemetry_csv_path = os.path.join(output_dir, f"{run_id}_telemetry.csv")
    screenshot_png_path = os.path.join(output_dir, f"{run_id}_screenshot.png")
    overlay_png_path = os.path.join(output_dir, f"{run_id}_overlay.png")

    print("=" * 80)
    print(f"RUNNING EXPERIMENT: Preset '{preset_name.upper()}' | Seed: {seed}")
    print(f"  Disturbance Config: enabled={cfg.enabled}")
    if cfg.enabled:
        print(f"    - Link offsets      : dL1 = {cfg.delta_L1_cm:+.2f} cm ({cfg.delta_L1_cm*10:+.1f} mm), "
              f"dL2 = {cfg.delta_L2_cm:+.2f} cm ({cfg.delta_L2_cm*10:+.1f} mm)")
        print(f"    - Joint zero offsets: dth1 = {cfg.delta_theta1_deg:+.2f} deg, dth2 = {cfg.delta_theta2_deg:+.2f} deg")
        print(f"    - Backlash deadband : B1 = {cfg.backlash_theta1_deg:.2f} deg, B2 = {cfg.backlash_theta2_deg:.2f} deg")
        print(f"    - Actuator noise std: sig1 = {cfg.noise_std_theta1_deg:.2f} deg, sig2 = {cfg.noise_std_theta2_deg:.2f} deg")
    print("=" * 80)

    # 1. Ensure scene objects exist
    setup_scene()

    # 2. Ingest StrokeIR JSON & transform to workspace coordinates
    if not os.path.exists(ir_json_path):
        raise FileNotFoundError(f"IR JSON not found: {ir_json_path}")
    ir = StrokeIR.from_json(ir_json_path)

    arm_strokes = []
    target_strokes_arm = []
    for stroke_px in ir.strokes:
        if len(stroke_px) < 2:
            continue
        pts_arm = image_to_arm_space(
            stroke_px,
            image_width=workspace_config.image_width,
            image_height=workspace_config.image_height,
            origin_x=workspace_config.draw_origin_x,
            origin_y=workspace_config.draw_origin_y,
            draw_width=workspace_config.draw_width_cm,
            draw_height=workspace_config.draw_height_cm,
        )
        valid_pts = []
        for pt in pts_arm:
            x, y = float(pt[0]), float(pt[1])
            try:
                validate_workspace_point(x, y)
                validate_joint_configuration(x, y)
                valid_pts.append((x, y))
            except ValueError:
                pass
        if len(valid_pts) >= 2:
            arm_strokes.append(valid_pts)
            target_strokes_arm.append(np.array(valid_pts))

    # 3. Plan nominal continuous trajectory
    motion_plan = plan_strokes(
        arm_strokes,
        branch="positive",
        spacing_cm=spacing_cm,
        skip_invalid_strokes=False,
    )

    # 4. Initialize CoppeliaController
    controller = CoppeliaController(verbose=False)
    controller.stop_simulation()
    controller.clear_drawing_object()
    controller.set_stepping(True)

    telemetry_records = []
    prev_tip_pos = None

    try:
        # Wrap simulation run in disturbance context manager to ensure restoration
        with mgr.active_context(controller.sim):
            controller.start_simulation()
            controller.clear_drawing_object()
            controller.create_drawing_object(line_width=3.5, color=(0.05, 0.15, 0.70))

            start_exec_time = time.time()
            for step_idx, pt in enumerate(motion_plan):
                # Pass nominal command through plant disturbance model
                cmd_th1, cmd_th2, dist_telem = mgr.transform_plant_command(pt.theta1, pt.theta2)

                # Command joint targets to plant
                controller.set_joint_targets(cmd_th1, cmd_th2)

                # Settle in sim (tolerance_deg=0.25 matches Phase 2 baseline)
                controller.wait_for_settle(
                    cmd_th1, cmd_th2, timeout_s=1.5, tolerance_deg=0.25, step_simulation=True
                )

                # Read actual physical state from plant
                act_th1, act_th2 = controller.get_joint_angles()
                act_x, act_y, act_z = controller.get_tip_position()

                err_x = act_x - pt.x
                err_y = act_y - pt.y
                err_euclidean_mm = math.hypot(err_x, err_y) * 10.0

                # Draw ink segment from previous tip position if pen is down
                if pt.pen_down and prev_tip_pos is not None:
                    ink_p1 = (prev_tip_pos[0], prev_tip_pos[1], 1.55)
                    ink_p2 = (act_x, act_y, 1.55)
                    controller.add_ink_segment(ink_p1, ink_p2, in_meters=False)

                prev_tip_pos = (act_x, act_y, act_z)

                telemetry_records.append({
                    "step": step_idx,
                    "stroke_id": pt.stroke_id,
                    "pen_down": 1 if pt.pen_down else 0,
                    "target_x_cm": round(pt.x, 4),
                    "target_y_cm": round(pt.y, 4),
                    "target_theta1_deg": round(math.degrees(pt.theta1), 4),
                    "target_theta2_deg": round(math.degrees(pt.theta2), 4),
                    "plant_cmd_theta1_deg": round(math.degrees(cmd_th1), 4),
                    "plant_cmd_theta2_deg": round(math.degrees(cmd_th2), 4),
                    "actual_theta1_deg": round(math.degrees(act_th1), 4),
                    "actual_theta2_deg": round(math.degrees(act_th2), 4),
                    "actual_x_cm": round(act_x, 4),
                    "actual_y_cm": round(act_y, 4),
                    "actual_z_cm": round(act_z, 4),
                    "error_x_cm": round(err_x, 5),
                    "error_y_cm": round(err_y, 5),
                    "error_euclidean_mm": round(err_euclidean_mm, 4),
                    "backlash_error1_deg": round(dist_telem["backlash_error1_deg"], 4),
                    "backlash_error2_deg": round(dist_telem["backlash_error2_deg"], 4),
                    "noise1_deg": round(dist_telem["noise1_deg"], 4),
                    "noise2_deg": round(dist_telem["noise2_deg"], 4),
                    "delta_L1_cm": cfg.delta_L1_cm,
                    "delta_L2_cm": cfg.delta_L2_cm,
                    "delta_theta1_deg": cfg.delta_theta1_deg,
                    "delta_theta2_deg": cfg.delta_theta2_deg,
                    "preset": preset_name,
                    "seed": seed if seed is not None else -1,
                })

            exec_duration = time.time() - start_exec_time
            print(f"  Execution finished in {exec_duration:.2f} s across {len(motion_plan)} steps.")

            # Capture top-down camera screenshot
            sim_cam_img = controller.get_vision_sensor_image()
            cv2.imwrite(screenshot_png_path, cv2.cvtColor(sim_cam_img, cv2.COLOR_RGB2BGR))
            print(f"  Screenshot saved to: {screenshot_png_path}")

            controller.clear_drawing_object()
            controller.stop_simulation()
            controller.set_stepping(False)

    except Exception as exc:
        print(f"  Execution error encountered: {exc}")
        controller.stop_simulation()
        controller.set_stepping(False)
        raise exc

    # 5. Export Telemetry CSV
    fieldnames = list(telemetry_records[0].keys())
    with open(telemetry_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(telemetry_records)
    print(f"  Telemetry logged to: {telemetry_csv_path}")

    # 6. Compute Comprehensive Metrics
    plan_pts = np.array([[r["target_x_cm"], r["target_y_cm"]] for r in telemetry_records])
    meas_pts = np.array([[r["actual_x_cm"], r["actual_y_cm"]] for r in telemetry_records])
    pen_states = np.array([r["pen_down"] for r in telemetry_records])
    stroke_ids = np.array([r["stroke_id"] for r in telemetry_records])

    # Decomposed errors
    decomp = decompose_tracking_errors(
        target_strokes_arm=target_strokes_arm,
        plan_positions_cm=plan_pts,
        measured_positions_cm=meas_pts,
        pen_states=pen_states,
        stroke_ids=stroke_ids,
    )
    plan_err = decomp["planning_error"]
    exec_err = decomp["execution_error"]
    total_err = decomp["total_error"]

    # Smoothness & Travel
    smoothness = trajectory_smoothness(meas_pts, dt=DEFAULT_EVAL_CONFIG.dt_seconds)
    travel = pen_travel_metrics(meas_pts, pen_states)

    # Vision Camera Evaluation
    cam_img = cv2.imread(screenshot_png_path)
    target_mask = rasterize_strokes_to_camera_mask(
        strokes_cm=target_strokes_arm,
        camera_config=camera_config,
        line_thickness_px=camera_config.line_thickness_px,
    )
    obs_mask = extract_ink_mask_from_camera_image(cam_img, is_bgr=True)
    iou_res = raster_iou(target_mask, obs_mask, tolerance_px=2)
    pixel_size_mm = camera_config.view_size_cm * 10.0 / camera_config.resolution_x
    cam_chamfer = raster_chamfer_and_contour_error(target_mask, obs_mask, pixel_size_mm=pixel_size_mm)

    # 7. Generate Target vs Drawn Overlay Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#ffffff")

    # Left: Trajectory Overlay
    ax0 = axes[0]
    ax0.set_title(f"Trajectory Overlay: Preset '{preset_name.upper()}' (Seed {seed})\n"
                  f"RMSE: {exec_err['rmse_mm']:.2f} mm | Total Chamfer: {total_err['chamfer_symmetric_mm']:.2f} mm",
                  fontsize=11, weight="bold")
    ax0.set_xlabel("Robot X (cm)", fontsize=10)
    ax0.set_ylabel("Robot Y (cm)", fontsize=10)
    ax0.set_aspect("equal")
    ax0.grid(True, linestyle="--", alpha=0.5)

    # Window box
    w_x, w_y = workspace_config.draw_origin_x, workspace_config.draw_origin_y - workspace_config.draw_height_cm
    rect = plt.Rectangle((w_x, w_y), workspace_config.draw_width_cm, workspace_config.draw_height_cm,
                         edgecolor="black", facecolor="#f8f9fa", linestyle="--", linewidth=1.2, label="6x6cm Window")
    ax0.add_patch(rect)

    for s_idx, s in enumerate(target_strokes_arm):
        lbl = "Target Strokes (JSON)" if s_idx == 0 else ""
        ax0.plot(s[:, 0], s[:, 1], color="#28a745", linewidth=2.5, alpha=0.7, label=lbl)

    drawn_added = False
    for i in range(1, len(meas_pts)):
        if pen_states[i] == 1:
            lbl = "Disturbed Executed Tip" if not drawn_added else ""
            ax0.plot([meas_pts[i-1, 0], meas_pts[i, 0]], [meas_pts[i-1, 1], meas_pts[i, 1]],
                     color="#0d6efd", linewidth=1.8, label=lbl)
            drawn_added = True
        else:
            ax0.plot([meas_pts[i-1, 0], meas_pts[i, 0]], [meas_pts[i-1, 1], meas_pts[i, 1]],
                     color="#adb5bd", linestyle=":", linewidth=1.0)
    ax0.legend(loc="upper right", fontsize=8)

    # Right: Camera Mask Inspection Overlay
    ax1 = axes[1]
    ax1.set_title(f"Camera Mask Inspection (512x512)\n"
                  f"Tolerance IoU: {iou_res['tolerance_iou_pct']:.1f}% | Plain IoU: {iou_res['plain_iou_pct']:.1f}%",
                  fontsize=11, weight="bold")
    ax1.set_xlabel("Camera X (px)", fontsize=10)
    ax1.set_ylabel("Camera Y (px)", fontsize=10)

    overlay = np.zeros((camera_config.resolution_y, camera_config.resolution_x, 3), dtype=np.uint8)
    overlay[:, :] = [245, 245, 245]  # Paper white
    overlay[obs_mask > 0] = [180, 50, 20]     # Blue drawn ink
    overlay[target_mask > 0] = [30, 160, 30]  # Green target
    overlap_idx = np.logical_and(target_mask > 0, obs_mask > 0)
    overlay[overlap_idx] = [0, 220, 255]     # Yellow overlap

    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_xlim(130, 380)
    ax1.set_ylim(380, 130)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1eb41e", label="Target Mask"),
        Patch(facecolor="#1432b4", label="Observed Disturbed Ink"),
        Patch(facecolor="#ffd200", label=f"Exact Overlap ({iou_res['intersection_pixels']} px)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(overlay_png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overlay figure saved to: {overlay_png_path}")

    metrics = {
        "preset": preset_name,
        "seed": seed,
        "execution_rmse_mm": exec_err["rmse_mm"],
        "mean_tracking_error_mm": exec_err["mean_error_mm"],
        "max_tracking_error_mm": exec_err["max_error_mm"],
        "execution_chamfer_mm": exec_err["chamfer_symmetric_mm"],
        "total_chamfer_mm": total_err["chamfer_symmetric_mm"],
        "planning_endpoint_error_mm": plan_err["mean_endpoint_error_mm"],
        "execution_endpoint_error_mm": exec_err["mean_endpoint_error_mm"],
        "total_endpoint_error_mm": total_err["mean_endpoint_error_mm"],
        "tolerance_iou_pct": iou_res["tolerance_iou_pct"],
        "tolerance_dice_pct": iou_res["tolerance_dice"] * 100.0,
        "plain_iou_pct": iou_res["plain_iou_pct"],
        "camera_chamfer_mm": cam_chamfer["symmetric_chamfer_mm"],
        "camera_contour_error_rmse_mm": cam_chamfer["contour_error_rmse_mm"],
        "camera_contour_error_max_mm": cam_chamfer["contour_error_max_mm"],
        "telemetry_csv": telemetry_csv_path,
        "screenshot_png": screenshot_png_path,
        "overlay_png": overlay_png_path,
    }

    print(f"\n--- Metrics Summary for {preset_name.upper()} (Seed {seed}) ---")
    print(f"  Execution RMSE          : {metrics['execution_rmse_mm']:.4f} mm")
    print(f"  Total Contour Chamfer   : {metrics['total_chamfer_mm']:.4f} mm")
    print(f"  Tolerance-Matched IoU   : {metrics['tolerance_iou_pct']:.2f}% (Headline, tau=2px)")
    print(f"  Plain IoU               : {metrics['plain_iou_pct']:.2f}% (Secondary)")
    print(f"  Execution Endpoint Err  : {metrics['execution_endpoint_error_mm']:.4f} mm")
    print(f"  Total Endpoint Error    : {metrics['total_endpoint_error_mm']:.4f} mm")
    print(f"  Camera Chamfer Distance : {metrics['camera_chamfer_mm']:.4f} mm")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run disturbed sketch plotting in CoppeliaSim.")
    parser.add_argument("--preset", type=str, default="none", choices=["none", "mild", "moderate", "severe"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    execute_disturbed_run(preset_name=args.preset, seed=args.seed)
