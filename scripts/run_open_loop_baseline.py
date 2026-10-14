"""
run_open_loop_baseline.py

Executes Phase 2 Open-Loop Baseline Drawing in CoppeliaSim:
1. Loads 01_robot_arm_sketch_skeleton.json (Intermediate Representation).
2. Maps pixel strokes to the configured robot workspace (default: 6x6 cm).
3. Validates every point against reachability and joint limits; reports any rejected points.
4. Generates continuous trajectory with pen-up repositioning and pen-down drawing.
5. Renders visible ink traces in CoppeliaSim using sim.addDrawingObject during pen-down.
6. Captures top-down vision sensor screenshot and logs full telemetry to CSV.
7. Generates comparative trajectory plots and prints complete tracking error metrics.
"""

import csv
import math
import os
import sys
import time

sys.path.insert(0, ".")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scripts.setup_scene import setup_scene
from src.config import DEFAULT_SIM_CONFIG, DEFAULT_WORKSPACE_CONFIG, WorkspaceConfig
from src.coordinate_transform import image_to_arm_space, validate_workspace_point, validate_joint_configuration
from src.coppelia_controller import CoppeliaController
from src.path_planner import plan_strokes
from src.trajectory_metrics import calculate_trajectory_metrics, calculate_reachability_percentage
from src.vision.image_processor import StrokeIR


def run_baseline_drawing(
    ir_json_path: str = "data/extracted_strokes/01_robot_arm_sketch_skeleton.json",
    workspace_config: WorkspaceConfig = DEFAULT_WORKSPACE_CONFIG,
    output_csv_path: str = "logs/open_loop_baseline_telemetry.csv",
    output_plot_path: str = "logs/baseline_trajectory_plot.png",
    output_screenshot_path: str = "logs/baseline_sim_screenshot.png",
    spacing_cm: float = 0.25,
):
    print("=" * 80)
    print("PHASE 2: OPEN-LOOP BASELINE DRAWING EXECUTION")
    print(f"Input IR: {ir_json_path}")
    print(
        f"Drawing Window: {workspace_config.draw_width_cm:.1f} x {workspace_config.draw_height_cm:.1f} cm "
        f"at origin ({workspace_config.draw_origin_x:.1f}, {workspace_config.draw_origin_y:.1f}) cm"
    )
    print("=" * 80)

    # 1. Ensure scene objects exist (idempotent)
    setup_scene()

    # 2. Ingest StrokeIR JSON
    if not os.path.exists(ir_json_path):
        raise FileNotFoundError(f"IR JSON not found: {ir_json_path}")
    ir = StrokeIR.from_json(ir_json_path)
    print(f"Loaded StrokeIR with {len(ir.strokes)} strokes from {ir.metadata.source_image}")

    # 3. Transform canvas pixel strokes into arm workspace coordinates
    arm_strokes = []
    total_input_points = 0
    rejected_points = []

    for stroke_idx, stroke_px in enumerate(ir.strokes):
        if len(stroke_px) < 2:
            continue
        total_input_points += len(stroke_px)
        pts_arm = image_to_arm_space(
            stroke_px,
            image_width=workspace_config.image_width,
            image_height=workspace_config.image_height,
            origin_x=workspace_config.draw_origin_x,
            origin_y=workspace_config.draw_origin_y,
            draw_width=workspace_config.draw_width_cm,
            draw_height=workspace_config.draw_height_cm,
        )

        valid_stroke_pts = []
        for pt_idx, pt in enumerate(pts_arm):
            x, y = float(pt[0]), float(pt[1])
            try:
                validate_workspace_point(x, y)
                validate_joint_configuration(x, y)
                valid_stroke_pts.append((x, y))
            except ValueError as exc:
                rejected_points.append({
                    "stroke_idx": stroke_idx,
                    "point_idx": pt_idx,
                    "point_px": stroke_px[pt_idx],
                    "point_cm": (x, y),
                    "reason": str(exc),
                })

        if len(valid_stroke_pts) >= 2:
            arm_strokes.append(valid_stroke_pts)

    print("\n--- Workspace Validation Report ---")
    print(f"Total Input Stroke Points : {total_input_points}")
    print(f"Accepted Valid Points     : {total_input_points - len(rejected_points)}")
    print(f"Rejected / Clipped Points : {len(rejected_points)}")
    if rejected_points:
        print("Detailed Rejections:")
        for r in rejected_points[:10]:
            print(f"  Stroke {r['stroke_idx']}, Point {r['point_idx']}: {r['point_cm']} cm - {r['reason']}")
        if len(rejected_points) > 10:
            print(f"  ... and {len(rejected_points) - 10} more.")
    else:
        print("  All points fit 100% within the safe workspace with zero clipping!")

    # 4. Plan continuous motion sequence (pen-up repositioning and pen-down drawing)
    motion_plan = plan_strokes(
        arm_strokes,
        branch="positive",
        spacing_cm=spacing_cm,
        skip_invalid_strokes=False,
    )
    print(f"\nMotion Plan Generated: {len(motion_plan)} trajectory points across {len(arm_strokes)} strokes.")

    # 5. Initialize controller & start simulation in synchronous stepping mode
    controller = CoppeliaController(verbose=False)
    controller.stop_simulation()
    controller.set_stepping(True)
    controller.start_simulation()

    # Create visible ink trace in CoppeliaSim (dark blue ink)
    controller.create_drawing_object(line_width=3.5, color=(0.05, 0.15, 0.70))

    telemetry_records = []
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    prev_tip_pos = None
    prev_pen_down = False
    start_exec_time = time.time()

    print("\nExecuting drawing in CoppeliaSim...")
    for step_idx, pt in enumerate(motion_plan):
        # Command joint targets
        controller.set_joint_targets(pt.theta1, pt.theta2)

        # Step simulation until settled within tight tolerance
        controller.wait_for_settle(
            pt.theta1, pt.theta2, timeout_s=1.5, tolerance_deg=0.25, step_simulation=True
        )

        # Read actual robot state
        act_theta1, act_theta2 = controller.get_joint_angles()
        act_x, act_y, act_z = controller.get_tip_position()

        # Telemetry calculations
        err_x = act_x - pt.x
        err_y = act_y - pt.y
        err_euclidean_mm = math.hypot(err_x, err_y) * 10.0

        # Draw visible ink segment if pen is down
        if pt.pen_down and prev_pen_down and prev_tip_pos is not None:
            # Draw on top of paper surface (z = 1.55 cm)
            ink_p1 = (prev_tip_pos[0], prev_tip_pos[1], 1.55)
            ink_p2 = (act_x, act_y, 1.55)
            controller.add_ink_segment(ink_p1, ink_p2, in_meters=False)

        prev_tip_pos = (act_x, act_y, act_z)
        prev_pen_down = pt.pen_down

        telemetry_records.append({
            "step": step_idx,
            "stroke_id": pt.stroke_id,
            "pen_down": 1 if pt.pen_down else 0,
            "target_x_cm": round(pt.x, 4),
            "target_y_cm": round(pt.y, 4),
            "target_theta1_deg": round(math.degrees(pt.theta1), 4),
            "target_theta2_deg": round(math.degrees(pt.theta2), 4),
            "actual_theta1_deg": round(math.degrees(act_theta1), 4),
            "actual_theta2_deg": round(math.degrees(act_theta2), 4),
            "actual_x_cm": round(act_x, 4),
            "actual_y_cm": round(act_y, 4),
            "actual_z_cm": round(act_z, 4),
            "error_x_cm": round(err_x, 5),
            "error_y_cm": round(err_y, 5),
            "error_euclidean_mm": round(err_euclidean_mm, 4),
        })

        if (step_idx + 1) % 50 == 0 or (step_idx + 1) == len(motion_plan):
            print(f"  Step {step_idx + 1:3d}/{len(motion_plan)} | Stroke {pt.stroke_id:2d} | "
                  f"Target: ({pt.x:.2f}, {pt.y:.2f}) | Actual: ({act_x:.2f}, {act_y:.2f}) cm | "
                  f"Tip Error: {err_euclidean_mm:.3f} mm")

    exec_duration = time.time() - start_exec_time
    print(f"Drawing execution finished in {exec_duration:.2f} seconds.")

    # 6. Capture top-down vision sensor image
    os.makedirs(os.path.dirname(os.path.abspath(output_screenshot_path)), exist_ok=True)
    sim_cam_img = controller.get_vision_sensor_image()
    cv2.imwrite(output_screenshot_path, cv2.cvtColor(sim_cam_img, cv2.COLOR_RGB2BGR))
    print(f"Captured top-down simulation screenshot saved to: {output_screenshot_path}")

    # 7. Stop simulation
    controller.stop_simulation()
    controller.set_stepping(False)

    # 8. Write Telemetry to CSV
    fieldnames = list(telemetry_records[0].keys())
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(telemetry_records)
    print(f"Full execution telemetry logged to: {output_csv_path}")

    # 9. Calculate Quantitative Metrics
    targets = np.array([[r["target_x_cm"], r["target_y_cm"]] for r in telemetry_records])
    actuals = np.array([[r["actual_x_cm"], r["actual_y_cm"]] for r in telemetry_records])
    pen_states = np.array([r["pen_down"] for r in telemetry_records])

    traj_metrics = calculate_trajectory_metrics(targets, actuals)

    # Compute distances
    pen_down_dist = 0.0
    pen_up_dist = 0.0
    for i in range(1, len(targets)):
        seg_dist = float(np.linalg.norm(actuals[i] - actuals[i - 1]))
        if pen_states[i] == 1:
            pen_down_dist += seg_dist
        else:
            pen_up_dist += seg_dist

    print("\n" + "=" * 60)
    print("PHASE 2 OPEN-LOOP BASELINE METRICS REPORT")
    print("=" * 60)
    print(f"Total Trajectory Points       : {len(targets)}")
    print(f"Trajectory RMSE               : {traj_metrics['rmse_cm'] * 10.0:.3f} mm ({traj_metrics['rmse_cm']:.4f} cm)")
    print(f"Mean Tracking Error           : {traj_metrics['mean_error_cm'] * 10.0:.3f} mm ({traj_metrics['mean_error_cm']:.4f} cm)")
    print(f"Max Tracking Error            : {traj_metrics['max_error_cm'] * 10.0:.3f} mm ({traj_metrics['max_error_cm']:.4f} cm)")
    print(f"Pen-Down Drawing Distance     : {pen_down_dist:.2f} cm")
    print(f"Pen-Up Repositioning Distance : {pen_up_dist:.2f} cm")
    print(f"Pen Travel Efficiency Ratio   : {pen_down_dist / (pen_down_dist + pen_up_dist) * 100:.1f}%")
    print(f"Execution Duration            : {exec_duration:.2f} s")
    print("=" * 60)

    # 10. Generate High-Resolution Trajectory Plots
    os.makedirs(os.path.dirname(os.path.abspath(output_plot_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#ffffff")

    # Left: Path Comparison
    ax0 = axes[0]
    ax0.set_title("Open-Loop Trajectory: Target vs Executed Path", fontsize=12, weight="bold")
    ax0.set_xlabel("Robot X (cm)", fontsize=11)
    ax0.set_ylabel("Robot Y (cm)", fontsize=11)
    ax0.set_aspect("equal")
    ax0.grid(True, linestyle="--", alpha=0.5)

    # Drawing workspace box
    w_x = workspace_config.draw_origin_x
    w_y = workspace_config.draw_origin_y - workspace_config.draw_height_cm
    w_w = workspace_config.draw_width_cm
    w_h = workspace_config.draw_height_cm
    rect = plt.Rectangle((w_x, w_y), w_w, w_h, edgecolor="black", facecolor="#f8f9fa", linestyle="--", linewidth=1.5, label="6x6cm Window")
    ax0.add_patch(rect)

    # Plot pen-up repositioning segments
    for i in range(1, len(targets)):
        if pen_states[i] == 0:
            ax0.plot([actuals[i - 1, 0], actuals[i, 0]], [actuals[i - 1, 1], actuals[i, 1]], color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

    # Plot pen-down drawing segments
    drawn_label_added = False
    for i in range(1, len(targets)):
        if pen_states[i] == 1:
            lbl = "Executed Trace (Pen Down)" if not drawn_label_added else ""
            ax0.plot([actuals[i - 1, 0], actuals[i, 0]], [actuals[i - 1, 1], actuals[i, 1]], color="#0d6efd", linewidth=2.2, label=lbl)
            drawn_label_added = True

    # Plot targets
    ax0.scatter(targets[:, 0], targets[:, 1], color="#dc3545", s=14, zorder=4, label="Target Points", alpha=0.6)
    ax0.legend(loc="upper right", fontsize=9)

    # Right: Error Distribution along path
    ax1 = axes[1]
    errors_mm = np.array([r["error_euclidean_mm"] for r in telemetry_records])
    ax1.plot(errors_mm, color="#0d6efd", linewidth=1.5, label="Tip Position Error (mm)")
    ax1.axhline(traj_metrics["rmse_cm"] * 10.0, color="#dc3545", linestyle="--", linewidth=1.5, label=f"RMSE: {traj_metrics['rmse_cm']*10.0:.3f} mm")
    ax1.axhline(traj_metrics["mean_error_cm"] * 10.0, color="#fd7e14", linestyle=":", linewidth=1.5, label=f"Mean: {traj_metrics['mean_error_cm']*10.0:.3f} mm")
    ax1.set_title("Pointwise Euclidean Tracking Error (mm)", fontsize=12, weight="bold")
    ax1.set_xlabel("Trajectory Step Index", fontsize=11)
    ax1.set_ylabel("Error (mm)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Trajectory comparison plot saved to: {output_plot_path}")

    return traj_metrics


if __name__ == "__main__":
    run_baseline_drawing()
