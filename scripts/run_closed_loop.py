"""
scripts/run_closed_loop.py

Phase 5: Closed-Loop Vision-Guided Drawing in CoppeliaSim.

Executes the sketch with optional per-stroke visual feedback correction.
The corrector uses ONLY:
  - TopDownVisionSensor images
  - Target strokes (from IR JSON)
  - Nominal robot FK/IK model

It does NOT access:
  - True tip position (get_tip_position is called ONLY for telemetry logging AFTER correction)
  - Disturbance parameters or config
  - Recorded telemetry from previous runs

Two correction variants are supported:
  - 'per_stroke': Observe each drawn stroke, ICP-match to target, update running correction.
  - 'calib_prepass': Draw a small cross first, fit global offset, then apply to entire drawing.
"""

import argparse
import csv
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

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
from src.kinematics import forward_kinematics, inverse_kinematics
from src.path_planner import plan_strokes, MotionPoint
from src.sim.disturbances import (
    get_disturbance_preset,
    PlantDisturbanceManager,
)
from src.sim.vision_corrector import (
    CorrectorConfig,
    CalibrationPrePass,
    PerStrokeObserver,
    RunningCorrector,
    StrokeMatcher,
)
from src.vision.image_processor import StrokeIR


def _plan_stroke_segment(
    pts: np.ndarray,
    branch: str = "positive",
    spacing_cm: float = 0.25,
) -> List[MotionPoint]:
    """Plan a single stroke with interpolation, used for calibration pattern."""
    return plan_strokes([pts.tolist()], branch=branch, spacing_cm=spacing_cm)


def execute_closed_loop_run(
    preset_name: str = "none",
    seed: int = 42,
    correction_variant: str = "per_stroke",   # 'per_stroke' or 'calib_prepass' or 'none'
    ir_json_path: str = "data/extracted_strokes/01_robot_arm_sketch_skeleton.json",
    output_dir: str = "logs/closed_loop",
    workspace_config: WorkspaceConfig = DEFAULT_WORKSPACE_CONFIG,
    camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
    corrector_config: Optional[CorrectorConfig] = None,
    spacing_cm: float = 0.25,
) -> Dict[str, Any]:
    """
    Execute closed-loop drawing in CoppeliaSim under the specified disturbance preset.

    The corrector accesses ONLY: camera images, target strokes, nominal IK model.
    It never reads true tip position, telemetry, or disturbance parameters.

    Returns:
        Comprehensive evaluation metrics dictionary.
    """
    if corrector_config is None:
        corrector_config = CorrectorConfig(
            enabled=(correction_variant != "none"),
            calib_prepass=(correction_variant == "calib_prepass"),
        )

    dist_cfg = get_disturbance_preset(preset_name, seed=seed)
    disturbance_mgr = PlantDisturbanceManager(dist_cfg)

    run_id = f"cl_{correction_variant}_{preset_name}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)
    telemetry_csv = os.path.join(output_dir, f"{run_id}_telemetry.csv")
    screenshot_png = os.path.join(output_dir, f"{run_id}_screenshot.png")
    overlay_png = os.path.join(output_dir, f"{run_id}_overlay.png")

    print("=" * 80)
    print(f"CLOSED-LOOP EXPERIMENT: variant='{correction_variant}' | preset='{preset_name}' | seed={seed}")
    print(f"  Corrector enabled={corrector_config.enabled}, EMA alpha={corrector_config.ema_alpha}, "
          f"max_offset={corrector_config.max_offset_cm*10:.1f} mm, "
          f"use_affine={corrector_config.use_affine}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 1. Load target strokes
    # -----------------------------------------------------------------------
    setup_scene()
    ir = StrokeIR.from_json(ir_json_path)

    arm_strokes: List[np.ndarray] = []
    target_strokes_arm: List[np.ndarray] = []
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
        valid = [pt for pt in pts_arm
                 if _valid_pt(float(pt[0]), float(pt[1]))]
        if len(valid) >= 2:
            arr = np.array(valid, dtype=float)
            arm_strokes.append(arr)
            target_strokes_arm.append(arr)

    # -----------------------------------------------------------------------
    # 2. Setup vision corrector components
    # -----------------------------------------------------------------------
    observer = PerStrokeObserver(camera_config, corrector_config.min_observed_pixels)
    matcher = StrokeMatcher(corrector_config)
    corrector = RunningCorrector(corrector_config)
    calib_pp = CalibrationPrePass(corrector_config, camera_config) if corrector_config.calib_prepass else None

    # -----------------------------------------------------------------------
    # 3. Execute drawing
    # -----------------------------------------------------------------------
    controller = CoppeliaController(verbose=False)
    controller.stop_simulation()
    controller.clear_drawing_object()
    controller.set_stepping(True)

    telemetry_records: List[Dict] = []
    correction_log: List[Dict] = []

    try:
        with disturbance_mgr.active_context(controller.sim):
            controller.start_simulation()
            controller.clear_drawing_object()
            controller.create_drawing_object(line_width=3.5, color=(0.05, 0.15, 0.70))

            # Step: capture blank canvas baseline before any drawing
            controller.step()
            baseline_img_bgr = cv2.cvtColor(
                controller.get_vision_sensor_image(), cv2.COLOR_RGB2BGR
            )
            observer.capture_baseline(baseline_img_bgr)

            # ----------------------------------------------------------------
            # Calibration pre-pass (optional)
            # ----------------------------------------------------------------
            if corrector_config.calib_prepass and calib_pp is not None:
                print("  [CalibPrePass] Drawing calibration cross pattern...")
                calib_strokes = calib_pp.get_pattern_strokes()
                calib_plan = plan_strokes(
                    [s.tolist() for s in calib_strokes],
                    branch="positive",
                    spacing_cm=spacing_cm,
                )
                _run_motion_plan(calib_plan, controller, disturbance_mgr, {})

                # Park arm so it doesn't occlude calibration ink
                park_th1, park_th2 = inverse_kinematics(4.0, 6.5, branch="positive")
                park_c1, park_c2, _ = disturbance_mgr.transform_plant_command(park_th1, park_th2)
                controller.set_joint_targets(park_c1, park_c2)
                controller.wait_for_settle(park_c1, park_c2, timeout_s=1.0, tolerance_deg=0.25, step_simulation=True)

                # Observe the calibration pattern
                calib_img_bgr = cv2.cvtColor(
                    controller.get_vision_sensor_image(), cv2.COLOR_RGB2BGR
                )
                calib_offset = calib_pp.estimate_offset_from_image(calib_img_bgr, baseline_img_bgr)

                if calib_offset is not None:
                    corrector.update(calib_offset)
                    print(f"  [CalibPrePass] Calibration offset: ({calib_offset[0]*10:.2f}, {calib_offset[1]*10:.2f}) mm")
                else:
                    print("  [CalibPrePass] Insufficient ink pixels, using zero offset.")

                # Clear calibration pattern ink and reset clean paper baseline for main drawing
                controller.clear_drawing_object()
                controller.create_drawing_object(line_width=3.5, color=(0.05, 0.15, 0.70))
                controller.step()
                clean_img_bgr = cv2.cvtColor(controller.get_vision_sensor_image(), cv2.COLOR_RGB2BGR)
                observer.capture_baseline(clean_img_bgr)

            # ----------------------------------------------------------------
            # Main drawing loop: per-stroke with vision correction
            # ----------------------------------------------------------------
            step_counter = 0
            prev_tip_pos = None
            start_exec_time = time.time()

            for stroke_idx, stroke_pts in enumerate(arm_strokes):
                # Nominal plan for this stroke (unperturbed target trajectory)
                nom_stroke_plan = plan_strokes(
                    [stroke_pts.tolist()],
                    branch="positive",
                    spacing_cm=spacing_cm,
                )

                # Apply current running correction to this stroke's waypoints
                corrected_pts = _apply_correction_to_stroke(
                    stroke_pts, corrector, corrector_config, workspace_config
                )

                # Plan this corrected stroke
                try:
                    stroke_plan = plan_strokes(
                        [corrected_pts.tolist()],
                        branch="positive",
                        spacing_cm=spacing_cm,
                        skip_invalid_strokes=True,
                    )
                    if len(stroke_plan) != len(nom_stroke_plan):
                        stroke_plan = nom_stroke_plan
                except ValueError:
                    stroke_plan = nom_stroke_plan

                # Execute this stroke
                for nom_pt, pt in zip(nom_stroke_plan, stroke_plan):
                    cmd_th1, cmd_th2, dist_telem = disturbance_mgr.transform_plant_command(
                        pt.theta1, pt.theta2
                    )
                    controller.set_joint_targets(cmd_th1, cmd_th2)
                    controller.wait_for_settle(
                        cmd_th1, cmd_th2, timeout_s=1.5, tolerance_deg=0.25, step_simulation=True
                    )

                    # For telemetry ONLY (corrector never reads this mid-loop):
                    act_th1, act_th2 = controller.get_joint_angles()
                    act_x, act_y, act_z = controller.get_tip_position()

                    err_euclidean_mm = math.hypot(act_x - nom_pt.x, act_y - nom_pt.y) * 10.0

                    if pt.pen_down and prev_tip_pos is not None:
                        controller.add_ink_segment(
                            (prev_tip_pos[0], prev_tip_pos[1], 1.55),
                            (act_x, act_y, 1.55),
                            in_meters=False,
                        )
                    prev_tip_pos = (act_x, act_y, act_z)

                    telemetry_records.append({
                        "step": step_counter,
                        "stroke_id": stroke_idx,
                        "pen_down": 1 if pt.pen_down else 0,
                        "target_x_cm": round(nom_pt.x, 4),
                        "target_y_cm": round(nom_pt.y, 4),
                        "planned_x_cm": round(nom_pt.x, 4),
                        "planned_y_cm": round(nom_pt.y, 4),
                        "corrected_x_cm": round(pt.x, 4),
                        "corrected_y_cm": round(pt.y, 4),
                        "actual_x_cm": round(act_x, 4),
                        "actual_y_cm": round(act_y, 4),
                        "error_x_cm": round(act_x - nom_pt.x, 5),
                        "error_y_cm": round(act_y - nom_pt.y, 5),
                        "error_euclidean_mm": round(err_euclidean_mm, 4),
                        "corr_dx_cm": round(corrector.current_offset[0], 5),
                        "corr_dy_cm": round(corrector.current_offset[1], 5),
                        "preset": preset_name,
                        "seed": seed,
                        "correction_variant": correction_variant,
                    })
                    step_counter += 1

                # --------------------------------------------------------
                # After stroke: move arm to park pose, capture image, update corrector
                # --------------------------------------------------------
                if corrector_config.enabled and correction_variant == "per_stroke":
                    # Park arm in corner so it does not occlude the ink
                    park_th1, park_th2 = inverse_kinematics(4.0, 6.5, branch="positive")
                    park_c1, park_c2, _ = disturbance_mgr.transform_plant_command(park_th1, park_th2)
                    controller.set_joint_targets(park_c1, park_c2)
                    controller.wait_for_settle(park_c1, park_c2, timeout_s=1.0, tolerance_deg=0.25, step_simulation=True)
                    prev_tip_pos = None  # prevent drawing during park

                    cur_img_bgr = cv2.cvtColor(
                        controller.get_vision_sensor_image(), cv2.COLOR_RGB2BGR
                    )
                    _, obs_pts_cm = observer.observe_stroke(cur_img_bgr)

                    if (len(obs_pts_cm) >= corrector_config.min_observed_pixels
                            and stroke_idx >= corrector_config.correction_start_stroke):
                        # ICP match observed ink to TARGET stroke (not corrected stroke)
                        raw_offset = matcher.estimate_translation(
                            obs_pts_cm,
                            target_strokes_arm[stroke_idx],
                            initial_offset=corrector.current_offset,
                        )
                        if raw_offset is not None:
                            smoothed = corrector.update(raw_offset)
                            correction_log.append({
                                "stroke_idx": stroke_idx,
                                "obs_pixels": len(obs_pts_cm),
                                "raw_dx_mm": raw_offset[0] * 10,
                                "raw_dy_mm": raw_offset[1] * 10,
                                "smooth_dx_mm": smoothed[0] * 10,
                                "smooth_dy_mm": smoothed[1] * 10,
                            })
                        else:
                            correction_log.append({
                                "stroke_idx": stroke_idx,
                                "obs_pixels": len(obs_pts_cm),
                                "raw_dx_mm": 0.0, "raw_dy_mm": 0.0,
                                "smooth_dx_mm": corrector.current_offset[0] * 10,
                                "smooth_dy_mm": corrector.current_offset[1] * 10,
                            })

            exec_duration = time.time() - start_exec_time
            print(f"  Execution finished in {exec_duration:.2f} s, {step_counter} steps, {len(arm_strokes)} strokes.")
            if correction_log:
                print(f"  Final correction: dx={corrector.current_offset[0]*10:.2f} mm, "
                      f"dy={corrector.current_offset[1]*10:.2f} mm  "
                      f"(from {corrector.n_observations} observations)")

            # Final camera capture
            final_img_rgb = controller.get_vision_sensor_image()
            cv2.imwrite(screenshot_png, cv2.cvtColor(final_img_rgb, cv2.COLOR_RGB2BGR))

            controller.clear_drawing_object()
            controller.stop_simulation()
            controller.set_stepping(False)

    except Exception as exc:
        controller.stop_simulation()
        controller.set_stepping(False)
        raise exc

    # -----------------------------------------------------------------------
    # 4. Save telemetry
    # -----------------------------------------------------------------------
    with open(telemetry_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(telemetry_records[0].keys()))
        writer.writeheader()
        writer.writerows(telemetry_records)

    # -----------------------------------------------------------------------
    # 5. Compute metrics
    # -----------------------------------------------------------------------
    plan_pts = np.array([[r["planned_x_cm"], r["planned_y_cm"]] for r in telemetry_records])
    meas_pts = np.array([[r["actual_x_cm"], r["actual_y_cm"]] for r in telemetry_records])
    pen_states = np.array([r["pen_down"] for r in telemetry_records])
    stroke_ids = np.array([r["stroke_id"] for r in telemetry_records])

    decomp = decompose_tracking_errors(
        target_strokes_arm=target_strokes_arm,
        plan_positions_cm=plan_pts,
        measured_positions_cm=meas_pts,
        pen_states=pen_states,
        stroke_ids=stroke_ids,
    )
    exec_err = decomp["execution_error"]
    total_err = decomp["total_error"]

    travel = pen_travel_metrics(meas_pts, pen_states)

    # Camera vision metrics
    cam_img_bgr = cv2.imread(screenshot_png)
    target_mask = rasterize_strokes_to_camera_mask(
        target_strokes_arm, camera_config, camera_config.line_thickness_px
    )
    obs_mask = extract_ink_mask_from_camera_image(cam_img_bgr, is_bgr=True)
    iou_res = raster_iou(target_mask, obs_mask, tolerance_px=2)
    pixel_size_mm = camera_config.view_size_cm * 10.0 / camera_config.resolution_x
    cam_chamfer = raster_chamfer_and_contour_error(target_mask, obs_mask, pixel_size_mm=pixel_size_mm)

    # -----------------------------------------------------------------------
    # 6. Overlay plot
    # -----------------------------------------------------------------------
    _save_overlay_plot(
        overlay_png, target_strokes_arm, meas_pts, pen_states,
        target_mask, obs_mask, iou_res, exec_err, correction_variant, preset_name, seed,
        camera_config, workspace_config,
    )

    metrics = {
        "preset": preset_name,
        "seed": seed,
        "correction_variant": correction_variant,
        "execution_rmse_mm": exec_err["rmse_mm"],
        "mean_tracking_error_mm": exec_err["mean_error_mm"],
        "max_tracking_error_mm": exec_err["max_error_mm"],
        "execution_chamfer_mm": exec_err["chamfer_symmetric_mm"],
        "total_chamfer_mm": total_err["chamfer_symmetric_mm"],
        "execution_endpoint_error_mm": exec_err["mean_endpoint_error_mm"],
        "total_endpoint_error_mm": total_err["mean_endpoint_error_mm"],
        "tolerance_iou_pct": iou_res["tolerance_iou_pct"],
        "plain_iou_pct": iou_res["plain_iou_pct"],
        "camera_chamfer_mm": cam_chamfer["symmetric_chamfer_mm"],
        "camera_contour_error_rmse_mm": cam_chamfer["contour_error_rmse_mm"],
        "final_correction_dx_mm": corrector.current_offset[0] * 10,
        "final_correction_dy_mm": corrector.current_offset[1] * 10,
        "n_correction_observations": corrector.n_observations,
        "telemetry_csv": telemetry_csv,
        "screenshot_png": screenshot_png,
        "overlay_png": overlay_png,
    }

    print(f"\n--- Metrics: {correction_variant.upper()} | {preset_name.upper()} | seed={seed} ---")
    print(f"  Execution RMSE        : {metrics['execution_rmse_mm']:.4f} mm")
    print(f"  Total Chamfer         : {metrics['total_chamfer_mm']:.4f} mm")
    print(f"  Tolerance IoU         : {metrics['tolerance_iou_pct']:.2f}% (tau=2px)")
    print(f"  Plain IoU             : {metrics['plain_iou_pct']:.2f}%")
    print(f"  Endpoint Error        : {metrics['total_endpoint_error_mm']:.4f} mm")
    print(f"  Camera Chamfer        : {metrics['camera_chamfer_mm']:.4f} mm")

    return metrics


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _valid_pt(x: float, y: float) -> bool:
    try:
        validate_workspace_point(x, y)
        validate_joint_configuration(x, y)
        return True
    except ValueError:
        return False


def _apply_correction_to_stroke(
    stroke_pts: np.ndarray,
    corrector: RunningCorrector,
    config: CorrectorConfig,
    workspace_config: WorkspaceConfig,
) -> np.ndarray:
    """Apply correction offset to all points of a stroke, keeping in valid workspace."""
    if not config.enabled or corrector.n_observations == 0:
        return stroke_pts

    corrected = []
    for pt in stroke_pts:
        cx, cy = corrector.correct_waypoint(float(pt[0]), float(pt[1]))
        if _valid_pt(cx, cy):
            corrected.append([cx, cy])
        else:
            corrected.append([float(pt[0]), float(pt[1])])
    return np.array(corrected, dtype=float)


def _run_motion_plan(
    motion_plan: List[MotionPoint],
    controller: CoppeliaController,
    disturbance_mgr: PlantDisturbanceManager,
    record_dict: Dict,
) -> None:
    """Execute a motion plan for calibration purposes (no telemetry tracking)."""
    prev_tip_pos = None
    for pt in motion_plan:
        cmd_th1, cmd_th2, _ = disturbance_mgr.transform_plant_command(pt.theta1, pt.theta2)
        controller.set_joint_targets(cmd_th1, cmd_th2)
        controller.wait_for_settle(cmd_th1, cmd_th2, timeout_s=1.5, tolerance_deg=0.25, step_simulation=True)
        act_x, act_y, act_z = controller.get_tip_position()
        if pt.pen_down and prev_tip_pos is not None:
            controller.add_ink_segment((prev_tip_pos[0], prev_tip_pos[1], 1.55), (act_x, act_y, 1.55), in_meters=False)
        prev_tip_pos = (act_x, act_y, act_z)


def _save_overlay_plot(
    path: str,
    target_strokes_arm: List[np.ndarray],
    meas_pts: np.ndarray,
    pen_states: np.ndarray,
    target_mask: np.ndarray,
    obs_mask: np.ndarray,
    iou_res: Dict,
    exec_err: Dict,
    correction_variant: str,
    preset_name: str,
    seed: int,
    camera_config: CameraConfig,
    workspace_config: WorkspaceConfig,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#ffffff")

    ax0 = axes[0]
    ax0.set_title(
        f"Trajectory Overlay: {correction_variant.upper()} | {preset_name.upper()} (seed {seed})\n"
        f"RMSE: {exec_err['rmse_mm']:.2f} mm | Chamfer: {exec_err['chamfer_symmetric_mm']:.2f} mm",
        fontsize=11, weight="bold"
    )
    ax0.set_xlabel("Robot X (cm)", fontsize=10)
    ax0.set_ylabel("Robot Y (cm)", fontsize=10)
    ax0.set_aspect("equal")
    ax0.grid(True, linestyle="--", alpha=0.5)

    rect = plt.Rectangle(
        (workspace_config.draw_origin_x, workspace_config.draw_origin_y - workspace_config.draw_height_cm),
        workspace_config.draw_width_cm, workspace_config.draw_height_cm,
        edgecolor="black", facecolor="#f8f9fa", linestyle="--", linewidth=1.0,
    )
    ax0.add_patch(rect)

    for s_idx, s in enumerate(target_strokes_arm):
        ax0.plot(s[:, 0], s[:, 1], color="#28a745", linewidth=2.5, alpha=0.7,
                 label="Target" if s_idx == 0 else "")

    drawn_added = False
    for i in range(1, len(meas_pts)):
        if pen_states[i] == 1:
            lbl = "Executed (CL)" if not drawn_added else ""
            ax0.plot([meas_pts[i-1, 0], meas_pts[i, 0]], [meas_pts[i-1, 1], meas_pts[i, 1]],
                     color="#0d6efd", linewidth=1.8, label=lbl)
            drawn_added = True
        else:
            ax0.plot([meas_pts[i-1, 0], meas_pts[i, 0]], [meas_pts[i-1, 1], meas_pts[i, 1]],
                     color="#adb5bd", linestyle=":", linewidth=0.8)
    ax0.set_xlim(3.5, 10.5)
    ax0.set_ylim(6.0, 13.0)
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    ax1.set_title(
        f"Camera Mask (tau=2px)\nTol-IoU: {iou_res['tolerance_iou_pct']:.1f}% | Plain-IoU: {iou_res['plain_iou_pct']:.1f}%",
        fontsize=11, weight="bold"
    )
    overlay = np.full((camera_config.resolution_y, camera_config.resolution_x, 3), 245, dtype=np.uint8)
    overlay[obs_mask > 0] = [180, 50, 20]
    overlay[target_mask > 0] = [30, 160, 30]
    overlap_idx = np.logical_and(target_mask > 0, obs_mask > 0)
    overlay[overlap_idx] = [0, 220, 255]
    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_xlim(130, 380)
    ax1.set_ylim(380, 130)

    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor="#1eb41e", label="Target"),
        Patch(facecolor="#1432b4", label="Executed (CL)"),
        Patch(facecolor="#ffd200", label="Overlap"),
    ], loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="moderate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="per_stroke",
                        choices=["none", "per_stroke", "calib_prepass"])
    args = parser.parse_args()
    execute_closed_loop_run(
        preset_name=args.preset,
        seed=args.seed,
        correction_variant=args.variant,
    )
