"""
scripts/run_phase6_benchmarks.py

Phase 6 Comprehensive Evaluation Suite:
1. Multi-Image Benchmark (all 3 skeleton sketches):
   - 01_robot_arm_sketch
   - 02_technical_gears
   - 03_natural_sketch
   Compares Naive vs NN+Reversal vs Kinematics-Aware 2-Opt.
2. Multi-Objective Weight Sweep:
   Tests sensitivity across Cartesian, Joint, Reversal, and Singularity weights.
3. Exact Optimality Benchmarking:
   Compares Exact DP against heuristics on small instances (N=6 and N=8).
4. Physical Simulation Study under Moderate Disturbance:
   Evaluates whether reducing joint reversals (and thus backlash activations) improves
   tracking RMSE, Chamfer, and Tolerance IoU under Open-Loop and Closed-Loop control.
5. High-Resolution Visual Overlays:
   Generates side-by-side air-travel and trajectory overlay plots.
"""

import argparse
import csv
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, ".")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.setup_scene import setup_scene
from src.config import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_WORKSPACE_CONFIG,
    DEFAULT_STROKE_ORDER_CONFIG,
    StrokeOrderConfig,
    WorkspaceConfig,
)
from src.coordinate_transform import image_to_arm_space
from src.coppelia_controller import CoppeliaController
from src.eval.metrics import (
    decompose_tracking_errors,
    extract_ink_mask_from_camera_image,
    pen_travel_metrics,
    raster_chamfer_and_contour_error,
    raster_iou,
    rasterize_strokes_to_camera_mask,
)
from src.kinematics import inverse_kinematics
from src.path_planner import plan_strokes, MotionPoint
from src.planning.stroke_optimizer import (
    OrientedStroke,
    create_oriented_stroke,
    order_strokes_naive,
    order_strokes_nn,
    order_strokes_2opt,
    order_strokes_exact_dp,
    evaluate_tour_metrics,
    compute_tour_total_cost,
)
from src.sim.disturbances import (
    get_disturbance_preset,
    PlantDisturbanceManager,
)
from src.sim.vision_corrector import (
    CalibrationPrePass,
    CorrectorConfig,
)
from src.vision.image_processor import StrokeIR


OUTPUT_DIR = "logs/phase6_eval"


# -----------------------------------------------------------------------------
# 1. Multi-Image Benchmark
# -----------------------------------------------------------------------------

def run_multi_image_benchmark() -> pd.DataFrame:
    """Benchmark Naive, NN+Reversal, and Kinematics 2-Opt across all 3 skeleton sketches."""
    print("=" * 80)
    print("PHASE 6: MULTI-IMAGE STROKE ORDERING BENCHMARK")
    print("=" * 80)

    dataset = [
        ("01_robot_arm_sketch", "data/extracted_strokes/01_robot_arm_sketch_skeleton.json"),
        ("02_technical_gears", "data/extracted_strokes/02_technical_gears_skeleton.json"),
        ("03_natural_sketch", "data/extracted_strokes/03_natural_sketch_skeleton.json"),
    ]

    wc = DEFAULT_WORKSPACE_CONFIG
    cfg = DEFAULT_STROKE_ORDER_CONFIG
    records: List[Dict[str, Any]] = []

    for name, json_path in dataset:
        ir = StrokeIR.from_json(json_path)
        strokes = []
        for s in ir.strokes:
            if len(s) >= 2:
                pts = image_to_arm_space(
                    s,
                    wc.image_width,
                    wc.image_height,
                    wc.draw_origin_x,
                    wc.draw_origin_y,
                    wc.draw_width_cm,
                    wc.draw_height_cm,
                )
                strokes.append(pts)

        print(f"\n--- Processing Image: {name} ({len(strokes)} strokes) ---")

        # 1. Naive (Phase 2 Baseline)
        t0 = time.time()
        naive_tour = order_strokes_naive(strokes, branch=cfg.branch)
        t_naive = time.time() - t0
        m_naive = evaluate_tour_metrics(naive_tour, cfg)

        # 2. Nearest-Neighbour with Reversal
        t0 = time.time()
        nn_tour = order_strokes_nn(strokes, cfg, multi_start=True)
        t_nn = time.time() - t0
        m_nn = evaluate_tour_metrics(nn_tour, cfg)

        # 3. Kinematics-Aware 2-Opt
        t0 = time.time()
        opt_tour = order_strokes_2opt(strokes, cfg, initial_tour=nn_tour)
        t_opt = time.time() - t0
        m_opt = evaluate_tour_metrics(opt_tour, cfg)

        for mode, m, elapsed in [
            ("Naive (Baseline)", m_naive, t_naive),
            ("NN + Reversal", m_nn, t_nn),
            ("Kinematic 2-Opt", m_opt, t_opt),
        ]:
            records.append({
                "Image": name,
                "Ordering Method": mode,
                "Stroke Count": len(strokes),
                "Pen-Down (cm)": round(m["pen_down_distance_cm"], 2),
                "Pen-Up (cm)": round(m["pen_up_distance_cm"], 2),
                "Total Travel (cm)": round(m["total_distance_cm"], 2),
                "Efficiency (%)": round(m["pen_travel_efficiency_pct"], 1),
                "Joint Disp Proxy (rad)": round(m["total_joint_displacement_rad"], 2),
                "Joint Reversals": m["joint_reversals_count"],
                "Min |det(J)|": round(m["min_det_jacobian"], 2),
                "Mean |det(J)|": round(m["mean_det_jacobian"], 2),
                "Solve Time (ms)": round(elapsed * 1000.0, 1),
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "phase6_multi_image_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved multi-image summary to {csv_path}")
    print(df.to_string(index=False))

    # Generate 3-image visual comparison overlays
    _plot_multi_image_tours(dataset, wc, cfg)

    return df


# -----------------------------------------------------------------------------
# 2. Configurable Weight Sweep
# -----------------------------------------------------------------------------

def run_weight_sweep() -> pd.DataFrame:
    """Evaluate optimization trade-offs under different multi-objective weight combinations."""
    print("\n" + "=" * 80)
    print("PHASE 6: MULTI-OBJECTIVE WEIGHT SWEEP")
    print("=" * 80)

    ir = StrokeIR.from_json("data/extracted_strokes/01_robot_arm_sketch_skeleton.json")
    wc = DEFAULT_WORKSPACE_CONFIG
    strokes = [
        image_to_arm_space(s, wc.image_width, wc.image_height, wc.draw_origin_x, wc.draw_origin_y, wc.draw_width_cm, wc.draw_height_cm)
        for s in ir.strokes if len(s) >= 2
    ]

    profiles = [
        ("Cartesian-Heavy", StrokeOrderConfig(w_cartesian=1.0, w_joint=0.0, w_reversal=0.0, w_singularity=0.0)),
        ("Balanced (Default)", StrokeOrderConfig(w_cartesian=1.0, w_joint=0.5, w_reversal=0.3, w_singularity=0.05)),
        ("Reversal-Minimized", StrokeOrderConfig(w_cartesian=0.5, w_joint=0.2, w_reversal=1.0, w_singularity=0.05)),
        ("Singularity-Averse", StrokeOrderConfig(w_cartesian=1.0, w_joint=0.5, w_reversal=0.3, w_singularity=0.20)),
    ]

    records = []
    for pname, cfg in profiles:
        tour = order_strokes_2opt(strokes, cfg)
        m = evaluate_tour_metrics(tour, cfg)
        records.append({
            "Profile": pname,
            "w_cart": cfg.w_cartesian,
            "w_joint": cfg.w_joint,
            "w_rev": cfg.w_reversal,
            "w_sing": cfg.w_singularity,
            "Pen-Up (cm)": round(m["pen_up_distance_cm"], 2),
            "Efficiency (%)": round(m["pen_travel_efficiency_pct"], 1),
            "Joint Disp (rad)": round(m["total_joint_displacement_rad"], 2),
            "Joint Reversals": m["joint_reversals_count"],
            "Min |det(J)|": round(m["min_det_jacobian"], 2),
            "Mean |det(J)|": round(m["mean_det_jacobian"], 2),
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "phase6_weight_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved weight sweep to {csv_path}")
    print(df.to_string(index=False))
    return df


# -----------------------------------------------------------------------------
# 3. Exact Optimality Benchmarking
# -----------------------------------------------------------------------------

def run_exact_optimality_benchmark() -> pd.DataFrame:
    """Benchmark Exact DP vs NN and 2-Opt on small stroke instances."""
    print("\n" + "=" * 80)
    print("PHASE 6: EXACT OPTIMALITY BENCHMARK (ATSP with Direction Selection)")
    print("=" * 80)

    ir = StrokeIR.from_json("data/extracted_strokes/01_robot_arm_sketch_skeleton.json")
    wc = DEFAULT_WORKSPACE_CONFIG
    strokes = [
        image_to_arm_space(s, wc.image_width, wc.image_height, wc.draw_origin_x, wc.draw_origin_y, wc.draw_width_cm, wc.draw_height_cm)
        for s in ir.strokes if len(s) >= 2
    ]

    records = []
    cfg = DEFAULT_STROKE_ORDER_CONFIG

    for n_sub in [4, 6, 8]:
        sub_strokes = strokes[:n_sub]

        # Exact DP
        t0 = time.time()
        dp_tour, dp_cost = order_strokes_exact_dp(sub_strokes, cfg)
        t_dp = time.time() - t0
        m_dp = evaluate_tour_metrics(dp_tour, cfg)

        # NN with multi-start
        t0 = time.time()
        nn_tour = order_strokes_nn(sub_strokes, cfg, multi_start=True)
        t_nn = time.time() - t0
        c_nn = compute_tour_total_cost(nn_tour, cfg)
        m_nn = evaluate_tour_metrics(nn_tour, cfg)

        # 2-Opt
        t0 = time.time()
        opt_tour = order_strokes_2opt(sub_strokes, cfg, initial_tour=nn_tour)
        t_opt = time.time() - t0
        c_opt = compute_tour_total_cost(opt_tour, cfg)
        m_opt = evaluate_tour_metrics(opt_tour, cfg)

        gap_nn_pct = (c_nn - dp_cost) / dp_cost * 100.0 if dp_cost > 0 else 0.0
        gap_opt_pct = (c_opt - dp_cost) / dp_cost * 100.0 if dp_cost > 0 else 0.0

        records.append({
            "Instance Size (N)": n_sub,
            "State Space (2^N * N!)": f"{((2**n_sub) * math.factorial(n_sub)):,}",
            "Exact DP Cost": round(dp_cost, 3),
            "Exact DP Pen-Up (cm)": round(m_dp["pen_up_distance_cm"], 2),
            "Exact DP Time (ms)": round(t_dp * 1000.0, 1),
            "NN Cost": round(c_nn, 3),
            "NN Optimality Gap (%)": f"{gap_nn_pct:.1f}%",
            "2-Opt Cost": round(c_opt, 3),
            "2-Opt Optimality Gap (%)": f"{gap_opt_pct:.1f}%",
            "2-Opt Time (ms)": round(t_opt * 1000.0, 1),
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "phase6_exact_optimality.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved exact optimality benchmark to {csv_path}")
    print(df.to_string(index=False))
    return df


# -----------------------------------------------------------------------------
# 4. Simulation Study: Backlash & Reversals under Moderate Disturbance
# -----------------------------------------------------------------------------

def run_simulation_backlash_study(seeds: List[int] = [42, 43, 44]) -> pd.DataFrame:
    """
    Execute simulation under the moderate disturbance preset to test whether reducing
    joint direction reversals decreases physical backlash accumulation in open-loop and closed-loop.
    """
    print("\n" + "=" * 80)
    print("PHASE 6: JOINT REVERSALS & BACKLASH SIMULATION STUDY (Moderate Preset)")
    print("=" * 80)

    ir = StrokeIR.from_json("data/extracted_strokes/01_robot_arm_sketch_skeleton.json")
    wc = DEFAULT_WORKSPACE_CONFIG
    strokes = [
        image_to_arm_space(s, wc.image_width, wc.image_height, wc.draw_origin_x, wc.draw_origin_y, wc.draw_width_cm, wc.draw_height_cm)
        for s in ir.strokes if len(s) >= 2
    ]

    naive_tour = order_strokes_naive(strokes)
    opt_tour = order_strokes_2opt(strokes, DEFAULT_STROKE_ORDER_CONFIG)

    results = []

    for seed in seeds:
        for mode_name, tour in [("Naive (Baseline)", naive_tour), ("Optimized (2-Opt)", opt_tour)]:
            # Convert tour back to stroke lists for execution
            tour_strokes = [s.points.tolist() for s in tour]

            # 1. Open-Loop Run
            res_ol = _execute_ordered_simulation(
                tour_strokes=tour_strokes,
                preset_name="moderate",
                seed=seed,
                is_closed_loop=False,
                wc=wc,
            )
            # 2. Closed-Loop Run (Calib PrePass)
            res_cl = _execute_ordered_simulation(
                tour_strokes=tour_strokes,
                preset_name="moderate",
                seed=seed,
                is_closed_loop=True,
                wc=wc,
            )

            results.append({
                "Seed": seed,
                "Ordering": mode_name,
                "Control": "Open-Loop",
                "Reversals": evaluate_tour_metrics(tour)["joint_reversals_count"],
                "RMSE (mm)": round(res_ol["rmse_mm"], 3),
                "Chamfer (mm)": round(res_ol["chamfer_mm"], 3),
                "Tol-IoU (%)": round(res_ol["tol_iou_pct"], 2),
                "Pen-Up (cm)": round(res_ol["pen_up_dist_cm"], 2),
                "Exec Time (s)": round(res_ol["exec_time_s"], 2),
            })
            results.append({
                "Seed": seed,
                "Ordering": mode_name,
                "Control": "Closed-Loop (CL)",
                "Reversals": evaluate_tour_metrics(tour)["joint_reversals_count"],
                "RMSE (mm)": round(res_cl["rmse_mm"], 3),
                "Chamfer (mm)": round(res_cl["chamfer_mm"], 3),
                "Tol-IoU (%)": round(res_cl["tol_iou_pct"], 2),
                "Pen-Up (cm)": round(res_cl["pen_up_dist_cm"], 2),
                "Exec Time (s)": round(res_cl["exec_time_s"], 2),
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "phase6_reversal_backlash_study.csv")
    df.to_csv(csv_path, index=False)

    # Group by Ordering and Control for aggregate presentation
    agg = df.groupby(["Ordering", "Control"]).agg({
        "Reversals": "first",
        "RMSE (mm)": ["mean", "std"],
        "Chamfer (mm)": ["mean", "std"],
        "Tol-IoU (%)": ["mean", "std"],
        "Pen-Up (cm)": "first",
        "Exec Time (s)": "mean",
    }).reset_index()

    print(f"\nSaved backlash study to {csv_path}")
    print(df.to_string(index=False))

    return df


def _execute_ordered_simulation(
    tour_strokes: List[List[List[float]]],
    preset_name: str,
    seed: int,
    is_closed_loop: bool,
    wc: WorkspaceConfig,
    camera_config = DEFAULT_CAMERA_CONFIG,
    spacing_cm: float = 0.25,
) -> Dict[str, float]:
    """Execute simulation run with given ordered strokes."""
    setup_scene()
    dist_cfg = get_disturbance_preset(preset_name, seed=seed)
    disturbance_mgr = PlantDisturbanceManager(dist_cfg)

    corrector_config = CorrectorConfig(enabled=is_closed_loop, calib_prepass=is_closed_loop)
    calib_pp = CalibrationPrePass(corrector_config, camera_config) if is_closed_loop else None

    target_strokes_arm = [np.array(s, dtype=float) for s in tour_strokes]
    nominal_plan = plan_strokes(tour_strokes, branch="positive", spacing_cm=spacing_cm)

    controller = CoppeliaController(verbose=False)
    controller.stop_simulation()
    controller.clear_drawing_object()
    controller.set_stepping(True)

    telemetry = []

    with disturbance_mgr.active_context(controller.sim):
        controller.start_simulation()
        controller.clear_drawing_object()
        controller.create_drawing_object(line_width=3.5, color=(0.05, 0.15, 0.70))
        controller.step()

        # Calibration pre-pass if closed loop
        corr_offset = np.zeros(2)
        if is_closed_loop and calib_pp is not None:
            baseline_img_bgr = cv2.cvtColor(controller.get_vision_sensor_image(), cv2.COLOR_RGB2BGR)
            calib_strokes = calib_pp.get_pattern_strokes()
            calib_plan = plan_strokes([s.tolist() for s in calib_strokes], branch="positive", spacing_cm=spacing_cm)
            calib_prev_pos = None
            for pt in calib_plan:
                c1, c2, _ = disturbance_mgr.transform_plant_command(pt.theta1, pt.theta2)
                controller.set_joint_targets(c1, c2)
                controller.wait_for_settle(c1, c2, timeout_s=1.0, tolerance_deg=0.25, step_simulation=True)
                act_x, act_y, act_z = controller.get_tip_position()
                if pt.pen_down and calib_prev_pos is not None:
                    controller.add_ink_segment((calib_prev_pos[0], calib_prev_pos[1], 1.55), (act_x, act_y, 1.55), in_meters=False)
                calib_prev_pos = (act_x, act_y, act_z)

            # Park arm
            park_th1, park_th2 = inverse_kinematics(4.0, 6.5, branch="positive")
            park_c1, park_c2, _ = disturbance_mgr.transform_plant_command(park_th1, park_th2)
            controller.set_joint_targets(park_c1, park_c2)
            controller.wait_for_settle(park_c1, park_c2, timeout_s=1.0, tolerance_deg=0.25, step_simulation=True)

            calib_img_bgr = cv2.cvtColor(controller.get_vision_sensor_image(), cv2.COLOR_RGB2BGR)
            offset = calib_pp.estimate_offset_from_image(calib_img_bgr, baseline_img_bgr)
            if offset is not None:
                corr_offset = offset

            # Clear calibration ink
            controller.clear_drawing_object()
            controller.create_drawing_object(line_width=3.5, color=(0.05, 0.15, 0.70))
            controller.step()

        # Execute tour motion plan
        start_t = time.time()
        prev_tip_pos = None

        for pt in nominal_plan:
            # Apply correction if closed-loop
            if is_closed_loop and np.linalg.norm(corr_offset) > 0:
                cmd_x = pt.x - corr_offset[0]
                cmd_y = pt.y - corr_offset[1]
                try:
                    cmd_th1, cmd_th2 = inverse_kinematics(cmd_x, cmd_y, branch="positive")
                except ValueError:
                    cmd_th1, cmd_th2 = pt.theta1, pt.theta2
            else:
                cmd_th1, cmd_th2 = pt.theta1, pt.theta2

            p_th1, p_th2, _ = disturbance_mgr.transform_plant_command(cmd_th1, cmd_th2)
            controller.set_joint_targets(p_th1, p_th2)
            controller.wait_for_settle(p_th1, p_th2, timeout_s=1.5, tolerance_deg=0.25, step_simulation=True)

            act_x, act_y, act_z = controller.get_tip_position()

            if pt.pen_down and prev_tip_pos is not None:
                controller.add_ink_segment((prev_tip_pos[0], prev_tip_pos[1], 1.55), (act_x, act_y, 1.55), in_meters=False)
            prev_tip_pos = (act_x, act_y, act_z)

            telemetry.append({
                "target_x": pt.x,
                "target_y": pt.y,
                "actual_x": act_x,
                "actual_y": act_y,
                "pen_down": 1 if pt.pen_down else 0,
                "stroke_id": pt.stroke_id,
            })

        exec_time = time.time() - start_t
        cam_img_rgb = controller.get_vision_sensor_image()
        controller.clear_drawing_object()
        controller.stop_simulation()
        controller.set_stepping(False)

    # Compute metrics
    target_pts = np.array([[r["target_x"], r["target_y"]] for r in telemetry])
    meas_pts = np.array([[r["actual_x"], r["actual_y"]] for r in telemetry])
    pen_states = np.array([r["pen_down"] for r in telemetry])
    stroke_ids = np.array([r["stroke_id"] for r in telemetry])

    decomp = decompose_tracking_errors(
        target_strokes_arm=target_strokes_arm,
        plan_positions_cm=target_pts,
        measured_positions_cm=meas_pts,
        pen_states=pen_states,
        stroke_ids=stroke_ids,
    )
    travel = pen_travel_metrics(meas_pts, pen_states)

    # Camera IoU & Chamfer
    cam_img_bgr = cv2.cvtColor(cam_img_rgb, cv2.COLOR_RGB2BGR)
    target_mask = rasterize_strokes_to_camera_mask(target_strokes_arm, camera_config, camera_config.line_thickness_px)
    obs_mask = extract_ink_mask_from_camera_image(cam_img_bgr, is_bgr=True)
    iou_res = raster_iou(target_mask, obs_mask, tolerance_px=2)
    px_size_mm = camera_config.view_size_cm * 10.0 / camera_config.resolution_x
    cam_chamfer = raster_chamfer_and_contour_error(target_mask, obs_mask, pixel_size_mm=px_size_mm)

    return {
        "rmse_mm": decomp["execution_error"]["rmse_mm"],
        "chamfer_mm": decomp["total_error"]["chamfer_symmetric_mm"],
        "tol_iou_pct": iou_res["tolerance_iou_pct"],
        "endpoint_error_mm": decomp["total_error"]["mean_endpoint_error_mm"],
        "pen_up_dist_cm": travel["pen_up_distance_cm"],
        "exec_time_s": exec_time,
    }


# -----------------------------------------------------------------------------
# 5. Visual Overlays: Side-by-Side Comparison Plots
# -----------------------------------------------------------------------------

def _plot_multi_image_tours(dataset, wc: WorkspaceConfig, cfg: StrokeOrderConfig) -> None:
    """Generate side-by-side plots for each image showing Naive vs 2-Opt pen-up repositioning paths."""
    for name, json_path in dataset:
        ir = StrokeIR.from_json(json_path)
        strokes = []
        for s in ir.strokes:
            if len(s) >= 2:
                pts = image_to_arm_space(
                    s,
                    wc.image_width,
                    wc.image_height,
                    wc.draw_origin_x,
                    wc.draw_origin_y,
                    wc.draw_width_cm,
                    wc.draw_height_cm,
                )
                strokes.append(pts)

        naive_tour = order_strokes_naive(strokes, branch=cfg.branch)
        opt_tour = order_strokes_2opt(strokes, cfg)

        m_naive = evaluate_tour_metrics(naive_tour, cfg)
        m_opt = evaluate_tour_metrics(opt_tour, cfg)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#ffffff")

        for idx, (ax, tour, m, title) in enumerate([
            (axes[0], naive_tour, m_naive, f"Naive Baseline ({name})\nPen-Up: {m_naive['pen_up_distance_cm']:.2f} cm | Efficiency: {m_naive['pen_travel_efficiency_pct']:.1f}% | Reversals: {m_naive['joint_reversals_count']}"),
            (axes[1], opt_tour, m_opt, f"Kinematic 2-Opt ({name})\nPen-Up: {m_opt['pen_up_distance_cm']:.2f} cm | Efficiency: {m_opt['pen_travel_efficiency_pct']:.1f}% | Reversals: {m_opt['joint_reversals_count']}"),
        ]):
            ax.set_title(title, fontsize=10, weight="bold")
            ax.set_xlabel("Robot X (cm)")
            ax.set_ylabel("Robot Y (cm)")
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.5)

            # Drawing window box
            rect = plt.Rectangle(
                (wc.draw_origin_x, wc.draw_origin_y - wc.draw_height_cm),
                wc.draw_width_cm, wc.draw_height_cm,
                edgecolor="black", facecolor="#f8f9fa", linestyle="--", linewidth=1.0,
            )
            ax.add_patch(rect)

            # Draw drawing strokes (solid blue/green)
            for s in tour:
                ax.plot(s.points[:, 0], s.points[:, 1], color="#198754", linewidth=2.0)
                # Mark stroke start (circle) and end (triangle)
                ax.scatter(s.start_pt[0], s.start_pt[1], color="#0d6efd", s=18, zorder=4)
                ax.scatter(s.end_pt[0], s.end_pt[1], color="#dc3545", s=18, marker="^", zorder=4)

            # Draw pen-up air repositioning lines (dashed red)
            for i in range(len(tour) - 1):
                p_end = tour[i].end_pt
                p_start = tour[i + 1].start_pt
                ax.plot(
                    [p_end[0], p_start[0]], [p_end[1], p_start[1]],
                    color="#dc3545", linestyle=":", linewidth=1.2, alpha=0.85
                )

            ax.set_xlim(3.5, 10.5)
            ax.set_ylim(6.0, 13.0)

        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color="#198754", lw=2, label="Drawing Stroke (Pen-Down)"),
            Line2D([0], [0], color="#dc3545", lw=1.5, linestyle=":", label="Air Repositioning (Pen-Up)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#0d6efd", markersize=6, label="Stroke Start"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#dc3545", markersize=6, label="Stroke End"),
        ]
        fig.legend(handles=custom_lines, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

        plot_path = os.path.join(OUTPUT_DIR, f"phase6_tour_comparison_{name}.png")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.10)
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved tour comparison figure to {plot_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_multi_image_benchmark()
    run_weight_sweep()
    run_exact_optimality_benchmark()
    run_simulation_backlash_study(seeds=[42, 43, 44])
