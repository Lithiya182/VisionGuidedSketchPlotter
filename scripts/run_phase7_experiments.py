"""
scripts/run_phase7_experiments.py

Phase 7: End-to-End Experiments
================================
Full cross-factor matrix:
  Images    : 01_robot_arm_sketch, 02_technical_gears, 03_natural_sketch, 04_complex_logo
  Modes     : open_loop_naive, open_loop_optimized, cl_prepass_naive, cl_prepass_optimized
  Presets   : none, mild, moderate, severe
  Seeds     : 42, 43, 44  (only for mild/moderate/severe; none is deterministic)

Metrics collected per run:
  RMSE (mm), Chamfer (mm), Tol-IoU (%), Endpoint error (mm),
  Pen-up dist (cm), Exec time (s)

Outputs:
  logs/phase7/phase7_aggregate.csv   -- one row per (image, mode, preset, seed)
  logs/phase7/phase7_summary.csv     -- mean ± std over seeds grouped by (image, mode, preset)
  logs/phase7/phase7_summary_table.txt
  logs/phase7/<image>_<mode>_<preset>_<seed>_overlay.png
"""

import csv
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.setup_scene import setup_scene
from src.config import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_STROKE_ORDER_CONFIG,
    DEFAULT_WORKSPACE_CONFIG,
    WorkspaceConfig,
    CameraConfig,
)
from src.coordinate_transform import image_to_arm_space, validate_workspace_point
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
from src.path_planner import plan_strokes
from src.planning.stroke_optimizer import (
    order_strokes_naive,
    order_strokes_2opt,
    evaluate_tour_metrics,
)
from src.sim.disturbances import get_disturbance_preset, PlantDisturbanceManager
from src.sim.vision_corrector import (
    CalibrationPrePass,
    CorrectorConfig,
    PerStrokeObserver,
    RunningCorrector,
    StrokeMatcher,
)
from src.vision.image_processor import StrokeIR

OUTPUT_DIR = "logs/phase7"

# ── Dataset ─────────────────────────────────────────────────────────────────────
DATASET = [
    ("01_robot_arm_sketch",
     "data/extracted_strokes/01_robot_arm_sketch_skeleton.json"),
    ("02_technical_gears",
     "data/extracted_strokes/02_technical_gears_skeleton.json"),
    ("03_natural_sketch",
     "data/extracted_strokes/03_natural_sketch_skeleton.json"),
    ("04_complex_logo",
     "data/extracted_strokes/04_complex_logo_skeleton.json"),
]

PRESETS = ["none", "mild", "moderate", "severe"]
SEEDS_STOCHASTIC = [42, 43, 44]
SEED_DETERMINISTIC = 42

# modes: (ordering, correction_variant)
MODES = [
    ("open_loop_naive",      "naive",     "none"),
    ("open_loop_optimized",  "optimized", "none"),
    ("cl_prepass_naive",     "naive",     "cl_prepass"),
    ("cl_prepass_optimized", "optimized", "cl_prepass"),
]


# ── Helper: load arm-space strokes from IR json ──────────────────────────────
def _load_strokes(json_path: str, wc: WorkspaceConfig) -> List[np.ndarray]:
    ir = StrokeIR.from_json(json_path)
    strokes = []
    for s in ir.strokes:
        if len(s) < 2:
            continue
        pts = image_to_arm_space(
            s,
            wc.image_width, wc.image_height,
            wc.draw_origin_x, wc.draw_origin_y,
            wc.draw_width_cm, wc.draw_height_cm,
        )
        # Filter reachable
        valid = [p for p in pts if _valid_pt(float(p[0]), float(p[1]))]
        if len(valid) >= 2:
            strokes.append(np.array(valid, dtype=float))
    return strokes


def _valid_pt(x: float, y: float) -> bool:
    try:
        from src.coordinate_transform import validate_joint_configuration
        validate_workspace_point(x, y)
        validate_joint_configuration(x, y)
        return True
    except ValueError:
        return False


# ── Single simulation run ─────────────────────────────────────────────────────
def _run_sim(
    tour_strokes: List[np.ndarray],
    preset_name: str,
    seed: int,
    correction_variant: str,  # 'none' or 'cl_prepass'
    wc: WorkspaceConfig = DEFAULT_WORKSPACE_CONFIG,
    camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
    spacing_cm: float = 0.25,
    output_overlay: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one simulation run and return metrics dict."""
    setup_scene()
    dist_cfg = get_disturbance_preset(preset_name, seed=seed)
    disturbance_mgr = PlantDisturbanceManager(dist_cfg)

    is_cl = correction_variant != "none"
    corrector_config = CorrectorConfig(enabled=is_cl, calib_prepass=is_cl)
    corrector = RunningCorrector(corrector_config)
    calib_pp = CalibrationPrePass(corrector_config, camera_config) if is_cl else None

    target_strokes_arm = tour_strokes  # already arm-space np arrays

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

        baseline_img_bgr = cv2.cvtColor(
            controller.get_vision_sensor_image(), cv2.COLOR_RGB2BGR
        )

        # ── Calibration pre-pass ────────────────────────────────────────────
        if is_cl and calib_pp is not None:
            calib_strokes = calib_pp.get_pattern_strokes()
            calib_plan = plan_strokes(
                [s.tolist() for s in calib_strokes],
                branch="positive", spacing_cm=spacing_cm,
            )
            calib_prev = None
            for pt in calib_plan:
                c1, c2, _ = disturbance_mgr.transform_plant_command(pt.theta1, pt.theta2)
                controller.set_joint_targets(c1, c2)
                controller.wait_for_settle(c1, c2, timeout_s=1.0, tolerance_deg=0.25, step_simulation=True)
                ax, ay, az = controller.get_tip_position()
                if pt.pen_down and calib_prev is not None:
                    controller.add_ink_segment(
                        (calib_prev[0], calib_prev[1], 1.55),
                        (ax, ay, 1.55), in_meters=False
                    )
                calib_prev = (ax, ay, az)

            # Park arm
            pk1, pk2 = inverse_kinematics(4.0, 6.5, branch="positive")
            pc1, pc2, _ = disturbance_mgr.transform_plant_command(pk1, pk2)
            controller.set_joint_targets(pc1, pc2)
            controller.wait_for_settle(pc1, pc2, timeout_s=1.0, tolerance_deg=0.25, step_simulation=True)

            calib_img = cv2.cvtColor(controller.get_vision_sensor_image(), cv2.COLOR_RGB2BGR)
            offset = calib_pp.estimate_offset_from_image(calib_img, baseline_img_bgr)
            if offset is not None:
                corrector.update(offset)

            # Wipe calibration ink
            controller.clear_drawing_object()
            controller.create_drawing_object(line_width=3.5, color=(0.05, 0.15, 0.70))
            controller.step()

        # ── Main drawing ────────────────────────────────────────────────────
        start_t = time.time()
        prev_tip = None

        for stroke_idx, stroke_arr in enumerate(target_strokes_arm):
            nom_plan = plan_strokes(
                [stroke_arr.tolist()], branch="positive", spacing_cm=spacing_cm
            )

            # Apply correction
            if is_cl and corrector.n_observations > 0:
                corr_pts = stroke_arr - corrector.current_offset
                corr_pts_valid = np.array(
                    [p for p in corr_pts if _valid_pt(float(p[0]), float(p[1]))],
                    dtype=float,
                )
                if len(corr_pts_valid) >= 2:
                    try:
                        exec_plan = plan_strokes(
                            [corr_pts_valid.tolist()], branch="positive",
                            spacing_cm=spacing_cm, skip_invalid_strokes=True,
                        )
                        if len(exec_plan) != len(nom_plan):
                            exec_plan = nom_plan
                    except ValueError:
                        exec_plan = nom_plan
                else:
                    exec_plan = nom_plan
            else:
                exec_plan = nom_plan

            for nom_pt, pt in zip(nom_plan, exec_plan):
                p1, p2, _ = disturbance_mgr.transform_plant_command(pt.theta1, pt.theta2)
                controller.set_joint_targets(p1, p2)
                controller.wait_for_settle(p1, p2, timeout_s=1.5, tolerance_deg=0.25, step_simulation=True)

                ax, ay, az = controller.get_tip_position()
                if pt.pen_down and prev_tip is not None:
                    controller.add_ink_segment(
                        (prev_tip[0], prev_tip[1], 1.55),
                        (ax, ay, 1.55), in_meters=False
                    )
                prev_tip = (ax, ay, az)

                telemetry.append({
                    "target_x": nom_pt.x, "target_y": nom_pt.y,
                    "actual_x": ax, "actual_y": ay,
                    "pen_down": 1 if pt.pen_down else 0,
                    "stroke_id": stroke_idx,
                })

        exec_time = time.time() - start_t
        cam_img_rgb = controller.get_vision_sensor_image()
        controller.clear_drawing_object()
        controller.stop_simulation()
        controller.set_stepping(False)

    # ── Metrics ──────────────────────────────────────────────────────────────
    tgt = np.array([[r["target_x"], r["target_y"]] for r in telemetry])
    meas = np.array([[r["actual_x"], r["actual_y"]] for r in telemetry])
    pens = np.array([r["pen_down"] for r in telemetry])
    sids = np.array([r["stroke_id"] for r in telemetry])

    decomp = decompose_tracking_errors(
        target_strokes_arm=target_strokes_arm,
        plan_positions_cm=tgt,
        measured_positions_cm=meas,
        pen_states=pens,
        stroke_ids=sids,
    )
    travel = pen_travel_metrics(meas, pens)

    cam_bgr = cv2.cvtColor(cam_img_rgb, cv2.COLOR_RGB2BGR)
    tgt_mask = rasterize_strokes_to_camera_mask(
        target_strokes_arm, camera_config, camera_config.line_thickness_px
    )
    obs_mask = extract_ink_mask_from_camera_image(cam_bgr, is_bgr=True)
    iou_res = raster_iou(tgt_mask, obs_mask, tolerance_px=2)
    px_mm = camera_config.view_size_cm * 10.0 / camera_config.resolution_x
    cam_ch = raster_chamfer_and_contour_error(tgt_mask, obs_mask, pixel_size_mm=px_mm)

    # Optional overlay save
    if output_overlay:
        _save_overlay(
            output_overlay, target_strokes_arm, meas, pens,
            tgt_mask, obs_mask, iou_res, decomp, camera_config, wc,
        )

    return {
        "rmse_mm": decomp["execution_error"]["rmse_mm"],
        "chamfer_mm": cam_ch["symmetric_chamfer_mm"],
        "tol_iou_pct": iou_res["tolerance_iou_pct"],
        "plain_iou_pct": iou_res["plain_iou_pct"],
        "endpoint_error_mm": decomp["total_error"]["mean_endpoint_error_mm"],
        "pen_up_dist_cm": travel["pen_up_distance_cm"],
        "exec_time_s": exec_time,
    }


# ── Overlay plot ─────────────────────────────────────────────────────────────
def _save_overlay(path, tgt_strokes, meas, pens, tgt_mask, obs_mask, iou, decomp, cam, wc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#fff")

    ax0 = axes[0]
    ax0.set_title(
        f"Trajectory: RMSE={decomp['execution_error']['rmse_mm']:.2f} mm",
        fontsize=10, weight="bold",
    )
    ax0.set_xlabel("X (cm)"); ax0.set_ylabel("Y (cm)")
    ax0.set_aspect("equal"); ax0.grid(True, linestyle="--", alpha=0.4)
    rect = plt.Rectangle(
        (wc.draw_origin_x, wc.draw_origin_y - wc.draw_height_cm),
        wc.draw_width_cm, wc.draw_height_cm,
        edgecolor="black", facecolor="#f8f9fa", linestyle="--", lw=1,
    )
    ax0.add_patch(rect)
    for s in tgt_strokes:
        ax0.plot(s[:, 0], s[:, 1], color="#28a745", lw=2.2, alpha=0.7)
    for i in range(1, len(meas)):
        if pens[i] == 1:
            ax0.plot([meas[i-1, 0], meas[i, 0]], [meas[i-1, 1], meas[i, 1]],
                     color="#0d6efd", lw=1.5)
    ax0.set_xlim(3.5, 10.5); ax0.set_ylim(6.0, 13.0)

    ax1 = axes[1]
    ax1.set_title(
        f"Camera mask  Tol-IoU={iou['tolerance_iou_pct']:.1f}%  Plain-IoU={iou['plain_iou_pct']:.1f}%",
        fontsize=10, weight="bold",
    )
    overlay = np.full((cam.resolution_y, cam.resolution_x, 3), 245, dtype=np.uint8)
    overlay[obs_mask > 0] = [180, 50, 20]
    overlay[tgt_mask > 0] = [30, 160, 30]
    overlay[np.logical_and(tgt_mask > 0, obs_mask > 0)] = [0, 200, 255]
    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_xlim(100, 380); ax1.set_ylim(380, 100)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Main experiment loop ──────────────────────────────────────────────────────
def run_experiments(
    images=None,
    presets=None,
    seeds=None,
    save_overlays: bool = True,
) -> pd.DataFrame:
    if images is None:
        images = DATASET
    if presets is None:
        presets = PRESETS
    if seeds is None:
        seeds = SEEDS_STOCHASTIC

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wc = DEFAULT_WORKSPACE_CONFIG
    cam = DEFAULT_CAMERA_CONFIG
    cfg = DEFAULT_STROKE_ORDER_CONFIG

    # ── Resume support: load any previously completed rows ──────────────────
    agg_path = os.path.join(OUTPUT_DIR, "phase7_aggregate.csv")
    records: List[Dict[str, Any]] = []
    completed: set = set()
    if os.path.exists(agg_path):
        try:
            prev_df = pd.read_csv(agg_path)
            for _, row in prev_df.iterrows():
                records.append(row.to_dict())
                completed.add((str(row["image"]), str(row["mode"]),
                                str(row["preset"]), int(row["seed"])))
            print(f"  Resuming: {len(records)} rows already done, {len(completed)} unique runs.")
        except Exception:
            pass  # if file is corrupted, start fresh

    print("=" * 110)
    print("PHASE 7: FULL EXPERIMENT MATRIX")
    print(f"  Images x Modes x Presets x Seeds = "
          f"{len(images)} x {len(MODES)} x {len(presets)} x {len(seeds)} runs")
    print("=" * 110)

    for img_name, json_path in images:
        print(f"\n{'='*50} IMAGE: {img_name} {'='*50}")
        raw_strokes = _load_strokes(json_path, wc)
        if not raw_strokes:
            print(f"  WARN: no valid strokes for {img_name}, skipping.")
            continue
        print(f"  Loaded {len(raw_strokes)} valid strokes.")

        # Pre-compute tours (ordering independent of seed/preset)
        naive_tour = order_strokes_naive(raw_strokes)
        opt_tour   = order_strokes_2opt(raw_strokes, cfg)

        naive_strokes = [s.points.tolist() for s in naive_tour]
        opt_strokes   = [s.points.tolist() for s in opt_tour]

        # Evaluate tour metrics
        m_naive = evaluate_tour_metrics(naive_tour, cfg)
        m_opt   = evaluate_tour_metrics(opt_tour, cfg)
        print(f"  Naive: pen-up={m_naive['pen_up_distance_cm']:.2f} cm, "
              f"revs={m_naive['joint_reversals_count']}")
        print(f"  2-Opt: pen-up={m_opt['pen_up_distance_cm']:.2f} cm, "
              f"revs={m_opt['joint_reversals_count']}")

        for preset in presets:
            run_seeds = [SEED_DETERMINISTIC] if preset == "none" else seeds
            for seed in run_seeds:
                for mode_name, ordering, cl_variant in MODES:
                    tour_strokes_list = opt_strokes if ordering == "optimized" else naive_strokes
                    # Convert to np arrays for metrics
                    tour_arm = [np.array(s, dtype=float) for s in tour_strokes_list]

                    overlay_path = None
                    if save_overlays:
                        overlay_path = os.path.join(
                            OUTPUT_DIR,
                            f"{img_name}_{mode_name}_{preset}_s{seed}_overlay.png"
                        )

                    print(f"  [{img_name}] {mode_name:25s} | {preset:8s} | seed={seed} ...", end=" ", flush=True)

                    # Skip if already completed
                    run_key = (img_name, mode_name, preset, seed)
                    if run_key in completed:
                        print("SKIP (already done)")
                        continue

                    t0 = time.time()
                    try:
                        res = _run_sim(
                            tour_arm, preset, seed, cl_variant, wc, cam,
                            output_overlay=overlay_path,
                        )
                        elapsed = time.time() - t0
                        print(f"RMSE={res['rmse_mm']:.3f} mm  "
                              f"Tol-IoU={res['tol_iou_pct']:.1f}%  "
                              f"t={elapsed:.1f}s")
                    except Exception as exc:
                        elapsed = time.time() - t0
                        print(f"ERROR: {exc}")
                        res = {
                            "rmse_mm": float("nan"), "chamfer_mm": float("nan"),
                            "tol_iou_pct": float("nan"), "plain_iou_pct": float("nan"),
                            "endpoint_error_mm": float("nan"),
                            "pen_up_dist_cm": float("nan"), "exec_time_s": float("nan"),
                        }

                    row = {
                        "image": img_name,
                        "mode": mode_name,
                        "ordering": ordering,
                        "cl_variant": cl_variant,
                        "preset": preset,
                        "seed": seed,
                        **{k: round(v, 4) if not math.isnan(v) else float("nan")
                           for k, v in res.items()},
                        "pen_up_naive_cm": round(m_naive["pen_up_distance_cm"], 3),
                        "pen_up_opt_cm":   round(m_opt["pen_up_distance_cm"], 3),
                        "reversals_naive": m_naive["joint_reversals_count"],
                        "reversals_opt":   m_opt["joint_reversals_count"],
                    }
                    records.append(row)
                    completed.add(run_key)

                    # ── Incremental save after every run ──
                    pd.DataFrame(records).to_csv(agg_path, index=False)

    df = pd.DataFrame(records)
    df.to_csv(agg_path, index=False)
    print(f"\nSaved aggregate CSV: {agg_path}  ({len(df)} rows)")

    # ── Summary: mean ± std over seeds ──────────────────────────────────────
    metric_cols = ["rmse_mm", "chamfer_mm", "tol_iou_pct", "endpoint_error_mm",
                   "pen_up_dist_cm", "exec_time_s"]
    summary_rows = []
    for (img, mode, preset), grp in df.groupby(["image", "mode", "preset"]):
        row: Dict[str, Any] = {"image": img, "mode": mode, "preset": preset,
                               "n_seeds": len(grp)}
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = round(vals.mean(), 4) if len(vals) else float("nan")
            row[f"{col}_std"]  = round(vals.std(ddof=0), 4) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    sum_path = os.path.join(OUTPUT_DIR, "phase7_summary.csv")
    df_sum.to_csv(sum_path, index=False)

    # ── Human-readable table ─────────────────────────────────────────────────
    _print_and_save_table(df_sum)
    return df


def _print_and_save_table(df: pd.DataFrame) -> None:
    lines = []
    lines.append("=" * 130)
    lines.append("PHASE 7 SUMMARY TABLE  (mean ± std over seeds)")
    lines.append("=" * 130)
    hdr = (
        f"{'Image':<22} {'Mode':<26} {'Preset':<9} "
        f"{'RMSE mm':>10} {'Chamfer mm':>12} {'Tol-IoU%':>10} "
        f"{'Endpt mm':>10} {'Pen-Up cm':>10} {'Time s':>8}"
    )
    lines.append(hdr)
    lines.append("-" * 130)

    prev_img = None
    for _, r in df.iterrows():
        if prev_img and r["image"] != prev_img:
            lines.append("")
        prev_img = r["image"]

        rmse   = f"{r['rmse_mm_mean']:.3f}±{r['rmse_mm_std']:.3f}"
        ch     = f"{r['chamfer_mm_mean']:.3f}±{r['chamfer_mm_std']:.3f}"
        iou    = f"{r['tol_iou_pct_mean']:.1f}±{r['tol_iou_pct_std']:.1f}"
        ep     = f"{r['endpoint_error_mm_mean']:.3f}±{r['endpoint_error_mm_std']:.3f}"
        pu     = f"{r['pen_up_dist_cm_mean']:.2f}±{r['pen_up_dist_cm_std']:.2f}"
        ts     = f"{r['exec_time_s_mean']:.1f}±{r['exec_time_s_std']:.1f}"

        lines.append(
            f"{r['image']:<22} {r['mode']:<26} {r['preset']:<9} "
            f"{rmse:>10} {ch:>12} {iou:>10} "
            f"{ep:>10} {pu:>10} {ts:>8}"
        )

    lines.append("=" * 130)
    table = "\n".join(lines)
    print(table)
    tbl_path = os.path.join(OUTPUT_DIR, "phase7_summary_table.txt")
    with open(tbl_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"\nSaved summary table: {tbl_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_experiments()
