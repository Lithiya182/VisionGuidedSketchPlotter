"""
scripts/run_demo.py

One-Command Demo: Vision-Guided Sketch Plotter
===============================================
Usage
-----
  python scripts/run_demo.py \\
      --image data/sample_images/01_robot_arm_sketch.png \\
      --mode skeleton \\
      --disturbance moderate \\
      --order optimized \\
      --control cl_prepass \\
      --seed 42

Outputs (in logs/demo/):
  <run_id>_open_loop_overlay.png   -- target vs open-loop drawing
  <run_id>_closed_loop_overlay.png -- target vs closed-loop drawing
  <run_id>_comparison.png          -- side-by-side 3-panel figure
  <run_id>_top_down.png            -- raw top-down camera capture
  <run_id>_metrics.csv             -- numeric metrics for both modes
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.setup_scene import setup_scene
_QUALITY = "std"
_INK = float(os.environ.get("SKETCH_INK", "3.5"))
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
    RunningCorrector,
)
from src.vision.image_processor import ImageProcessor, StrokeIR
from src.config import VisionConfig

OUTPUT_DIR = "logs/demo"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _valid_pt(x: float, y: float) -> bool:
    try:
        from src.coordinate_transform import validate_joint_configuration
        validate_workspace_point(x, y)
        validate_joint_configuration(x, y)
        return True
    except ValueError:
        return False


def _load_or_extract_strokes(
    image_path: str,
    mode: str,
    wc: WorkspaceConfig,
) -> List[np.ndarray]:
    """Extract strokes from a raw image (or from a pre-saved JSON if it exists)."""
    stem = Path(image_path).stem
    json_path = "data/extracted_strokes/%s_%s%s.json" % (stem, mode, "_hq" if _QUALITY == "hq" else "")

    if (not os.path.exists(json_path)) or os.path.getmtime(image_path) > os.path.getmtime(json_path):
        print(f"  Extracting strokes ({mode}) from {image_path} ...")
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        proc = ImageProcessor(VisionConfig(simplification_epsilon_px=0.75 if _QUALITY == "hq" else 1.5))
        ir = proc.process_image(image_path, mode=mode)
        os.makedirs("data/extracted_strokes", exist_ok=True)
        ir.to_json(json_path)
        n_strokes = len(ir.strokes)
        print(f"  Extracted {n_strokes} strokes, saved to {json_path}")
    else:
        ir = StrokeIR.from_json(json_path)
        print(f"  Loaded {len(ir.strokes)} strokes from {json_path}")

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
        valid = [p for p in pts if _valid_pt(float(p[0]), float(p[1]))]
        if len(valid) >= 2:
            strokes.append(np.array(valid, dtype=float))

    rejected = len(ir.strokes) - len(strokes)
    if rejected:
        print(f"  WARN: {rejected} strokes rejected (out of workspace / unreachable).")
    return strokes


def _run_one(
    tour_arm: List[np.ndarray],
    preset_name: str,
    seed: int,
    cl_variant: str,   # 'none' or 'cl_prepass'
    wc: WorkspaceConfig,
    cam: CameraConfig,
    spacing_cm: float = 0.25,
) -> Dict[str, Any]:
    """Single simulation run; returns metrics + raw camera image."""
    setup_scene()
    dist_cfg = get_disturbance_preset(preset_name, seed=seed)
    dist_mgr = PlantDisturbanceManager(dist_cfg)

    is_cl = cl_variant != "none"
    corr_cfg = CorrectorConfig(enabled=is_cl, calib_prepass=is_cl)
    corrector = RunningCorrector(corr_cfg)
    calib_pp  = CalibrationPrePass(corr_cfg, cam) if is_cl else None

    ctrl = CoppeliaController(verbose=False)
    ctrl.stop_simulation()
    ctrl.clear_drawing_object()
    ctrl.set_stepping(True)

    telemetry = []

    with dist_mgr.active_context(ctrl.sim):
        ctrl.start_simulation()
        ctrl.clear_drawing_object()
        ctrl.create_drawing_object(line_width=_INK, color=(0.05, 0.15, 0.70))
        ctrl.step()

        baseline_bgr = cv2.cvtColor(ctrl.get_vision_sensor_image(), cv2.COLOR_RGB2BGR)

        # Calibration pre-pass
        if is_cl and calib_pp:
            calib_plan = plan_strokes(
                [s.tolist() for s in calib_pp.get_pattern_strokes()],
                branch="positive", spacing_cm=spacing_cm,
            )
            prev = None
            for pt in calib_plan:
                c1, c2, _ = dist_mgr.transform_plant_command(pt.theta1, pt.theta2)
                ctrl.set_joint_targets(c1, c2)
                ctrl.wait_for_settle(c1, c2, timeout_s=1.0, tolerance_deg=0.25, step_simulation=True)
                ax, ay, az = ctrl.get_tip_position()
                if pt.pen_down and prev:
                    ctrl.add_ink_segment((prev[0], prev[1], 1.55), (ax, ay, 1.55), in_meters=False)
                prev = (ax, ay, az)

            pk1, pk2 = inverse_kinematics(4.0, 6.5, branch="positive")
            pc1, pc2, _ = dist_mgr.transform_plant_command(pk1, pk2)
            ctrl.set_joint_targets(pc1, pc2)
            ctrl.wait_for_settle(pc1, pc2, timeout_s=1.0, tolerance_deg=0.25, step_simulation=True)

            calib_bgr = cv2.cvtColor(ctrl.get_vision_sensor_image(), cv2.COLOR_RGB2BGR)
            off = calib_pp.estimate_offset_from_image(calib_bgr, baseline_bgr)
            if off is not None:
                corrector.update(off)
                print(f"    [CalibPrePass] offset=({off[0]*10:.2f}, {off[1]*10:.2f}) mm")
            else:
                print("    [CalibPrePass] offset undetected (using zero).")

            ctrl.clear_drawing_object()
            ctrl.create_drawing_object(line_width=_INK, color=(0.05, 0.15, 0.70))
            ctrl.step()

        # Main drawing
        t0 = time.time()
        prev = None
        for si, sarr in enumerate(tour_arm):
            nom = plan_strokes([sarr.tolist()], branch="positive", spacing_cm=spacing_cm)
            if is_cl and corrector.n_observations > 0:
                cp = sarr - corrector.current_offset
                vp = np.array([p for p in cp if _valid_pt(float(p[0]), float(p[1]))], dtype=float)
                if len(vp) >= 2:
                    try:
                        ep = plan_strokes([vp.tolist()], branch="positive",
                                          spacing_cm=spacing_cm, skip_invalid_strokes=True)
                        if len(ep) != len(nom):
                            ep = nom
                    except ValueError:
                        ep = nom
                else:
                    ep = nom
            else:
                ep = nom

            for np_pt, pt in zip(nom, ep):
                p1, p2, _ = dist_mgr.transform_plant_command(pt.theta1, pt.theta2)
                ctrl.set_joint_targets(p1, p2)
                ctrl.wait_for_settle(p1, p2, timeout_s=1.5, tolerance_deg=0.25, step_simulation=True)
                ax, ay, az = ctrl.get_tip_position()
                if pt.pen_down and prev:
                    ctrl.add_ink_segment((prev[0], prev[1], 1.55), (ax, ay, 1.55), in_meters=False)
                prev = (ax, ay, az)
                telemetry.append({
                    "tx": np_pt.x, "ty": np_pt.y,
                    "ax": ax, "ay": ay,
                    "pd": 1 if pt.pen_down else 0,
                    "sid": si,
                })

        exec_time = time.time() - t0
        # Park the arm outside the camera view so it cannot hide ink
        _pk1, _pk2 = inverse_kinematics(14.0, 0.0, branch="positive")
        _pc1, _pc2, _ = dist_mgr.transform_plant_command(_pk1, _pk2)
        ctrl.set_joint_targets(_pc1, _pc2)
        ctrl.wait_for_settle(_pc1, _pc2, timeout_s=3.0, tolerance_deg=0.25, step_simulation=True)
        cam_rgb = ctrl.get_vision_sensor_image()
        ctrl.clear_drawing_object()
        ctrl.stop_simulation()
        ctrl.set_stepping(False)

    tgt  = np.array([[r["tx"], r["ty"]] for r in telemetry])
    meas = np.array([[r["ax"], r["ay"]] for r in telemetry])
    pens = np.array([r["pd"] for r in telemetry])
    sids = np.array([r["sid"] for r in telemetry])

    decomp = decompose_tracking_errors(tour_arm, tgt, meas, pens, sids)
    travel = pen_travel_metrics(meas, pens)

    cam_bgr   = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
    tgt_mask  = rasterize_strokes_to_camera_mask(tour_arm, cam, cam.line_thickness_px)
    obs_mask  = extract_ink_mask_from_camera_image(cam_bgr, is_bgr=True)
    iou_res   = raster_iou(tgt_mask, obs_mask, tolerance_px=2)
    px_mm     = cam.view_size_cm * 10.0 / cam.resolution_x
    cam_ch    = raster_chamfer_and_contour_error(tgt_mask, obs_mask, pixel_size_mm=px_mm)

    return {
        "rmse_mm":           decomp["execution_error"]["rmse_mm"],
        "chamfer_mm":        cam_ch["symmetric_chamfer_mm"],
        "tol_iou_pct":       iou_res["tolerance_iou_pct"],
        "plain_iou_pct":     iou_res["plain_iou_pct"],
        "endpoint_error_mm": decomp["total_error"]["mean_endpoint_error_mm"],
        "pen_up_dist_cm":    travel["pen_up_distance_cm"],
        "exec_time_s":       exec_time,
        "cam_bgr":           cam_bgr,
        "tgt_mask":          tgt_mask,
        "obs_mask":          obs_mask,
        "meas":              meas,
        "pens":              pens,
    }


# ── Figure generation ─────────────────────────────────────────────────────────
def _save_comparison_figure(
    path: str,
    image_name: str,
    tour_arm: List[np.ndarray],
    res_ol: Dict,
    res_cl: Optional[Dict],
    preset: str, seed: int,
    wc: WorkspaceConfig,
    cam: CameraConfig,
) -> None:
    n_panels = 3 if res_cl else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6), facecolor="#fff")
    fig.suptitle(
        f"{image_name}  |  Disturbance: {preset.upper()}  |  seed={seed}",
        fontsize=13, weight="bold", y=1.01,
    )

    def _overlay_ax(ax, meas, pens, tgt_mask, obs_mask, iou_res, title_extra):
        overlay = np.full((cam.resolution_y, cam.resolution_x, 3), 245, dtype=np.uint8)
        overlay[obs_mask > 0] = [180, 50, 20]
        overlay[tgt_mask > 0] = [30, 160, 30]
        overlay[np.logical_and(tgt_mask > 0, obs_mask > 0)] = [0, 200, 255]
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_xlim(80, 400); ax.set_ylim(400, 80)
        ax.set_title(
            f"{title_extra}\nTol-IoU={iou_res['tolerance_iou_pct']:.1f}%  "
            f"Plain-IoU={iou_res['plain_iou_pct']:.1f}%",
            fontsize=10, weight="bold",
        )
        ax.axis("off")

    # Panel 0: Target mask
    tgt_only = np.full((cam.resolution_y, cam.resolution_x, 3), 245, dtype=np.uint8)
    tgt_only[res_ol["tgt_mask"] > 0] = [30, 160, 30]
    axes[0].imshow(cv2.cvtColor(tgt_only, cv2.COLOR_BGR2RGB))
    axes[0].set_xlim(80, 400); axes[0].set_ylim(400, 80)
    axes[0].set_title("Target Strokes", fontsize=10, weight="bold")
    axes[0].axis("off")

    # Panel 1: Open-loop
    _overlay_ax(
        axes[1],
        res_ol["meas"], res_ol["pens"],
        res_ol["tgt_mask"], res_ol["obs_mask"],
        {"tolerance_iou_pct": res_ol["tol_iou_pct"],
         "plain_iou_pct":     res_ol["plain_iou_pct"]},
        f"Open-Loop  RMSE={res_ol['rmse_mm']:.3f} mm",
    )

    # Panel 2: Closed-loop (optional)
    if res_cl and n_panels == 3:
        _overlay_ax(
            axes[2],
            res_cl["meas"], res_cl["pens"],
            res_cl["tgt_mask"], res_cl["obs_mask"],
            {"tolerance_iou_pct": res_cl["tol_iou_pct"],
             "plain_iou_pct":     res_cl["plain_iou_pct"]},
            f"Closed-Loop  RMSE={res_cl['rmse_mm']:.3f} mm",
        )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#1eb41e", label="Target"),
        Patch(facecolor="#b43214", label="Drawn"),
        Patch(facecolor="#00c8ff", label="Overlap"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison figure: {path}")


# ── CLI entry-point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command demo for the Vision-Guided Sketch Plotter."
    )
    parser.add_argument(
        "--image",
        default="data/sample_images/01_robot_arm_sketch.png",
        help="Path to input image (PNG/JPG).",
    )
    parser.add_argument(
        "--mode",
        default="skeleton",
        choices=["skeleton", "canny", "boundary"],
        help="Stroke-extraction mode.",
    )
    parser.add_argument(
        "--disturbance",
        default="moderate",
        choices=["none", "mild", "moderate", "severe"],
        help="Disturbance preset to inject into the simulated plant.",
    )
    parser.add_argument(
        "--order",
        default="optimized",
        choices=["naive", "optimized"],
        help="Stroke ordering: naive baseline or kinematic 2-opt.",
    )
    parser.add_argument(
        "--control",
        default="cl_prepass",
        choices=["open_loop", "cl_prepass"],
        help="Control mode: open_loop or closed-loop with calibration pre-pass.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for disturbance reproducibility.",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--quality", default="std", choices=["std", "hq"],
        help="std = benchmark settings; hq = 8.5 cm window, finer detail.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_name = Path(args.image).stem + ("-hq" if args.quality == "hq" else "")
    run_id = f"{image_name}_{args.mode}_{args.disturbance}_{args.order}_{args.control}_s{args.seed}"
    print(f"\n{'='*80}")
    print(f"DEMO RUN: {run_id}")
    print(f"  image={args.image}  mode={args.mode}")
    print(f"  disturbance={args.disturbance}  order={args.order}  control={args.control}  seed={args.seed}")
    print(f"{'='*80}\n")

    wc  = DEFAULT_WORKSPACE_CONFIG
    sys.modules[__name__]._QUALITY = args.quality
    if args.quality == "hq":
        wc = __import__("dataclasses").replace(
            wc, draw_origin_x=2.75, draw_origin_y=13.75,
            draw_width_cm=8.5, draw_height_cm=8.5)
        _run_one.__defaults__ = (0.1,)
    cam = DEFAULT_CAMERA_CONFIG
    cfg = DEFAULT_STROKE_ORDER_CONFIG

    # 1. Stroke extraction
    raw_strokes = _load_or_extract_strokes(args.image, args.mode, wc)
    if not raw_strokes:
        print("ERROR: No valid strokes after workspace filtering. Exiting.")
        sys.exit(1)
    print(f"  Valid strokes: {len(raw_strokes)}")

    # 2. Stroke ordering
    if args.order == "optimized":
        tour = order_strokes_2opt(raw_strokes, cfg)
        print(f"  Ordering: Kinematic 2-Opt")
    else:
        tour = order_strokes_naive(raw_strokes)
        print(f"  Ordering: Naive Baseline")

    tour_arm = [s.points.tolist() if hasattr(s, "points") else s for s in tour]
    tour_arm_np = [np.array(s, dtype=float) for s in tour_arm]

    m = evaluate_tour_metrics(tour, cfg)
    print(f"  Pen-up={m['pen_up_distance_cm']:.2f} cm, "
          f"efficiency={m['pen_travel_efficiency_pct']:.1f}%, "
          f"reversals={m['joint_reversals_count']}")

    # 3. Open-loop run (always run for comparison baseline)
    print(f"\nRunning OPEN-LOOP ({args.disturbance}, seed={args.seed}) ...")
    res_ol = _run_one(tour_arm_np, args.disturbance, args.seed, "none", wc, cam)
    res_ol["tgt_mask"] = res_ol["tgt_mask"]  # already in dict

    print(f"  OL RMSE={res_ol['rmse_mm']:.3f} mm  "
          f"Tol-IoU={res_ol['tol_iou_pct']:.1f}%  "
          f"Chamfer={res_ol['chamfer_mm']:.3f} mm")

    # Save top-down camera capture (open-loop)
    td_path = os.path.join(args.output_dir, f"{run_id}_top_down_ol.png")
    cv2.imwrite(td_path, res_ol["cam_bgr"])
    print(f"  Top-down capture saved: {td_path}")

    # 4. Closed-loop run (if requested)
    res_cl = None
    if args.control == "cl_prepass":
        print(f"\nRunning CLOSED-LOOP/CalibPrePass ({args.disturbance}, seed={args.seed}) ...")
        res_cl = _run_one(tour_arm_np, args.disturbance, args.seed, "cl_prepass", wc, cam)
        print(f"  CL RMSE={res_cl['rmse_mm']:.3f} mm  "
              f"Tol-IoU={res_cl['tol_iou_pct']:.1f}%  "
              f"Chamfer={res_cl['chamfer_mm']:.3f} mm")

        td_cl = os.path.join(args.output_dir, f"{run_id}_top_down_cl.png")
        cv2.imwrite(td_cl, res_cl["cam_bgr"])
        print(f"  Top-down capture (CL) saved: {td_cl}")

    # 5. Side-by-side figure
    cmp_path = os.path.join(args.output_dir, f"{run_id}_comparison.png")
    _save_comparison_figure(
        cmp_path, image_name, tour_arm_np, res_ol, res_cl,
        args.disturbance, args.seed, wc, cam,
    )

    # 6. Metrics CSV
    csv_path = os.path.join(args.output_dir, f"{run_id}_metrics.csv")
    metric_keys = ["rmse_mm", "chamfer_mm", "tol_iou_pct", "plain_iou_pct",
                   "endpoint_error_mm", "pen_up_dist_cm", "exec_time_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode"] + metric_keys)
        w.writeheader()
        w.writerow({"mode": "open_loop", **{k: round(res_ol[k], 4) for k in metric_keys}})
        if res_cl:
            w.writerow({"mode": "closed_loop_prepass", **{k: round(res_cl[k], 4) for k in metric_keys}})
    print(f"  Metrics CSV saved: {csv_path}")

    # 7. Print final summary table
    print(f"\n{'-'*70}")
    print(f"{'Metric':<28} {'Open-Loop':>14} {'Closed-Loop':>14}")
    print(f"{'-'*70}")
    for k in metric_keys:
        cl_val = f"{res_cl[k]:.4f}" if res_cl else "  N/A"
        print(f"  {k:<26} {res_ol[k]:>14.4f} {cl_val:>14}")
    if res_cl:
        rmse_recov = (res_ol["rmse_mm"] - res_cl["rmse_mm"]) / max(res_ol["rmse_mm"], 1e-6) * 100
        print(f"\n  RMSE reduction (raw)     : {rmse_recov:.1f}%")
    print(f"{'-'*70}")
    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
