"""
scripts/run_phase4_experiments.py

Phase 4 Full Experimental Matrix Runner:
Executes the open-loop baseline across 4 disturbance levels:
- NONE     : Nominal baseline (1 run)
- MILD     : Mild disturbance (seeds 42, 43, 44)
- MODERATE : Moderate disturbance (seeds 42, 43, 44)
- SEVERE   : Severe disturbance (seeds 42, 43, 44)

Computes mean and std of:
- Trajectory RMSE (mm)
- Total Contour Chamfer Distance (mm)
- Headline Tolerance-Matched IoU (%)
- Secondary Plain IoU (%)
- Stroke Endpoint Error (mm)
- Camera Contour Error (mm)

Exports:
- logs/disturbances/phase4_disturbance_summary.csv
- logs/disturbances/phase4_presets_comparison.png
- Individual run telemetry and overlays
- Synchronizes artifacts to brain artifact directory.
"""

import csv
import os
import shutil
import sys
from typing import Any, Dict, List

sys.path.insert(0, ".")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_disturbed_baseline import execute_disturbed_run


def run_phase4_experiments(
    output_dir: str = "logs/disturbances",
    artifact_dir: str = "C:/Users/Asus/.gemini/antigravity/brain/ac1ebb6f-18f3-406c-aa61-f89d373e5abd",
    seeds: List[int] = [42, 43, 44],
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    print("=" * 80)
    print("PHASE 4: EXPERIMENTAL DISTURBANCE MATRIX EXECUTION")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    # 1. Preset 'none' (Nominal baseline, 1 run with seed 42)
    print("\n>>> Running Preset: NONE (Nominal Baseline)")
    res_none = execute_disturbed_run(preset_name="none", seed=42, output_dir=output_dir)
    all_results.append(res_none)

    # 2. Presets with noise: mild, moderate, severe across 3 seeds
    presets_to_test = ["mild", "moderate", "severe"]
    for preset in presets_to_test:
        for seed in seeds:
            print(f"\n>>> Running Preset: {preset.upper()} (Seed: {seed})")
            res = execute_disturbed_run(preset_name=preset, seed=seed, output_dir=output_dir)
            all_results.append(res)

    # 3. Aggregate results and compute mean & std
    df_raw = pd.DataFrame(all_results)
    raw_csv_path = os.path.join(output_dir, "phase4_raw_runs.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"\nAll individual runs saved to: {raw_csv_path}")

    # Aggregation
    summary_rows = []
    presets_order = ["none", "mild", "moderate", "severe"]

    for preset in presets_order:
        sub_df = df_raw[df_raw["preset"] == preset]
        n_runs = len(sub_df)

        rmse_mean = float(sub_df["execution_rmse_mm"].mean())
        rmse_std = float(sub_df["execution_rmse_mm"].std(ddof=1)) if n_runs > 1 else 0.0

        chamfer_mean = float(sub_df["total_chamfer_mm"].mean())
        chamfer_std = float(sub_df["total_chamfer_mm"].std(ddof=1)) if n_runs > 1 else 0.0

        tol_iou_mean = float(sub_df["tolerance_iou_pct"].mean())
        tol_iou_std = float(sub_df["tolerance_iou_pct"].std(ddof=1)) if n_runs > 1 else 0.0

        plain_iou_mean = float(sub_df["plain_iou_pct"].mean())
        plain_iou_std = float(sub_df["plain_iou_pct"].std(ddof=1)) if n_runs > 1 else 0.0

        end_err_mean = float(sub_df["total_endpoint_error_mm"].mean())
        end_err_std = float(sub_df["total_endpoint_error_mm"].std(ddof=1)) if n_runs > 1 else 0.0

        cam_chamfer_mean = float(sub_df["camera_chamfer_mm"].mean())
        cam_chamfer_std = float(sub_df["camera_chamfer_mm"].std(ddof=1)) if n_runs > 1 else 0.0

        summary_rows.append({
            "Preset": preset.capitalize(),
            "Runs": n_runs,
            "RMSE_Mean_mm": round(rmse_mean, 4),
            "RMSE_Std_mm": round(rmse_std, 4),
            "Chamfer_Mean_mm": round(chamfer_mean, 4),
            "Chamfer_Std_mm": round(chamfer_std, 4),
            "Tolerance_IoU_Mean_pct": round(tol_iou_mean, 2),
            "Tolerance_IoU_Std_pct": round(tol_iou_std, 2),
            "Plain_IoU_Mean_pct": round(plain_iou_mean, 2),
            "Plain_IoU_Std_pct": round(plain_iou_std, 2),
            "Endpoint_Err_Mean_mm": round(end_err_mean, 4),
            "Endpoint_Err_Std_mm": round(end_err_std, 4),
            "Cam_Chamfer_Mean_mm": round(cam_chamfer_mean, 4),
            "Cam_Chamfer_Std_mm": round(cam_chamfer_std, 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "phase4_disturbance_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary table exported to: {summary_csv_path}")

    # 4. Generate 4-Panel Preset Comparison Dashboard
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), facecolor="#ffffff")
    plt.subplots_adjust(wspace=0.25)

    representative_runs = {
        "None": ("none", 42),
        "Mild": ("mild", 42),
        "Moderate": ("moderate", 42),
        "Severe": ("severe", 42),
    }

    for idx, (label, (p_name, s_val)) in enumerate(representative_runs.items()):
        ax = axes[idx]
        run_data = df_raw[(df_raw["preset"] == p_name) & (df_raw["seed"] == s_val)].iloc[0]
        telem_df = pd.read_csv(run_data["telemetry_csv"])

        targets = telem_df[["target_x_cm", "target_y_cm"]].to_numpy()
        actuals = telem_df[["actual_x_cm", "actual_y_cm"]].to_numpy()
        pen_states = telem_df["pen_down"].to_numpy()

        ax.set_title(f"{label} Disturbance\n"
                     f"RMSE: {run_data['execution_rmse_mm']:.2f} mm | Tol-IoU: {run_data['tolerance_iou_pct']:.1f}%",
                     fontsize=11, weight="bold")
        ax.set_xlabel("Robot X (cm)", fontsize=10)
        ax.set_ylabel("Robot Y (cm)", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.5)

        # 6x6cm Window Box
        rect = plt.Rectangle((4.0, 6.5), 6.0, 6.0, edgecolor="black", facecolor="#f8f9fa",
                             linestyle="--", linewidth=1.0)
        ax.add_patch(rect)

        # Draw target in green
        ax.scatter(targets[pen_states == 1, 0], targets[pen_states == 1, 1],
                   color="#28a745", s=6, alpha=0.4, label="Target Strokes")

        # Draw executed tip in blue
        drawn_labeled = False
        for i in range(1, len(actuals)):
            if pen_states[i] == 1:
                lbl = "Executed Trace" if not drawn_labeled else ""
                ax.plot([actuals[i-1, 0], actuals[i, 0]], [actuals[i-1, 1], actuals[i, 1]],
                        color="#0d6efd", linewidth=1.6, label=lbl)
                drawn_labeled = True
            else:
                ax.plot([actuals[i-1, 0], actuals[i, 0]], [actuals[i-1, 1], actuals[i, 1]],
                        color="#ced4da", linestyle=":", linewidth=0.8)

        ax.set_xlim(3.5, 10.5)
        ax.set_ylim(6.0, 13.0)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    comp_plot_path = os.path.join(output_dir, "phase4_presets_comparison.png")
    plt.savefig(comp_plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison dashboard saved to: {comp_plot_path}")

    # 5. Copy artifacts to brain artifact directory
    if os.path.exists(artifact_dir):
        dest_dir = os.path.join(artifact_dir, "disturbances")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(summary_csv_path, os.path.join(dest_dir, os.path.basename(summary_csv_path)))
        shutil.copy2(comp_plot_path, os.path.join(dest_dir, os.path.basename(comp_plot_path)))
        for p_name, s_val in representative_runs.values():
            overlay_src = os.path.join(output_dir, f"{p_name}_seed{s_val}_overlay.png")
            if os.path.exists(overlay_src):
                shutil.copy2(overlay_src, os.path.join(dest_dir, os.path.basename(overlay_src)))
        print(f"Artifacts successfully synchronized to: {dest_dir}")

    # 6. Display formatted table
    print("\n" + "=" * 95)
    print("PHASE 4 DISTURBANCE EXPERIMENT RESULTS SUMMARY (MEAN +/- STD)")
    print("=" * 95)
    header = f"{'Preset':<10} | {'RMSE (mm)':<17} | {'Chamfer (mm)':<17} | {'Tol-IoU (%)':<16} | {'Plain-IoU (%)':<16} | {'Endpoint Err (mm)':<17}"
    print(header)
    print("-" * 95)
    for r in summary_rows:
        row_str = (
            f"{r['Preset']:<10} | "
            f"{r['RMSE_Mean_mm']:>6.3f} +/- {r['RMSE_Std_mm']:<5.3f} | "
            f"{r['Chamfer_Mean_mm']:>6.3f} +/- {r['Chamfer_Std_mm']:<5.3f} | "
            f"{r['Tolerance_IoU_Mean_pct']:>5.2f} +/- {r['Tolerance_IoU_Std_pct']:<4.2f}% | "
            f"{r['Plain_IoU_Mean_pct']:>5.2f} +/- {r['Plain_IoU_Std_pct']:<4.2f}% | "
            f"{r['Endpoint_Err_Mean_mm']:>6.3f} +/- {r['Endpoint_Err_Std_mm']:<5.3f}"
        )
        print(row_str)
    print("=" * 95)

    return df_summary


if __name__ == "__main__":
    run_phase4_experiments()
