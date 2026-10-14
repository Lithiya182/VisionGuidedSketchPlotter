"""
scripts/run_phase5_experiments.py

Phase 5 Comprehensive Experimental Suite:
1. Open-Loop vs Closed-Loop evaluation across presets:
   - none (ceiling reference)
   - mild (seeds 42, 43, 44)
   - moderate (seeds 42, 43, 44)
   - severe (seeds 42, 43, 44)
2. Disturbance Ablation Study:
   - offset_only
   - backlash_only
   - noise_only
3. Statistical Aggregation:
   - Mean +- Std for all metrics (RMSE, Chamfer, Tolerance IoU, Endpoint Error, Camera Chamfer)
   - Percentage of error recovered relative to nominal ceiling
4. Multi-Condition Overlay Visualizations:
   - Side-by-side Target vs Open-Loop vs Closed-Loop comparison plots
"""

import argparse
import csv
import json
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

from scripts.run_closed_loop import execute_closed_loop_run
from src.config import DEFAULT_CAMERA_CONFIG, DEFAULT_WORKSPACE_CONFIG
from src.sim.disturbances import DISTURBANCE_PRESETS


OUTPUT_DIR = "logs/phase5_eval"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "phase5_comparison_summary.csv")
ABLATION_CSV = os.path.join(OUTPUT_DIR, "phase5_ablation_summary.csv")


def run_full_evaluation(seeds: List[int] = [42, 43, 44]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full suite of Phase 5 closed-loop vs open-loop experiments."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Main Presets: none, mild, moderate, severe
    # -------------------------------------------------------------------------
    presets = ["none", "mild", "moderate", "severe"]
    variants = ["none", "calib_prepass", "per_stroke"]

    print("=" * 80)
    print("PHASE 5: STARTING CLOSED-LOOP EVALUATION SUITE")
    print("=" * 80)

    for preset in presets:
        run_seeds = [42] if preset == "none" else seeds
        for s in run_seeds:
            for variant in variants:
                print(f"\n>>> Running: preset='{preset}', seed={s}, variant='{variant}'")
                res = execute_closed_loop_run(
                    preset_name=preset,
                    seed=s,
                    correction_variant=variant,
                    output_dir=OUTPUT_DIR,
                )
                all_results.append(res)

    # -------------------------------------------------------------------------
    # 2. Disturbance Ablation Study: offset_only, backlash_only, noise_only
    # -------------------------------------------------------------------------
    ablation_presets = ["offset_only", "backlash_only", "noise_only"]
    ablation_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("PHASE 5: STARTING DISTURBANCE ABLATION STUDY")
    print("=" * 80)

    for abl_preset in ablation_presets:
        for variant in ["none", "calib_prepass"]:
            print(f"\n>>> Ablation Run: preset='{abl_preset}', variant='{variant}'")
            res = execute_closed_loop_run(
                preset_name=abl_preset,
                seed=42,
                correction_variant=variant,
                output_dir=OUTPUT_DIR,
            )
            ablation_results.append(res)

    # -------------------------------------------------------------------------
    # 3. Compile and Aggregate Main Presets Table
    # -------------------------------------------------------------------------
    df_main = pd.DataFrame(all_results)
    df_abl = pd.DataFrame(ablation_results)

    df_main.to_csv(SUMMARY_CSV, index=False)
    df_abl.to_csv(ABLATION_CSV, index=False)

    print(f"\nSaved raw results to {SUMMARY_CSV} and {ABLATION_CSV}")

    # Generate aggregate summary table
    summary_rows = []
    
    # Baseline 'none' values as ceiling
    none_row = df_main[df_main["preset"] == "none"].iloc[0]
    ceil_rmse = none_row["execution_rmse_mm"]
    ceil_chamfer = none_row["total_chamfer_mm"]
    ceil_iou = none_row["tolerance_iou_pct"]

    for preset in presets:
        for variant in variants:
            sub = df_main[(df_main["preset"] == preset) & (df_main["correction_variant"] == variant)]
            if len(sub) == 0:
                continue
            
            mean_rmse = sub["execution_rmse_mm"].mean()
            std_rmse = sub["execution_rmse_mm"].std(ddof=0)
            mean_chamfer = sub["total_chamfer_mm"].mean()
            std_chamfer = sub["total_chamfer_mm"].std(ddof=0)
            mean_iou = sub["tolerance_iou_pct"].mean()
            std_iou = sub["tolerance_iou_pct"].std(ddof=0)
            mean_ep = sub["total_endpoint_error_mm"].mean()
            std_ep = sub["total_endpoint_error_mm"].std(ddof=0)

            # Error recovered relative to open loop
            # Find open loop mean for this preset
            ol_sub = df_main[(df_main["preset"] == preset) & (df_main["correction_variant"] == "none")]
            ol_rmse = ol_sub["execution_rmse_mm"].mean() if len(ol_sub) > 0 else mean_rmse
            ol_chamfer = ol_sub["total_chamfer_mm"].mean() if len(ol_sub) > 0 else mean_chamfer

            if variant == "none" or preset == "none" or math.isclose(ol_rmse, ceil_rmse, abs_tol=1e-3):
                recov_rmse_pct = 0.0
                recov_chamfer_pct = 0.0
            else:
                recov_rmse_pct = max(0.0, (ol_rmse - mean_rmse) / (ol_rmse - ceil_rmse) * 100.0)
                recov_chamfer_pct = max(0.0, (ol_chamfer - mean_chamfer) / (ol_chamfer - ceil_chamfer) * 100.0)

            summary_rows.append({
                "Preset": preset.capitalize(),
                "Mode": "Open-Loop" if variant == "none" else ("CL-PrePass" if variant == "calib_prepass" else "CL-PerStroke"),
                "RMSE (mm)": f"{mean_rmse:.3f} +- {std_rmse:.3f}",
                "Chamfer (mm)": f"{mean_chamfer:.3f} +- {std_chamfer:.3f}",
                "Tol-IoU (%)": f"{mean_iou:.2f} +- {std_iou:.2f}",
                "Endpoint (mm)": f"{mean_ep:.3f} +- {std_ep:.3f}",
                "RMSE Recov (%)": f"{recov_rmse_pct:.1f}%",
                "Chamfer Recov (%)": f"{recov_chamfer_pct:.1f}%",
            })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print("PHASE 5: AGGREGATE PERFORMANCE COMPARISON TABLE")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    agg_csv = os.path.join(OUTPUT_DIR, "phase5_aggregate_table.csv")
    summary_df.to_csv(agg_csv, index=False)

    # -------------------------------------------------------------------------
    # 4. Generate 3-Way Comparison Overlay Figures
    # -------------------------------------------------------------------------
    _generate_comparison_figures(df_main, df_abl)

    return df_main, df_abl


def _generate_comparison_figures(df_main: pd.DataFrame, df_abl: pd.DataFrame) -> None:
    """Generate high-resolution 3-way comparison figures for presets and ablation."""
    presets_to_plot = ["moderate", "severe"]

    for p in presets_to_plot:
        # Load telemetry for seed 42
        ol_csv = os.path.join(OUTPUT_DIR, f"cl_none_{p}_seed42_telemetry.csv")
        cl_csv = os.path.join(OUTPUT_DIR, f"cl_calib_prepass_{p}_seed42_telemetry.csv")

        if not os.path.exists(ol_csv) or not os.path.exists(cl_csv):
            continue

        df_ol = pd.read_csv(ol_csv)
        df_cl = pd.read_csv(cl_csv)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#ffffff")

        # 1. Target Nominal Path
        ax0 = axes[0]
        ax0.set_title(f"Target Reference Sketch\n(Nominal 6x6 cm Window)", fontsize=11, weight="bold")
        ax0.set_xlabel("X (cm)")
        ax0.set_ylabel("Y (cm)")
        ax0.set_aspect("equal")
        ax0.grid(True, linestyle="--", alpha=0.5)

        tgt_down = df_ol[df_ol["pen_down"] == 1]
        for s_id in sorted(tgt_down["stroke_id"].unique()):
            sub = tgt_down[tgt_down["stroke_id"] == s_id]
            ax0.plot(sub["target_x_cm"], sub["target_y_cm"], color="#28a745", linewidth=2.0)
        ax0.set_xlim(3.8, 10.2)
        ax0.set_ylim(6.2, 12.8)

        # 2. Open-Loop Disturbed Trace
        ax1 = axes[1]
        ol_rmse = df_main[(df_main["preset"] == p) & (df_main["correction_variant"] == "none") & (df_main["seed"] == 42)]["execution_rmse_mm"].values[0]
        ax1.set_title(f"Open-Loop Baseline ({p.upper()})\nRMSE: {ol_rmse:.2f} mm (No Vision)", fontsize=11, weight="bold")
        ax1.set_xlabel("X (cm)")
        ax1.set_ylabel("Y (cm)")
        ax1.set_aspect("equal")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Plot target faint
        for s_id in sorted(tgt_down["stroke_id"].unique()):
            sub = tgt_down[tgt_down["stroke_id"] == s_id]
            ax1.plot(sub["target_x_cm"], sub["target_y_cm"], color="#28a745", linewidth=1.0, alpha=0.35)

        ol_down = df_ol[df_ol["pen_down"] == 1]
        for s_id in sorted(ol_down["stroke_id"].unique()):
            sub = ol_down[ol_down["stroke_id"] == s_id]
            ax1.plot(sub["actual_x_cm"], sub["actual_y_cm"], color="#dc3545", linewidth=1.8)
        ax1.set_xlim(3.8, 10.2)
        ax1.set_ylim(6.2, 12.8)

        # 3. Closed-Loop Trace
        ax2 = axes[2]
        cl_rmse = df_main[(df_main["preset"] == p) & (df_main["correction_variant"] == "calib_prepass") & (df_main["seed"] == 42)]["execution_rmse_mm"].values[0]
        ax2.set_title(f"Closed-Loop Corrected ({p.upper()})\nRMSE: {cl_rmse:.2f} mm (Visual Feedback)", fontsize=11, weight="bold")
        ax2.set_xlabel("X (cm)")
        ax2.set_ylabel("Y (cm)")
        ax2.set_aspect("equal")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Plot target faint
        for s_id in sorted(tgt_down["stroke_id"].unique()):
            sub = tgt_down[tgt_down["stroke_id"] == s_id]
            ax2.plot(sub["target_x_cm"], sub["target_y_cm"], color="#28a745", linewidth=1.0, alpha=0.35)

        cl_down = df_cl[df_cl["pen_down"] == 1]
        for s_id in sorted(cl_down["stroke_id"].unique()):
            sub = cl_down[cl_down["stroke_id"] == s_id]
            ax2.plot(sub["actual_x_cm"], sub["actual_y_cm"], color="#0d6efd", linewidth=1.8)
        ax2.set_xlim(3.8, 10.2)
        ax2.set_ylim(6.2, 12.8)

        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color="#28a745", lw=2, label="Target Contour"),
            Line2D([0], [0], color="#dc3545", lw=2, label="Open-Loop Error"),
            Line2D([0], [0], color="#0d6efd", lw=2, label="Closed-Loop Corrected"),
        ]
        fig.legend(handles=custom_lines, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

        fig_path = os.path.join(OUTPUT_DIR, f"phase5_3way_comparison_{p}.png")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.10)
        plt.savefig(fig_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 3-way comparison figure to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    run_full_evaluation(seeds=args.seeds)
