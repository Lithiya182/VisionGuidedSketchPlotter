"""
run_vision_pipeline.py

Executes the Phase 1 computer vision extraction pipeline on the 3 sample images,
evaluating all 3 extraction modes (boundary, canny, skeleton).
Generates versioned StrokeIR JSON files and multi-panel visual previews.
"""

import math
import os
import sys
import time

sys.path.insert(0, ".")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import VisionConfig
from src.vision.image_processor import ImageProcessor, StrokeIR


def render_preview_figure(
    img_path: str,
    processor: ImageProcessor,
    mode: str,
    output_png_path: str,
) -> StrokeIR:
    """
    Run pipeline on image in the specified mode and save a comprehensive 4-panel visual preview.
    """
    img = cv2.imread(img_path)
    canvas_img, scale, pad_x, pad_y = processor.resize_and_pad(img)
    gray = processor.preprocess_grayscale(canvas_img)

    # Intermediate binary/feature map
    if mode == "boundary":
        _, feat_map = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mode_title = "Otsu Binary Silhouette"
    elif mode == "canny":
        feat_map = cv2.Canny(gray, processor.config.canny_threshold_low, processor.config.canny_threshold_high)
        mode_title = "Canny Edge Map"
    elif mode == "skeleton":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        feat_map = np.zeros(binary.shape, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_img = binary.copy()
        while True:
            eroded = cv2.erode(temp_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_img, temp)
            feat_map = cv2.bitwise_or(feat_map, temp)
            temp_img = eroded.copy()
            if cv2.countNonZero(temp_img) == 0:
                break
        mode_title = "Morphological Skeleton (Centerline)"

    # Process image to get StrokeIR
    t0 = time.perf_counter()
    ir = processor.process_image(img_path, mode=mode)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Build 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#1e1e1e")
    fig.suptitle(
        f"Mode: {mode.upper()} | {os.path.basename(img_path)} | "
        f"Strokes: {len(ir.strokes)} | Pts: {ir.metadata.simplification_stats.raw_points_count} -> {ir.metadata.simplification_stats.simplified_points_count} "
        f"(-{ir.metadata.simplification_stats.compression_ratio*100:.1f}%) | "
        f"Dev: max {ir.metadata.simplification_stats.max_deviation_px:.2f}px, mean {ir.metadata.simplification_stats.mean_deviation_px:.2f}px | {dt_ms:.1f}ms",
        fontsize=13,
        color="white",
        weight="bold",
    )

    # Panel 1: Original Canvas (480x480)
    axes[0].imshow(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Standardized 480x480 Canvas", color="white", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Intermediate Feature Map
    axes[1].imshow(feat_map, cmap="gray")
    axes[1].set_title(f"2. {mode_title}", color="white", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Simplified Vector Strokes (color-coded)
    vector_canvas = np.full((480, 480, 3), 250, dtype=np.uint8)
    axes[2].imshow(vector_canvas)
    axes[2].set_xlim(0, 480)
    axes[2].set_ylim(480, 0)  # Top-left origin, Y down
    axes[2].set_title(f"3. Vector Strokes ({len(ir.strokes)} strokes)", color="white", fontsize=11)
    axes[2].set_facecolor("#fafafa")

    cmap = plt.get_cmap("tab20")
    for idx, stroke in enumerate(ir.strokes):
        if len(stroke) < 2:
            continue
        pts = np.array(stroke)
        color = cmap(idx % 20)
        axes[2].plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.0)
        axes[2].scatter(pts[:, 0], pts[:, 1], color=color, s=12, zorder=3)
        # Highlight stroke start
        axes[2].scatter(pts[0, 0], pts[0, 1], color="green", s=30, marker="o", zorder=4)

    # Panel 4: Vector Stroke Overlay atop original image
    axes[3].imshow(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))
    axes[3].set_xlim(0, 480)
    axes[3].set_ylim(480, 0)
    axes[3].set_title("4. Stroke Alignment Overlay", color="white", fontsize=11)
    for idx, stroke in enumerate(ir.strokes):
        if len(stroke) < 2:
            continue
        pts = np.array(stroke)
        axes[3].plot(pts[:, 0], pts[:, 1], color="magenta", linewidth=1.5, alpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.savefig(output_png_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return ir


def main():
    sample_dir = "data/sample_images"
    json_dir = "data/extracted_strokes"
    preview_dir = "data/previews"
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    images = [
        "01_robot_arm_sketch.png",
        "02_technical_gears.png",
        "03_natural_sketch.png",
    ]

    modes = ["boundary", "canny", "skeleton"]

    print("=" * 115)
    print("PHASE 1 COMPUTER VISION STROKE EXTRACTION BENCHMARK")
    print("=" * 115)
    header = (
        f"{'Image':<26} | {'Mode':<10} | {'Strokes':<8} | {'Raw Pts':<8} | "
        f"{'Poly Pts':<9} | {'Compress':<9} | {'MaxDev(px)':<11} | {'MeanDev(px)':<12}"
    )
    print(header)
    print("-" * 115)

    processor = ImageProcessor(VisionConfig(simplification_epsilon_px=1.5))

    for img_name in images:
        img_path = os.path.join(sample_dir, img_name)
        base_stem = os.path.splitext(img_name)[0]

        for mode in modes:
            json_path = os.path.join(json_dir, f"{base_stem}_{mode}.json")
            preview_path = os.path.join(preview_dir, f"{base_stem}_{mode}_preview.png")

            ir = render_preview_figure(img_path, processor, mode, preview_path)
            ir.to_json(json_path)

            stats = ir.metadata.simplification_stats
            row = (
                f"{img_name:<26} | {mode:<10} | {len(ir.strokes):<8} | "
                f"{stats.raw_points_count:<8} | {stats.simplified_points_count:<9} | "
                f"{stats.compression_ratio * 100:6.1f}%   | "
                f"{stats.max_deviation_px:<11.2f} | {stats.mean_deviation_px:<12.2f}"
            )
            print(row)

    print("=" * 115)
    print(f"Extraction completed. Stroke JSONs saved to '{json_dir}', previews saved to '{preview_dir}'.")


if __name__ == "__main__":
    main()
