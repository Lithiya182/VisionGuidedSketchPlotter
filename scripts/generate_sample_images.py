"""
generate_sample_images.py

Generates 3 realistic test sketch images in data/sample_images/:
1. 01_robot_arm_sketch.png: Technical line-art sketch of a robotic arm
2. 02_technical_gears.png: Precision engineering gear drawing with keyway and bolt circle
3. 03_natural_sketch.png: Organic hand-drawn sketch with curves and variable line weights
"""

import math
import os
import cv2
import numpy as np


def create_robot_sketch(output_path: str):
    canvas = np.full((600, 600, 3), 255, dtype=np.uint8)
    ink = (30, 30, 30)

    # Base stand
    cv2.rectangle(canvas, (180, 480), (420, 520), ink, 3)
    cv2.rectangle(canvas, (240, 440), (360, 480), ink, 3)
    cv2.circle(canvas, (300, 440), 25, ink, 3)

    # Link 1 (angled up-right)
    j1 = (300, 440)
    j2 = (420, 260)
    cv2.line(canvas, j1, j2, ink, 5)
    cv2.circle(canvas, j2, 20, ink, 3)

    # Link 2 (angled down-right to drawing pad)
    tip = (260, 180)
    cv2.line(canvas, j2, tip, ink, 4)
    cv2.circle(canvas, tip, 8, ink, -1)

    # Pen tool and stylus holder
    cv2.line(canvas, tip, (240, 150), ink, 3)
    cv2.circle(canvas, (240, 150), 4, ink, -1)

    # Measurement / angle indicator arc
    cv2.ellipse(canvas, j1, (60, 60), 0, -90, -40, ink, 2)
    cv2.ellipse(canvas, j2, (50, 50), 0, 40, 120, ink, 2)

    # Drawing paper outline
    cv2.rectangle(canvas, (100, 100), (320, 320), ink, 2)

    cv2.imwrite(output_path, canvas)
    print(f"Created: {output_path}")


def create_gear_schematic(output_path: str):
    canvas = np.full((640, 640, 3), 255, dtype=np.uint8)
    ink = (20, 20, 20)
    cx, cy = 320, 320

    # Outer pitch circle and gear teeth
    num_teeth = 12
    r_inner = 180
    r_outer = 220
    gear_pts = []
    for i in range(num_teeth):
        theta_start = i * (2 * math.pi / num_teeth)
        theta_mid1 = theta_start + (0.25 * 2 * math.pi / num_teeth)
        theta_mid2 = theta_start + (0.50 * 2 * math.pi / num_teeth)
        theta_end = theta_start + (0.75 * 2 * math.pi / num_teeth)

        gear_pts.append([cx + r_inner * math.cos(theta_start), cy + r_inner * math.sin(theta_start)])
        gear_pts.append([cx + r_outer * math.cos(theta_mid1), cy + r_outer * math.sin(theta_mid1)])
        gear_pts.append([cx + r_outer * math.cos(theta_mid2), cy + r_outer * math.sin(theta_mid2)])
        gear_pts.append([cx + r_inner * math.cos(theta_end), cy + r_inner * math.sin(theta_end)])

    gear_arr = np.array(gear_pts, dtype=np.int32)
    cv2.polylines(canvas, [gear_arr], isClosed=True, color=ink, thickness=3)

    # Inner bore and central axle with keyway
    cv2.circle(canvas, (cx, cy), 120, ink, 3)
    cv2.circle(canvas, (cx, cy), 60, ink, 3)
    # Keyway notch
    cv2.rectangle(canvas, (cx - 15, cy - 75), (cx + 15, cy - 50), ink, 3)

    # 4 bolt holes
    for angle in [45, 135, 225, 315]:
        rad = math.radians(angle)
        bx = int(cx + 90 * math.cos(rad))
        by = int(cy + 90 * math.sin(rad))
        cv2.circle(canvas, (bx, by), 15, ink, 2)
        # Center marks
        cv2.line(canvas, (bx - 20, by), (bx + 20, by), ink, 1)
        cv2.line(canvas, (bx, by - 20), (bx, by + 20), ink, 1)

    cv2.imwrite(output_path, canvas)
    print(f"Created: {output_path}")


def create_natural_sketch(output_path: str):
    canvas = np.full((550, 700, 3), 255, dtype=np.uint8)
    ink = (25, 25, 25)
    cx, cy = 350, 275

    # Butterfly body
    cv2.ellipse(canvas, (cx, cy), (16, 90), 0, 0, 360, ink, 4)
    cv2.circle(canvas, (cx, cy - 100), 20, ink, 4)

    # Antennae
    ant1 = np.array([[cx - 8, cy - 115], [cx - 30, cy - 160], [cx - 60, cy - 175], [cx - 80, cy - 165]], dtype=np.int32)
    ant2 = np.array([[cx + 8, cy - 115], [cx + 30, cy - 160], [cx + 60, cy - 175], [cx + 80, cy - 165]], dtype=np.int32)
    cv2.polylines(canvas, [ant1, ant2], isClosed=False, color=ink, thickness=3)
    cv2.circle(canvas, (cx - 80, cy - 165), 6, ink, -1)
    cv2.circle(canvas, (cx + 80, cy - 165), 6, ink, -1)

    # Upper Left Wing
    wing_ul = np.array([
        [cx - 15, cy - 30],
        [cx - 120, cy - 150],
        [cx - 240, cy - 160],
        [cx - 280, cy - 80],
        [cx - 250, cy + 20],
        [cx - 160, cy + 50],
        [cx - 15, cy + 10]
    ], dtype=np.int32)
    cv2.polylines(canvas, [wing_ul], isClosed=True, color=ink, thickness=4)

    # Upper Right Wing
    wing_ur = np.array([
        [cx + 15, cy - 30],
        [cx + 120, cy - 150],
        [cx + 240, cy - 160],
        [cx + 280, cy - 80],
        [cx + 250, cy + 20],
        [cx + 160, cy + 50],
        [cx + 15, cy + 10]
    ], dtype=np.int32)
    cv2.polylines(canvas, [wing_ur], isClosed=True, color=ink, thickness=4)

    # Lower Left Wing
    wing_ll = np.array([
        [cx - 15, cy + 20],
        [cx - 140, cy + 60],
        [cx - 190, cy + 140],
        [cx - 140, cy + 200],
        [cx - 60, cy + 170],
        [cx - 15, cy + 70]
    ], dtype=np.int32)
    cv2.polylines(canvas, [wing_ll], isClosed=True, color=ink, thickness=4)

    # Lower Right Wing
    wing_lr = np.array([
        [cx + 15, cy + 20],
        [cx + 140, cy + 60],
        [cx + 190, cy + 140],
        [cx + 140, cy + 200],
        [cx + 60, cy + 170],
        [cx + 15, cy + 70]
    ], dtype=np.int32)
    cv2.polylines(canvas, [wing_lr], isClosed=True, color=ink, thickness=4)

    # Inner wing cell vein curves
    cv2.ellipse(canvas, (cx - 140, cy - 50), (60, 40), 30, 0, 360, ink, 2)
    cv2.ellipse(canvas, (cx + 140, cy - 50), (60, 40), -30, 0, 360, ink, 2)
    cv2.circle(canvas, (cx - 110, cy + 130), 25, ink, 2)
    cv2.circle(canvas, (cx + 110, cy + 130), 25, ink, 2)

    cv2.imwrite(output_path, canvas)
    print(f"Created: {output_path}")


def create_complex_logo(output_path: str) -> None:
    """Create a realistic corporate/engineering robotics badge with filled elements, text, and gradients."""
    h, w = 600, 600
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    ink = (25, 25, 25)

    # Subtle radial shading/gradient background typical of a photo/scanned badge
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    bg_vignette = np.clip(255 - dist_from_center * 0.05, 235, 255).astype(np.uint8)
    for c in range(3):
        canvas[:, :, c] = np.minimum(canvas[:, :, c], bg_vignette)

    # Outer border double circles
    cv2.circle(canvas, (cx, cy), 260, ink, 6)
    cv2.circle(canvas, (cx, cy), 245, ink, 2)
    cv2.circle(canvas, (cx, cy), 170, ink, 3)

    # 12 notched gear teeth along outer ring
    for i in range(12):
        angle = i * (2 * math.pi / 12)
        tx = int(cx + 252 * math.cos(angle))
        ty = int(cy + 252 * math.sin(angle))
        cv2.circle(canvas, (tx, ty), 6, ink, -1)

    # Central hexagonal shield
    pts_hex = []
    for i in range(6):
        angle = i * (2 * math.pi / 6) - math.pi / 6
        pts_hex.append([int(cx + 120 * math.cos(angle)), int(cy + 120 * math.sin(angle))])
    cv2.polylines(canvas, [np.array(pts_hex, dtype=np.int32)], isClosed=True, color=ink, thickness=4)

    # Bold central gripper icon (solid filled arms)
    # Left claw
    claw_l = np.array([
        [cx - 70, cy - 60],
        [cx - 40, cy - 80],
        [cx - 20, cy - 50],
        [cx - 40, cy - 20],
        [cx - 70, cy - 30],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [claw_l], ink)

    # Right claw
    claw_r = np.array([
        [cx + 70, cy - 60],
        [cx + 40, cy - 80],
        [cx + 20, cy - 50],
        [cx + 40, cy - 20],
        [cx + 70, cy - 30],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [claw_r], ink)

    # Base cylinder
    cv2.rectangle(canvas, (cx - 35, cy - 20), (cx + 35, cy + 30), ink, -1)
    cv2.line(canvas, (cx - 35, cy + 30), (cx, cy + 80), ink, 5)
    cv2.line(canvas, (cx + 35, cy + 30), (cx, cy + 80), ink, 5)

    # Text lettering "ROBOTICS LAB" in circular banner
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "ROBOTICS", (cx - 110, cy - 195), font, 1.0, ink, 3, cv2.LINE_AA)
    cv2.putText(canvas, "RESEARCH", (cx - 105, cy + 225), font, 0.9, ink, 3, cv2.LINE_AA)

    # Star emblems
    cv2.circle(canvas, (cx - 205, cy), 12, ink, -1)
    cv2.circle(canvas, (cx + 205, cy), 12, ink, -1)

    cv2.imwrite(output_path, canvas)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    out_dir = "data/sample_images"
    os.makedirs(out_dir, exist_ok=True)
    create_robot_sketch(os.path.join(out_dir, "01_robot_arm_sketch.png"))
    create_gear_schematic(os.path.join(out_dir, "02_technical_gears.png"))
    create_natural_sketch(os.path.join(out_dir, "03_natural_sketch.png"))
    create_complex_logo(os.path.join(out_dir, "04_complex_logo.png"))
