# Vision-Guided Sketch Plotter

A complete research platform for **kinematics-aware contour planning and camera-feedback error correction** on a simulated low-cost 2-DOF planar robotic arm in CoppeliaSim (Python ZeroMQ Remote API).

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Setup](#setup)
3. [Quick Start – One-Command Demo](#quick-start--one-command-demo)
4. [Running Individual Phases](#running-individual-phases)
5. [Results Summary](#results-summary)
6. [Configuration](#configuration)
7. [Limitations](#limitations)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT IMAGE  (PNG/JPG)                                             │
│       │                                                             │
│       ▼  vision/image_processor.py                                  │
│  STROKE IR  (pixel-space JSON, 480×480)                             │
│       │  coordinate_transform.py                                    │
│       ▼                                                             │
│  ARM-SPACE STROKES  (cm, reachability-filtered)                     │
│       │                                                             │
│   ┌───┴──────────────────────────────┐                              │
│   │  planning/stroke_optimizer.py    │                              │
│   │  ATSP-DS + 2-Opt ordering        │  ◄─ StrokeOrderConfig        │
│   └───┬──────────────────────────────┘                              │
│       │                                                             │
│   kinematics.py  →  path_planner.py  →  MOTION PLAN                │
│                             │                                       │
│                   ┌─────────┴──────────┐                            │
│                   │ OPEN-LOOP          │  CLOSED-LOOP               │
│                   │ (baseline)         │  sim/vision_corrector.py   │
│                   │                   │  ├─ CalibrationPrePass      │
│                   │                   │  ├─ PerStrokeObserver (ICP) │
│                   │                   │  └─ RunningCorrector (EMA)  │
│                   └─────────┬──────────┘                            │
│                             │                                       │
│              coppelia_controller.py                                 │
│              ┌──────────────────────────────────┐                   │
│              │  CoppeliaSim ZeroMQ Remote API   │                   │
│              │  • Joint angle actuation         │                   │
│              │  • sim.addDrawingObject ink trace │                   │
│              │  • TopDownVisionSensor capture   │                   │
│              │  • sim/disturbances.py (plant)   │                   │
│              └──────────────────────────────────┘                   │
│                             │                                       │
│              eval/metrics.py                                        │
│              RMSE, Chamfer, Tol-IoU, Endpoint Error                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Map

| Directory / File | Purpose |
|---|---|
| `src/vision/image_processor.py` | Grayscale → Canny/boundary/skeleton → IR JSON |
| `src/coordinate_transform.py` | Pixel ↔ arm-workspace mapping (no aspect distortion) |
| `src/kinematics.py` | Forward & inverse kinematics (elbow-branch selection) |
| `src/path_planner.py` | Stroke interpolation, pen-up/down motion plan |
| `src/planning/stroke_optimizer.py` | ATSP-DS with multi-start NN + 2-opt + exact DP |
| `src/sim/disturbances.py` | Calibration offset, backlash deadband, Gaussian noise (plant only) |
| `src/sim/vision_corrector.py` | Camera observation, ICP matching, EMA running corrector |
| `src/eval/metrics.py` | RMSE, Chamfer, Tol-IoU (τ=2px), endpoint error, pen travel |
| `src/coppelia_controller.py` | ZeroMQ bridge + joint control + drawing object |
| `src/config.py` | Central dataclass config (all parameters) |
| `scripts/setup_scene.py` | Idempotent CoppeliaSim scene setup |
| `scripts/run_demo.py` | One-command demo |
| `scripts/run_phase7_experiments.py` | Full experiment matrix |

---

## Setup

### Prerequisites
- **CoppeliaSim ≥ 4.6** with ZeroMQ Remote API enabled (port 23000)
- **Python ≥ 3.10**

### Install
```bash
git clone <repo>
cd VisionGuidedSketchPlotter
pip install -r requirements.txt
```

### Start CoppeliaSim
```
1. Open CoppeliaSim
2. File → Open → VisionGuidedSketchPlotter.ttt
3. Start the simulation (Play button)
```

### Verify the scene
```bash
python scripts/setup_scene.py
```
Expected output: audit table listing `Joint1`, `Joint2`, `Tip`, `DrawingPaper`, `TopDownVisionSensor` with correct handles and positions.

---

## Quick Start – One-Command Demo

```bash
python scripts/run_demo.py \
    --image  data/sample_images/01_robot_arm_sketch.png \
    --mode   skeleton \
    --disturbance moderate \
    --order  optimized \
    --control cl_prepass \
    --seed   42
```

### CLI Flags

| Flag | Choices | Default | Description |
|---|---|---|---|
| `--image` | any PNG/JPG | `01_robot_arm_sketch.png` | Input image |
| `--mode` | `skeleton` / `canny` / `boundary` | `skeleton` | CV extraction mode |
| `--disturbance` | `none` / `mild` / `moderate` / `severe` | `moderate` | Plant disturbance preset |
| `--order` | `naive` / `optimized` | `optimized` | Stroke ordering (baseline vs 2-opt) |
| `--control` | `open_loop` / `cl_prepass` | `cl_prepass` | Open-loop or closed-loop |
| `--seed` | integer | `42` | Disturbance RNG seed |
| `--output-dir` | path | `logs/demo/` | Output directory |

### Outputs (in `logs/demo/`)
| File | Contents |
|---|---|
| `*_comparison.png` | 3-panel: Target / Open-loop / Closed-loop camera overlays |
| `*_top_down_ol.png` | Raw top-down camera capture (open-loop) |
| `*_top_down_cl.png` | Raw top-down camera capture (closed-loop) |
| `*_metrics.csv` | Numeric metrics for both modes |

---

## Running Individual Phases

```bash
# Phase 1 – Vision pipeline (all 3 images × 3 modes)
python scripts/run_vision_pipeline.py

# Phase 2 – Open-loop baseline
python scripts/run_open_loop_baseline.py

# Phase 4 – Disturbance presets
python scripts/run_phase4_experiments.py

# Phase 5 – Closed-loop evaluation
python scripts/run_phase5_experiments.py --preset moderate --seed 42

# Phase 6 – Stroke-order benchmarks
python scripts/run_phase6_benchmarks.py

# Phase 7 – Full experiment matrix
python scripts/run_phase7_experiments.py

# Unit tests
pytest tests/ -v
```

---

## Results Summary

### Phase 5 – Closed-Loop Correction (per-stroke vs calibration pre-pass)

> **Preset parameters** – `moderate`: δL₁=+1.5 mm, δL₂=-1.0 mm, δθ₁=+0.8°, δθ₂=-0.8°, backlash 0.50°/0.60°, noise σ=0.12°.  
> **Correction variant** – `CL-PrePass`: draws a calibration cross, estimates global plant offset from camera, then applies inverse correction to all remaining waypoints.

| Preset | Mode | RMSE (mm) | Chamfer (mm) | Tol-IoU (%) | RMSE Recovered |
|---|---|---|---|---|---|
| None | Open-Loop | 0.318 ± 0.000 | 0.228 ± 0.000 | 84.4 ± 0.0 | — |
| Mild | Open-Loop | 0.737 ± 0.004 | 0.436 ± 0.000 | 67.8 ± 0.2 | — |
| Mild | CL-PrePass | 0.656 ± 0.017 | 0.365 ± 0.005 | 79.5 ± 0.9 | 19.3% |
| **Moderate** | **Open-Loop** | **1.830 ± 0.018** | **0.962 ± 0.016** | **30.8 ± 0.3** | — |
| **Moderate** | **CL-PrePass** | **1.034 ± 0.024** | **0.505 ± 0.013** | **67.9 ± 0.3** | **52.6%** |
| Severe | Open-Loop | 3.715 ± 0.044 | 1.574 ± 0.039 | 22.3 ± 0.3 | — |
| Severe | CL-PrePass | 1.699 ± 0.062 | 0.711 ± 0.019 | 53.3 ± 1.2 | 59.4% |

*Note: Recoverable-error recovery (vs nominal floor 0.318 mm) at moderate = **87.7%**. Raw RMSE reduction = **72.2%**. Both figures must be read together.*

### Phase 6 – Stroke-Order Optimization

| Image | Method | Pen-Up | Efficiency | Reversals | Solve Time |
|---|---|---|---|---|---|
| robot_arm_sketch | Naive baseline | 28.10 cm | 45.0% | 64 | 2 ms |
| robot_arm_sketch | **Kinematic 2-Opt** | **5.41 cm** | **81.0%** | **17** | 406 ms |
| technical_gears | Naive baseline | 81.68 cm | 26.8% | 107 | 6 ms |
| technical_gears | **Kinematic 2-Opt** | **12.28 cm** | **70.9%** | **19** | 4.9 s |

**Backlash & reversal study (moderate, seeds 42–44):**

| Ordering | Control | Reversals | RMSE (mm) | Tol-IoU (%) | Pen-Up (cm) |
|---|---|---|---|---|---|
| Naive | Open-Loop | 64 | 1.830 ± 0.018 | 30.8 ± 0.3 | 24.6 |
| Naive | **CL-PrePass** | 64 | **1.034 ± 0.024** | **67.9 ± 0.3** | 24.6 |
| 2-Opt | Open-Loop | 17 | 2.094 ± 0.018 | 30.6 ± 0.2 | 5.2 |
| 2-Opt | **CL-PrePass** | 17 | **1.092 ± 0.029** | **52.5 ± 1.3** | 5.1 |

---

## Configuration

All parameters are in [`src/config.py`](src/config.py):

```python
WorkspaceConfig(
    draw_width_cm  = 6.0,    # Drawing window (cm)
    draw_height_cm = 6.0,
    draw_origin_x  = 4.0,   # Bottom-left corner in robot frame (cm)
    draw_origin_y  = 12.5,
)

StrokeOrderConfig(
    w_cartesian  = 1.0,   # Air-distance weight
    w_joint      = 0.5,   # Joint-displacement weight
    w_reversal   = 0.3,   # Joint-direction-reversal penalty
    w_singularity= 0.05,  # 1/|det(J)| singularity proximity penalty
    branch       = "positive",  # Elbow-up
)
```

### Disturbance Presets

| Preset | δL₁ | δL₂ | δθ₁ | δθ₂ | Backlash θ₁/θ₂ | Noise σ |
|---|---|---|---|---|---|---|
| none | 0 | 0 | 0° | 0° | 0° / 0° | 0° |
| mild | +0.5 mm | −0.5 mm | +0.30° | −0.30° | 0.25° / 0.25° | 0.05° |
| **moderate** | +1.5 mm | −1.0 mm | +0.80° | −0.80° | 0.50° / 0.60° | 0.12° |
| severe | +3.0 mm | −2.5 mm | +1.50° | −1.50° | 1.00° / 1.20° | 0.25° |

---

## Limitations

> [!IMPORTANT]
> The following are inherent constraints of this research prototype. They are reported here honestly, not hidden.

1. **Simulation only.** All experiments run in CoppeliaSim. Real hardware would introduce additional mechanical effects (friction, motor dynamics, inertia, cable drag) not modelled here.

2. **Simulated disturbances are approximations.** Calibration offsets, backlash, and noise are injected by modifying the commanded joint angles. A real joint's backlash deadband is direction-history-dependent and hysteretic; the model here is a simplified deadband on reversals only.

3. **Virtual top-down camera.** The `TopDownVisionSensor` is an ideal pinhole camera with zero lens distortion, no rolling shutter, no exposure variation, and perfect lighting. Real camera calibration, lens distortion, and illumination variation would degrade ICP matching.

4. **Singularity weight has no effect in this window** (min |det J| ~82.5 across profiles, see Phase 6). The drawing window sits in the robot's upper workspace (θ₂ ≈ 30°–90°) where the Jacobian determinant is uniformly large; the singularity penalty w_singularity=0.05 contributes negligibly to the stroke-order cost across all disturbance profiles. |det J| = L1·L2·sin(θ2) = 0 at the park/home pose (θ2 = 0°), and |det J| ~82.5 is the minimum over the actual drawn waypoints in the Phase 6 study. It is retained for generality when the window is enlarged or the robot configuration changed.

**Min |det J| over drawn waypoints per image (Phase 6):**
- robot_arm_sketch: min=82.49, mean=93.82 (30 strokes)
- technical_gears: min=85.30, mean=94.96 (61 strokes)
- natural_sketch: min=84.99, mean=95.08 (38 strokes)
- complex_logo: not evaluated in Phase 6 (N=101 strokes, computationally tractable only with heuristics).

5. **Heuristic optimality gap at N = 8+.** Kinematic 2-Opt achieves a **36.9% gap** relative to the exact bitmask DP solution at N = 8. For the full sketches (N = 30–101), exact DP is computationally intractable (≥ 10⁴¹ states). The 2-Opt heuristic is the only practical option; the optimality gap on full instances is unknown.

6. **Stroke ordering does not eliminate backlash amplitude.** Reducing joint reversals by 73% (64 → 17) cuts the frequency of backlash events but cannot eliminate the per-event deadband (~0.9 mm). The corrector is required to compensate for the residual systematic offset.

7. **Closed-loop relies on calibration pre-pass, not per-stroke streaming.** The `CL-PrePass` variant uses a one-shot global offset estimate. It cannot adapt to spatially varying or nonlinear distortions across the workspace. Affine correction (`use_affine=True`) is implemented but not used in the headline results due to insufficient ink pixels in short strokes.

8. **Skeleton mode degrades quality on complex images.** The logo image (04_complex_logo) produces 101 noisy strokes in skeleton mode vs 37 (boundary) and 71 (canny) strokes with more total coverage. Skeleton extraction fragments thin structures in high-detail images, inflating stroke count without improving coverage. Boundary or canny modes are recommended for complex/natural images.

9. **No physical effort model.** Joint displacement (radians) is used as a proxy for mechanical effort and motor wear. Actual torque, speed, and motor heating are not modelled.
