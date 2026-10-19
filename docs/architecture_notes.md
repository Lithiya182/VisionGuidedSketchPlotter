# Vision-Guided Sketch Plotter - Architecture Notes

## End-to-End Workflow

The system processes images through a staged pipeline from input to drawn output:

### 1. Image Input
- Receives either a file path or numpy array (BGR / RGB grayscale)
- OpenCV-based preprocessing: resize-and-pad to 480×480 canvas, Gaussian smoothing

### 2. Computer Vision / IR Extraction
- **Vision layer** (`src/vision/image_processor.py`) extracts vectorized strokes in one of three modes:
  - **Boundary**: Outline/silhouette contours via Otsu thresholding + `cv2.findContours`
  - **Canny**: Edge boundary contours via Canny edge detection
  - **Skeleton**: 1-pixel wide centerlines via morphological skeletonization + graph traversal
- Output: `StrokeIR` dataclass with versioned metadata and pixel-space strokes on a standardized 480×480 canvas
- Strict CV/IR separation enforced: pixel coordinates use top-left origin, Y-down convention
- Douglas-Peucker simplification with deviation tracking (max/mean px, compression ratio)

### 3. Coordinate Mapping
- **Coordinate transform** (`src/coordinate_transform.py`) maps pixel coordinates to robot workspace (cm):
  - `image_to_arm_space()`: affine scaling + Y-axis inversion (image Y down → robot Y up)
  - Drawing origin at (4.0, 12.5) cm, 6.0×6.0 cm workspace from top-left image corner
  - Validation: radial reachability bounds [2.0, 18.0] cm (with 2 cm inner margin, 2 cm outer margin)
  - Joint limit validation: theta1 ∈ [-90°, 90°], theta2 ∈ [-150°, 150°]

### 4. Inverse Kinematics
- **Kinematics layer** (`src/kinematics.py`) computes joint angles for target Cartesian positions:
  - Default robot: L1 = L2 = 10 cm 2-DOF planar arm
  - Two elbow branches: "positive" (theta2 > 0) and "negative" (theta2 < 0)
  - Joint limit validation enforced
  - Rejection of unreachable targets and invalid branches

### 5. Path Planning / Stroke Optimization
- **Path planner** (`src/path_planner.py`) converts contours into MotionPoint sequences with pen-up/pen-down states:
  - First point of each stroke approached with pen up
  - Segments interpolated at configurable spacing (default 0.5 cm)
  - Full workspace and joint validation per point
- **Stroke optimizer** (`src/planning/stroke_optimizer.py`) Phase 6:
  - ATSP with multi-objective transition cost: cartesian distance, joint movement, reversals, singularity proximity
  - Algorithms: naive ordering, nearest-neighbour, 2-opt local search, exact bitmask DP (N ≤ 12)
  - OrientedStrokes with direction selection (forward/reversed) and kinematics validation

### 6. Simulation & Closed-Loop Correction
- **CoppeliaSim controller** (`src/coppelia_controller.py`): Remote API client for 2-DOF planar robot:
  - Object handle discovery and dynamic configuration
  - Joint target positioning with settle tolerance (0.25 deg)
  - Vision sensor image capture from `/TopDownVisionSensor`
  - Drawing object management (ink line rendering)
- **Plant disturbance manager** (`src/sim/disturbances.py`):
  - Calibration offsets (link length errors, joint zero offsets)
  - Direction-dependent gear backlash deadband operator
  - Seeded Gaussian joint actuator noise
  - Scene apply/restore idempotence (saves/restores nominal transforms)
- **Vision corrector** (`src/sim/vision_corrector.py` Phase 5):
  - **PerStrokeObserver**: Image differencing to extract per-stroke ink pixels
  - **StrokeMatcher**: ICP-style nearest-point refinement → 2D translation offset estimation
  - **RunningCorrector**: EMA-smoothed offset estimation, magnitude capping at 5.0 mm per stroke
  - **CalibrationPrePass**: Optional cross/grid pattern drawing to fit global plant transform
  - **Critical architectural constraint**: Uses ONLY TopDownVisionSensor image, target strokes, and nominal model. Must NOT access get_tip_position, disturbance parameters, or telemetry records (enforced by AST tests).

### 7. Robot Actuation / Drawing Output
- **Joint commands** sent to CoppeliaSim via `set_joint_targets()` with `wait_for_settle(tolerance 0.25 deg)`
- **Ink segments** rendered via `add_ink_segment()` as visible lines in the simulation
- Nominal (no disturbance) baseline must reproduce: RMSE 0.318 mm, Chamfer 0.228 mm

## Logical Layer Grouping

| Layer | Responsibility | Key Files |
|------|---|---|
| **Vision/IR** | CV extraction, IR generation, simplification stats | `src/vision/image_processor.py` |
| **Coordinate Mapping** | Pixel → cm, workspace/joint validation | `src/coordinate_transform.py` |
| **Kinematics** | FK/IK, joint limits, elbow branching | `src/kinematics.py` |
| **Path Planning** | Stroke planning, pen-up/pen-down, interpolation | `src/path_planner.py` |
| **Stroke Optimization** | ATSP ordering, 2-opt, exact DP, metrics | `src/planning/stroke_optimizer.py` |
| **Simulation** | Disturbances, vision correction, CoppeliaSim bridge | `src/sim/` |
| **Controller** | Remote API, object handles, joint control, vision capture | `src/coppelia_controller.py` |

## Implemented, Simulated, and Planned Features

### Implemented (Phase 1-6 complete)
- Vision extraction (boundary, canny, skeleton modes)
- Coordinate mapping and workspace validation
- Forward and inverse kinematics with 2-DOF planar arm
- Stroke planning with pen-up/pen-down states
- Stroke ordering: naive, NN, 2-opt, exact DP (N ≤ 12)
- Plant disturbance models (backlash, Gaussian noise, calibration offsets)
- Vision corrector (PerStrokeObserver, StrokeMatcher, RunningCorrector, CalibrationPrePass)
- CoppeliaSim controller with Remote API interface
- Evaluation metrics (Chamfer, IoU, pen travel, smoothness, error decomposition)
- AST-enforced architectural constraints (vision corrector forbidden accesses, settle tolerance 0.25 deg)

### Simulated / Phase-Gated
- Closed-loop vision correction requires CoppeliaSim running with scene loaded
- Plant disturbance injection applied only on simulated plant (not in planner/Kinematics)
- Stroke optimization algorithms require CoppeliaSim scene for full trajectory evaluation
- Exact DP optimizer computationally restricted to N ≤ 12 strokes

### Planned / Future
- Real hardware actuation (physical 2-DOF robotic arm)
- Adaptive disturbance compensation using real-time vision feedback
- Real-time stroke optimization during active drawing
- Multi-robot coordination
- GPU-accelerated ICP for faster vision correction

## Missing Integrations / Known Gaps

Based on code inspection, the following integrations are noted as missing or unimplemented:

1. **Real Robot Hardware Interface**: No physical robot driver exists. Actuation currently targets CoppeliaSim only via Remote API.

2. **Live Camera Integration**: Vision pipeline currently operates on static images or numpy arrays. No integration with live camera feed (e.g., USB camera, Raspberry Pi camera).

3. **CoppeliaSim Scene Dependency**: Full pipeline requires CoppeliaSim open with `VisionGuidedSketchPlotter.ttt` scene loaded. Scene objects must be configured via `scripts/setup_scene.py`.

4. **Real-Time Disturbance Compensation**: While the vision corrector estimates and corrects per-stroke offsets, there is no adaptive loop that re-plans subsequent strokes based on corrected waypoints during active drawing.

5. **Closed-Loop End-to-End Pipeline**: The individual components (vision → IK → planning → simulation) exist but a unified end-to-end script that chains them all (e.g., `run_closed_loop.py`) has not been verified to produce the nominal baseline metrics (RMSE 0.318 mm, Chamfer 0.228 mm).

6. **Online Stroke Re-optimization**: After vision correction updates the plant offset, there is no automatic re-optimization of remaining stroke order. The corrected offsets are applied to waypoints, but the stroke sequence itself is not re-optimized.

7. **Disturbance-Aware Path Planning**: The planner, FK, and IK modules are completely unaware of disturbance parameters. Disturbances act only on the simulated plant (enforced by architectural constraint).

8. **Sensor Calibration Workflow**: While `CalibrationPrePass` exists in the vision corrector, there is no documented calibration workflow for real-world system identification (beyond the synthetic cross/grid patterns).

## Key Architectural Constraints (from AGENTS.md)

- **Settle tolerance**: 0.25 deg. Nominal baseline must reproduce RMSE 0.318 mm, Chamfer 0.228 mm.
- **CV/IR separation**: Vision outputs pixel-space strokes (480×480, top-left origin, Y down); mapping stays in `coordinate_transform.py`.
- **Disturbances act only on simulated plant**: The vision corrector may use ONLY the TopDownVisionSensor image, target strokes, and the nominal model (enforced by AST test; must keep passing).
- **No assumption of physical hardware**: All components are designed for the simulated CoppeliaSim environment.

## Data Flow Summary

```
Image Input
        ↓
Computer Vision (ImageProcessor) → StrokeIR [480×480 px, Y-down]
        ↓
Coordinate Mapping (image_to_arm_space) → Arm Space [cm]
        ↓
Inverse Kinematics → Joint Angles [rad, theta1/theta2]
        ↓
Forward Kinematics / Simulation → Pen Tip Position [cm]
        ↓
Robot Actuation (CoppeliaController.set_joint_targets) → Drawing Output
        ↑
Closed Loop: Vision Sensor → Vision Corrector (PerStrokeObserver, StrokeMatcher, RunningCorrector)
        |
Disturbance Config → Plant DisturbanceManager → Simulated Plant Modifications
```