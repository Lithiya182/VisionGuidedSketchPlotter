"""
src/planning/stroke_optimizer.py

Phase 6: Kinematics-Aware Stroke-Order Optimization.

Formulates stroke sequencing and orientation selection as an asymmetric Traveling
Salesperson Problem (ATSP) with a multi-objective kinematic transition cost:
  1. Pen-up Cartesian distance: ||x_to - x_from||
  2. Joint-space movement: |d_theta1| + |d_theta2| (proxy for actuator excursion)
  3. Joint-reversal penalty: count of direction changes triggering mechanical backlash
  4. Singularity-proximity penalty: 1 / (|det(J)| + eps), where det(J) = L1 * L2 * sin(theta2)

Hard constraints:
  - Unreachable targets and joint limit violations receive infinite cost.
  - Branch reachability is validated across all points of every candidate stroke.

Algorithms:
  - order_strokes_naive: Preserve input sequence (Phase 2 baseline)
  - order_strokes_nn: Greedy nearest-neighbour with bidirectional stroke reversal
  - order_strokes_2opt: Kinematics-aware 2-opt local search with stroke inversion
  - order_strokes_exact_dp: Exact bitmask DP (O(N^2 2^N)) for optimality benchmarking
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import StrokeOrderConfig, DEFAULT_STROKE_ORDER_CONFIG
from src.coordinate_transform import (
    validate_workspace_point,
    THETA1_LIMITS_DEG,
    THETA2_LIMITS_DEG,
)
from src.kinematics import forward_kinematics, inverse_kinematics, L1, L2


@dataclass
class OrientedStroke:
    """
    A single polyline stroke with an assigned traversal direction.

    Attributes:
        stroke_id: Original stroke index from source IR JSON.
        is_forward: True if drawn start->end, False if drawn reversed (end->start).
        points: (K, 2) array of Cartesian waypoints in traversal order.
        start_pt: First waypoint (x, y) in cm.
        end_pt: Last waypoint (x, y) in cm.
        start_joints: (theta1, theta2) in radians at start.
        end_joints: (theta1, theta2) in radians at end.
        start_dir_joint: (d_theta1, d_theta2) joint displacement on first drawing segment.
        end_dir_joint: (d_theta1, d_theta2) joint displacement on final drawing segment.
        det_jacobian_start: det(J) = L1*L2*sin(theta2) at start.
        det_jacobian_end: det(J) = L1*L2*sin(theta2) at end.
        length_cm: Arc-length of this stroke in cm.
        is_valid: True if all waypoints are kinematically reachable within joint limits.
    """
    stroke_id: int
    is_forward: bool
    points: np.ndarray
    start_pt: Tuple[float, float]
    end_pt: Tuple[float, float]
    start_joints: Tuple[float, float]
    end_joints: Tuple[float, float]
    start_dir_joint: Tuple[float, float]
    end_dir_joint: Tuple[float, float]
    det_jacobian_start: float
    det_jacobian_end: float
    length_cm: float
    is_valid: bool


def create_oriented_stroke(
    stroke_pts: np.ndarray,
    stroke_id: int,
    is_forward: bool = True,
    branch: str = "positive",
    l1: float = L1,
    l2: float = L2,
) -> OrientedStroke:
    """
    Instantiate an OrientedStroke, evaluating kinematics, Jacobian determinants,
    and stroke-boundary joint velocities.
    """
    raw_pts = np.asarray(stroke_pts, dtype=float)
    if not is_forward:
        pts = raw_pts[::-1].copy()
    else:
        pts = raw_pts.copy()

    start_pt = (float(pts[0, 0]), float(pts[0, 1]))
    end_pt = (float(pts[-1, 0]), float(pts[-1, 1]))

    # Calculate stroke arc length
    if len(pts) >= 2:
        diffs = np.diff(pts, axis=0)
        length_cm = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        length_cm = 0.0

    # Solve IK across all stroke vertices to guarantee reachability
    is_valid = True
    joint_angles = []
    for p in pts:
        try:
            validate_workspace_point(float(p[0]), float(p[1]))
            th1, th2 = inverse_kinematics(
                float(p[0]),
                float(p[1]),
                l1=l1,
                l2=l2,
                branch=branch,
                theta1_limits_deg=THETA1_LIMITS_DEG,
                theta2_limits_deg=THETA2_LIMITS_DEG,
            )
            joint_angles.append((th1, th2))
        except ValueError:
            is_valid = False
            break

    if not is_valid or len(joint_angles) < 2:
        return OrientedStroke(
            stroke_id=stroke_id,
            is_forward=is_forward,
            points=pts,
            start_pt=start_pt,
            end_pt=end_pt,
            start_joints=(0.0, 0.0),
            end_joints=(0.0, 0.0),
            start_dir_joint=(0.0, 0.0),
            end_dir_joint=(0.0, 0.0),
            det_jacobian_start=0.0,
            det_jacobian_end=0.0,
            length_cm=length_cm,
            is_valid=False,
        )

    start_joints = joint_angles[0]
    end_joints = joint_angles[-1]

    # Initial drawing direction in joint space: segment 0 -> 1
    start_dir_joint = (
        joint_angles[1][0] - joint_angles[0][0],
        joint_angles[1][1] - joint_angles[0][1],
    )
    # Final drawing direction in joint space: segment (K-2) -> (K-1)
    end_dir_joint = (
        joint_angles[-1][0] - joint_angles[-2][0],
        joint_angles[-1][1] - joint_angles[-2][1],
    )

    det_j_start = l1 * l2 * math.sin(start_joints[1])
    det_j_end = l1 * l2 * math.sin(end_joints[1])

    return OrientedStroke(
        stroke_id=stroke_id,
        is_forward=is_forward,
        points=pts,
        start_pt=start_pt,
        end_pt=end_pt,
        start_joints=start_joints,
        end_joints=end_joints,
        start_dir_joint=start_dir_joint,
        end_dir_joint=end_dir_joint,
        det_jacobian_start=det_j_start,
        det_jacobian_end=det_j_end,
        length_cm=length_cm,
        is_valid=True,
    )


def compute_transition_cost(
    s1: OrientedStroke,
    s2: OrientedStroke,
    cfg: StrokeOrderConfig = DEFAULT_STROKE_ORDER_CONFIG,
) -> float:
    """
    Compute multi-objective transition cost for pen-up movement from s1 to s2:
      Cost = w_cart * d_cart + w_joint * d_joint + w_rev * n_rev + w_sing * p_sing

    Returns infinity if either stroke is kinematically unreachable.
    """
    if not s1.is_valid or not s2.is_valid:
        return float("inf")

    # 1. Cartesian pen-up distance (cm)
    dx = s2.start_pt[0] - s1.end_pt[0]
    dy = s2.start_pt[1] - s1.end_pt[1]
    d_cart = math.hypot(dx, dy)

    # 2. Joint-space movement proxy (radians)
    d_th1 = s2.start_joints[0] - s1.end_joints[0]
    d_th2 = s2.start_joints[1] - s1.end_joints[1]
    d_joint = abs(d_th1) + abs(d_th2)

    # 3. Joint direction reversals
    # Counts direction reversals between stroke drawing motion and repositioning motion
    n_rev = 0
    # Reversal on exiting s1:
    if s1.end_dir_joint[0] * d_th1 < -1e-5:
        n_rev += 1
    if s1.end_dir_joint[1] * d_th2 < -1e-5:
        n_rev += 1
    # Reversal on entering s2:
    if d_th1 * s2.start_dir_joint[0] < -1e-5:
        n_rev += 1
    if d_th2 * s2.start_dir_joint[1] < -1e-5:
        n_rev += 1

    # 4. Singularity-proximity penalty: 1 / (|det(J)| + eps)
    p_sing = (
        1.0 / (abs(s1.det_jacobian_end) + cfg.eps_singularity)
        + 1.0 / (abs(s2.det_jacobian_start) + cfg.eps_singularity)
    )

    cost = (
        cfg.w_cartesian * d_cart
        + cfg.w_joint * d_joint
        + cfg.w_reversal * float(n_rev)
        + cfg.w_singularity * p_sing
    )
    return cost


def compute_tour_total_cost(
    tour: List[OrientedStroke],
    cfg: StrokeOrderConfig = DEFAULT_STROKE_ORDER_CONFIG,
) -> float:
    """Compute total transition cost across all sequential stroke connections."""
    if len(tour) < 2:
        return 0.0
    total = 0.0
    for i in range(len(tour) - 1):
        c = compute_transition_cost(tour[i], tour[i + 1], cfg)
        if math.isinf(c):
            return float("inf")
        total += c
    return total


# -----------------------------------------------------------------------------
# 1. Naive Ordering (Phase 2 Baseline)
# -----------------------------------------------------------------------------

def order_strokes_naive(
    strokes: List[np.ndarray],
    branch: str = "positive",
) -> List[OrientedStroke]:
    """
    Preserve stroke order and forward directions as given by image extraction.
    Reproduces the Phase 2 baseline.
    """
    return [
        create_oriented_stroke(s, idx, is_forward=True, branch=branch)
        for idx, s in enumerate(strokes)
    ]


# -----------------------------------------------------------------------------
# 2. Nearest-Neighbour (NN) with Stroke Reversal
# -----------------------------------------------------------------------------

def order_strokes_nn(
    strokes: List[np.ndarray],
    cfg: StrokeOrderConfig = DEFAULT_STROKE_ORDER_CONFIG,
    multi_start: bool = True,
) -> List[OrientedStroke]:
    """
    Greedy Nearest-Neighbour baseline with bidirectional stroke reversal.

    When multi_start is True, evaluates starting from each candidate stroke and
    selects the tour with minimum total transition cost.
    """
    n = len(strokes)
    if n == 0:
        return []

    # Pre-generate forward and reversed candidate strokes
    candidates: List[Tuple[OrientedStroke, OrientedStroke]] = []
    for idx, s in enumerate(strokes):
        fwd = create_oriented_stroke(s, idx, is_forward=True, branch=cfg.branch)
        rev = (
            create_oriented_stroke(s, idx, is_forward=False, branch=cfg.branch)
            if cfg.enable_reversal
            else fwd
        )
        candidates.append((fwd, rev))

    start_indices = list(range(n)) if multi_start else [0]
    num_dirs = 2 if cfg.enable_reversal else 1

    best_tour: Optional[List[OrientedStroke]] = None
    best_cost = float("inf")

    for start_i in start_indices:
        for start_d in range(num_dirs):
            visited = [False] * n
            tour: List[OrientedStroke] = [candidates[start_i][start_d]]
            visited[start_i] = True
            current = tour[0]

            for _ in range(n - 1):
                best_next = None
                best_step_cost = float("inf")
                best_idx = -1

                for i in range(n):
                    if visited[i]:
                        continue
                    for d in range(num_dirs):
                        cand = candidates[i][d]
                        c = compute_transition_cost(current, cand, cfg)
                        if c < best_step_cost:
                            best_step_cost = c
                            best_next = cand
                            best_idx = i

                if best_next is not None and best_idx != -1:
                    visited[best_idx] = True
                    tour.append(best_next)
                    current = best_next
                else:
                    # Fallback for remaining unreachable/disconnected strokes
                    for i in range(n):
                        if not visited[i]:
                            visited[i] = True
                            tour.append(candidates[i][0])
                            current = candidates[i][0]
                            break

            tour_cost = compute_tour_total_cost(tour, cfg)
            if tour_cost < best_cost:
                best_cost = tour_cost
                best_tour = tour

    return best_tour if best_tour is not None else []


# -----------------------------------------------------------------------------
# 3. Kinematics-Aware 2-Opt Local Search Refinement
# -----------------------------------------------------------------------------

def order_strokes_2opt(
    strokes: List[np.ndarray],
    cfg: StrokeOrderConfig = DEFAULT_STROKE_ORDER_CONFIG,
    initial_tour: Optional[List[OrientedStroke]] = None,
) -> List[OrientedStroke]:
    """
    Kinematics-aware 2-Opt local search refinement starting from Nearest-Neighbour.

    Applies:
      1. Segment reversals with direction flips: reverses intermediate tour segments
         and tests whether inverting stroke directions lowers transition cost.
      2. Relocate/insertion moves: moves individual strokes to optimal tour slots.
      3. Guaranteed non-increasing cost: output cost is <= initial tour cost.
    """
    n = len(strokes)
    if n <= 2:
        return order_strokes_nn(strokes, cfg) if initial_tour is None else list(initial_tour)

    # Pre-generate candidate forward and reversed strokes
    candidates: Dict[int, Tuple[OrientedStroke, OrientedStroke]] = {}
    for idx, s in enumerate(strokes):
        fwd = create_oriented_stroke(s, idx, is_forward=True, branch=cfg.branch)
        rev = (
            create_oriented_stroke(s, idx, is_forward=False, branch=cfg.branch)
            if cfg.enable_reversal
            else fwd
        )
        candidates[idx] = (fwd, rev)

    if initial_tour is None:
        tour = order_strokes_nn(strokes, cfg, multi_start=True)
    else:
        tour = list(initial_tour)

    best_cost = compute_tour_total_cost(tour, cfg)
    improved = True
    iteration = 0

    while improved and iteration < cfg.max_2opt_iterations:
        improved = False
        iteration += 1

        # 1. Standard 2-opt segment reversals
        for i in range(n - 1):
            for j in range(i + 1, n):
                seg = tour[i:j+1]
                # Test with and without inverting stroke internal directions
                flip_options = [False, True] if cfg.enable_reversal else [False]
                for flip_dirs in flip_options:
                    if flip_dirs:
                        cand_seg = [
                            candidates[s.stroke_id][1 if s.is_forward else 0]
                            for s in seg[::-1]
                        ]
                    else:
                        cand_seg = seg[::-1]

                    cand_tour = tour[:i] + cand_seg + tour[j+1:]
                    cand_cost = compute_tour_total_cost(cand_tour, cfg)

                    if cand_cost < best_cost - 1e-4:
                        tour = cand_tour
                        best_cost = cand_cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        # 2. Relocate moves (remove stroke i and insert at position j)
        if not improved:
            for i in range(n):
                s_i = tour[i]
                rem_tour = tour[:i] + tour[i+1:]
                for j in range(n):
                    dir_options = [s_i.is_forward]
                    if cfg.enable_reversal:
                        dir_options.append(not s_i.is_forward)
                    for d_fwd in dir_options:
                        cand_s = candidates[s_i.stroke_id][0 if d_fwd else 1]
                        cand_tour = rem_tour[:j] + [cand_s] + rem_tour[j:]
                        cand_cost = compute_tour_total_cost(cand_tour, cfg)

                        if cand_cost < best_cost - 1e-4:
                            tour = cand_tour
                            best_cost = cand_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return tour


# -----------------------------------------------------------------------------
# 4. Exact Bitmask Dynamic Programming Optimizer (O(N^2 2^N))
# -----------------------------------------------------------------------------

def order_strokes_exact_dp(
    strokes: List[np.ndarray],
    cfg: StrokeOrderConfig = DEFAULT_STROKE_ORDER_CONFIG,
) -> Tuple[List[OrientedStroke], float]:
    """
    Exact bitmask dynamic programming solver for ATSP with stroke reversals.
    Guaranteed to find the globally optimal stroke ordering and directions for N <= 12.

    State: dp[visited_mask][last_stroke_idx][last_direction]
    Returns:
        (optimal_tour, optimal_cost)
    """
    n = len(strokes)
    if n > 12:
        raise ValueError(f"Exact DP is computationally restricted to N <= 12 strokes (got {n}).")

    if n <= 1:
        single = [create_oriented_stroke(s, i, True, cfg.branch) for i, s in enumerate(strokes)]
        return single, 0.0

    # Pre-build candidates: candidates[i][0] = fwd, candidates[i][1] = rev
    candidates: List[Tuple[OrientedStroke, OrientedStroke]] = []
    for idx, s in enumerate(strokes):
        fwd = create_oriented_stroke(s, idx, is_forward=True, branch=cfg.branch)
        rev = create_oriented_stroke(s, idx, is_forward=False, branch=cfg.branch)
        candidates.append((fwd, rev))

    num_dirs = 2 if cfg.enable_reversal else 1
    num_states = 1 << n

    # dp[mask][i][d]: min cost to cover mask, ending at stroke i with direction d
    dp = np.full((num_states, n, num_dirs), np.inf, dtype=float)
    parent = np.full((num_states, n, num_dirs, 3), -1, dtype=int)

    # Initialize: can start at any stroke i with direction d
    for i in range(n):
        for d in range(num_dirs):
            mask = 1 << i
            if candidates[i][d].is_valid:
                dp[mask, i, d] = 0.0

    # DP over bitmasks
    for mask in range(1, num_states):
        for last_i in range(n):
            if not (mask & (1 << last_i)):
                continue
            for last_d in range(num_dirs):
                cur_cost = dp[mask, last_i, last_d]
                if math.isinf(cur_cost):
                    continue
                cur_stroke = candidates[last_i][last_d]

                for next_i in range(n):
                    if mask & (1 << next_i):
                        continue
                    next_mask = mask | (1 << next_i)
                    for next_d in range(num_dirs):
                        next_stroke = candidates[next_i][next_d]
                        trans_cost = compute_transition_cost(cur_stroke, next_stroke, cfg)
                        if math.isinf(trans_cost):
                            continue
                        new_cost = cur_cost + trans_cost
                        if new_cost < dp[next_mask, next_i, next_d]:
                            dp[next_mask, next_i, next_d] = new_cost
                            parent[next_mask, next_i, next_d] = [mask, last_i, last_d]

    # Find optimal ending state
    full_mask = (1 << n) - 1
    best_cost = np.inf
    best_end = (-1, -1)
    for i in range(n):
        for d in range(num_dirs):
            if dp[full_mask, i, d] < best_cost:
                best_cost = dp[full_mask, i, d]
                best_end = (i, d)

    if best_end[0] == -1 or math.isinf(best_cost):
        # Fallback to NN if graph is disconnected
        fallback = order_strokes_nn(strokes, cfg)
        return fallback, compute_tour_total_cost(fallback, cfg)

    # Reconstruct optimal sequence
    tour = []
    cur_m = full_mask
    cur_i, cur_d = best_end
    while cur_m != -1 and cur_i != -1:
        tour.append(candidates[cur_i][cur_d])
        prev = parent[cur_m, cur_i, cur_d]
        if prev[0] == -1:
            break
        cur_m, cur_i, cur_d = int(prev[0]), int(prev[1]), int(prev[2])

    tour.reverse()
    return tour, float(best_cost)


# -----------------------------------------------------------------------------
# 5. Tour Metrics Evaluation
# -----------------------------------------------------------------------------

def evaluate_tour_metrics(
    tour: List[OrientedStroke],
    cfg: StrokeOrderConfig = DEFAULT_STROKE_ORDER_CONFIG,
    initial_tip_pos: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive kinematic, geometric, and efficiency metrics for a stroke tour.

    Metrics:
      - pen_down_distance_cm: Useful drawing distance
      - pen_up_distance_cm: Air repositioning travel
      - total_distance_cm: Total Cartesian tip travel
      - pen_travel_efficiency_pct: (pen_down / total) * 100
      - total_joint_displacement_rad: Proxy for joint excursion (|d_th1| + |d_th2|)
      - joint_reversals_count: Total direction flips on Joint1 and Joint2
      - min_det_jacobian: Minimum |det(J)| along transitions
      - mean_det_jacobian: Mean |det(J)| across waypoints
      - singularity_penalty_sum: Sum of 1/(|det(J)| + eps)
    """
    if not tour:
        return {
            "pen_down_distance_cm": 0.0,
            "pen_up_distance_cm": 0.0,
            "total_distance_cm": 0.0,
            "pen_travel_efficiency_pct": 0.0,
            "total_joint_displacement_rad": 0.0,
            "joint_reversals_count": 0.0,
            "min_det_jacobian": 0.0,
            "mean_det_jacobian": 0.0,
            "singularity_penalty_sum": 0.0,
        }

    pen_down_dist = sum(s.length_cm for s in tour)
    pen_up_dist = 0.0
    total_joint_disp = 0.0
    total_reversals = 0
    det_j_values = []

    # Optional move from initial position to first stroke start
    if initial_tip_pos is not None:
        p0 = initial_tip_pos
        p1 = tour[0].start_pt
        pen_up_dist += math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    # Accumulate internal joint motion inside each stroke
    for s in tour:
        det_j_values.append(abs(s.det_jacobian_start))
        det_j_values.append(abs(s.det_jacobian_end))

    # Transitions between strokes
    for i in range(len(tour) - 1):
        s1 = tour[i]
        s2 = tour[i + 1]

        # Cartesian pen-up distance
        dx = s2.start_pt[0] - s1.end_pt[0]
        dy = s2.start_pt[1] - s1.end_pt[1]
        pen_up_dist += math.hypot(dx, dy)

        # Joint displacement proxy
        d_th1 = s2.start_joints[0] - s1.end_joints[0]
        d_th2 = s2.start_joints[1] - s1.end_joints[1]
        total_joint_disp += abs(d_th1) + abs(d_th2)

        # Joint reversals
        if s1.end_dir_joint[0] * d_th1 < -1e-5:
            total_reversals += 1
        if s1.end_dir_joint[1] * d_th2 < -1e-5:
            total_reversals += 1
        if d_th1 * s2.start_dir_joint[0] < -1e-5:
            total_reversals += 1
        if d_th2 * s2.start_dir_joint[1] < -1e-5:
            total_reversals += 1

    total_dist = pen_down_dist + pen_up_dist
    efficiency = (pen_down_dist / total_dist * 100.0) if total_dist > 0.0 else 0.0

    min_det = min(det_j_values) if det_j_values else 0.0
    mean_det = float(np.mean(det_j_values)) if det_j_values else 0.0
    sing_penalty = sum(1.0 / (val + cfg.eps_singularity) for val in det_j_values)

    return {
        "pen_down_distance_cm": float(pen_down_dist),
        "pen_up_distance_cm": float(pen_up_dist),
        "total_distance_cm": float(total_dist),
        "pen_travel_efficiency_pct": float(efficiency),
        "total_joint_displacement_rad": float(total_joint_disp),
        "joint_reversals_count": int(total_reversals),
        "min_det_jacobian": float(min_det),
        "mean_det_jacobian": float(mean_det),
        "singularity_penalty_sum": float(sing_penalty),
    }
