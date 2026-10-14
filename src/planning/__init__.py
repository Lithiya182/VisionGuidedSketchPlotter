"""
src/planning/__init__.py

Kinematics-aware path planning and stroke order optimization.
"""

from src.planning.stroke_optimizer import (
    OrientedStroke,
    create_oriented_stroke,
    compute_transition_cost,
    compute_tour_total_cost,
    order_strokes_naive,
    order_strokes_nn,
    order_strokes_2opt,
    order_strokes_exact_dp,
    evaluate_tour_metrics,
)

__all__ = [
    "OrientedStroke",
    "create_oriented_stroke",
    "compute_transition_cost",
    "compute_tour_total_cost",
    "order_strokes_naive",
    "order_strokes_nn",
    "order_strokes_2opt",
    "order_strokes_exact_dp",
    "evaluate_tour_metrics",
]
