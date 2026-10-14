"""
vision package

Computer vision image ingestion, contour extraction, skeletonization,
and Intermediate Representation (StrokeIR) generation.
"""

from src.vision.image_processor import (
    ImageProcessor,
    StrokeIR,
    ContourMetadata,
    SimplificationStats,
    compute_simplification_deviation,
)

__all__ = [
    "ImageProcessor",
    "StrokeIR",
    "ContourMetadata",
    "SimplificationStats",
    "compute_simplification_deviation",
]
