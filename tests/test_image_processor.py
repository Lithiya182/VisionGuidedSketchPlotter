"""
test_image_processor.py

Unit test suite for Phase 1 Computer Vision pipeline:
- Canvas letterboxing & aspect preservation
- Boundary, Canny, and Skeleton extraction modes
- Tiny-contour noise removal
- Douglas-Peucker simplification deviation tracking
- StrokeIR versioned JSON serialization and round-trip fidelity
- Verification of strict CV/IR separation (pixel coordinates on 480x480 canvas)
"""

import json
import math
import numpy as np
import pytest
import cv2

from src.config import VisionConfig
from src.vision.image_processor import (
    ImageProcessor,
    StrokeIR,
    compute_simplification_deviation,
    point_to_segment_distance,
)


@pytest.fixture
def processor():
    return ImageProcessor(VisionConfig(
        canvas_width=480,
        canvas_height=480,
        simplification_epsilon_px=1.5,
        min_contour_length_px=10.0,
        min_contour_area_px=10.0,
    ))


def test_point_to_segment_distance():
    """Verify perpendicular distance calculation to line segments."""
    # Point above middle of horizontal segment (0, 0) -> (10, 0)
    d = point_to_segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx(3.0)

    # Point past endpoint B
    d_past = point_to_segment_distance(14.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert d_past == pytest.approx(5.0)  # hypot(4, 3) = 5.0

    # Point before endpoint A
    d_before = point_to_segment_distance(-3.0, 4.0, 0.0, 0.0, 10.0, 0.0)
    assert d_before == pytest.approx(5.0)


def test_letterbox_padding_preserves_aspect_ratio(processor):
    """
    Test that rectangular images of arbitrary aspect ratios are scaled
    and centered on the 480x480 canvas without geometric distortion.
    """
    # 1. Wide landscape image: 800 x 400 (aspect 2:1)
    wide_img = np.full((400, 800, 3), 255, dtype=np.uint8)
    canvas_w, scale_w, pad_x_w, pad_y_w = processor.resize_and_pad(wide_img)
    assert canvas_w.shape == (480, 480, 3)
    assert scale_w == pytest.approx(480.0 / 800.0)
    assert pad_x_w == pytest.approx(0.0)
    assert pad_y_w == pytest.approx(120.0)

    # 2. Tall portrait image: 300 x 600 (aspect 1:2)
    tall_img = np.full((600, 300, 3), 255, dtype=np.uint8)
    canvas_t, scale_t, pad_x_t, pad_y_t = processor.resize_and_pad(tall_img)
    assert canvas_t.shape == (480, 480, 3)
    assert scale_t == pytest.approx(480.0 / 600.0)
    assert pad_x_t == pytest.approx(120.0)
    assert pad_y_t == pytest.approx(0.0)


def test_tiny_contour_removal(processor):
    """Verify that isolated salt-and-pepper noise specs are filtered out."""
    img = np.full((480, 480), 255, dtype=np.uint8)

    # Valid large square: 100 x 100
    cv2.rectangle(img, (100, 100), (200, 200), 0, -1)

    # Tiny noise specs (e.g. 2x2 and 3x3 pixels)
    cv2.rectangle(img, (20, 20), (22, 22), 0, -1)
    cv2.rectangle(img, (400, 50), (403, 53), 0, -1)

    ir = processor.process_image(img, mode="boundary")
    # Only the main square boundary should be extracted
    assert len(ir.strokes) == 1
    pts = np.array(ir.strokes[0])
    # The bounding box of the remaining contour must match the large square
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    assert min_x == pytest.approx(100.0, abs=2.0)
    assert max_x == pytest.approx(200.0, abs=2.0)
    assert min_y == pytest.approx(100.0, abs=2.0)
    assert max_y == pytest.approx(200.0, abs=2.0)


@pytest.mark.parametrize("mode", ["boundary", "canny", "skeleton"])
def test_all_extraction_modes_produce_valid_ir(processor, mode):
    """
    Test that boundary, canny, and skeleton modes all execute successfully,
    respect the 480x480 coordinate bounds, and produce valid StrokeIR.
    """
    # Create test image with circle and line
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(img, (150, 150), 60, (0, 0, 0), 4)
    cv2.line(img, (50, 150), (250, 150), (0, 0, 0), 3)

    ir = processor.process_image(img, mode=mode, epsilon=1.5)

    assert isinstance(ir, StrokeIR)
    assert ir.version == "1.0"
    assert ir.metadata.mode == mode
    assert len(ir.strokes) > 0

    stats = ir.metadata.simplification_stats
    assert stats.raw_points_count > 0
    assert stats.simplified_points_count > 0
    assert 0.0 <= stats.compression_ratio < 1.0

    # Ensure all coordinates strictly lie within the 480x480 canvas bounds
    for stroke in ir.strokes:
        for x, y in stroke:
            assert 0.0 <= x <= 480.0
            assert 0.0 <= y <= 480.0


def test_simplification_deviation_bounds(processor):
    """
    Test that the recorded simplification deviation does not exceed the
    Douglas-Peucker epsilon tolerance.
    """
    # Create a smooth circle curve
    img = np.full((480, 480), 255, dtype=np.uint8)
    cv2.circle(img, (240, 240), 120, 0, 2)

    eps = 2.0
    ir = processor.process_image(img, mode="canny", epsilon=eps)
    stats = ir.metadata.simplification_stats

    # Maximum deviation must be bounded by epsilon (with small numerical float tolerance)
    assert stats.max_deviation_px <= eps + 0.05
    assert 0.0 <= stats.mean_deviation_px <= stats.max_deviation_px
    assert stats.compression_ratio > 0.70  # Substantial point reduction


def test_versioned_json_roundtrip(processor, tmp_path):
    """
    Verify StrokeIR JSON serialization and deserialization fidelity.
    """
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), 3)

    ir = processor.process_image(img, mode="skeleton")
    json_path = tmp_path / "test_stroke_ir.json"

    # Serialize to JSON file
    ir.to_json(str(json_path))
    assert json_path.exists()

    # Deserialize
    reloaded_ir = StrokeIR.from_json(str(json_path))

    assert reloaded_ir.version == ir.version
    assert reloaded_ir.metadata.mode == ir.metadata.mode
    assert reloaded_ir.metadata.canvas_size == (480, 480)
    assert len(reloaded_ir.strokes) == len(ir.strokes)

    for orig_stroke, reloaded_stroke in zip(ir.strokes, reloaded_ir.strokes):
        assert len(orig_stroke) == len(reloaded_stroke)
        for (ox, oy), (rx, ry) in zip(orig_stroke, reloaded_stroke):
            assert ox == pytest.approx(rx, abs=1e-3)
            assert oy == pytest.approx(ry, abs=1e-3)
