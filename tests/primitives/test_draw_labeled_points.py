"""Tests for draw_labeled_points primitive."""

from __future__ import annotations

import numpy as np

from robocode.primitives.draw_labeled_points import draw_labeled_points


def _make_image(h: int = 100, w: int = 150) -> np.ndarray:
    """Create a solid test image."""
    return np.full((h, w, 3), fill_value=80, dtype=np.uint8)


def test_empty_points_returns_copy() -> None:
    """Empty points list returns a copy, not the same object."""
    img = _make_image()
    result = draw_labeled_points(img, [])
    assert result is not img
    np.testing.assert_array_equal(result, img)


def test_output_shape_and_dtype() -> None:
    """Output shape and dtype match input."""
    img = _make_image(64, 96)
    result = draw_labeled_points(img, [(10, 10, "A")])
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_drawing_modifies_image() -> None:
    """Drawing a point produces a different image."""
    img = _make_image()
    result = draw_labeled_points(img, [(50, 50, "center")])
    assert not np.array_equal(result, img)


def test_marker_visible_near_point() -> None:
    """Marker is visible in the region around the specified location."""
    img = _make_image(200, 200)
    result = draw_labeled_points(img, [(100, 100, "X")])
    region = result[90:110, 90:110]
    original_region = img[90:110, 90:110]
    assert not np.array_equal(region, original_region)


def test_multiple_points() -> None:
    """Multiple points can be drawn."""
    img = _make_image(200, 200)
    points = [(30, 30, "A"), (170, 170, "B")]
    result = draw_labeled_points(img, points)
    assert not np.array_equal(result, img)
    assert result.shape == img.shape


def test_input_not_mutated() -> None:
    """Input array is not mutated."""
    img = _make_image()
    original = img.copy()
    draw_labeled_points(img, [(25, 25, "test")])
    np.testing.assert_array_equal(img, original)


def test_custom_marker_radius_and_font_size() -> None:
    """Custom marker_radius and font_size are accepted."""
    img = _make_image()
    result = draw_labeled_points(img, [(50, 50, "big")], marker_radius=15, font_size=24)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, img)
