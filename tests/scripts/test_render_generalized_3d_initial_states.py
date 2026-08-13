"""Tests for the generalized 3D initial-state GIF renderer."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).parents[2] / "scripts" / "render_generalized_3d_initial_states.py"
)
_SPEC = importlib.util.spec_from_file_location("generalized_3d_renderer", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_annotate = getattr(_MODULE, "_annotate")
_render_config = getattr(_MODULE, "_render_config")


def test_render_config_requires_a_seed(tmp_path: Path) -> None:
    """An empty train-and-eval request fails before any environment is built."""
    with pytest.raises(ValueError, match="At least one"):
        _render_config("unused", tmp_path, 0, 0, 42, 5, 480)


def test_annotate_resizes_frame() -> None:
    """The render-width option changes the stored frame resolution."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    annotated = _annotate(frame, "test", "eval", 1, 1, 42, 3, 480)
    assert annotated.shape == (240, 480, 3)
