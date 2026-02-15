"""Render-state primitive."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def render_state(env: Any, state: Any) -> NDArray[np.uint8]:
    """Render the given *state* as an RGB image without mutating the env."""
    saved = env.get_state()
    env.set_state(state)
    frame: NDArray[np.uint8] = env.render()
    env.set_state(saved)
    return frame
