"""Render-state primitive."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from kinder.envs.utils import render_2dstate
from matplotlib.axes import Axes
from numpy.typing import NDArray

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv

AxCallback = Callable[[Axes], Any]


def _render_kinder_with_callback(
    env: KinderGeom2DEnv,
    state: Any,
    ax_callback: AxCallback,
) -> NDArray[np.uint8]:
    """Render a KinderGeom2DEnv state with an arbitrary axes callback."""
    saved = env.get_state()
    env.set_state(state)

    inner = env._kinder_env._object_centric_env  # type: ignore[attr-defined]  # pylint: disable=protected-access
    render_input_state = inner._current_state.copy()  # type: ignore[attr-defined]  # pylint: disable=protected-access
    render_input_state.data.update(inner.initial_constant_state.data)

    config = inner.config
    img: NDArray[np.uint8] = render_2dstate(
        render_input_state,
        inner._static_object_body_cache,  # type: ignore[attr-defined]  # pylint: disable=protected-access
        config.world_min_x,
        config.world_max_x,
        config.world_min_y,
        config.world_max_y,
        config.render_dpi,
        ax_callback=ax_callback,
    )

    env.set_state(saved)
    return img


def render_state(
    env: Any,
    state: Any,
    ax_callback: AxCallback | None = None,
) -> NDArray[np.uint8]:
    """Render the given *state* as an RGB image without mutating the env.

    Parameters
    ----------
    state:
        Environment state (as returned by ``env.get_state()``).
    ax_callback:
        Optional callback that receives the matplotlib ``Axes`` and can draw
        arbitrary overlays (markers, lines, annotations, etc.).
        Only supported for ``KinderGeom2DEnv``.
    """
    if ax_callback is not None:
        if not isinstance(env, KinderGeom2DEnv):
            raise NotImplementedError(
                f"ax_callback is not supported for {type(env).__name__}"
            )
        return _render_kinder_with_callback(env, state, ax_callback)

    saved = env.get_state()
    env.set_state(state)
    frame: NDArray[np.uint8] = env.render()
    env.set_state(saved)
    return frame
