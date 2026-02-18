"""Render-state primitive."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from kinder.envs.utils import render_2dstate
from numpy.typing import NDArray

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv


def _draw_labels(labels: list[tuple[float, float, str]]) -> Any:
    """Return an ax_callback that draws labeled markers."""

    def _callback(ax: plt.Axes) -> None:
        for wx, wy, text in labels:
            ax.plot(
                wx,
                wy,
                "o",
                markersize=7,
                markerfacecolor="red",
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=999,
            )
            ax.annotate(
                text,
                (wx, wy),
                textcoords="offset points",
                xytext=(8, -8),
                fontsize=8,
                color="white",
                bbox={"boxstyle": "round,pad=0.2", "fc": "black", "alpha": 0.6},
                zorder=1000,
            )

    return _callback


def _render_kinder_with_labels(
    env: KinderGeom2DEnv,
    state: Any,
    labels: list[tuple[float, float, str]],
) -> NDArray[np.uint8]:
    """Render a KinderGeom2DEnv state with labels in world coordinates."""
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
        ax_callback=_draw_labels(labels),
    )

    env.set_state(saved)
    return img


def render_state(
    env: Any,
    state: Any,
    labels: list[tuple[float, float, str]] | None = None,
) -> NDArray[np.uint8]:
    """Render the given *state* as an RGB image without mutating the env.

    Parameters
    ----------
    state:
        Environment state (as returned by ``env.get_state()``).
    labels:
        Optional list of ``(world_x, world_y, text)`` tuples. Each label is
        drawn as a red marker with a text annotation at the given world
        coordinates. Only supported for ``KinderGeom2DEnv``.
    """
    if labels:
        if not isinstance(env, KinderGeom2DEnv):
            raise NotImplementedError(
                f"Labels are not supported for {type(env).__name__}"
            )
        return _render_kinder_with_labels(env, state, labels)

    saved = env.get_state()
    env.set_state(state)
    frame: NDArray[np.uint8] = env.render()
    env.set_state(saved)
    return frame
