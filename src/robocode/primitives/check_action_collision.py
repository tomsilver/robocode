"""Collision-checking primitive."""

from __future__ import annotations

from typing import Any

import numpy as np

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.environments.maze_env import MazeEnv

# Action-index to (row-delta, col-delta) for MazeEnv.
_MAZE_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _maze_check(state: Any, action: Any) -> bool:
    """Optimised collision check for MazeEnv (pure-state, no stepping)."""
    r, c = state.agent
    dr, dc = _MAZE_DELTAS[int(action)]
    nr, nc = r + dr, c + dc
    return (not (0 <= nr < state.height and 0 <= nc < state.width)) or (
        (nr, nc) in state.obstacles
    )


def _kinder_check(env: Any, state: Any, action: Any) -> bool:
    """Collision check for KinderGeom2DEnv using kinder reference identity."""
    saved = env.get_state()
    env.set_state(state)
    inner = env._kinder_env._object_centric_env  # type: ignore[attr-defined]  # pylint: disable=protected-access
    ref_before = inner._current_state  # type: ignore[attr-defined]  # pylint: disable=protected-access
    env.step(np.array(action, dtype=np.float32))
    ref_after = inner._current_state  # type: ignore[attr-defined]  # pylint: disable=protected-access
    env.set_state(saved)
    return ref_after is ref_before  # type: ignore[no-any-return]


def _generic_check(env: Any, state: Any, action: Any) -> bool:
    """Fallback: step the env and compare states."""
    saved = env.get_state()
    env.set_state(state)
    next_state, _, _, _, _ = env.step(action)
    env.set_state(saved)
    return bool(np.array_equal(np.asarray(state), np.asarray(next_state)))


def check_action_collision(env: Any, state: Any, action: Any) -> bool:
    """Return True if taking *action* in *state* causes a collision."""
    if isinstance(env, MazeEnv):
        return _maze_check(state, action)
    if isinstance(env, KinderGeom2DEnv):
        return _kinder_check(env, state, action)
    return _generic_check(env, state, action)
