"""Collision-checking primitive."""

from __future__ import annotations

from typing import Any

import numpy as np
from kinder.envs.kinematic2d.base_env import ObjectCentricKinematic2DRobotEnv
from kinder.envs.kinematic2d.object_types import CRVRobotType
from kinder.envs.kinematic2d.utils import (
    get_suctioned_objects,
    snap_suctioned_objects,
)
from kinder.envs.utils import state_2d_has_collision
from prpl_utils.utils import wrap_angle
from relational_structs import ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.environments.maze_env import MazeEnv
from robocode.environments.variable_object_count_env import VariableObjectCountEnv

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


def _kinder_collision_after_action(
    inner: ObjectCentricKinematic2DRobotEnv,
    ocs: ObjectCentricState,
    action: Any,
) -> bool:
    """True iff applying ``action`` in ``ocs`` collides, by kinder's own step predicate.

    Mirrors ``ObjectCentricKinematic2DRobotEnv.step`` up to its collision test, without
    running the reward/observation work and without mutating ``inner`` or ``ocs``. Reuses
    the backend's warm ``_static_object_body_cache`` so static geometry is built once per
    episode rather than rebuilt on every call.
    """
    # pylint: disable=protected-access
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    if act.shape != (5,):
        raise ValueError(f"Expected action of shape (5,), got {act.shape}")
    dx, dy, dtheta, darm, vac = act
    robots = [o for o in ocs if o.is_instance(CRVRobotType)]
    if len(robots) != 1:
        raise ValueError(f"Expected exactly one CRV robot, found {len(robots)}")
    robot = robots[0]

    state = ocs.copy()
    state.set(robot, "x", state.get(robot, "x") + float(dx))
    state.set(robot, "y", state.get(robot, "y") + float(dy))
    state.set(robot, "theta", wrap_angle(state.get(robot, "theta") + float(dtheta)))
    min_arm = state.get(robot, "base_radius")
    max_arm = state.get(robot, "arm_length")
    new_arm = float(
        np.clip(state.get(robot, "arm_joint") + float(darm), min_arm, max_arm)
    )
    state.set(robot, "arm_joint", new_arm)
    state.set(robot, "vacuum", float(vac))

    # Suction and contact are read from the pre-action state, matching step()'s ordering.
    suctioned = get_suctioned_objects(ocs, robot)
    snap_suctioned_objects(state, robot, suctioned)
    state, moved = inner.get_objects_to_move(state, suctioned)

    moving = {robot} | {o for o, _ in suctioned} | {o for o, _ in moved}
    full_state = inner.get_state_with_constant_objects(state)
    obstacles = set(full_state) - moving
    return state_2d_has_collision(
        full_state, moving, obstacles, inner._static_object_body_cache
    )


def check_action_collision(env: Any, state: Any, action: Any) -> bool:
    """Return True if taking *action* in *state* causes a collision."""
    # pylint: disable=protected-access
    if isinstance(env, MazeEnv):
        return _maze_check(state, action)
    if isinstance(env, VariableObjectCountEnv):
        inner = env._backend_for(env.infer_count(state))._object_centric_env
        ocs = state
    elif isinstance(env, KinderGeom2DEnv):
        box_space = env._kinder_env.observation_space
        assert isinstance(box_space, ObjectCentricBoxSpace)
        ocs = box_space.devectorize(np.asarray(state, dtype=np.float32))
        inner = env._kinder_env._object_centric_env
    else:
        raise NotImplementedError(
            f"check_action_collision supports MazeEnv and the kinematic2d kinder "
            f"envs (KinderGeom2DEnv, VariableObjectCountEnv), not "
            f"{type(env).__name__}. Collision checking for kinder 3D is not "
            f"implemented yet."
        )
    assert isinstance(inner, ObjectCentricKinematic2DRobotEnv)
    return _kinder_collision_after_action(inner, ocs, action)
