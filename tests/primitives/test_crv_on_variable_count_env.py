"""Crv motion/grasp primitives on states from a variable-object-count env.

The crv planners are count-agnostic -- they iterate whatever objects the state holds.
These exercise them on a real multi-object ObjectCentricState pulled from
VariableObjectCountEnv (robot + target + surface + several obstructions), rather than a
hand-built one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.primitives import build_primitives
from robocode.primitives.crv_motion_planning_grasp import (
    SuctionFailedEmptySpaceError,
    SuctionFailedNoCollisionFreePathError,
)

_OBSTRUCTION2D: dict[str, Any] = {
    "constant_object_env_path": "kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",
    "count_kwarg": "num_obstructions",
    "count_object_prefix": "obstruction",
    "design_counts": [1, 2, 3],
    "eval_counts": [1, 2, 3],
    "bilevel_env_name": "obstruction2d",
}


def _multi_object_state(env: VariableObjectCountEnv):
    """A 3-obstruction instance, so the planners face several objects."""
    state, _ = env.reset(seed=0, options={"object_count": 3})
    n = sum(1 for nm in state.get_object_names() if nm.startswith("obstruction"))
    assert n == 3
    return state


def test_plan_crv_actions_on_variable_count_state() -> None:
    """Base-motion planning succeeds on a multi-object variable-count state."""
    env = VariableObjectCountEnv(**_OBSTRUCTION2D)
    crv = build_primitives(env, ["crv_motion_planning"])["crv_motion_planning"]
    state = _multi_object_state(env)
    robot = state.get_object_from_name("robot")
    goal = crv.CRVConfig(
        float(state.get(robot, "x")),
        float(state.get(robot, "y")),
        float(state.get(robot, "theta")),
    )
    actions = crv.plan_crv_actions(state, goal, carrying=False, seed=0)
    # A plan computed over all the obstructions, not a crash on the state structure.
    assert actions is not None
    assert all(isinstance(a, np.ndarray) for a in actions)
    env.close()


def test_plan_crv_grasp_on_variable_count_state() -> None:
    """Grasp planning reaches a geometric outcome on a multi-object variable-count
    state -- a suction sequence, or a crv-typed planning failure when obstructions
    block the approach -- rather than choking on the state structure."""
    env = VariableObjectCountEnv(**_OBSTRUCTION2D)
    prims = build_primitives(env, ["crv_motion_planning_grasp"])
    grasp = prims["crv_motion_planning_grasp"]
    state = _multi_object_state(env)
    # Either a suction sequence or a crv-typed failure is acceptable; any other
    # exception (e.g. mishandling the state) propagates and fails the test.
    try:
        waypoints = grasp.plan_crv_grasp(
            state,
            "target_block",
            grasp.RelativeGraspPose(x=-0.56, y=0.0, theta=0.0),
            0.5,
            seed=0,
        )
        assert waypoints  # found a grasp
    except (SuctionFailedEmptySpaceError, SuctionFailedNoCollisionFreePathError):
        pass
    env.close()
