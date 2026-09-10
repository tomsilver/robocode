"""Tests for the PDDLStream per-instance baseline on ``rovers``.

The plan-to-action conversion is exercised without planning, because the defect that
cost the most to find was there: the upstream plan's motions are trajectories, and
keeping only their endpoints drives straight through whatever they were routed around.

The end-to-end tests plan, so they need the ``pddlstream`` extra. They do not need
``kinder_pddlstream_planning``: the rovers domain comes from the stock tree, so this
module skips independently of the Packing3D tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pddlstream")

# pylint: disable=wrong-import-position
from robocode.approaches.pddlstream_planning_approach import (  # noqa: E402
    PDDLStreamPlanningApproach,
)
from robocode.environments.rovers_env import NOOP, SAMPLE, RoversEnv  # noqa: E402
from robocode.environments.rovers_variable_count_env import (  # noqa: E402
    RoversVariableCountEnv,
)
from robocode.planners.pddlstream_rovers import (  # noqa: E402
    RoversPDDLStreamPlanner,
)


def _make_env() -> RoversVariableCountEnv:
    return RoversVariableCountEnv(design_counts=[1], eval_counts=[1])


def _make_approach(
    env: RoversVariableCountEnv, timeout: float
) -> PDDLStreamPlanningApproach:
    return PDDLStreamPlanningApproach(
        env.action_space,
        env.observation_space,
        seed=0,
        primitives={},
        env=env,
        max_steps=1000,
        eval_timeout=timeout,
    )


def test_every_waypoint_of_a_motion_is_followed() -> None:
    """A plan's motion is a route, not a destination.

    Upstream's motion streams check a whole trajectory against the obstacles. Following
    only its final configuration means driving straight at that configuration and into
    whatever the trajectory went around, which is exactly what happened: rovers wedged
    against a pillar and burned the rest of the episode pushing into it.
    """
    env = RoversEnv(num_objectives=1)
    try:
        env.reset(seed=0)
        planner = RoversPDDLStreamPlanner(env)
        start = env.rover_conf(0)
        # Two hops that are not collinear, so cutting the corner is detectable.
        steps = [
            {
                "rover": 0,
                "waypoints": [
                    [start[0], start[1] + 0.4, 0.0],
                    [start[0] + 0.4, start[1] + 0.4, 0.0],
                ],
                "op": "noop",
            }
        ]
        visited = []
        for action in planner.actions(steps):
            env.step(action)
            visited.append(env.rover_conf(0)[:2].copy())
        assert visited, "no actions were produced"
        # The rover passed near the intermediate waypoint rather than heading straight
        # for the last one.
        corner = np.array([start[0], start[1] + 0.4])
        assert min(float(np.linalg.norm(v - corner)) for v in visited) < 0.15
    finally:
        env.close()


def test_operators_are_emitted_in_their_own_band() -> None:
    """A step's operator has to decode back to the operator the plan asked for."""
    env = RoversEnv(num_objectives=1)
    try:
        env.reset(seed=0)
        planner = RoversPDDLStreamPlanner(env)
        actions = list(planner.actions([{"rover": 1, "waypoints": [], "op": "sample"}]))
        assert len(actions) == 1
        # pylint: disable=protected-access
        assert env._operator(actions[0][7]) == SAMPLE
        # The other rover is told to do nothing, not left at whatever zero decodes to.
        assert env._operator(actions[0][3]) == NOOP
    finally:
        env.close()


def test_solve_instance_plans_and_executes(tmp_path: Path) -> None:
    """A one-objective instance is planned upstream and the plan runs.

    Solving is not asserted: the upstream plan is optimal in its own action count, not
    in the environment's steps, and it routinely needs more of them than the budget
    allows. What has to hold is that a plan is found and replayed inside the budget
    without crashing.
    """
    env = _make_env()
    approach = _make_approach(env, timeout=120.0)
    try:
        result = approach.solve_instance(
            env=env, seed=0, budget_usd=0.0, output_subdir=tmp_path, count=1
        )
    finally:
        env.close()
    assert result.crashed is False
    assert result.cost_usd == 0.0
    assert result.extras["plan_found"] is True, "planner found no plan"
    assert result.extras["object_count"] == 1
    assert result.num_steps == result.extras["plan_length"]
    assert result.num_steps <= result.extras["step_budget"]


def test_no_plan_within_timeout_is_unsolved_not_crashed(tmp_path: Path) -> None:
    """An expired planning budget scores as unsolved rather than crashing."""
    env = _make_env()
    approach = _make_approach(env, timeout=0.01)
    try:
        result = approach.solve_instance(
            env=env, seed=0, budget_usd=0.0, output_subdir=tmp_path, count=1
        )
    finally:
        env.close()
    assert result.solved is False
    assert result.crashed is False
    assert result.extras["plan_found"] is False
    assert result.num_steps is None


def test_other_families_are_still_refused() -> None:
    """Dispatching rovers must not have taken over the families it does not cover."""
    # pylint: disable=import-outside-toplevel
    from robocode.environments.variable_object_count_env import VariableObjectCountEnv

    env: Any = VariableObjectCountEnv(
        constant_object_env_path=(
            "kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv"
        ),
        count_kwarg="num_obstructions",
        count_object_prefix="obstruction",
        design_counts=[0],
        eval_counts=[0],
    )
    approach = PDDLStreamPlanningApproach(
        env.action_space,
        env.observation_space,
        seed=0,
        primitives={},
        env=env,
        max_steps=10,
        eval_timeout=1.0,
    )
    try:
        with pytest.raises(NotImplementedError):
            approach.solve_instance(
                env=env, seed=0, budget_usd=0.0, output_subdir=Path("."), count=0
            )
    finally:
        env.close()
