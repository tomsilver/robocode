"""Tests for the PDDLStream per-instance baseline on PR2Packed.

The plan-to-action conversion is exercised directly, without planning, because the
two defects that cost the most to find were both there: a grasp read as a release,
and a densely interpolated path spent one action per node. Those tests need neither
the stock PDDLStream tree nor a physics client and run in milliseconds.

The end-to-end tests do plan, so they need the ``pddlstream`` extra. They do not
need ``kinder_pddlstream_planning``: PR2Packed's domain comes from the stock tree,
not from the Packing3D package, so this module skips independently of the
Packing3D tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robocode.planners.pddlstream_pr2packed_worker import gripper_closes

# The helpers above are pure; everything below drives a real planner and env.
pytest.importorskip("pddlstream")

# pylint: disable=wrong-import-position
from robocode.approaches.pddlstream_planning_approach import (  # noqa: E402
    PDDLStreamPlanningApproach,
)
from robocode.environments.pr2_tamp_variable_count_env import (  # noqa: E402
    PR2PackedVariableCountEnv,
)
from robocode.planners.pddlstream_pr2packed import (  # noqa: E402
    PR2PackedPDDLStreamPlanner,
)

# The gripper joint's max limit for this benchmark's PR2, and the width a top grasp
# on a 0.07m block closes to. They are close enough together that a "half the limit"
# split misreads the grasp, which is what makes this worth pinning.
_GRIPPER_MAX = 0.548
_GRASP_WIDTH = 0.4298039215686276


def _make_env() -> PR2PackedVariableCountEnv:
    return PR2PackedVariableCountEnv(design_counts=[1], eval_counts=[1])


def _make_approach(
    env: PR2PackedVariableCountEnv, timeout: float
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


def test_grasp_width_close_to_the_limit_is_still_a_close() -> None:
    """A grasp commands a width below the limit, and may be only just below it."""
    assert gripper_closes((_GRASP_WIDTH,) * 4, _GRIPPER_MAX) is True
    # The regression this pins: the grasp width is above half the limit, so a
    # "below half" test calls it a release and the plan never picks anything up.
    assert _GRASP_WIDTH > _GRIPPER_MAX / 2.0


def test_opening_commands_the_limit_exactly_and_is_not_a_close() -> None:
    """Releasing commands the joint's max limit, as a scalar rather than a tuple."""
    assert gripper_closes(_GRIPPER_MAX, _GRIPPER_MAX) is False


def test_subsample_thins_a_dense_path_to_the_action_resolution() -> None:
    """A densely interpolated path costs one action per node unless it is thinned.

    Upstream returns nodes a few thousandths of a radian apart; the environment accepts
    a delta of 0.2 per joint, so following every node spends many actions retracing one
    hop.
    """
    circular = np.zeros(7, dtype=bool)
    dense = [np.full(7, 0.004 * i) for i in range(250)]
    # pylint: disable=protected-access
    kept = PR2PackedPDDLStreamPlanner._subsample(dense, circular)
    assert len(kept) < len(dense) / 10
    # The regularly kept nodes are a full action apart, so each costs one step. The
    # endpoint is appended whatever its spacing, so it is excluded from that check.
    for before, after in zip(kept[:-1], kept[1:-1]):
        assert np.max(np.abs(after - before)) >= 0.2 - 1e-9
    assert np.array_equal(kept[-1], dense[-1])


def test_subsample_keeps_the_endpoint_of_a_short_path() -> None:
    """A hop shorter than one action still has to arrive at its endpoint."""
    circular = np.zeros(3, dtype=bool)
    short = [np.zeros(3), np.full(3, 0.01)]
    # pylint: disable=protected-access
    kept = PR2PackedPDDLStreamPlanner._subsample(short, circular)
    assert np.array_equal(kept[-1], short[-1])


def test_segments_split_trajectories_at_gripper_commands() -> None:
    """Thinning is only sound within one trajectory.

    Consecutive commands are separated by a gripper action that has to land at the exact
    configuration the plan grasps or releases from, so the runs on either side must not
    be merged and thinned together.
    """
    steps: list[dict[str, Any]] = [
        {"kind": "arm", "values": [0.0] * 7},
        {"kind": "arm", "values": [0.1] * 7},
        {"kind": "gripper", "close": True},
        {"kind": "arm", "values": [0.2] * 7},
        {"kind": "base", "values": [1.0, 1.0, 0.0]},
    ]
    # pylint: disable=protected-access
    segments = list(PR2PackedPDDLStreamPlanner._segments(steps))
    assert [kind for kind, _ in segments] == ["arm", "gripper", "arm", "base"]
    assert len(segments[0][1]) == 2, "the pre-grasp trajectory must stay intact"


def test_gripper_steps_become_open_and_close_actions() -> None:
    """A close drives the gripper slot negative and a release drives it positive."""
    env = _make_env()
    try:
        env.reset(seed=0, options={"object_count": 1})
        planner = PR2PackedPDDLStreamPlanner(env.current_backend)
        actions = list(
            planner.actions(
                [
                    {"kind": "gripper", "close": True},
                    {"kind": "gripper", "close": False},
                ]
            )
        )
    finally:
        env.close()
    assert len(actions) == 2
    assert actions[0][10] == -1.0
    assert actions[1][10] == 1.0
    # A gripper command moves nothing else.
    assert np.allclose(actions[0][:10], 0.0)


def test_solve_instance_plans_and_executes_one_block(tmp_path: Path) -> None:
    """A one-block instance is planned upstream and the plan solves the episode."""
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
    assert result.solved


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
    assert result.extras["plan_length"] == 0
    assert result.num_steps is None


def test_execution_stops_at_the_step_budget(tmp_path: Path) -> None:
    """A plan longer than the episode's budget is cut off, not run past it."""
    env = _make_env()
    approach = _make_approach(env, timeout=120.0)
    try:
        result = approach.solve_instance(
            env=env,
            seed=0,
            budget_usd=0.0,
            output_subdir=tmp_path,
            count=1,
            max_steps=5,
        )
    finally:
        env.close()
    assert result.crashed is False
    assert result.extras["plan_found"] is True
    assert result.num_steps == 5
    assert result.solved is False
