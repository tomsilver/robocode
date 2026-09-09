"""PDDLStream planning on the PR2 ``blocked`` environment.

The planner's pure helpers are covered by ``test_pddlstream_pr2packed``; what is
checked here is that the shared planner path handles the second stock scene: the
payload names ``blocked`` and counts spares rather than blocks, and the plans that
come back execute on the environment, both with and without a spare.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pddlstream")

# pylint: disable=wrong-import-position
from robocode.approaches.pddlstream_planning_approach import (  # noqa: E402
    PDDLStreamPlanningApproach,
)
from robocode.environments.pr2_tamp_blocked_variable_count_env import (  # noqa: E402
    PR2BlockedVariableCountEnv,
)
from robocode.planners.pddlstream_pr2packed import (  # noqa: E402
    PR2PackedPDDLStreamPlanner,
)


def _make_env(base_steps: int = 700) -> PR2BlockedVariableCountEnv:
    """Spare counts 0 and 1 with a generous step budget (see the packed tests)."""
    return PR2BlockedVariableCountEnv(
        design_counts=[0, 1], eval_counts=[0, 1], base_steps=base_steps
    )


def _make_approach(
    env: PR2BlockedVariableCountEnv, timeout: float
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


def test_unknown_problem_is_rejected() -> None:
    """The planner only knows the stock scenes it can rebuild."""
    with pytest.raises(ValueError, match="problem must be one of"):
        PR2PackedPDDLStreamPlanner(object(), problem="stacked")  # type: ignore[arg-type]


def test_instance_payload_counts_spares_and_lists_every_movable() -> None:
    """A blocked payload names its scene, counts spares, and poses every movable."""
    env = _make_env()
    try:
        env.reset(seed=0, options={"object_count": 1})
        planner = PR2PackedPDDLStreamPlanner(env.current_backend, problem="blocked")
        payload = planner._instance(  # pylint: disable=protected-access
            max_time=1.0, seed=0
        )
    finally:
        env.close()
    assert payload["problem"] == "blocked"
    assert payload["count"] == 1
    # The penned green, one spare, and the red blocker, in observation order.
    assert len(payload["block_poses"]) == 3
    # The pen's three walls travel with the instance too.
    assert len(payload["wall_poses"]) == 3
    assert len(payload["base"]) == 3
    assert len(payload["arm"]) == 7


def _solve_all(count: int, seeds: tuple[int, ...], tmp_path: Path) -> list[bool]:
    env = _make_env()
    approach = _make_approach(env, timeout=120.0)
    solved = []
    try:
        for seed in seeds:
            result = approach.solve_instance(
                env=env, seed=seed, budget_usd=0.0, output_subdir=tmp_path, count=count
            )
            assert result.crashed is False
            assert result.cost_usd == 0.0
            assert result.extras["plan_found"] is True, f"seed={seed}: no plan"
            assert result.extras["object_count"] == count
            assert result.num_steps == result.extras["plan_length"]
            assert result.num_steps <= result.extras["step_budget"]
            solved.append(bool(result.solved))
    finally:
        env.close()
    return solved


def test_solve_instance_moves_the_blocker_when_there_is_no_spare(
    tmp_path: Path,
) -> None:
    """At count 0 the only green block is penned, so every plan clears the blocker.

    As in the packed tests, solving is asserted over the set of seeds: replay is open-
    loop, and a grasp the upstream model accepts can be refused here.
    """
    solved = _solve_all(0, (0, 1, 2), tmp_path)
    assert any(solved), f"no zero-spare instance was solved (solved={solved})"


def test_solve_instance_with_a_spare(tmp_path: Path) -> None:
    """With a spare on the far table the planner may fetch it instead."""
    solved = _solve_all(1, (0, 1), tmp_path)
    assert any(solved), f"no one-spare instance was solved (solved={solved})"


def test_no_plan_within_timeout_is_unsolved_not_crashed(tmp_path: Path) -> None:
    """An expired planning budget scores as unsolved rather than crashing."""
    env = _make_env()
    approach = _make_approach(env, timeout=0.01)
    try:
        result = approach.solve_instance(
            env=env, seed=0, budget_usd=0.0, output_subdir=tmp_path, count=0
        )
    finally:
        env.close()
    assert result.solved is False
    assert result.crashed is False
    assert result.extras["plan_found"] is False
    assert result.num_steps is None
