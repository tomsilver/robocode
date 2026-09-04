"""Tests for the PDDLStream per-instance baseline on Packing3D."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pddlstream")
pytest.importorskip("kinder_pddlstream_planning")

# pylint: disable=wrong-import-position
from robocode.approaches.pddlstream_planning_approach import (  # noqa: E402
    PDDLStreamPlanningApproach,
)
from robocode.environments.variable_object_count_env import (  # noqa: E402
    VariableObjectCountEnv,
)


def _make_env() -> VariableObjectCountEnv:
    return VariableObjectCountEnv(
        constant_object_env_path="kinder.envs.kinematic3d.packing3d:Packing3DEnv",
        count_kwarg="num_parts",
        count_object_prefix="part",
        design_counts=[1],
        eval_counts=[1],
    )


def _make_approach(
    env: VariableObjectCountEnv, timeout: float
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


def test_solve_instance_plans_and_executes_one_part(tmp_path: Path) -> None:
    """A one-part instance is planned and the plan replays on the evaluated env.

    The twin sim and the evaluated env must stay in step for the whole rollout.
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
    assert result.extras["twin_divergences"] == 0
    assert result.extras["plan_length"] == result.num_steps
    assert result.solved


def test_no_plan_within_timeout_is_unsolved_not_crashed(tmp_path: Path) -> None:
    """An expired planning budget scores as unsolved with plan_found False."""
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


def test_other_envs_are_refused() -> None:
    """The baseline covers the variable-count Packing3D family only."""
    env = VariableObjectCountEnv(
        constant_object_env_path=(
            "kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv"
        ),
        count_kwarg="num_obstructions",
        count_object_prefix="obstruction",
        design_counts=[0],
        eval_counts=[0],
    )
    approach = _make_approach(env, timeout=1.0)
    try:
        with pytest.raises(NotImplementedError, match="Packing3D"):
            approach.solve_instance(
                env=env, seed=0, budget_usd=0.0, output_subdir=Path("."), count=0
            )
    finally:
        env.close()


def test_train_is_not_supported() -> None:
    """Per-instance approaches do not train."""
    env = _make_env()
    try:
        with pytest.raises(NotImplementedError):
            _make_approach(env, timeout=1.0).train()
    finally:
        env.close()
