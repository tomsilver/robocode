"""Tests for the PDDLStream per-instance baseline on Packing3D."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
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


def test_planning_matches_episode_rng_and_isolates_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning gets the evaluated reset's RNG state and a private scratch cwd."""
    env = _make_env()
    approach = _make_approach(env, timeout=1.0)
    observed: list[tuple[float, float, Path]] = []

    def _fake_plan(sim: Any, state: Any, max_time: float) -> None:
        del state, max_time
        # pylint: disable=protected-access
        backend = env._current_backend
        assert backend is not None
        eval_rng = backend._object_centric_env.np_random
        # pylint: enable=protected-access
        expected_rng = np.random.default_rng()
        expected_rng.bit_generator.state = copy.deepcopy(eval_rng.bit_generator.state)
        cwd = Path.cwd()
        observed.append((sim.np_random.uniform(), expected_rng.uniform(), cwd))
        Path("temp").mkdir()
        Path("statistics").mkdir()

    monkeypatch.setattr(
        "kinder_pddlstream_planning.packing3d.run.plan_packing3d", _fake_plan
    )
    try:
        for _ in range(2):
            result = approach.solve_instance(
                env=env,
                seed=123,
                budget_usd=0.0,
                output_subdir=tmp_path / "instance",
                count=1,
            )
            assert result.extras["plan_found"] is False
    finally:
        env.close()

    assert len(observed) == 2
    assert observed[0][0] == observed[0][1]
    assert observed[1][0] == observed[1][1]
    assert observed[0][0] == observed[1][0]
    assert all(cwd.parent == tmp_path / "instance" for _, _, cwd in observed)
    assert all(not cwd.exists() for _, _, cwd in observed)


def test_twin_client_is_closed_after_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every return path disconnects the per-instance twin's PyBullet client."""
    env = _make_env()
    approach = _make_approach(env, timeout=1.0)
    client_ids: list[int] = []

    def _fake_plan(sim: Any, state: Any, max_time: float) -> None:
        del state, max_time
        client_ids.append(sim.physics_client_id)
        assert p.isConnected(sim.physics_client_id)

    monkeypatch.setattr(
        "kinder_pddlstream_planning.packing3d.run.plan_packing3d", _fake_plan
    )
    try:
        approach.solve_instance(
            env=env,
            seed=0,
            budget_usd=0.0,
            output_subdir=tmp_path,
            count=1,
        )
        assert len(client_ids) == 1
        assert not p.isConnected(client_ids[0])
    finally:
        env.close()


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
