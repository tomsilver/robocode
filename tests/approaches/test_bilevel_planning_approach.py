"""Tests for the bilevel planning per-instance baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robocode.approaches.bilevel_planning_approach import BilevelPlanningApproach
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv


def _make_env() -> KinderGeom2DEnv:
    # obstruction2d with 0 obstructions is reliably solvable and fast to plan.
    return KinderGeom2DEnv(
        "kinder/Obstruction2D-o0-v0",
        bilevel_env_name="obstruction2d",
        bilevel_env_model_kwargs={"num_obstructions": 0},
    )


def _make_approach(env: KinderGeom2DEnv) -> BilevelPlanningApproach:
    return BilevelPlanningApproach(
        env.action_space,
        env.observation_space,
        seed=0,
        primitives={},
        env=env,
        max_steps=1000,
        planning_timeout=30.0,
    )


def test_solve_instance_solves_and_reports_metrics(tmp_path: Path) -> None:
    """A solvable instance is solved and reports the planning/execution split."""
    env = _make_env()
    approach = _make_approach(env)
    result = approach.solve_instance(
        env=env, seed=0, budget_usd=0.0, output_subdir=tmp_path
    )
    assert result.solved
    assert result.cost_usd == 0.0
    assert result.crashed is False
    assert result.extras["plan_found"] is True
    assert result.extras["plan_length"] == result.num_steps
    for key in ("planning_time", "execution_time", "env_step_time"):
        assert result.extras[key] >= 0.0


def test_solve_instance_captures_frames_when_rendering(tmp_path: Path) -> None:
    """Render=True populates InstanceResult.frames so the runner can save a gif."""
    env = _make_env()
    approach = _make_approach(env)
    result = approach.solve_instance(
        env=env, seed=0, budget_usd=0.0, output_subdir=tmp_path, render=True
    )
    assert result.solved
    assert result.frames is not None
    assert result.num_steps is not None
    # One frame captured at reset plus one per executed step.
    assert len(result.frames) == result.num_steps + 1
    assert isinstance(result.frames[0], np.ndarray)


def test_solve_instance_skips_frames_without_rendering(tmp_path: Path) -> None:
    """Without render, no frames are captured (frames is None)."""
    env = _make_env()
    approach = _make_approach(env)
    result = approach.solve_instance(
        env=env, seed=0, budget_usd=0.0, output_subdir=tmp_path
    )
    assert result.frames is None


def test_models_are_built_once(tmp_path: Path) -> None:
    """SesameModels depend only on the fixed env, so they are cached across seeds."""
    env = _make_env()
    approach = _make_approach(env)
    approach.solve_instance(env=env, seed=0, budget_usd=0.0, output_subdir=tmp_path)
    models = approach._models  # pylint: disable=protected-access
    assert models is not None
    approach.solve_instance(env=env, seed=1, budget_usd=0.0, output_subdir=tmp_path)
    assert approach._models is models  # pylint: disable=protected-access


def test_planning_failure_is_unsolved_not_crashed(tmp_path: Path) -> None:
    """A planning timeout scores as an unsolved (not crashed) attempt.

    obstruction2d with 2 obstructions plus a tiny timeout reliably yields no plan (the
    degradation path). It must return solved=False, crashed=False, no episode metrics,
    and still report the planning time it spent failing.
    """
    env = KinderGeom2DEnv(
        "kinder/Obstruction2D-o2-v0",
        bilevel_env_name="obstruction2d",
        bilevel_env_model_kwargs={"num_obstructions": 2},
    )
    approach = BilevelPlanningApproach(
        env.action_space,
        env.observation_space,
        seed=0,
        primitives={},
        env=env,
        max_steps=1000,
        planning_timeout=0.001,
    )
    result = approach.solve_instance(
        env=env, seed=0, budget_usd=0.0, output_subdir=tmp_path
    )
    assert result.solved is False
    assert result.crashed is False
    assert result.cost_usd == 0.0
    assert result.total_reward is None
    assert result.num_steps is None
    assert result.extras["plan_found"] is False
    assert result.extras["plan_length"] == 0
    assert result.extras["planning_time"] >= 0.0


def test_planning_is_deterministic(tmp_path: Path) -> None:
    """Same seed, same models -> same plan (the planner is seeded)."""
    env = _make_env()
    approach = _make_approach(env)
    r1 = approach.solve_instance(
        env=env, seed=3, budget_usd=0.0, output_subdir=tmp_path
    )
    r2 = approach.solve_instance(
        env=env, seed=3, budget_usd=0.0, output_subdir=tmp_path
    )
    assert r1.solved == r2.solved
    assert r1.num_steps == r2.num_steps
    assert r1.total_reward == r2.total_reward


def test_missing_mapping_fails_loudly(tmp_path: Path) -> None:
    """An env without the bilevel mapping raises rather than planning silently."""
    env = KinderGeom2DEnv("kinder/Obstruction2D-o0-v0")  # no bilevel_env_name
    approach = _make_approach(env)
    with pytest.raises(AssertionError, match="bilevel_env_name"):
        approach.solve_instance(env=env, seed=0, budget_usd=0.0, output_subdir=tmp_path)


def test_train_is_not_supported() -> None:
    """The per-instance baseline does not train a generalized policy."""
    env = _make_env()
    approach = _make_approach(env)
    with pytest.raises(NotImplementedError):
        approach.train()
