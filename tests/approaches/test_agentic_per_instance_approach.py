"""Tests for agentic_per_instance_approach.py."""

from functools import partial

import pytest

from robocode.approaches.agentic_approach import AgenticApproach
from robocode.approaches.agentic_per_instance_approach import (
    AgenticPerInstanceApproach,
)
from robocode.environments.maze_env import MazeEnv
from robocode.primitives.check_action_collision import check_action_collision
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.sandbox_types import SandboxResult

_GENERATED_RETURNS_ZERO = (
    "class GeneratedApproach:\n"
    "    def __init__(self, action_space, observation_space, primitives):\n"
    "        pass\n"
    "    def reset(self, state, info):\n"
    "        pass\n"
    "    def get_action(self, state):\n"
    "        return 0\n"
)


def _make_approach(tmp_path, **kwargs):
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticPerInstanceApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=7,
        primitives={"check_action_collision": partial(check_action_collision, env)},
        backend=DEFAULT_BACKEND_CFG,
        max_steps=50,
        output_dir=str(tmp_path),
        **kwargs,
    )
    return env, approach


def test_per_instance_flag_distinguishes_lifecycles():
    """The per-instance approach advertises per_instance; the generalized one does
    not."""
    assert AgenticPerInstanceApproach.per_instance is True
    assert AgenticApproach.per_instance is False


def test_train_raises(tmp_path):
    """Train() is unused for per-instance approaches and fails loudly."""
    _env, approach = _make_approach(tmp_path)
    with pytest.raises(NotImplementedError, match="solve_instance"):
        approach.train()


def test_solve_instance_targets_seed_and_scores_program(tmp_path, monkeypatch):
    """solve_instance prompts for the target seed, then scores the written program."""
    env, approach = _make_approach(tmp_path)

    def fake_run(*, sandbox_dir, prompt, system_prompt, max_budget_usd, init_files):
        del system_prompt, init_files
        # The prompt targets the specific eval seed and carries budget stewardship.
        assert "env.reset(seed=99)" in prompt
        assert "BUDGET:" in prompt
        assert max_budget_usd == pytest.approx(2.0)
        approach_file = sandbox_dir / "approach.py"
        approach_file.write_text(_GENERATED_RETURNS_ZERO)
        return SandboxResult(
            success=True, output_file=approach_file, error=None, total_cost_usd=1.23
        )

    monkeypatch.setattr(approach, "_run_sandbox", fake_run)
    result = approach.solve_instance(
        env=env, seed=99, budget_usd=2.0, output_subdir=tmp_path / "instance_0"
    )
    assert result.cost_usd == pytest.approx(1.23)
    assert result.crashed is False
    assert isinstance(result.solved, bool)
    assert result.num_steps is not None


def test_solve_instance_pins_count_in_prompt(tmp_path, monkeypatch):
    """For a variable-count env the prompt names the pinned (seed, count) instance, so
    the agent develops against exactly what it is scored on."""
    env, approach = _make_approach(tmp_path)
    captured = {}

    def fake_run(*, sandbox_dir, prompt, system_prompt, max_budget_usd, init_files):
        del sandbox_dir, system_prompt, max_budget_usd, init_files
        captured["prompt"] = prompt
        # Stop before scoring: this test only checks the prompt the agent receives.
        return SandboxResult(
            success=False, output_file=None, error="stop", total_cost_usd=0.0
        )

    monkeypatch.setattr(approach, "_run_sandbox", fake_run)
    approach.solve_instance(
        env=env,
        seed=99,
        budget_usd=2.0,
        output_subdir=tmp_path / "instance_0",
        count=4,
    )
    assert "env.reset(seed=99, options={'object_count': 4})" in captured["prompt"]
    assert "env.reset(seed=99)`" not in captured["prompt"]  # not the unpinned form


def test_solve_instance_no_program_is_crashed(tmp_path, monkeypatch):
    """When the agent commits no approach.py, the attempt is crashed but charged."""
    env, approach = _make_approach(tmp_path)

    def fake_run(**_kwargs):
        return SandboxResult(
            success=False, output_file=None, error="boom", total_cost_usd=0.5
        )

    monkeypatch.setattr(approach, "_run_sandbox", fake_run)
    result = approach.solve_instance(
        env=env, seed=1, budget_usd=1.0, output_subdir=tmp_path / "instance_0"
    )
    assert result.crashed is True
    assert result.solved is False
    assert result.cost_usd == pytest.approx(0.5)


def test_solve_instance_scoring_crash_is_isolated(tmp_path, monkeypatch):
    """A generated program that crashes at load is scored as a failure, not raised."""
    env, approach = _make_approach(tmp_path)

    def fake_run(*, sandbox_dir, **_kwargs):
        approach_file = sandbox_dir / "approach.py"
        # Missing GeneratedApproach symbol -> load raises inside solve_instance.
        approach_file.write_text("x = 1\n")
        return SandboxResult(
            success=True, output_file=approach_file, error=None, total_cost_usd=0.75
        )

    monkeypatch.setattr(approach, "_run_sandbox", fake_run)
    result = approach.solve_instance(
        env=env, seed=2, budget_usd=1.0, output_subdir=tmp_path / "instance_0"
    )
    assert result.crashed is True
    assert result.solved is False
    assert result.cost_usd == pytest.approx(0.75)
