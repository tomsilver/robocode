"""Tests for llm_genplan_approach.py."""

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box
from omegaconf import DictConfig

from robocode.approaches.llm_genplan_approach import (
    LLMGenPlanApproach,
    _parse_python_code,
)
from robocode.utils.llm import LLMResponse, create_llm_client


class _ToyEnv(Env):
    """1D env: reach position >= 3.0 by adding the action each step."""

    def __init__(self):
        self.observation_space = Box(0.0, 10.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._pos = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = 0.0
        return np.array([self._pos], dtype=np.float32), {}

    def step(self, action):
        self._pos += float(action[0])
        obs = np.array([self._pos], dtype=np.float32)
        return obs, -1.0, self._pos >= 3.0, False, {}

    def render(self):
        return None


class _FakeClient:
    """Returns canned responses in order, recording the messages it saw."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, messages):
        """Record the roles seen and return the next canned response."""
        self.calls.append([m["role"] for m in messages])
        return LLMResponse(text=self._responses.pop(0), cost_usd=0.01)


_BROKEN = """```python
import numpy as np
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        pass
    def reset(self, state, info):
        pass
    def get_action(self, state):
        return np.array([0.0], dtype=np.float32)
```"""

_FIXED = _BROKEN.replace("[0.0]", "[1.0]")


def test_parse_python_code():
    """The parser extracts a fenced block and falls back to raw text."""
    assert _parse_python_code("pre ```python\nx = 1\n``` post") == "x = 1"
    assert _parse_python_code("no fence here") == "no fence here"


def test_create_llm_client_dispatch(monkeypatch):
    """The factory routes providers and rejects unknown ones."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    anthropic = create_llm_client(
        DictConfig({"provider": "anthropic", "model": "claude-sonnet-4-6"})
    )
    assert type(anthropic).__name__ == "AnthropicClient"
    cli = create_llm_client(DictConfig({"provider": "cli", "model": "sonnet"}))
    assert type(cli).__name__ == "ClaudeCLIClient"
    with pytest.raises(ValueError):
        create_llm_client(DictConfig({"provider": "bogus", "model": "x"}))


def _make_approach(env, client, tmp_path):
    approach = LLMGenPlanApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=0,
        primitives={},
        completion=DictConfig({"provider": "cli", "model": "x"}),
        env=env,
        output_dir=str(tmp_path),
        max_steps=10,
        num_train_tasks=2,
        num_prompt_tasks=1,
        max_debug_attempts=4,
        chain_of_thought=False,
        use_docker=False,
    )
    approach._client = client  # pylint: disable=protected-access
    return approach


def test_debug_loop_fixes_broken_policy(tmp_path):
    """A broken first attempt is debugged into a solving policy."""
    env = _ToyEnv()
    client = _FakeClient([_BROKEN, _FIXED])
    approach = _make_approach(env, client, tmp_path)
    approach.train()

    # Two completions: the broken attempt, then the fix after feedback.
    assert len(client.calls) == 2
    assert approach.num_generations == 2  # recorded for results.json
    approach_py = (tmp_path / "sandbox" / "approach.py").read_text()
    # The fix must REPLACE the broken code, not be appended to it: exactly one
    # GeneratedApproach class should remain (regression guard for accumulation).
    assert approach_py.count("class GeneratedApproach") == 1
    assert "[1.0]" in approach_py and "[0.0]" not in approach_py

    # The loaded policy solves the env.
    state, info = env.reset(seed=42)
    approach.reset(state, info)
    terminated = False
    for _ in range(10):
        state, _, terminated, _, _ = env.step(approach.step())
        approach.update(state, 0.0, terminated, {})
        if terminated:
            break
    assert terminated


def test_docker_cost_readback(tmp_path, monkeypatch):
    """With use_docker, the cost written by the container is read back."""
    env = _ToyEnv()
    approach = LLMGenPlanApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=0,
        primitives={},
        completion=DictConfig({"provider": "cli", "model": "x"}),
        env_cfg="{}",
        output_dir=str(tmp_path),
        use_docker=True,
    )

    def fake_run(sandbox_dir, completion_cfg, image):
        del completion_cfg, image
        sandbox_dir.joinpath("approach.py").write_text(_parse_python_code(_FIXED))
        sandbox_dir.joinpath("cost.json").write_text(
            '{"total_cost_usd": 0.05, "num_generations": 3}'
        )

    monkeypatch.setattr(
        "robocode.approaches.llm_genplan_approach.run_genplan_in_docker", fake_run
    )
    approach.train()
    assert approach.total_cost_usd == pytest.approx(0.05)
    assert approach.num_generations == 3  # read back from the container


def test_cot_adds_summary_and_strategy_turns(tmp_path):
    """With CoT on, summary and strategy exchanges precede the code."""
    env = _ToyEnv()
    client = _FakeClient(["a summary", "a strategy", _FIXED])
    approach = _make_approach(env, client, tmp_path)
    approach._chain_of_thought = True  # pylint: disable=protected-access
    approach.train()

    assert len(client.calls) == 3  # summary, strategy, implement
    assert approach.total_cost_usd == pytest.approx(0.03)
    assert approach.num_generations == 1  # CoT exchanges are not code generations


def test_budget_stops_debug_loop(tmp_path):
    """The dollar budget bounds the debug loop before max_debug_attempts."""
    env = _ToyEnv()
    client = _FakeClient([_BROKEN] * 10)  # never fixed -> always re-prompts
    approach = LLMGenPlanApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=0,
        primitives={},
        completion=DictConfig({"provider": "cli", "model": "x"}),
        env=env,
        output_dir=str(tmp_path),
        max_steps=10,
        num_train_tasks=2,
        num_prompt_tasks=1,
        max_debug_attempts=20,
        max_budget_usd=0.025,
        chain_of_thought=False,
        use_docker=False,
    )
    approach._client = client  # pylint: disable=protected-access
    approach.train()

    # 0.01/attempt against a 0.025 budget -> 3 attempts, not the 21-attempt step cap.
    assert len(client.calls) == 3
    assert approach.num_generations == 3
