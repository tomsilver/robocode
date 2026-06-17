"""Tests for best_of_k_approach.py."""

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box
from omegaconf import DictConfig

from robocode.approaches.best_of_k_approach import BestOfKApproach
from robocode.utils.llm import LLMResponse


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


def _program(action_value: float) -> str:
    return (
        "```python\n"
        "import numpy as np\n"
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        pass\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        f"        return np.array([{action_value}], dtype=np.float32)\n"
        "```"
    )


_SOLVE = _program(1.0)  # reaches the goal -> solves every seed
_STALL = _program(0.0)  # never moves -> solves nothing


class _FakeClient:
    """Returns canned responses in order, each with a fixed cost."""

    def __init__(self, responses, cost_usd=0.01):
        self._responses = list(responses)
        self._cost = cost_usd
        self.calls = []

    def complete(self, messages):
        """Record the roles seen and return the next canned response."""
        self.calls.append([m["role"] for m in messages])
        return LLMResponse(text=self._responses.pop(0), cost_usd=self._cost)


def _make(env, client, tmp_path, **kwargs):
    approach = BestOfKApproach(
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
        use_docker=False,
        **kwargs,
    )
    approach._client = client  # pylint: disable=protected-access
    return approach


def test_picks_solver_and_stops_early(tmp_path):
    """A solving candidate is selected and stops the loop before the budget."""
    env = _ToyEnv()
    client = _FakeClient([_STALL, _SOLVE, _SOLVE])
    approach = _make(env, client, tmp_path, max_generation_steps=5, max_budget_usd=5.0)
    approach.train()

    # stall (solves 0), then solve (solves all) -> stop; the 3rd is never sampled.
    assert len(client.calls) == 2
    assert approach.num_generations == 2  # recorded for results.json
    approach_py = (tmp_path / "sandbox" / "approach.py").read_text()
    assert "[1.0]" in approach_py and "[0.0]" not in approach_py

    # The kept policy solves the env.
    state, info = env.reset(seed=0)
    approach.reset(state, info)
    terminated = False
    for _ in range(10):
        state, _, terminated, _, _ = env.step(approach.step())
        approach.update(state, 0.0, terminated, {})
        if terminated:
            break
    assert terminated


def test_step_cap_stops_loop(tmp_path):
    """With no solver, the loop stops at max_generation_steps and keeps a candidate."""
    env = _ToyEnv()
    client = _FakeClient([_STALL, _STALL, _STALL, _STALL])
    approach = _make(env, client, tmp_path, max_generation_steps=3, max_budget_usd=5.0)
    approach.train()

    assert len(client.calls) == 3  # the step cap, not the 4th response
    assert approach.num_generations == 3
    assert (tmp_path / "sandbox" / "approach.py").exists()


def test_budget_cap_stops_loop(tmp_path):
    """The dollar budget bounds the loop when it binds before the step cap."""
    env = _ToyEnv()
    client = _FakeClient([_STALL] * 10, cost_usd=0.01)
    # 0.01/candidate, budget 0.025 -> 3 candidates; high step cap so the budget binds.
    approach = _make(
        env, client, tmp_path, max_generation_steps=20, max_budget_usd=0.025
    )
    approach.train()

    assert len(client.calls) == 3
    assert approach.num_generations == 3


def test_unbounded_loop_raises(tmp_path):
    """No step cap + a backend that reports no cost is a loud error, not a hang."""
    env = _ToyEnv()
    client = _FakeClient([_STALL, _STALL], cost_usd=None)
    approach = _make(
        env, client, tmp_path, max_generation_steps=None, max_budget_usd=5.0
    )
    with pytest.raises(RuntimeError):
        approach.train()
