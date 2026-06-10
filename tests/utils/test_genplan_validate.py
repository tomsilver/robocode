"""Tests for genplan_validate.py."""

from pathlib import Path

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

from robocode.utils.genplan_validate import validate_tasks


class _ToyEnv(Env):
    """1D env: reach position >= 3.0 by adding the action each step."""

    def __init__(self) -> None:
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


_HEADER = """\
import numpy as np
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        pass
    def reset(self, state, info):
        pass
"""


def _validate(tmp_path: Path, get_action_body: str, max_steps=10, timeout=10.0):
    env = _ToyEnv()
    approach_path = tmp_path / "approach.py"
    approach_path.write_text(
        f"{_HEADER}    def get_action(self, state):\n        {get_action_body}\n"
    )
    return validate_tasks(
        env,
        approach_path,
        env.action_space,
        env.observation_space,
        primitives={},
        seeds=[0],
        max_steps=max_steps,
        timeout=timeout,
    )


def test_solving_policy_passes(tmp_path):
    """A policy that reaches the goal returns no failure."""
    failure = _validate(tmp_path, "return np.array([1.0], dtype=np.float32)")
    assert failure is None


def test_not_solved_reports_final_state(tmp_path):
    """A stalling policy is classified not-solved with the final state shown."""
    failure = _validate(tmp_path, "return np.array([0.0], dtype=np.float32)")
    assert failure["error_type"] == "not-solved"
    assert "10 steps" in failure["feedback"]
    assert "final state" in failure["feedback"]


def test_invalid_action_reports_step(tmp_path):
    """An out-of-space action is classified invalid-action with the step index."""
    failure = _validate(tmp_path, "return np.array([5.0], dtype=np.float32)")
    assert failure["error_type"] == "invalid-action"
    assert "at step 0" in failure["feedback"]


def test_exception_reports_traceback(tmp_path):
    """A raising policy is classified python-exception with the traceback."""
    failure = _validate(tmp_path, "raise ValueError('boom')")
    assert failure["error_type"] == "python-exception"
    assert "ValueError: boom" in failure["feedback"]


def test_hard_worker_death_reports_crash(tmp_path):
    """A worker killed before reporting yields worker-crashed, not a KeyError."""
    failure = _validate(tmp_path, "import os; os._exit(13)")
    assert failure["error_type"] == "worker-crashed"
    assert "exit code 13" in failure["feedback"]


def test_infinite_loop_times_out(tmp_path):
    """A non-terminating get_action is killed and classified as timeout."""
    failure = _validate(tmp_path, "while True: pass", timeout=0.5)
    assert failure["error_type"] == "timeout"
