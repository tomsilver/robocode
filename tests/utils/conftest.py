"""Fixtures shared by the generated-approach tests in this directory."""

from typing import Any, Iterator

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box


class StatefulGoalEnv(Env):  # type: ignore[type-arg]
    """Toy env with get_state/set_state; step terminates once position reaches 5.0.

    Honest play needs 51 steps of +0.1; a teleport to the goal state solves in one, so
    the step count alone says whether a program reached the goal through its actions.
    """

    def __init__(self) -> None:
        self.observation_space = Box(0.0, 10.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._pos = 0.0

    def reset(self, *, seed: Any = None, options: Any = None) -> Any:
        super().reset(seed=seed)
        self._pos = 0.0
        return np.array([self._pos], dtype=np.float32), {}

    def step(self, action: Any) -> Any:
        del action
        self._pos = min(10.0, self._pos + 0.1)
        obs = np.array([self._pos], dtype=np.float32)
        return obs, 0.0, self._pos >= 5.0, False, {}

    def get_state(self) -> Any:
        """Snapshot the current position."""
        return np.array([self._pos], dtype=np.float32)

    def set_state(self, state: Any) -> None:
        """Restore the position from a snapshot."""
        self._pos = float(state[0])

    def goal_state(self) -> Any:
        """A state sitting at the goal position (5.0)."""
        return np.array([5.0], dtype=np.float32)

    def render(self) -> None:
        return None


@pytest.fixture(name="goal_env")
def _goal_env() -> Iterator[StatefulGoalEnv]:
    """A freshly reset :class:`StatefulGoalEnv`."""
    env = StatefulGoalEnv()
    env.reset(seed=0)
    yield env
