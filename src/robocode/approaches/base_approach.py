"""An approach that takes random actions."""

import abc
from typing import Any, Generic, SupportsFloat, TypeVar

import numpy as np

from robocode.environments.base_env import BaseEnv

_StateType = TypeVar("_StateType")
_ActType = TypeVar("_ActType")


class BaseApproach(Generic[_StateType, _ActType], abc.ABC):
    """Base class for a sequential decision-making agent."""

    def __init__(
        self,
        simulator: BaseEnv[_StateType, _ActType],
        seed: int,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._simulator = simulator
        self._state_space = simulator.observation_space
        self._action_space = simulator.action_space
        self._action_space.seed(seed)
        self._last_state: _StateType | None = None
        self._last_action: _ActType | None = None
        self._last_reward: SupportsFloat | None = None
        self._last_done: bool | None = None
        self._last_info: dict[str, Any] | None = None

    @abc.abstractmethod
    def _get_action(self) -> _ActType:
        """Produce an action to execute now."""

    def reset(
        self,
        state: _StateType,
        info: dict[str, Any],
    ) -> None:
        """Start a new episode."""
        self._last_state = state
        self._last_info = info

    def step(self) -> _ActType:
        """Get the next action to take."""
        self._last_action = self._get_action()
        return self._last_action

    def update(
        self, state: _StateType, reward: float, done: bool, info: dict[str, Any]
    ) -> None:
        """Record the reward and next state following an action."""
        self._last_state = state
        self._last_info = info
        self._last_reward = reward
        self._last_done = done

    def seed(self, seed: int) -> None:
        """Reset the random number generator."""
        self._rng = np.random.default_rng(seed)
