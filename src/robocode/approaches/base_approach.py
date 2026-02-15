"""Base class for approaches."""

import abc
import copy
from collections.abc import Callable
from typing import Any, Generic, SupportsFloat, TypeVar

import numpy as np
from gymnasium.spaces import Space

_StateType = TypeVar("_StateType")
_ActType = TypeVar("_ActType")


class BaseApproach(Generic[_StateType, _ActType], abc.ABC):
    """Base class for a sequential decision-making agent."""

    def __init__(
        self,
        action_space: Space[_ActType],
        observation_space: Space[_StateType],
        seed: int,
        visible_filepaths: list[str] | None = None,
        env_description_path: str | None = None,
        check_action_collision: Callable[[Any, Any], bool] | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._state_space = copy.deepcopy(observation_space)
        self._action_space = copy.deepcopy(action_space)
        self._action_space.seed(seed)
        # Paths the approach is allowed to read. All transitive dependencies
        # of these files (i.e., any module they import) are also visible.
        self._visible_filepaths = list(visible_filepaths or [])
        self._env_description_path = env_description_path
        self._check_action_collision = check_action_collision
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

    def train(self) -> None:
        """Train the approach.

        Override to implement learning.
        """

    def seed(self, seed: int) -> None:
        """Reset the random number generator."""
        self._rng = np.random.default_rng(seed)
