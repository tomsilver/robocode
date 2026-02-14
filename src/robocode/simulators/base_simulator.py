"""Base class for simulators."""

import abc
from typing import Generic, TypeVar

import numpy as np
from gymnasium.spaces import Space

_StateType = TypeVar("_StateType")
_ActType = TypeVar("_ActType")


class BaseSimulator(Generic[_StateType, _ActType], abc.ABC):
    """A simulator exposes only a sample_next_state function."""

    @property
    @abc.abstractmethod
    def action_space(self) -> Space[_ActType]:
        """The action space of the simulator."""

    @property
    @abc.abstractmethod
    def observation_space(self) -> Space[_StateType]:
        """The observation space of the simulator."""

    @abc.abstractmethod
    def sample_next_state(
        self, state: _StateType, action: _ActType, rng: np.random.Generator
    ) -> _StateType:
        """Sample a next state."""
