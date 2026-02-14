"""Base class for simulators."""

import abc
from typing import Generic, TypeVar

import numpy as np

_StateType = TypeVar("_StateType")
_ActType = TypeVar("_ActType")


class BaseSimulator(Generic[_StateType, _ActType], abc.ABC):
    """A simulator exposes only a sample_next_state function."""

    @abc.abstractmethod
    def sample_next_state(
        self, state: _StateType, action: _ActType, rng: np.random.Generator
    ) -> _StateType:
        """Sample a next state."""
