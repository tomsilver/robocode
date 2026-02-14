"""Defines the base class for an environment."""

import abc
from typing import TypeVar

import numpy as np
from gymnasium.core import Env

_StateType = TypeVar("_StateType")
_ActType = TypeVar("_ActType")


class BaseEnv(Env[_StateType, _ActType]):
    """Base class for environments, are assumed fully observed and resettable."""

    @abc.abstractmethod
    def set_state(self, state: _StateType) -> None:
        """Reset the internal state of the environment to the given one."""

    @abc.abstractmethod
    def get_state(self) -> _StateType:
        """Get the internal state of the environment."""

    def sample_next_state(
        self, state: _StateType, action: _ActType, rng: np.random.Generator
    ) -> _StateType:
        """Sample a next state given a state, action, and RNG."""
        old_rng = self.np_random
        self.np_random = rng
        self.set_state(state)
        next_state, _, _, _, _ = self.step(action)
        self.np_random = old_rng
        return next_state
