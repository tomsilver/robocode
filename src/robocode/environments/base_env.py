"""Defines the base class for an environment."""

import abc
from typing import TypeVar

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
