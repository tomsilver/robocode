"""Common data structures."""

from __future__ import annotations

import abc
from typing import Callable, Generic, TypeVar

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class Behavior(abc.ABC, Generic[_X, _U]):
    """A behavior has a single goal, a single pre-condition, 
    and a feedforward policy designed to achieve that goal."""

    subgoal: Callable[[_X], bool]
    precondition: Callable[[_X], bool]
    policy: Callable[[_X], _U]


    @abc.abstractmethod
    def reset(self, x: _X) -> None:
        """Reset the internal state and current parameters."""

    @abc.abstractmethod
    def initializable(self, x: _X) -> bool:
        """Check if the pre-condition is satisfied."""

    @abc.abstractmethod
    def terminated(self, x: _X) -> bool:
        """Check if the subgoal has been achieved."""

    @abc.abstractmethod
    def step(self, x: _X) -> _U:
        """Return the next action to execute."""