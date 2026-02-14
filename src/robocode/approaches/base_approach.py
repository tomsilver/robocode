"""A base class defining the API for an approach."""

import abc
from typing import Callable

import numpy as np

from robocode.structs import Action, Goal, State, Task


class Approach(abc.ABC):
    """A base class defining the API for an approach.

    In this example, an approach has access to the transition function, cost function,
    and goal function of a benchmark, but it does not have access to the task
    distribution.
    """

    def __init__(
        self,
        actions: list[Action],
        transition_fn: Callable[[State, Action], State],
        cost_fn: Callable[[State, Action, State], float],
        goal_fn: Callable[[State, Goal], bool],
    ) -> None:

        self._actions = actions
        self._transition_fn = transition_fn
        self._cost_fn = cost_fn
        self._goal_fn = goal_fn

    @abc.abstractmethod
    def train(self, training_tasks: list[Task]) -> None:
        """Learn something on training tasks."""

    @abc.abstractmethod
    def generate_plan(
        self, task: Task, train_or_test: str, timeout: float, rng: np.random.Generator
    ) -> list[Action]:
        """Generate a plan to solve the given task."""
