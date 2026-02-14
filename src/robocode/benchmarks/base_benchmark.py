"""A base class defining the API for a benchmark."""

import abc

import numpy as np

from robocode.structs import Action, Goal, State, Task


class Benchmark(abc.ABC):
    """A base class defining the API for a benchmark.

    In this example, a benchmark is a sequential decision-making problem with a
    distribution over initial states and goals. There is a train- test split in the
    distribution.
    """

    @abc.abstractmethod
    def get_actions(self) -> list[Action]:
        """Define the valid actions for the task."""

    @abc.abstractmethod
    def generate_tasks(
        self, num_tasks: int, train_or_test: str, rng: np.random.Generator
    ) -> list[Task]:
        """Randomly generate a number of tasks."""

    @abc.abstractmethod
    def get_next_state(self, state: State, action: Action) -> State:
        """The transition function."""

    @abc.abstractmethod
    def get_cost(self, state: State, action: Action, next_state: State) -> float:
        """Get the cost of a transition."""

    @abc.abstractmethod
    def check_goal(self, state: State, goal: Goal) -> bool:
        """The goal function."""
