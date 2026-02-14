"""An approach that attempts to make a plan by taking random actions."""

import time

import numpy as np

from robocode.approaches.base_approach import Approach
from robocode.structs import Action, Task


class RandomApproach(Approach):
    """An approach that attempts to make a plan by taking random actions."""

    def train(self, training_tasks: list[Task]) -> None:
        pass

    def generate_plan(
        self, task: Task, train_or_test: str, timeout: float, rng: np.random.Generator
    ) -> list[Action]:
        start_time = time.perf_counter()

        plan: list[Action] = []
        state = task.init

        while time.perf_counter() - start_time < timeout:
            if self._goal_fn(state, task.goal):
                return plan
            action = rng.choice(self._actions)
            plan.append(action)
            state = self._transition_fn(state, action)

        return plan
