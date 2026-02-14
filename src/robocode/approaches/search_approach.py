"""An approach that uses uniform-cost search to find a plan."""

from typing import Iterator

import numpy as np
from prpl_utils.search import run_astar

from robocode.approaches.base_approach import Approach
from robocode.structs import Action, State, Task


class SearchApproach(Approach):
    """An approach that uses uniform-cost search to find a plan."""

    def train(self, training_tasks: list[Task]) -> None:
        pass

    def generate_plan(
        self, task: Task, train_or_test: str, timeout: float, rng: np.random.Generator
    ) -> list[Action]:

        def _successor_fn(s: State) -> Iterator[tuple[Action, State, float]]:
            for a in self._actions:
                ns = self._transition_fn(s, a)
                c = self._cost_fn(s, a, ns)
                yield (a, ns, c)

        _, plan, _ = run_astar(
            initial_state=task.init,
            check_goal=lambda s: self._goal_fn(s, task.goal),
            get_successors=_successor_fn,
            heuristic=lambda _: 0,  # == uniform-cost search
            timeout=timeout,
        )

        return plan
