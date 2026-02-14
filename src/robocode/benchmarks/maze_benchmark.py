"""A 2D maze benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from robocode.benchmarks.base_benchmark import Benchmark
from robocode.structs import Action, Goal, State, Task


@dataclass(frozen=True)
class _MazeState:

    agent: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]
    height: int
    width: int

    def copywith(self, agent: tuple[int, int]) -> _MazeState:
        """Return a copy of the state with the agent changed."""
        return _MazeState(agent, self.obstacles, self.height, self.width)


class MazeBenchmark(Benchmark):
    """A 2D maze benchmark."""

    _empty: ClassVar[int] = 0
    _obstacle: ClassVar[int] = 1
    _agent: ClassVar[int] = 2

    _up: ClassVar[int] = 0
    _down: ClassVar[int] = 1
    _left: ClassVar[int] = 2
    _right: ClassVar[int] = 3

    def __init__(
        self,
        min_height: int,
        max_height: int,
        min_width: int,
        max_width: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._min_height = min_height
        self._max_height = max_height
        self._min_width = min_width
        self._max_width = max_width

    def get_actions(self) -> list[Action]:
        return [self._up, self._down, self._left, self._right]

    def generate_tasks(
        self, num_tasks: int, train_or_test: str, rng: np.random.Generator
    ) -> list[Task]:
        assert train_or_test in ("train", "test")  # does not actually matter
        # Generate mazes independently.
        return [self._generate_task(rng) for _ in range(num_tasks)]

    def _generate_task(self, rng: np.random.Generator) -> Task:
        # Generate an empty obstacle grid of random size.
        height = rng.integers(self._min_height, self._max_height + 1, dtype=int)
        width = rng.integers(self._min_width, self._max_width + 1, dtype=int)

        # Generate a random start position.
        start = (
            rng.integers(0, height, dtype=int),
            rng.integers(0, width, dtype=int),
        )

        # Do a random walk to get an end position.
        visited = {start}
        walk_state = _MazeState(start, frozenset(), height, width)
        while True:
            action = rng.choice(self.get_actions())
            next_state = self.get_next_state(walk_state, action)
            assert isinstance(next_state, _MazeState)
            walk_state = next_state
            current = walk_state.agent
            visited.add(current)
            if start != current and rng.uniform() > 0.99:
                target = current
                break

        # Add random obstacles. Choose a quarter of the safe cells.
        all_positions = {(r, c) for r in range(height) for c in range(width)}
        obstacle_candidates = sorted(all_positions - visited)
        num_obstacles = int(len(obstacle_candidates) * 0.25)
        obstacles = frozenset(
            (r, c)
            for r, c in rng.choice(
                obstacle_candidates, size=num_obstacles, replace=False
            )
        )
        state = _MazeState(start, obstacles, height, width)

        # Finish the task.
        task = Task(state, f"Go to {target}")

        return task

    def get_next_state(self, state: State, action: Action) -> State:
        assert isinstance(state, _MazeState)
        r, c = state.agent
        dr, dc = {
            self._up: (-1, 0),
            self._down: (1, 0),
            self._left: (0, -1),
            self._right: (0, 1),
        }[action]
        nr, nc = r + dr, c + dc
        if (not (0 <= nr < state.height and 0 <= nc < state.width)) or (
            (nr, nc) in state.obstacles
        ):
            nr, nc = r, c
        return state.copywith(agent=(nr, nc))

    def get_cost(self, state: State, action: Action, next_state: State) -> float:
        return 1.0

    def check_goal(self, state: State, goal: Goal) -> bool:
        assert isinstance(state, _MazeState)
        assert goal.startswith("Go to (") and goal.endswith(")")
        r, c = map(int, goal[len("Go to (") : -1].replace(" ", "").split(","))
        return state.agent == (r, c)
