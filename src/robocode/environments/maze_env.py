"""A 2D maze benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, SupportsFloat, SupportsInt

import numpy as np
from gymnasium.core import RenderFrame
from gymnasium.spaces import Discrete
from prpl_utils.spaces import FunctionalSpace

from robocode.environments.base_env import BaseEnv


@dataclass(frozen=True)
class _MazeState:

    agent: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]
    height: int
    width: int
    goal: tuple[int, int]

    def copywith(self, agent: tuple[int, int]) -> _MazeState:
        """Return a copy of the state with the agent changed."""
        return _MazeState(agent, self.obstacles, self.height, self.width, self.goal)


_MazeAction = SupportsInt


class MazeEnv(BaseEnv[_MazeState, _MazeAction]):
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
        self.action_space = Discrete(len(self._get_actions()))
        self.observation_space = FunctionalSpace(
            contains_fn=lambda x: isinstance(x, _MazeState)
        )
        self._current_state: _MazeState | None = None

    def reset(self, *args, **kwargs) -> tuple[_MazeState, dict[str, Any]]:
        super().reset(*args, **kwargs)
        self._current_state = self._generate_task(self.np_random)
        return self._current_state, {}

    def step(
        self, action: _MazeAction
    ) -> tuple[_MazeState, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._current_state is not None, "Must call reset() before step()"
        self._current_state = self._get_next_state(self._current_state, action)
        terminated = self._current_state.agent == self._current_state.goal
        return self._current_state, -1, terminated, False, {}

    def set_state(self, state: _MazeState) -> None:
        self._current_state = state

    def get_state(self) -> _MazeState:
        assert self._current_state is not None, "Must call reset()"
        return self._current_state

    def check_action_collision(self, state: _MazeState, action: _MazeAction) -> bool:
        """Return True if the action hits a wall or obstacle."""
        r, c = state.agent
        dr, dc = {
            self._up: (-1, 0),
            self._down: (1, 0),
            self._left: (0, -1),
            self._right: (0, 1),
        }[int(action)]
        nr, nc = r + dr, c + dc
        return (not (0 <= nr < state.height and 0 <= nc < state.width)) or (
            (nr, nc) in state.obstacles
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    def _get_actions(self) -> list[_MazeAction]:
        return [self._up, self._down, self._left, self._right]

    def _generate_task(self, rng: np.random.Generator) -> _MazeState:
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
        walk_state = _MazeState(start, frozenset(), height, width, (0, 0))
        actions = self._get_actions()
        while True:
            action = actions[rng.choice(len(actions))]
            next_state = self._get_next_state(walk_state, action)
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
        state = _MazeState(start, obstacles, height, width, target)

        return state

    def _get_next_state(self, state: _MazeState, action: _MazeAction) -> _MazeState:
        assert isinstance(state, _MazeState)
        r, c = state.agent
        dr, dc = {
            self._up: (-1, 0),
            self._down: (1, 0),
            self._left: (0, -1),
            self._right: (0, 1),
        }[int(action)]
        nr, nc = r + dr, c + dc
        if (not (0 <= nr < state.height and 0 <= nc < state.width)) or (
            (nr, nc) in state.obstacles
        ):
            nr, nc = r, c
        return state.copywith(agent=(nr, nc))
