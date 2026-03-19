"""Oracle approach for Obstruction2D-o2 (medium, 2 obstructions).

Sequences two behaviors:  ClearTargetRegion -> PickPlaceTargetBlock.
At reset, determines the starting behavior by checking preconditions
backwards: if PickPlaceTargetBlock is already initializable, skip clearing.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from gymnasium.spaces import Space
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach
from robocode.oracles.obstruction2d_medium.behaviors import (
    ClearTargetRegion,
    PickPlaceTargetBlock,
)
from robocode.primitives.behavior import Behavior


class Obstruction2DOracleApproach(BaseApproach[NDArray, NDArray]):
    """Oracle approach that chains ClearTargetRegion -> PickPlaceTargetBlock."""

    def __init__(
        self,
        action_space: Space[NDArray],
        observation_space: Space[NDArray],
        seed: int = 0,
        primitives: dict[str, Callable[..., Any]] | None = None,
        env_description_path: str | None = None,
    ) -> None:
        super().__init__(
            action_space,
            observation_space,
            seed,
            primitives or {},
            env_description_path,
        )
        self._behaviors: deque[Behavior[NDArray, NDArray]] = deque()
        self._current: Behavior[NDArray, NDArray] | None = None

    def reset(self, state: NDArray, info: dict[str, Any]) -> None:
        super().reset(state, info)

        pick_place = PickPlaceTargetBlock()
        clear = ClearTargetRegion()

        # Determine the behavior sequence by checking backwards.
        if pick_place.initializable(state):
            self._behaviors = deque([pick_place])
        else:
            self._behaviors = deque([clear, pick_place])

        # Activate the first behavior.
        self._current = self._behaviors.popleft()
        self._current.reset(state)

    def _get_action(self) -> NDArray:
        assert self._current is not None and self._last_state is not None
        return self._current.step(self._last_state)

    def update(
        self,
        state: NDArray,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        super().update(state, reward, done, info)

        if done or self._current is None:
            return

        # If the current behavior's subgoal is reached, advance to the next.
        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
