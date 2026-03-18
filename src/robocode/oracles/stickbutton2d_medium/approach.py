"""Oracle approach for StickButton2D-b3 (medium, 3 buttons).

Sequences up to three behaviors:
  RePositionStick  (only if stick bottom unreachable)
  -> GraspStickBottom
  -> TouchAllButtons

At reset, determines the starting behavior by checking preconditions
backwards: if TouchAllButtons is already initializable, skip to it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from gymnasium.spaces import Space
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach
from robocode.oracles.stickbutton2d_medium.behaviors import (
    GraspStickBottom,
    RePositionStick,
    TouchAllButtons,
)
from robocode.primitives.behavior import Behavior


class StickButton2DOracleApproach(BaseApproach[NDArray, NDArray]):
    """Oracle approach: RePositionStick? -> GraspStickBottom -> TouchAllButtons."""

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

        reposition = RePositionStick()
        grasp = GraspStickBottom()
        touch = TouchAllButtons()

        # Determine the behaviour sequence by checking backwards.
        if touch.initializable(state):
            self._behaviors = deque([touch])
        elif grasp.initializable(state):
            self._behaviors = deque([grasp, touch])
        elif reposition.initializable(state):
            self._behaviors = deque([reposition, grasp, touch])
        else:
            # Fallback: just try grasp → touch
            self._behaviors = deque([grasp, touch])

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
