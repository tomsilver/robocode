"""Oracle approach for PushPullHook2D.

Sequences four behaviors:
  GraspRotate -> Sweep -> PrePushPull -> Push or Pull

At reset, determines the starting behavior by checking preconditions
backwards.  The final behavior (Push vs Pull) is chosen based on
whether the movable button is above or below the target button.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from gymnasium.spaces import Space
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach
from robocode.oracles.pushpullhook2d.behaviors import (
    GraspRotate,
    PrePushPull,
    Pull,
    Push,
    Sweep,
)
from robocode.oracles.pushpullhook2d.obs_helpers import get_feature
from robocode.primitives.behavior import Behavior


class PushPullHook2DOracleApproach(BaseApproach[NDArray, NDArray]):
    """Oracle approach that chains GraspRotate -> Sweep -> PrePushPull -> Push/Pull."""

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

    @staticmethod
    def _make_push_or_pull(state: NDArray) -> Behavior[NDArray, NDArray]:
        """Choose Push or Pull based on button positions."""
        mov_y = get_feature(state, "movable_button", "y")
        tgt_y = get_feature(state, "target_button", "y")
        if mov_y > tgt_y:
            return Pull()
        return Push()

    def reset(self, state: NDArray, info: dict[str, Any]) -> None:
        super().reset(state, info)

        push_pull = self._make_push_or_pull(state)
        pre = PrePushPull()
        sweep = Sweep()
        grasp = GraspRotate()

        # Determine the behavior sequence by checking backwards.
        if push_pull.initializable(state):
            self._behaviors = deque([push_pull])
        elif pre.initializable(state):
            self._behaviors = deque([pre, push_pull])
        elif sweep.initializable(state):
            self._behaviors = deque([sweep, pre, push_pull])
        else:
            self._behaviors = deque([grasp, sweep, pre, push_pull])

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
