"""Oracle approach for ClutteredStorage2D-b3."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from gymnasium.spaces import Space
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach
from robocode.oracles.clutteredstorage2d_medium.behaviors import (
    StoreRemainingBlocks,
)
from robocode.primitives.behavior import Behavior


class ClutteredStorage2DOracleApproach(BaseApproach[NDArray, NDArray]):
    """Oracle approach for storing all blocks in the shelf."""

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

        store = StoreRemainingBlocks()
        if store.initializable(state):
            self._behaviors = deque([store])
            self._current = self._behaviors.popleft()
            self._current.reset(state)
            return

        self._behaviors = deque()
        self._current = store

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

        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)

    def debug_snapshot(self) -> dict[str, Any]:
        """Return lightweight debug metadata for the active behavior."""
        if self._current is None or not hasattr(self._current, "debug_snapshot"):
            return {}
        return self._current.debug_snapshot()
