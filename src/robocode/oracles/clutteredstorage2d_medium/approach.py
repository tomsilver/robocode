"""Oracle approach for ClutteredStorage2D-b3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from gymnasium.spaces import Space
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach
from robocode.oracles.clutteredstorage2d_medium.behaviors import (
    ClearCompactBlocker,
    CompactShelfBlocks,
    StoreOutsideBlock,
)
from robocode.oracles.clutteredstorage2d_medium.obs_helpers import (
    all_blocks_inside_shelf,
)
from robocode.primitives import crv_motion_planning as crv_motion_planning_module
from robocode.primitives import crv_motion_planning_grasp as crv_grasp_module
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
            primitives
            or {
                "crv_motion_planning": cast(
                    Callable[..., Any], crv_motion_planning_module
                ),
                "crv_motion_planning_grasp": cast(Callable[..., Any], crv_grasp_module),
            },
            env_description_path,
        )
        self._seed = seed
        self._current: Behavior[NDArray, NDArray] | None = None
        self._previous_behavior: str | None = None
        self._previous_result = "running"
        self._allow_startup_compact = True

    def _new_behavior(
        self,
        cls: Any,
        **kwargs: Any,
    ) -> Behavior[NDArray, NDArray]:
        return cls(self._primitives, self._seed, **kwargs)

    def _behavior_result(self, behavior: Behavior[NDArray, NDArray]) -> str:
        if hasattr(behavior, "result"):
            return cast(Any, behavior).result()
        return "running"

    def _behavior_name(self, behavior: Behavior[NDArray, NDArray]) -> str:
        return str(getattr(behavior, "behavior_name", behavior.__class__.__name__))

    def _select_behavior(
        self,
        state: NDArray,
        previous_result: str | None,
    ) -> Behavior[NDArray, NDArray] | None:
        last_result = (
            self._previous_result if previous_result is None else previous_result
        )
        if all_blocks_inside_shelf(state):
            return None

        if self._allow_startup_compact:
            compact = self._new_behavior(CompactShelfBlocks)
            if compact.initializable(state):
                if (
                    self._previous_behavior == CompactShelfBlocks.behavior_name
                    and last_result == "blocked"
                ):
                    clear = self._new_behavior(
                        ClearCompactBlocker, use_store_selection=False
                    )
                    if clear.initializable(state):
                        return clear
                if (
                    self._previous_behavior == ClearCompactBlocker.behavior_name
                    and last_result == "failed"
                ):
                    clear = self._new_behavior(
                        ClearCompactBlocker, use_store_selection=True
                    )
                    if clear.initializable(state):
                        return clear
                if (
                    self._previous_behavior == CompactShelfBlocks.behavior_name
                    and last_result == "failed"
                ):
                    store = self._new_behavior(StoreOutsideBlock)
                    if store.initializable(state):
                        return store
                return compact

        store = self._new_behavior(StoreOutsideBlock)
        if store.initializable(state):
            return store
        return None

    def _activate_behavior(
        self,
        state: NDArray,
        previous_result: str | None,
    ) -> None:
        for _ in range(4):
            self._current = self._select_behavior(state, previous_result)
            if self._current is None:
                return
            if self._behavior_name(self._current) == StoreOutsideBlock.behavior_name:
                self._allow_startup_compact = False
            self._current.reset(state)
            current_result = self._behavior_result(self._current)
            if current_result == "running":
                return
            self._previous_behavior = self._behavior_name(self._current)
            self._previous_result = current_result
            previous_result = current_result
        self._current = None

    def reset(self, state: NDArray, info: dict[str, Any]) -> None:
        super().reset(state, info)
        self._previous_behavior = None
        self._previous_result = "running"
        self._allow_startup_compact = True
        self._activate_behavior(state, None)

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

        current_result = self._behavior_result(self._current)
        if self._current.terminated(state) or current_result != "running":
            self._previous_behavior = self._behavior_name(self._current)
            self._previous_result = current_result
            self._activate_behavior(state, current_result)

    def debug_snapshot(self) -> dict[str, Any]:
        """Return lightweight debug metadata for the active behavior."""
        if self._current is None or not hasattr(self._current, "debug_snapshot"):
            return {}
        return self._current.debug_snapshot()
