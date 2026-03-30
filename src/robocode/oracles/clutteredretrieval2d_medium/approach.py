"""Oracle approach for ClutteredRetrieval2D-o10 (medium)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gymnasium.spaces import Space
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach
from robocode.oracles.clutteredretrieval2d_medium.behaviors import (
    AcquireTargetBlock,
    PlaceTargetBlockInRegion,
    RemoveSingleBlockingObstruction,
)
from robocode.oracles.clutteredretrieval2d_medium.obs_helpers import (
    candidate_pick_poses,
    filter_feasible_grasp_poses,
    holding_target_block,
    rank_blockers_for_removal,
    target_inside_region,
)
from robocode.primitives.behavior import Behavior


class ClutteredRetrieval2DOracleApproach(BaseApproach[NDArray, NDArray]):
    """Planner-driven oracle with blocker-removal recovery."""

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
        self._acquire = AcquireTargetBlock(self._primitives, seed=seed)
        self._place = PlaceTargetBlockInRegion(self._primitives, seed=seed + 1)
        self._remove = RemoveSingleBlockingObstruction(self._primitives, seed=seed + 2)
        self._current: Behavior[NDArray, NDArray] | None = None
        self._mode = "acquire"
        self._resume_mode = "acquire"
        self._blocked_removals: set[str] = set()

    def reset(self, state: NDArray, info: dict[str, Any]) -> None:
        super().reset(state, info)
        self._mode = "place" if holding_target_block(state) else "acquire"
        self._resume_mode = self._mode
        self._blocked_removals.clear()
        self._activate_for_mode(state)

    def _choose_fallback_blocker(self, state: NDArray) -> str | None:
        for name in rank_blockers_for_removal(state, excluded=self._blocked_removals):
            feasible = filter_feasible_grasp_poses(
                state,
                name,
                candidate_pick_poses(state, name),
            )
            if feasible:
                return name
        return None

    def _activate_for_mode(self, state: NDArray) -> None:
        if self._mode == "remove":
            self._current = self._remove
            self._current.reset(state)
            return
        if self._mode == "place":
            self._current = self._place
            self._current.reset(state)
            return
        self._current = self._acquire
        self._current.reset(state)

    def _get_action(self) -> NDArray:
        assert self._current is not None and self._last_state is not None

        if self._mode == "place" and not holding_target_block(self._last_state):
            self._mode = "acquire"
            self._activate_for_mode(self._last_state)

        if self._mode == "acquire" and self._acquire.required_blocker is not None:
            blocker = self._acquire.required_blocker
            if blocker in self._blocked_removals:
                blocker = self._choose_fallback_blocker(self._last_state)
                self._acquire.required_blocker = blocker
            self._remove.blocker_name = blocker
            self._remove.remove_context = "acquire"
            self._resume_mode = "acquire"
            if blocker is not None:
                self._mode = "remove"
                self._activate_for_mode(self._last_state)
        elif self._mode == "place" and self._place.required_blocker is not None:
            self._remove.blocker_name = self._place.required_blocker
            self._remove.remove_context = "place"
            self._resume_mode = "place"
            self._mode = "remove"
            self._activate_for_mode(self._last_state)

        assert self._current is not None
        return self._current.step(self._last_state)

    def update(
        self,
        state: NDArray,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        prev_state = None if self._last_state is None else self._last_state.copy()
        super().update(state, reward, done, info)
        if done or self._current is None:
            return
        if self._mode == "remove" and prev_state is not None:
            self._remove.observe_transition(prev_state, state)
        if self._mode == "place" and not holding_target_block(state) and not target_inside_region(state):
            self._mode = "acquire"
            self._activate_for_mode(state)
            return
        if self._mode == "remove" and self._remove.blocker_grasp_infeasible:
            if self._remove.blocker_name is not None:
                self._blocked_removals.add(self._remove.blocker_name)
            self._remove.clear()
            if self._resume_mode == "acquire":
                self._acquire.required_blocker = self._choose_fallback_blocker(state)
                self._mode = "remove" if self._acquire.required_blocker is not None else "acquire"
            else:
                self._place.required_blocker = None
                self._mode = self._resume_mode
            self._activate_for_mode(state)
            return
        if target_inside_region(state):
            return
        if self._mode == "remove" and self._remove.terminated(state):
            self._remove.clear()
            if self._resume_mode == "acquire":
                self._acquire.required_blocker = None
            elif self._resume_mode == "place":
                self._place.required_blocker = None
            self._mode = self._resume_mode
            self._activate_for_mode(state)
            return
        if self._mode == "acquire" and self._acquire.terminated(state):
            self._mode = "place"
            self._activate_for_mode(state)
            return
        if self._mode == "place" and self._place.terminated(state):
            return
