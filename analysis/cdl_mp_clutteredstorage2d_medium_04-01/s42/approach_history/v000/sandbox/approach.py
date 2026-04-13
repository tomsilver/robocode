"""GeneratedApproach for kinder/ClutteredStorage2D-b3-v0.

Builds a behavior deque by checking preconditions backwards.
All intelligence lives in behaviors.py — this file is intentionally thin.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from behaviors import PickBlockBehavior, PlaceBlockBehavior
from obs_helpers import get_outside_blocks, BLOCK_NAMES


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors: deque = deque()
        self._current = None

    def reset(self, state: np.ndarray, info: dict) -> None:
        """Build behavior deque by checking preconditions BACKWARDS."""
        self._behaviors = deque()
        outside = get_outside_blocks(state)

        # Build (pick, place) pairs for each block that needs to go into shelf.
        # Check preconditions backwards: if already placed, skip its pick+place pair.
        slot = 0
        pairs = []
        for block_name in outside:
            pick = PickBlockBehavior(block_name, self._primitives)
            place = PlaceBlockBehavior(slot, self._primitives)
            pairs.append((pick, place))
            slot += 1

        # Reconstruct behavior deque (first pair first):
        for pick, place in pairs:
            self._behaviors.append(pick)
            self._behaviors.append(place)

        if self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
        else:
            self._current = None

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Delegate to current behavior; advance when subgoal is reached."""
        if self._current is None:
            return np.zeros(5, dtype=np.float32)

        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)

        return self._current.step(state)
