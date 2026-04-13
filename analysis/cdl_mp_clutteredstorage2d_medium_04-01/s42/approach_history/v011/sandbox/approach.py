"""GeneratedApproach for kinder/ClutteredStorage2D-b3-v0.

Behavior sequence (backward precondition checking):
  For each block outside the shelf (determined at reset):
    1. PickBlockBehavior(i)  - grasp block i
    2. PlaceBlockBehavior(i) - place block i inside shelf

approach.py is THIN: only builds behavior deque and delegates.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from behaviors import AllDoneBehavior, PickBlockBehavior, PlaceBlockBehavior
from obs_helpers import get_outside_block_indices, is_block_in_shelf, is_holding_block


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors: deque = deque()
        self._current = None

    def reset(self, state: np.ndarray, info: dict) -> None:
        self._behaviors = deque()

        # Determine which blocks are outside shelf (backward precondition)
        outside = get_outside_block_indices(state)

        if not outside:
            # All blocks already in shelf
            b = AllDoneBehavior()
            b.reset(state)
            self._behaviors.append(b)
        else:
            for block_idx in outside:
                # Check if we are already holding this block (skip pick)
                if is_holding_block(state, block_idx):
                    place = PlaceBlockBehavior(block_idx, self._primitives)
                    self._behaviors.append(place)
                else:
                    pick = PickBlockBehavior(block_idx, self._primitives)
                    place = PlaceBlockBehavior(block_idx, self._primitives)
                    self._behaviors.append(pick)
                    self._behaviors.append(place)

        self._current = self._behaviors.popleft()
        self._current.reset(state)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        # Advance to next behavior when subgoal reached
        while self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)

        return self._current.step(state)
