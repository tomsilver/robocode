"""GeneratedApproach for ClutteredStorage2D-b3-v0."""

from collections import deque
import numpy as np
from behaviors import PickBlock, PlaceInShelf
from obs_helpers import blocks_outside_shelf, is_block_in_shelf


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors: deque = deque()
        self._current = None

    def reset(self, state, info):
        outside = blocks_outside_shelf(state)
        self._behaviors = deque()
        for slot, block_name in enumerate(outside):
            pick = PickBlock(block_name, self._primitives, place_slot=slot)
            place = PlaceInShelf(block_name, self._primitives, slot_index=slot)
            # Check if pick precondition is satisfied (need to pick then place)
            if place.initializable(state):
                self._behaviors.append(place)
            elif pick.initializable(state):
                self._behaviors.append(pick)
                self._behaviors.append(place)
        if not self._behaviors:
            # All blocks in shelf; add a no-op
            self._behaviors.append(_NoOp())
        self._current = self._behaviors.popleft()
        self._current.reset(state)

    def get_action(self, state):
        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
        return self._current.step(state)


class _NoOp:
    def reset(self, state): pass
    def initializable(self, state): return True
    def terminated(self, state): return True
    def step(self, state): return np.zeros(5, dtype=np.float32)
