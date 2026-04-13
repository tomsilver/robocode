"""GeneratedApproach for ClutteredStorage2D-b3-v0."""

from collections import deque
import numpy as np
from behaviors import PickBlock, PlaceInShelf, MoveBlockToTemp
from obs_helpers import blocks_outside_shelf, is_block_in_shelf


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors: deque = deque()
        self._current = None

    def reset(self, state, info):
        outside = blocks_outside_shelf(state)
        self._behaviors = deque()

        block0_in_shelf = is_block_in_shelf(state, 'block0')

        if outside and block0_in_shelf:
            # block0 occupies shelf space; must move it out first, then stack top-down.
            # Step 1: pick block0 from shelf and carry to temp floor location
            pick0 = PickBlock('block0', self._primitives, place_slot=99, allow_in_shelf=True)
            move0 = MoveBlockToTemp('block0', self._primitives, temp_x=2.5, temp_y=1.0)
            self._behaviors.extend([pick0, move0])
            # Step 2: place outside blocks top-down (slot 0 = highest y, slot 1 = next, ...)
            for slot, block_name in enumerate(outside):
                pick = PickBlock(block_name, self._primitives, place_slot=slot)
                place = PlaceInShelf(block_name, self._primitives, slot_index=slot)
                self._behaviors.extend([pick, place])
            # Step 3: place block0 at bottom slot
            n = len(outside)
            pick0_back = PickBlock('block0', self._primitives, place_slot=n)
            place0_back = PlaceInShelf('block0', self._primitives, slot_index=n)
            self._behaviors.extend([pick0_back, place0_back])
        else:
            # No block0 in shelf (or everything already placed): place outside blocks normally
            for slot, block_name in enumerate(outside):
                pick = PickBlock(block_name, self._primitives, place_slot=slot)
                place = PlaceInShelf(block_name, self._primitives, slot_index=slot)
                self._behaviors.extend([pick, place])

        if not self._behaviors:
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
