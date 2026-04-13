"""GeneratedApproach for kinder/ClutteredStorage2D-b3-v0.

Strategy:
  1. Pick block0 OUT of the shelf (clears the way).
  2. Temp-drop block0 at a safe location.
  3. Pick block1 with a reorienting pick angle → deposit at highest shelf slot.
  4. Pick block2 with a reorienting pick angle → deposit at middle shelf slot.
  5. Pick block0 from temp → HoldInShelfBehavior extends arm slowly until
     all 3 blocks are inside the shelf and the env fires terminated.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from behaviors import (
    DEPOSIT_ARM_SLOT,
    HoldInShelfBehavior,
    PickBlockBehavior,
    PlaceBlockBehavior,
    TempDropBehavior,
)
from obs_helpers import BLOCK_NAMES, get_outside_blocks, is_block_in_shelf


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors: deque = deque()
        self._current = None

    def reset(self, state: np.ndarray, info: dict) -> None:
        self._behaviors = deque()

        outside = get_outside_blocks(state)
        in_shelf = [b for b in BLOCK_NAMES if is_block_in_shelf(state, b)]

        if not in_shelf:
            # Unusual: no blocks in shelf initially. Just pick+place each.
            for i, bn in enumerate(outside):
                arm_j = DEPOSIT_ARM_SLOT[i] if i < len(DEPOSIT_ARM_SLOT) else 0.625
                self._behaviors.append(PickBlockBehavior(bn, self._primitives))
                self._behaviors.append(PlaceBlockBehavior(self._primitives, deposit_arm_joint=arm_j,
                                                          held_block_name=bn))
        else:
            # Standard b3 strategy:
            shelf_block = in_shelf[0]   # block0 starts in shelf

            # 1. Pick shelf block out
            self._behaviors.append(PickBlockBehavior(shelf_block, self._primitives))
            # 2. Temp drop
            self._behaviors.append(TempDropBehavior(self._primitives))
            # 3. Pick+place each outside block (highest arm first)
            for i, bn in enumerate(outside):
                arm_j = DEPOSIT_ARM_SLOT[i] if i < len(DEPOSIT_ARM_SLOT) else 0.625
                self._behaviors.append(PickBlockBehavior(bn, self._primitives))
                self._behaviors.append(PlaceBlockBehavior(self._primitives, deposit_arm_joint=arm_j,
                                                          held_block_name=bn))
            # 4. Pick shelf block from temp, hold in shelf → terminated
            self._behaviors.append(PickBlockBehavior(shelf_block, self._primitives))
            self._behaviors.append(HoldInShelfBehavior(self._primitives,
                                                       held_block_name=shelf_block))

        if self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
        else:
            self._current = None

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if self._current is None:
            return np.zeros(5, dtype=np.float32)

        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)

        return self._current.step(state)
