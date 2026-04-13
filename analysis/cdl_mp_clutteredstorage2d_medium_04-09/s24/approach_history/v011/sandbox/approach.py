"""GeneratedApproach for kinder/ClutteredStorage2D-b3-v0.

Strategy:
1. Pick up block0 from shelf (approach from below, allow_shelf=True)
2. Drop block0 on floor
3. Pick up block1, place in shelf via arm-push stacking
4. Pick up block2, place in shelf (pushes block1 up)
5. Pick up block0 (now on floor), place in shelf (pushes block2+block1 up)

All three blocks end up inside the shelf via upward pushing.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from behaviors import PickupBlock, PlaceBlock, DropBlock
from obs_helpers import get_blocks_outside_shelf, extract_robot, is_block_in_shelf, get_shelf_slot
from act_helpers import make_action, VAC_OFF


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors: deque = deque()
        self._current = None

    def _build_behaviors(self, obs):
        self._behaviors = deque()
        robot = extract_robot(obs)
        holding = robot.vacuum > 0.5

        # If holding, place first
        if holding:
            self._behaviors.append(PlaceBlock(self._primitives))

        outside = get_blocks_outside_shelf(obs)

        if is_block_in_shelf(obs, 'block0'):
            # Remove block0, stack block1+block2, then block0 last
            # Drop block0 at a safe location away from other blocks and shelf
            slot = get_shelf_slot(obs)
            shelf_cx = slot[0] + slot[2] / 2
            # Drop block0 on the opposite side from blocks 1 and 2
            drop_x = max(0.3, min(4.7, shelf_cx if shelf_cx < 2.5 else 0.5))
            self._behaviors.append(PickupBlock('block0', self._primitives, allow_shelf=True))
            self._behaviors.append(DropBlock(drop_x=drop_x, drop_y=0.6))
            for name in ['block1', 'block2']:
                self._behaviors.append(PickupBlock(name, self._primitives))
                self._behaviors.append(PlaceBlock(self._primitives))
            # Block0 is now on floor
            self._behaviors.append(PickupBlock('block0', self._primitives))
            self._behaviors.append(PlaceBlock(self._primitives))
        else:
            # Block0 already out — normal sequence
            for name in outside:
                self._behaviors.append(PickupBlock(name, self._primitives))
                self._behaviors.append(PlaceBlock(self._primitives))
            # Make sure block0 gets placed if not in shelf
            if 'block0' in outside:
                pass  # already handled above
            elif not is_block_in_shelf(obs, 'block0'):
                self._behaviors.append(PickupBlock('block0', self._primitives))
                self._behaviors.append(PlaceBlock(self._primitives))

    def reset(self, state, info):
        obs = np.array(state, dtype=np.float32)
        self._build_behaviors(obs)
        if self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(obs)
        else:
            self._current = _IdleBehavior()

    def get_action(self, state):
        obs = np.array(state, dtype=np.float32)

        if self._current is None:
            return make_action(0., 0., 0., 0., VAC_OFF)

        if self._current.terminated(obs):
            if self._behaviors:
                self._current = self._behaviors.popleft()
                self._current.reset(obs)
            else:
                # Rebuild if any blocks still outside
                outside = get_blocks_outside_shelf(obs)
                if outside or not is_block_in_shelf(obs, 'block0'):
                    self._build_behaviors(obs)
                    if self._behaviors:
                        self._current = self._behaviors.popleft()
                        self._current.reset(obs)

        return self._current.step(obs)


class _IdleBehavior:
    def terminated(self, obs): return False
    def reset(self, obs): pass
    def step(self, obs): return make_action(0., 0., 0., 0., VAC_OFF)
