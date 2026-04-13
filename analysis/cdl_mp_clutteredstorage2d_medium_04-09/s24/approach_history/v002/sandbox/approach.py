"""GeneratedApproach for kinder/ClutteredStorage2D-b3-v0."""

from __future__ import annotations

from collections import deque

import numpy as np

from behaviors import PickupBlock, PlaceBlock
from obs_helpers import BLOCK_NAMES, get_blocks_outside_shelf, extract_robot


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors: deque = deque()
        self._current = None

    def reset(self, state, info):
        obs = np.array(state, dtype=np.float32)
        self._behaviors = deque()

        # Determine which blocks need to be placed (outside shelf)
        outside = get_blocks_outside_shelf(obs)

        # Check backward from last behavior
        robot = extract_robot(obs)
        holding = robot.vacuum > 0.5

        if holding:
            # Already holding a block — start with PlaceBlock
            self._behaviors.append(PlaceBlock(self._primitives))
            # Then handle remaining outside blocks
            for name in outside:
                self._behaviors.append(PickupBlock(name, self._primitives))
                self._behaviors.append(PlaceBlock(self._primitives))
        else:
            # Not holding anything — pick up each outside block in sequence
            for name in outside:
                self._behaviors.append(PickupBlock(name, self._primitives))
                self._behaviors.append(PlaceBlock(self._primitives))

        if not self._behaviors:
            # All blocks already in shelf — use a no-op
            self._behaviors.append(PlaceBlock(self._primitives))

        self._current = self._behaviors.popleft()
        self._current.reset(obs)

    def get_action(self, state):
        obs = np.array(state, dtype=np.float32)

        if self._current.terminated(obs) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(obs)

        return self._current.step(obs)
