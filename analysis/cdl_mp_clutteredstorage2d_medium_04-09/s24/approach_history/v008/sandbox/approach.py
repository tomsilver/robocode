"""GeneratedApproach for kinder/ClutteredStorage2D-b3-v0."""

from __future__ import annotations

from collections import deque

import numpy as np

from behaviors import PickupBlock, PlaceBlock
from obs_helpers import BLOCK_NAMES, get_blocks_outside_shelf, extract_robot
from act_helpers import make_action, VAC_OFF


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors: deque = deque()
        self._current = None

    def _build_behaviors(self, obs):
        """Build behavior queue for all blocks currently outside shelf."""
        self._behaviors = deque()
        outside = get_blocks_outside_shelf(obs)
        robot = extract_robot(obs)
        holding = robot.vacuum > 0.5

        if holding:
            self._behaviors.append(PlaceBlock(self._primitives))
            for name in outside:
                self._behaviors.append(PickupBlock(name, self._primitives))
                self._behaviors.append(PlaceBlock(self._primitives))
        else:
            for name in outside:
                self._behaviors.append(PickupBlock(name, self._primitives))
                self._behaviors.append(PlaceBlock(self._primitives))

        if not self._behaviors:
            self._behaviors.append(PlaceBlock(self._primitives))

    def reset(self, state, info):
        obs = np.array(state, dtype=np.float32)
        self._build_behaviors(obs)
        self._current = self._behaviors.popleft()
        self._current.reset(obs)

    def get_action(self, state):
        obs = np.array(state, dtype=np.float32)

        if self._current.terminated(obs):
            if self._behaviors:
                self._current = self._behaviors.popleft()
                self._current.reset(obs)
            else:
                # Check if any blocks still outside — rebuild if so
                outside = get_blocks_outside_shelf(obs)
                if outside:
                    self._build_behaviors(obs)
                    if self._behaviors:
                        self._current = self._behaviors.popleft()
                        self._current.reset(obs)

        return self._current.step(obs)
