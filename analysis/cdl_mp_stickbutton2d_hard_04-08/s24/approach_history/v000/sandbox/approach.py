"""GeneratedApproach for kinder/StickButton2D-b5-v0."""
import numpy as np
from collections import deque

from behaviors import PressAllButtons


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors  = deque()
        self._current    = None

    def reset(self, state, info):
        b_press = PressAllButtons()
        b_press.set_primitives(self._primitives)

        # Only one behavior: press all buttons.
        # Check backwards: if already done, start there; otherwise start fresh.
        if b_press.terminated(state):
            self._behaviors = deque([b_press])
        else:
            self._behaviors = deque([b_press])

        self._current = self._behaviors.popleft()
        self._current.reset(state)

    def get_action(self, state):
        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
        return self._current.step(state)
