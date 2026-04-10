"""GeneratedApproach for kinder/Obstruction2D-o4-v0."""
from collections import deque
from behaviors import ClearObstruction, PickBlock, PlaceBlock
from obs_helpers import obstruction_overlaps_surface, block_held, block_is_on_surface, NUM_OBSTRUCTIONS

NUM_OBSTRUCTIONS = 4


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._behaviors = deque()
        self._current = None

    def reset(self, state, info):
        # Build behavior sequence by checking preconditions BACKWARDS.
        behaviors = []

        # Last behavior: place block on surface
        b_place = PlaceBlock()
        if not b_place.initializable(state):
            # Need to pick block first
            b_pick = PickBlock()
            behaviors.insert(0, b_pick)

            # Need to clear obstructions first
            for i in range(NUM_OBSTRUCTIONS - 1, -1, -1):
                b_clear = ClearObstruction(i)
                if b_clear.initializable(state):
                    behaviors.insert(0, b_clear)

        behaviors.append(b_place)

        self._behaviors = deque(behaviors)
        self._current = self._behaviors.popleft()
        self._current.reset(state)

    def get_action(self, state):
        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
        return self._current.step(state)
