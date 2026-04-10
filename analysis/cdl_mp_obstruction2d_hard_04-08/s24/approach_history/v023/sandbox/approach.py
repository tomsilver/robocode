"""GeneratedApproach for kinder/Obstruction2D-o4-v0."""
from collections import deque
from behaviors import ClearObstruction, PickBlock, PlaceBlock
from obs_helpers import obstruction_overlaps_surface, block_held, NUM_OBSTRUCTIONS

NUM_OBS = NUM_OBSTRUCTIONS


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._behaviors = deque()
        self._current = None
        self._primitives = primitives

    def reset(self, state, info):
        behaviors = []

        b_place = PlaceBlock(self._primitives)
        b_pick  = PickBlock(self._primitives)

        if not b_place.initializable(state):
            if not b_pick.initializable(state):
                # Block already held — skip to place
                pass
            else:
                behaviors.append(b_pick)

            # Add clearing behaviors for obstructions that overlap surface
            clears = []
            for i in range(NUM_OBS):
                b = ClearObstruction(i, self._primitives)
                if b.initializable(state):
                    clears.append(b)
            # Insert clears before pick
            behaviors = clears + behaviors

        behaviors.append(b_place)

        self._behaviors = deque(behaviors)
        self._current = self._behaviors.popleft()
        self._current.reset(state)

    def get_action(self, state):
        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
        return self._current.step(state)
