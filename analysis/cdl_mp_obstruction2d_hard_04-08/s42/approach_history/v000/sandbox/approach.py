"""Generated approach for kinder/Obstruction2D-o4-v0."""
from collections import deque
from behaviors import ClearAllObstructions, PickupTargetBlock, PlaceTargetBlock


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors = deque()
        self._current = None

    def reset(self, state, info):
        b_clear = ClearAllObstructions(primitives=self._primitives)
        b_pick  = PickupTargetBlock(primitives=self._primitives)
        b_place = PlaceTargetBlock(primitives=self._primitives)

        # Check preconditions BACKWARDS
        if b_place.initializable(state):
            self._behaviors = deque([b_place])
        elif b_pick.initializable(state):
            self._behaviors = deque([b_pick, b_place])
        else:
            self._behaviors = deque([b_clear, b_pick, b_place])

        self._current = self._behaviors.popleft()
        self._current.reset(state)

    def get_action(self, state):
        # Advance to next behavior when subgoal reached
        while self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
        return self._current.step(state)
