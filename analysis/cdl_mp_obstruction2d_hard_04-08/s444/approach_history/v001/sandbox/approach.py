"""GeneratedApproach for kinder/Obstruction2D-o4-v0."""
from collections import deque
from behaviors import ClearAllObstructions, GraspBlock, PlaceBlock
from obs_helpers import any_obstruction_on_surface, is_holding_block


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._primitives = primitives
        self._behaviors = deque()
        self._current = None

    def reset(self, state, info):
        # Build behavior sequence by checking preconditions BACKWARDS.
        place = PlaceBlock(self._primitives)
        grasp = GraspBlock(self._primitives)
        clear = ClearAllObstructions(self._primitives)

        behaviors = []

        if place.initializable(state):
            # Already holding block — just place
            behaviors = [place]
        elif grasp.initializable(state):
            # Surface is clear — grasp then place
            behaviors = [grasp, place]
        else:
            # Need to clear surface first
            behaviors = [clear, grasp, place]

        self._behaviors = deque(behaviors)
        self._current = self._behaviors.popleft()
        self._current.reset(state)

    def get_action(self, state):
        # Advance to next behavior when subgoal reached.
        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
        return self._current.step(state)
