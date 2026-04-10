"""GeneratedApproach: thin wrapper that sequences behaviors."""
from collections import deque
from behaviors import PickAndDrop, PickAndPlace
from obs_helpers import (
    obstruction_on_surface, is_block_grasped,
    extract_obstruction, NUM_OBSTRUCTIONS,
)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._behaviors = deque()
        self._current   = None

    def reset(self, obs, info):
        b_place = PickAndPlace()

        if b_place.initializable(obs) and is_block_grasped(obs):
            # Block already held → just place
            self._behaviors = deque([b_place])
        elif b_place.initializable(obs):
            # Surface clear → pick then place
            self._behaviors = deque([b_place])
        else:
            # Need to clear obstructions on the surface
            on_surf = [i for i in range(NUM_OBSTRUCTIONS)
                       if obstruction_on_surface(obs, i)]
            # Sort tallest first: tall obs block robot access to shorter adjacent obs
            on_surf.sort(key=lambda i: -extract_obstruction(obs, i)['height'])
            # Assign distinct zone indices: k-th obstruction uses zones[k]
            to_clear = [PickAndDrop(i, zone_idx=k) for k, i in enumerate(on_surf)]
            self._behaviors = deque(to_clear + [b_place])

        self._current = self._behaviors.popleft()
        self._current.reset(obs)

    def get_action(self, obs):
        # Advance to next behavior when subgoal reached
        while self._current.terminated(obs) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(obs)
        return self._current.step(obs)
