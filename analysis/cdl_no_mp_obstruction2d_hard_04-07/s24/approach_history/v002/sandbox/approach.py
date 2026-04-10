"""GeneratedApproach: thin wrapper that sequences behaviors."""
from collections import deque
from behaviors import PickAndDrop, PickAndPlace
from obs_helpers import (
    get_drop_zones, obstruction_on_surface, is_block_grasped,
    NUM_OBSTRUCTIONS, IDX_OBS_BASES,
)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._behaviors = deque()
        self._current   = None

    def reset(self, obs, info):
        drop_zones = get_drop_zones(obs)

        b_place = PickAndPlace()
        b_clears = [
            PickAndDrop(i, drop_zones[i])
            for i in range(NUM_OBSTRUCTIONS)
        ]

        # Backward precondition checking
        if b_place.initializable(obs) and is_block_grasped(obs):
            # Block already held → just place
            self._behaviors = deque([b_place])
        elif b_place.initializable(obs):
            # Surface clear → pick then place
            self._behaviors = deque([b_place])
        else:
            # Need to clear some obstructions
            # Only queue obstructions that are actually on the surface
            to_clear = [
                b_clears[i]
                for i in range(NUM_OBSTRUCTIONS)
                if obstruction_on_surface(obs, i)
            ]
            # Always follow with place (which also does the pick)
            self._behaviors = deque(to_clear + [b_place])

        self._current = self._behaviors.popleft()
        self._current.reset(obs)

    def get_action(self, obs):
        # Advance to next behavior when subgoal reached
        while self._current.terminated(obs) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(obs)
        return self._current.step(obs)
