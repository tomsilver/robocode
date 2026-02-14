"""A simulator backed by a BaseEnv."""

from typing import TypeVar

import numpy as np
from gymnasium.spaces import Space

from robocode.environments.base_env import BaseEnv
from robocode.simulators.base_simulator import BaseSimulator

_StateType = TypeVar("_StateType")
_ActType = TypeVar("_ActType")


class EnvSimulator(BaseSimulator[_StateType, _ActType]):
    """Wraps a BaseEnv to expose only sample_next_state."""

    def __init__(self, env: BaseEnv[_StateType, _ActType]) -> None:
        self._env = env

    @property
    def action_space(self) -> Space[_ActType]:
        """The action space of the simulator."""
        return self._env.action_space

    @property
    def observation_space(self) -> Space[_StateType]:
        """The observation space of the simulator."""
        return self._env.observation_space

    def sample_next_state(
        self, state: _StateType, action: _ActType, rng: np.random.Generator
    ) -> _StateType:
        old_rng = self._env.np_random
        self._env.np_random = rng
        self._env.set_state(state)
        next_state, _, _, _, _ = self._env.step(action)
        self._env.np_random = old_rng
        return next_state
