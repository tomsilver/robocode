"""Wrapper around kinder geom2d environments."""

from typing import Any, SupportsFloat

import gymnasium
import kinder
from gymnasium.core import RenderFrame
from kinder.core import ConstantObjectKinDEREnv
from numpy.typing import NDArray
from relational_structs.spaces import ObjectCentricBoxSpace

from robocode.environments.base_env import BaseEnv

kinder.register_all_environments()


def _unwrap_to_kinder(env: gymnasium.Env) -> ConstantObjectKinDEREnv:
    """Unwrap gymnasium wrappers to get the underlying kinder env."""
    while isinstance(env, gymnasium.Wrapper):
        env = env.env  # type: ignore[assignment]
    assert isinstance(env, ConstantObjectKinDEREnv)
    return env


class KinderGeom2DEnv(BaseEnv[NDArray[Any], NDArray[Any]]):
    """A robocode environment backed by a kinder geom2d environment."""

    def __init__(self, env_id: str) -> None:
        self._kinder_env = _unwrap_to_kinder(kinder.make(env_id))
        self.observation_space = self._kinder_env.observation_space
        self.action_space = self._kinder_env.action_space
        self._current_obs: NDArray[Any] | None = None
        super().__init__()

    def reset(self, *args: Any, **kwargs: Any) -> tuple[NDArray[Any], dict[str, Any]]:
        obs, info = self._kinder_env.reset(*args, **kwargs)
        self._current_obs = obs
        return obs, info

    def step(
        self, action: NDArray[Any]
    ) -> tuple[NDArray[Any], SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._kinder_env.step(action)
        self._current_obs = obs
        return obs, reward, terminated, truncated, info

    def get_state(self) -> NDArray[Any]:
        assert self._current_obs is not None, "Must call reset()"
        return self._current_obs.copy()

    def set_state(self, state: NDArray[Any]) -> None:
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        obj_state = self.observation_space.devectorize(state)
        inner = self._kinder_env._object_centric_env  # type: ignore[attr-defined]  # pylint: disable=protected-access
        inner._current_state = obj_state  # type: ignore[attr-defined]  # pylint: disable=protected-access
        self._current_obs = state.copy()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._kinder_env.render()  # type: ignore[no-untyped-call]
