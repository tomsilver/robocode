"""Wrapper around kinder geom3d environments."""

from typing import Any, SupportsFloat

import gymnasium
import kinder
import numpy as np
from gymnasium.core import RenderFrame
from kinder.core import ConstantObjectKinDEREnv
from numpy.typing import NDArray

from robocode.environments.base_env import BaseEnv

kinder.register_all_environments()


def _unwrap_to_kinder(env: gymnasium.Env) -> ConstantObjectKinDEREnv:
    """Unwrap gymnasium wrappers to get the underlying kinder env."""
    while isinstance(env, gymnasium.Wrapper):
        env = env.env  # type: ignore[assignment]
    assert isinstance(env, ConstantObjectKinDEREnv)
    return env


class KinderGeom3DEnv(BaseEnv[NDArray[Any], NDArray[Any]]):
    """A robocode environment backed by a kinder geom3d environment."""

    def __init__(self, env_id: str) -> None:
        self._env_id = env_id
        self._kinder_env = _unwrap_to_kinder(kinder.make(env_id))
        self.observation_space = self._kinder_env.observation_space
        self.action_space = self._kinder_env.action_space
        self._current_obs: NDArray[Any] | None = None
        super().__init__()

    @property
    def env_description(self) -> str:
        """Markdown description of this environment for an agent."""
        md = self._kinder_env.metadata
        return (
            f"# {self._env_id}\n\n"
            f"{md.get('description', '')}\n\n"
            f"## Variant\n\n{md.get('variant_specific_description', '')}\n\n"
            f"## Observation Space\n\n{md.get('observation_space_description', '')}\n\n"
            f"## Action Space\n\n{md.get('action_space_description', '')}\n\n"
            f"## Reward\n\n{md.get('reward_description', '')}\n\n"
            f"## Example Usage\n\n"
            f"```python\n"
            f"import numpy as np\n"
            f"from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv\n\n"
            f'env = KinderGeom3DEnv("{self._env_id}")\n'
            f"obs, info = env.reset(seed=0)\n"
            f"print(obs.shape)  # {self._kinder_env.observation_space.shape}\n\n"
            f"# Take a random action\n"
            f"action = env.action_space.sample()\n"
            f"next_obs, reward, terminated, truncated, info = env.step(action)\n\n"
            f"# Save and restore state\n"
            f"saved = env.get_state()\n"
            f"env.step(env.action_space.sample())\n"
            f"env.set_state(saved)  # restores to the saved state\n\n"
            f"# Run an episode\n"
            f"obs, info = env.reset(seed=1)\n"
            f"done = False\n"
            f"while not done:\n"
            f"    action = env.action_space.sample()\n"
            f"    obs, reward, terminated, truncated, info = env.step(action)\n"
            f"    done = terminated or truncated\n"
            f"```\n\n"
            f"`obs` and `action` are numpy arrays matching the tables above.\n\n"
            f"## Source Code\n\n"
            f"`KinderGeom3DEnv` is a thin wrapper. The underlying environment "
            f"logic lives in the `kinder` package. To find the source files:\n\n"
            f"```python\n"
            f"import kinder.envs.geom3d\n"
            f"print(kinder.envs.geom3d.__path__)\n"
            f"```\n\n"
            f"Key files in that directory:\n"
            f"- `base_env.py` \u2014 `step()` transition dynamics and collision "
            f"handling\n"
            f"- The environment-specific module (e.g. `motion3d.py`) \u2014 "
            f"reward function (`_get_reward_and_done`), config, and "
            f"scene generation\n"
            f"- `object_types.py` \u2014 object type definitions and feature names"
        )

    def reset(self, *args: Any, **kwargs: Any) -> tuple[NDArray[Any], dict[str, Any]]:
        obs, info = self._kinder_env.reset(*args, **kwargs)
        self._current_obs = obs
        return obs, info

    def step(
        self, action: NDArray[Any]
    ) -> tuple[NDArray[Any], SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.array(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self._kinder_env.step(action)
        self._current_obs = obs
        return obs, reward, terminated, truncated, info

    def get_state(self) -> NDArray[Any]:
        assert self._current_obs is not None, "Must call reset()"
        return self._current_obs.copy()

    def set_state(self, state: NDArray[Any]) -> None:
        obs, _ = self._kinder_env.reset(options={"init_state": state})
        self._current_obs = obs

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._kinder_env.render()  # type: ignore[no-untyped-call]
