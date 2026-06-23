"""Wrapper around kinder geom2d environments."""

import os

# kinder.register_all_environments() (below) hardcodes MUJOCO_GL=osmesa and
# PYOPENGL_PLATFORM=osmesa on headless Linux and then imports mujoco to probe
# for Dynamic3D support, which permanently locks PyOpenGL to that platform.
# Default to EGL (present on most dev machines via libegl1, and expected by
# robosuite via LIBERO), but honor an explicit MUJOCO_GL chosen by the caller:
# the sandbox sets osmesa because headless EGL device displays need a GPU. This
# matters even in the 2D wrapper, since importing it (e.g. via robocode.rendering,
# which always imports this module) would otherwise flip a sandbox's osmesa back
# to egl and break the Dynamic3D mujoco renderer. Capture the chosen backend,
# preempt-import mujoco so PyOpenGL locks to it, restore it after kinder runs.
os.environ.setdefault("MUJOCO_GL", "egl")
# PyOpenGL must agree with the chosen MUJOCO_GL backend, so derive its platform
# from it rather than defaulting independently (glfw is on-screen and uses GLX
# on Linux). This keeps a caller-set MUJOCO_GL=osmesa from pairing with egl.
_PYOPENGL_FOR_MUJOCO = {"egl": "egl", "osmesa": "osmesa", "glfw": "glx"}
os.environ.setdefault(
    "PYOPENGL_PLATFORM", _PYOPENGL_FOR_MUJOCO.get(os.environ["MUJOCO_GL"], "egl")
)
_MUJOCO_GL = os.environ["MUJOCO_GL"]
_PYOPENGL_PLATFORM = os.environ["PYOPENGL_PLATFORM"]
try:
    import mujoco  # pylint: disable=unused-import

    _ = mujoco
except Exception:  # pylint: disable=broad-except
    # mujoco is optional; only needed for libero-style downstream use. Catch
    # broad Exception because mujoco's import chain touches ctypes/OpenGL and
    # can raise AttributeError / OSError when GL runtime libs are missing.
    pass

# pylint: disable=wrong-import-position
from typing import Any, SupportsFloat

import gymnasium
import kinder
import numpy as np
from gymnasium.core import RenderFrame
from kinder.core import ConstantObjectKinDEREnv
from numpy.typing import NDArray

from robocode.environments.base_env import BaseEnv

kinder.register_all_environments()
# kinder flips these to osmesa on headless Linux; restore the chosen backend so
# later mujoco/robosuite users in this process pick it up.
os.environ["MUJOCO_GL"] = _MUJOCO_GL
os.environ["PYOPENGL_PLATFORM"] = _PYOPENGL_PLATFORM


def _unwrap_to_kinder(env: gymnasium.Env) -> ConstantObjectKinDEREnv:
    """Unwrap gymnasium wrappers to get the underlying kinder env."""
    while isinstance(env, gymnasium.Wrapper):
        env = env.env  # type: ignore[assignment]
    assert isinstance(env, ConstantObjectKinDEREnv)
    return env


class KinderGeom2DEnv(BaseEnv[NDArray[Any], NDArray[Any]]):
    """A robocode environment backed by a kinder geom2d environment."""

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
        return self._describe(include_access=True)

    @property
    def env_description_blackbox(self) -> str:
        """Description for blackbox mode.

        Omits the direct-import example usage and the source-code pointers; in blackbox
        mode the agent has access to neither and interacts with the environment only
        through env_client.
        """
        return self._describe(include_access=False)

    def _describe(self, include_access: bool) -> str:
        md = self._kinder_env.metadata
        description = (
            f"# {self._env_id}\n\n"
            f"{md.get('description', '')}\n\n"
            f"## Variant\n\n{md.get('variant_specific_description', '')}\n\n"
            f"## Observation Space\n\n{md.get('observation_space_description', '')}\n\n"
            f"## Action Space\n\n{md.get('action_space_description', '')}\n\n"
            f"## Reward\n\n{md.get('reward_description', '')}\n\n"
        )
        if not include_access:
            return description
        return description + (
            f"## Example Usage\n\n"
            f"```python\n"
            f"import numpy as np\n"
            f"from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv\n\n"
            f'env = KinderGeom2DEnv("{self._env_id}")\n'
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
            f"`KinderGeom2DEnv` is a thin wrapper. The underlying environment "
            f"logic lives in the `kinder` package. To find the source files:\n\n"
            f"```python\n"
            f"import kinder.envs.kinematic2d\n"
            f"print(kinder.envs.kinematic2d.__path__)\n"
            f"```\n\n"
            f"Key files in that directory:\n"
            f"- `base_env.py` \u2014 `step()` transition dynamics and collision "
            f"handling\n"
            f"- The environment-specific module (e.g. `motion2d.py`) \u2014 "
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
