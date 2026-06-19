"""Tests for kinder_geom3d_env.py."""

import os
import subprocess
import sys

import numpy as np
import pytest

from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv

ALL_3D_ENV_IDS = [
    "kinder/Motion3D-v0",
    "kinder/Obstruction3D-o0-v0",
    "kinder/Obstruction3D-o2-v0",
    "kinder/Obstruction3D-o4-v0",
    "kinder/KinematicShelf3D-o1-v0",
    "kinder/KinematicShelf3D-o3-v0",
    "kinder/KinematicShelf3D-o5-v0",
    "kinder/Transport3D-o1-v0",
    "kinder/Transport3D-o2-v0",
    "kinder/Packing3D-p1-v0",
    "kinder/Packing3D-p2-v0",
    "kinder/Packing3D-p3-v0",
]


@pytest.mark.parametrize("env_id", ALL_3D_ENV_IDS)
def test_kinder_geom3d_basic(env_id: str) -> None:
    """Basic functionality: reset, step, get/set state."""
    env = KinderGeom3DEnv(env_id)
    env.action_space.seed(123)
    state, _ = env.reset(seed=123)
    assert env.observation_space.contains(state)

    # Step returns a valid observation.
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    next_state, _reward, _terminated, truncated, _ = env.step(action)
    assert env.observation_space.contains(next_state)
    assert not truncated

    # get_state reflects the latest observation.
    assert np.array_equal(env.get_state(), next_state)

    # set_state restores a previous state; stepping from it is reproducible
    # up to float32 vectorization tolerance.
    env.set_state(state)
    assert np.array_equal(env.get_state(), state)
    replayed_state, _, _, _, _ = env.step(action)
    np.testing.assert_allclose(replayed_state, next_state, atol=1e-6)

    env.close()


@pytest.mark.parametrize("env_id", ALL_3D_ENV_IDS)
def test_kinder_geom3d_sample_next_state(env_id: str) -> None:
    """sample_next_state produces a valid next state."""
    env = KinderGeom3DEnv(env_id)
    env.action_space.seed(42)
    state, _ = env.reset(seed=42)
    action = env.action_space.sample()

    rng = np.random.default_rng(0)
    next_state = env.sample_next_state(state, action, rng)
    assert env.observation_space.contains(next_state)

    env.close()


# Both kinder wrappers pin MUJOCO_GL at import time. The sandbox runs headless
# with no GPU, so it sets MUJOCO_GL=osmesa; the wrappers must NOT clobber that
# back to egl (egl device displays need a GPU and crash the Dynamic3D mujoco
# renderer). A fresh subprocess is required because the pin happens once on
# first import. See the geom2d/geom3d env wrappers.
_WRAPPER_MODULES = [
    "robocode.environments.kinder_geom3d_env",
    "robocode.environments.kinder_geom2d_env",
]


def _mujoco_gl_after_import(module: str, preset: str | None) -> str:
    """Import *module* in a fresh process and return the resulting MUJOCO_GL."""
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    if preset is not None:
        env["MUJOCO_GL"] = preset
        env["PYOPENGL_PLATFORM"] = preset
    code = f"import os; import {module}; print('RESULT=' + os.environ['MUJOCO_GL'])"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT="))
    return line.split("=", 1)[1]


@pytest.mark.parametrize("module", _WRAPPER_MODULES)
def test_wrapper_honors_preset_mujoco_gl(module: str) -> None:
    """A caller-set MUJOCO_GL (the sandbox sets osmesa) survives importing the wrapper,
    rather than being force-overridden to egl."""
    assert _mujoco_gl_after_import(module, "osmesa") == "osmesa"


@pytest.mark.parametrize("module", _WRAPPER_MODULES)
def test_wrapper_defaults_to_egl(module: str) -> None:
    """With no MUJOCO_GL set, the wrapper defaults to egl."""
    assert _mujoco_gl_after_import(module, None) == "egl"
