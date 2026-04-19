"""Tests for the LIBERO-PRO benchmark integration.

Skips cleanly when ``libero`` is not installed (it ships behind the
``[libero]`` extra). Pre-creates ``~/.libero/config.yaml`` so the package's
interactive first-run prompt never fires in CI.

The env-rollout test runs in a spawned subprocess because the kinder 3D
tests (tests/environments/test_kinder_geom3d_env.py) init pybullet's OpenGL
in the same interpreter, leaving PyOpenGL in a state where robosuite's
subsequent mujoco/EGL init fails with "NoneType has no attribute glGetError".
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import pytest
import yaml


def _ensure_libero_config() -> None:
    """Write ``~/.libero/config.yaml`` if absent so importing ``libero`` is non-
    interactive."""
    cfg_path = Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
    cfg_file = cfg_path / "config.yaml"
    if cfg_file.exists():
        return
    cfg_path.mkdir(parents=True, exist_ok=True)
    libero_pkg = (
        Path(__file__).resolve().parents[2]
        / "third-party"
        / "LIBERO-PRO"
        / "libero"
        / "libero"
    )
    cfg = {
        "benchmark_root": str(libero_pkg),
        "bddl_files": str(libero_pkg / "bddl_files"),
        "init_states": str(libero_pkg / "init_files"),
        "datasets": str(libero_pkg.parent / "datasets"),
        "assets": str(libero_pkg / "assets"),
    }
    cfg_file.write_text(yaml.safe_dump(cfg))


_ensure_libero_config()
libero = pytest.importorskip("libero", reason="install with `uv sync --extra libero`")


def test_libero_benchmark_dict_nonempty() -> None:
    """``benchmark.get_benchmark_dict()`` returns the full suite registry."""
    from libero import benchmark  # pylint: disable=import-outside-toplevel

    suites = benchmark.get_benchmark_dict()
    assert "libero_goal" in suites
    assert "libero_object" in suites
    assert "libero_spatial" in suites


def _run_goal_task_0_rollout() -> None:
    """Reset + 5 zero-action steps on libero_goal task 0, asserting obs/reward/done.

    Called as the target of a spawned subprocess, so all heavy imports live here.
    """
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    import numpy as np  # pylint: disable=import-outside-toplevel
    from libero import benchmark  # pylint: disable=import-outside-toplevel
    from libero.envs import (  # pylint: disable=import-outside-toplevel
        OffScreenRenderEnv,
    )

    task_suite = benchmark.get_benchmark_dict()["libero_goal"]()
    bddl_file = task_suite.get_task_bddl_file_path(0)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, camera_heights=64, camera_widths=64
    )
    try:
        env.seed(0)
        obs = env.reset()
        assert isinstance(obs, dict), type(obs)
        assert "agentview_image" in obs
        assert obs["agentview_image"].shape == (64, 64, 3)

        action = np.zeros(7, dtype=np.float32)
        for _ in range(5):
            obs, reward, done, _ = env.step(action)
            assert isinstance(obs, dict)
            assert np.isfinite(reward)
            assert isinstance(done, (bool, np.bool_))
    finally:
        env.close()


def test_libero_env_rollout() -> None:
    """Run the libero_goal task 0 rollout in a fresh interpreter and check exit."""
    proc = mp.get_context("spawn").Process(target=_run_goal_task_0_rollout)
    proc.start()
    proc.join(timeout=180)
    assert proc.exitcode == 0, f"rollout process exited with code {proc.exitcode}"
