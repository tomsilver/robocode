"""Tests for the in-sandbox primitive builder used by the render MCP server."""

from __future__ import annotations

import shutil
from pathlib import Path

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.mcp.local_render import _build_sandbox_primitives

_PRIMITIVES_SRC = Path("src/robocode/primitives")


def _sandbox_primitives_dir(tmp_path: Path, files: list[str]) -> Path:
    """Mirror what _setup_sandbox_dir copies: individual primitive .py files."""
    dest = tmp_path / "primitives"
    dest.mkdir()
    for name in files:
        shutil.copy2(_PRIMITIVES_SRC / name, dest / name)
    return dest


def test_build_sandbox_primitives_handles_class_and_function(tmp_path: Path) -> None:
    """The builder binds a function primitive by partial and a class one by
    instantiation, so both an env-bound function (check_action_collision) and an
    env-bound class (bilevel_models -> BilevelModels) are usable in-sandbox."""
    prims_dir = _sandbox_primitives_dir(
        tmp_path, ["check_action_collision.py", "bilevel_models.py"]
    )
    env = KinderGeom2DEnv(
        "kinder/Obstruction2D-o2-v0",
        bilevel_env_name="obstruction2d",
        bilevel_env_model_kwargs={"num_obstructions": 2},
    )
    prims = _build_sandbox_primitives(env, prims_dir)

    assert set(prims) == {"check_action_collision", "bilevel_models"}
    # function primitive -> callable (state, action) -> bool
    assert callable(prims["check_action_collision"])
    # class primitive -> an object exposing the bilevel_models API (built from a
    # copied module, so match by API, not isinstance).
    bm = prims["bilevel_models"]
    obs, _ = env.reset(seed=0)
    atoms = bm.get_abstract_state(obs)
    assert len(atoms) > 0
    assert bm.skill_names


def test_build_sandbox_primitives_empty_when_none(tmp_path: Path) -> None:
    """No copied primitives -> empty dict (render still works with no primitives)."""
    env = KinderGeom2DEnv("kinder/Obstruction2D-o0-v0")
    assert _build_sandbox_primitives(env, tmp_path / "missing") == {}
