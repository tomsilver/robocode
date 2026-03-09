"""Tests for the MCP server tools."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from robocode.mcp.server import create_server

_ENV_CONFIG = {
    "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
    "env_id": "kinder/Motion2D-p0-v0",
}


def _call_tool(srv: Any, name: str, arguments: dict[str, Any]) -> Any:
    """Call an MCP tool synchronously and return the result value."""
    _, meta = asyncio.run(srv.call_tool(name, arguments))
    return meta["result"]


@pytest.fixture()
def renders_dir(tmp_path: Path) -> Path:
    """Return a temporary renders directory."""
    return tmp_path / "mcp_renders"


def test_create_server_registers_requested_tools() -> None:
    """Only the requested tools are registered."""
    srv = create_server(_ENV_CONFIG, ["render_state"])
    tools = asyncio.run(srv.list_tools())
    names = {t.name for t in tools}
    assert "render_state" in names
    assert "render_policy" not in names


def test_create_server_registers_both_tools() -> None:
    """Both tools are registered when both are requested."""
    srv = create_server(_ENV_CONFIG, ["render_state", "render_policy"])
    tools = asyncio.run(srv.list_tools())
    names = {t.name for t in tools}
    assert names == {"render_state", "render_policy"}


def test_render_state_returns_png_path(renders_dir: Path) -> None:
    """render_state tool returns a path to a PNG that exists on disk."""
    srv = create_server(_ENV_CONFIG, ["render_state"], renders_dir=renders_dir)
    path = _call_tool(srv, "render_state", {"seed": 42})
    assert path.endswith(".png")
    assert Path(path).exists()


def test_render_policy_returns_frame_paths(tmp_path: Path, renders_dir: Path) -> None:
    """render_policy tool returns a list of PNG paths."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "approach.py").write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._action_space = action_space\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return self._action_space.sample()\n"
    )
    srv = create_server(_ENV_CONFIG, ["render_policy"], renders_dir=renders_dir)
    paths = _call_tool(
        srv,
        "render_policy",
        {
            "approach_dir": str(sandbox),
            "seed": 42,
            "max_steps": 5,
            "max_frames": 3,
        },
    )
    assert isinstance(paths, list)
    assert len(paths) == 3
    for p in paths:
        assert p.endswith(".png")
        assert Path(p).exists()
