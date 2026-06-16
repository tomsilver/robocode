"""Tests for the MCP server tools."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.mcp import (
    MCP_TOOLS_SYSTEM_PROMPT_SUFFIX,
    MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX,
    mcp_tool_descriptions,
)
from robocode.mcp.local_render import build_local_server
from robocode.mcp.server import build_blackbox_server
from robocode.utils.env_server import env_server_running, serialize_space

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
    srv = build_local_server(["render_state"], _ENV_CONFIG)
    tools = asyncio.run(srv.list_tools())
    names = {t.name for t in tools}
    assert "render_state" in names
    assert "render_policy" not in names


def test_create_server_registers_both_tools() -> None:
    """Both tools are registered when both are requested."""
    srv = build_local_server(["render_state", "render_policy"], _ENV_CONFIG)
    tools = asyncio.run(srv.list_tools())
    names = {t.name for t in tools}
    assert names == {"render_state", "render_policy"}


def test_render_state_returns_png_path(renders_dir: Path) -> None:
    """render_state tool returns a path to a PNG that exists on disk."""
    srv = build_local_server(["render_state"], _ENV_CONFIG, renders_dir=renders_dir)
    path = _call_tool(srv, "render_state", {"seed": 42})
    assert path.endswith(".png")
    assert Path(path).exists()


def test_render_state_with_arbitrary_state(renders_dir: Path) -> None:
    """render_state accepts an arbitrary state as a list of floats."""
    env = KinderGeom2DEnv(_ENV_CONFIG["env_id"])
    env.reset(seed=0)
    state_list = env.get_state().tolist()

    srv = build_local_server(["render_state"], _ENV_CONFIG, renders_dir=renders_dir)
    path = _call_tool(srv, "render_state", {"state": state_list})
    assert path.endswith(".png")
    assert Path(path).exists()
    assert "state_custom" in Path(path).name


def test_render_state_label(renders_dir: Path) -> None:
    """The label parameter is reflected in the output filename."""
    srv = build_local_server(["render_state"], _ENV_CONFIG, renders_dir=renders_dir)
    path = _call_tool(srv, "render_state", {"seed": 0, "label": "my_label"})
    assert "my_label" in Path(path).name


def test_render_state_deduplicates_filenames(renders_dir: Path) -> None:
    """Calling render_state twice with the same args produces distinct files."""
    srv = build_local_server(["render_state"], _ENV_CONFIG, renders_dir=renders_dir)
    path1 = _call_tool(srv, "render_state", {"seed": 7})
    path2 = _call_tool(srv, "render_state", {"seed": 7})
    assert path1 != path2
    assert Path(path1).exists()
    assert Path(path2).exists()


def _write_env_spaces(sandbox: Path, port: int, token: str) -> Path:
    """Write an env_spaces.json mirroring what the blackbox approach writes."""
    env = KinderGeom2DEnv(_ENV_CONFIG["env_id"])
    meta = {
        "host": "127.0.0.1",
        "port": port,
        "token": token,
        "observation_space": serialize_space(env.observation_space),
        "action_space": serialize_space(env.action_space),
        "max_steps": 5,
    }
    env.close()
    path = sandbox / "env_spaces.json"
    path.write_text(json.dumps(meta))
    return path


def test_blackbox_render_state_proxies_to_host(tmp_path: Path) -> None:
    """In blackbox mode, render_state proxies to the host env server."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    with env_server_running(json.dumps(_ENV_CONFIG), sandbox) as (port, token):
        env_spaces = _write_env_spaces(sandbox, port, token)
        srv = build_blackbox_server(["render_state"], env_spaces)
        path = _call_tool(srv, "render_state", {"seed": 42})
    assert path.endswith(".png")
    assert Path(path).exists()
    assert Path(path).parent == sandbox / "mcp_renders"


def test_blackbox_render_state_label_cannot_escape_sandbox(tmp_path: Path) -> None:
    """A label with path separators is sanitized; the PNG stays in mcp_renders."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    with env_server_running(json.dumps(_ENV_CONFIG), sandbox) as (port, token):
        env_spaces = _write_env_spaces(sandbox, port, token)
        srv = build_blackbox_server(["render_state"], env_spaces)
        path = _call_tool(srv, "render_state", {"seed": 0, "label": "../../escape"})
    written = Path(path)
    assert written.exists()
    assert written.parent == sandbox / "mcp_renders"
    assert ".." not in written.name


def test_blackbox_render_policy_runs_in_sandbox(tmp_path: Path) -> None:
    """In blackbox mode, render_policy runs approach.py locally, not on the host.

    The tool execs the sandbox's approach.py here and only renders the visited states on
    the host, returning absolute frame paths.
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "approach.py").write_text(
        "import numpy as np\n"
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._action_space = action_space\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return np.zeros(self._action_space.shape,"
        " dtype=self._action_space.dtype)\n"
    )
    with env_server_running(json.dumps(_ENV_CONFIG), sandbox) as (port, token):
        env_spaces = _write_env_spaces(sandbox, port, token)
        srv = build_blackbox_server(["render_policy"], env_spaces)
        paths = _call_tool(srv, "render_policy", {"seed": 1, "max_steps": 3})
    assert paths
    assert all(Path(p).exists() for p in paths)
    assert all(Path(p).parent == sandbox / "mcp_renders" for p in paths)


def test_blackbox_render_policy_honors_approach_dir(tmp_path: Path) -> None:
    """render_policy renders the approach.py under the requested approach_dir.

    The approach lives only in a candidate subdirectory, so a render that ignored
    approach_dir (and used the sandbox-root approach.py) would fail to find it.
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    candidate = sandbox / "candidate"
    candidate.mkdir()
    (candidate / "approach.py").write_text(
        "import numpy as np\n"
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._action_space = action_space\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return np.zeros(self._action_space.shape,"
        " dtype=self._action_space.dtype)\n"
    )
    with env_server_running(json.dumps(_ENV_CONFIG), sandbox) as (port, token):
        env_spaces = _write_env_spaces(sandbox, port, token)
        srv = build_blackbox_server(["render_policy"], env_spaces)
        paths = _call_tool(
            srv,
            "render_policy",
            {"approach_dir": "candidate", "seed": 1, "max_steps": 3},
        )
    assert paths
    assert all(Path(p).exists() for p in paths)


def test_local_render_state_label_cannot_escape_renders_dir(renders_dir: Path) -> None:
    """A label with path separators is sanitized; the PNG stays in renders_dir."""
    srv = build_local_server(["render_state"], _ENV_CONFIG, renders_dir=renders_dir)
    path = _call_tool(srv, "render_state", {"seed": 0, "label": "../../escape"})
    written = Path(path)
    assert written.exists()
    assert written.parent == renders_dir
    assert ".." not in written.name


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
    srv = build_local_server(["render_policy"], _ENV_CONFIG, renders_dir=renders_dir)
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


def test_mcp_tool_descriptions_blackbox_uses_handle_api() -> None:
    """Blackbox render_state keeps devectorize but describes the host handle API."""
    normal = mcp_tool_descriptions("claude")["render_state"]
    assert "devectorize" in normal
    assert "vectorize" in normal
    assert "ObjectCentricState" in normal
    # The from-scratch layout API needs the env source, which blackbox lacks.
    assert "constant_objects" in normal
    assert "type_features" in normal

    blackbox = mcp_tool_descriptions("claude", blackbox=True)["render_state"]
    # devectorize/vectorize are proxied to the host, so they stay available...
    assert "devectorize" in blackbox
    assert "vectorize" in blackbox
    assert "ObjectCentricState" in blackbox
    assert "get_object_names" in blackbox
    # ...but the build-from-scratch layout concepts do not.
    assert "constant_objects" not in blackbox
    assert "type_features" not in blackbox
    # render_policy is shared and does not reference layout concepts either way.
    assert (
        mcp_tool_descriptions("claude", blackbox=True)["render_policy"]
        == mcp_tool_descriptions("claude")["render_policy"]
    )


def test_mcp_system_prompt_suffix_blackbox_keeps_devectorize() -> None:
    """The blackbox system-prompt suffix keeps the devectorize/vectorize guidance."""
    assert "devectorize" in MCP_TOOLS_SYSTEM_PROMPT_SUFFIX
    assert "devectorize" in MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX
    assert "vectorize" in MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX
    assert "render_policy" in MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX


def _module_imports(module_name: str) -> set[str]:
    """Return the set of module names imported by *module_name*'s source.

    Parses the module file with ``ast`` (without executing it) and collects the
    target of every ``import`` and ``from ... import`` statement.
    """
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_local_render_does_not_import_robocode_primitives() -> None:
    """The normal-mode render modules must not import the stripped primitives pkg.

    The agentic Docker mount strips ``robocode/primitives/``, so the normal-mode
    MCP render server (and the render helpers it uses) must render without
    importing it. A source-level check is deterministic and does not require a
    sandbox; render primitives instead come from the in-sandbox ``primitives/``
    package copied by the sandbox setup.
    """
    for module_name in (
        "robocode.mcp.local_render",
        "robocode.rendering.render_state",
        "robocode.rendering.render_policy",
    ):
        imports = _module_imports(module_name)
        offending = {
            name
            for name in imports
            if name == "robocode.primitives" or name.startswith("robocode.primitives.")
        }
        assert not offending, f"{module_name} imports {offending}"
