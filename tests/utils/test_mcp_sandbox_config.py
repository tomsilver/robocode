"""Tests for the MCP config written into a sandbox."""

import json
from pathlib import Path

from robocode.mcp import _write_sandbox_env_config, setup_mcp_config
from robocode.mcp.local_render import _build_env

STICKBUTTON2D_CFG = {
    "_target_": "robocode.environments.variable_object_count_env.VariableObjectCountEnv",
    "constant_object_env_path": (
        "kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv"
    ),
    "count_kwarg": "num_buttons",
    "count_object_prefix": "button",
    "design_counts": [1, 2, 3],
    "eval_counts": [1, 2, 3, 5, 10],
    "bilevel_env_name": "stickbutton2d",
}


def _write_sandbox(tmp_path: Path, env_config: dict) -> Path:
    """Lay out a run directory the way the experiment runner does, return the
    sandbox."""
    (tmp_path / "env_config.json").write_text(json.dumps(env_config), encoding="utf-8")
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    setup_mcp_config(
        sandbox_dir,
        tool_names=("render_state",),
        python_cmd="python",
        env_config_path=str(sandbox_dir / ".mcp" / "env_config.json"),
        log_file_path=str(sandbox_dir / ".mcp" / "mcp_server.log"),
    )
    return sandbox_dir


def test_sandbox_env_config_holds_no_count_range(tmp_path: Path) -> None:
    """Everything reachable from the sandbox is free of the configured counts.

    The whole sandbox tree is scanned, not just the config: the render server's command
    line is written into the sandbox too, and the counts must not ride along in either.
    """
    sandbox_dir = _write_sandbox(tmp_path, STICKBUTTON2D_CFG)

    written = json.loads(
        (sandbox_dir / ".mcp" / "env_config.json").read_text(encoding="utf-8")
    )
    # The keys are present but carry placeholders: a variable-count env cannot be
    # constructed without them, and the render server instantiates the env from this
    # file. What must not leak is the configured values, which the scan below covers.
    assert written["design_counts"] == [1]
    assert written["eval_counts"] == [1]

    for path in sandbox_dir.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "2, 3" not in text
            assert "5, 10" not in text


def test_sandbox_env_config_still_instantiates(tmp_path: Path) -> None:
    """The reduced config builds an env through the render server, and a pinned count
    still resets to that size."""
    sandbox_dir = _write_sandbox(tmp_path, STICKBUTTON2D_CFG)
    written = json.loads(
        (sandbox_dir / ".mcp" / "env_config.json").read_text(encoding="utf-8")
    )

    env = _build_env(written)
    state, info = env.reset(seed=0, options={"object_count": 4})
    assert info["object_count"] == 4
    assert sum(1 for n in state.get_object_names() if n.startswith("button")) == 4
    env.close()


def test_env_config_without_counts_is_unchanged(tmp_path: Path) -> None:
    """A fixed-count env config reaches the render server as it was written."""
    fixed_cfg = {
        "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
        "env_id": "kinder/Motion2D-p0-v0",
    }
    sandbox_dir = _write_sandbox(tmp_path, fixed_cfg)
    written = json.loads(
        (sandbox_dir / ".mcp" / "env_config.json").read_text(encoding="utf-8")
    )
    assert written == fixed_cfg


def test_sandbox_env_config_replaces_count_ranges(tmp_path: Path) -> None:
    """Count ranges are substituted, not dropped.

    They name the evaluation protocol, so the real values must not reach the sandbox;
    but a variable-count env needs them to construct at all, and the in-sandbox render
    server instantiates the env from this file. Dropping them made the render server
    fail to boot for any env whose config lacked a kinder-specific marker key, which
    took the whole run down with it.
    """
    source = tmp_path / "env_config.json"
    source.write_text(
        json.dumps(
            {
                "_target_": "pkg.mod:Env",
                "design_counts": [1, 2, 3],
                "eval_counts": [1, 2, 3, 4, 5],
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "sandbox_env_config.json"
    _write_sandbox_env_config(source, dest)
    written = json.loads(dest.read_text(encoding="utf-8"))
    assert written["_target_"] == "pkg.mod:Env"
    assert written["design_counts"] == [1]
    assert written["eval_counts"] == [1]


def test_sandbox_env_config_leaves_fixed_count_configs_alone(tmp_path: Path) -> None:
    """An env without count ranges gains no keys it cannot accept."""
    source = tmp_path / "env_config.json"
    source.write_text(
        json.dumps({"_target_": "pkg.mod:Env", "num_blocks": 3}), encoding="utf-8"
    )
    dest = tmp_path / "sandbox_env_config.json"
    _write_sandbox_env_config(source, dest)
    assert json.loads(dest.read_text(encoding="utf-8")) == {
        "_target_": "pkg.mod:Env",
        "num_blocks": 3,
    }
