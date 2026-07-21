"""Tests for the MCP config written into a sandbox."""

import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from robocode.mcp import setup_mcp_config

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
    assert written["design_counts"] == [1]
    assert written["eval_counts"] == [1]

    for path in sandbox_dir.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "2, 3" not in text
            assert "5, 10" not in text


def test_sandbox_env_config_still_instantiates(tmp_path: Path) -> None:
    """The reduced config builds an env, and a pinned count still resets to that
    size."""
    sandbox_dir = _write_sandbox(tmp_path, STICKBUTTON2D_CFG)
    written = json.loads(
        (sandbox_dir / ".mcp" / "env_config.json").read_text(encoding="utf-8")
    )

    env = instantiate(OmegaConf.create(written))
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
