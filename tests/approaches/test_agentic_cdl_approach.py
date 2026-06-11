"""Tests for agentic_cdl_approach.py."""

import json
import socket

import pytest

from robocode.approaches.agentic_cdl_approach import AgenticCDLApproach
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.environments.maze_env import MazeEnv
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.sandbox_types import SandboxResult

_BLACKBOX_ENV_CFG = json.dumps(
    {
        "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
        "env_id": "kinder/Motion2D-p0-v0",
    }
)


def test_accepts_run_experiment_kwargs():
    """The extra kwargs run_experiment passes to every approach are absorbed."""
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticCDLApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
        primitives={},
        backend=DEFAULT_BACKEND_CFG,
        env=env,
        env_cfg="{}",
        max_steps=100,
    )
    assert approach.total_cost_usd is None


def test_blackbox_validation():
    """blackbox rejects mcp_tools, missing env_cfg, and non-Box spaces."""
    env = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
    with pytest.raises(ValueError, match="incompatible with blackbox"):
        AgenticCDLApproach(
            action_space=env.action_space,
            observation_space=env.observation_space,
            seed=123,
            primitives={},
            backend=DEFAULT_BACKEND_CFG,
            blackbox=True,
            env_cfg=_BLACKBOX_ENV_CFG,
            mcp_tools=("render_state",),
        )
    with pytest.raises(ValueError, match="env_cfg"):
        AgenticCDLApproach(
            action_space=env.action_space,
            observation_space=env.observation_space,
            seed=123,
            primitives={},
            backend=DEFAULT_BACKEND_CFG,
            blackbox=True,
        )
    env.close()
    maze = MazeEnv(5, 8, 5, 8)
    with pytest.raises(TypeError, match="serialize_space"):
        AgenticCDLApproach(
            action_space=maze.action_space,
            observation_space=maze.observation_space,
            seed=123,
            primitives={},
            backend=DEFAULT_BACKEND_CFG,
            blackbox=True,
            env_cfg=_BLACKBOX_ENV_CFG,
        )


def test_blackbox_train_wires_sandbox(tmp_path, monkeypatch):
    """train() in blackbox mode starts the server and prepares the sandbox."""
    env = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
    captured = {}

    def fake_run(docker_config, config, backend=None, apptainer_config=None):
        del config, backend, apptainer_config
        captured["config"] = docker_config
        # The env server must be live while the agent would run.
        meta = json.loads((docker_config.sandbox_dir / "env_spaces.json").read_text())
        socket.create_connection(("127.0.0.1", meta["port"]), timeout=5).close()
        return SandboxResult(success=False, output_file=None, error="skipped")

    monkeypatch.setattr(
        "robocode.approaches.agentic_cdl_approach.run_with_rate_limit_retry",
        fake_run,
    )
    approach = AgenticCDLApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
        primitives={},
        backend=DEFAULT_BACKEND_CFG,
        container_backend="docker",
        blackbox=True,
        env_cfg=_BLACKBOX_ENV_CFG,
        max_steps=50,
        output_dir=str(tmp_path),
    )
    approach.train()
    env.close()

    cfg = captured["config"]
    assert cfg.blackbox
    assert "env_client.py" in cfg.init_files
    assert "behavior.py" in cfg.init_files
    assert "BLACK BOX" in cfg.prompt
    assert "must NOT import `env_client`" in cfg.prompt
    assert "devectorize" not in cfg.prompt
    assert "map the observation layout empirically" in cfg.prompt
    assert "Read the environment source files" not in cfg.prompt
    assert "inspect the source code" not in cfg.prompt
    assert "black box" in cfg.system_prompt

    meta = json.loads((tmp_path / "sandbox" / "env_spaces.json").read_text())
    assert meta["host"] == "host.docker.internal"
    assert meta["max_steps"] == 50
