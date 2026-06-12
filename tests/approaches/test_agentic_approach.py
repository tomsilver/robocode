"""Tests for agentic_approach.py."""

import json
import socket
from functools import partial

import pytest

from robocode.approaches.agentic_approach import AgenticApproach
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.environments.maze_env import MazeEnv
from robocode.primitives.check_action_collision import check_action_collision
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.sandbox_types import SandboxResult

_BLACKBOX_ENV_CFG = json.dumps(
    {
        "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
        "env_id": "kinder/Motion2D-p0-v0",
    }
)


def test_agentic_approach_fallback():
    """Without training, AgenticApproach falls back to random actions."""
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
        primitives={"check_action_collision": partial(check_action_collision, env)},
        backend=DEFAULT_BACKEND_CFG,
    )
    state, info = env.reset(seed=123)
    approach.reset(state, info)
    action = approach.step()
    assert env.action_space.contains(action)


def test_agentic_approach_with_generated():
    """AgenticApproach delegates to a generated approach when loaded."""
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
        primitives={"check_action_collision": partial(check_action_collision, env)},
        backend=DEFAULT_BACKEND_CFG,
        output_dir="/tmp/test_agentic",
    )

    # Write a minimal generated approach and load it.
    sandbox_dir = approach._output_dir / "sandbox"  # pylint: disable=protected-access
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    approach_file = sandbox_dir / "approach.py"
    approach_file.write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space,"
        " observation_space,\n"
        "                 primitives):\n"
        "        self._n = action_space.n\n"
        "        self._step_count = 0\n"
        "    def reset(self, state, info):\n"
        "        self._step_count = 0\n"
        "    def get_action(self, state):\n"
        "        self._step_count += 1\n"
        "        return 0\n"
    )
    approach._load_generated(approach_file)  # pylint: disable=protected-access

    state, info = env.reset(seed=123)
    approach.reset(state, info)
    action = approach.step()
    assert action == 0
    assert env.action_space.contains(action)


def test_load_dir_skips_agent(tmp_path):
    """When load_dir is set, train() loads from it without calling the agent."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    approach_file = sandbox_dir / "approach.py"
    approach_file.write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space,"
        " observation_space,\n"
        "                 primitives):\n"
        "        pass\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return 0\n"
    )

    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=42,
        primitives={"check_action_collision": partial(check_action_collision, env)},
        backend=DEFAULT_BACKEND_CFG,
        load_dir=str(tmp_path),
    )
    approach.train()

    state, info = env.reset(seed=42)
    approach.reset(state, info)
    assert approach.step() == 0


def test_load_generated_with_sibling_modules(tmp_path):
    """approach.py can import sibling modules and subdirectories."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()

    # Sibling module.
    (sandbox_dir / "math_tools.py").write_text(
        "def multiply(a, b):\n    return a * b\n"
    )

    # Nested subpackage.
    utils_dir = sandbox_dir / "utils"
    utils_dir.mkdir()
    (utils_dir / "__init__.py").write_text("")
    (utils_dir / "helpers.py").write_text("def add(a, b):\n    return a + b\n")

    # approach.py imports from both and stores the computed value.
    (sandbox_dir / "approach.py").write_text(
        "from math_tools import multiply\n"
        "from utils.helpers import add\n"
        "\n"
        "COMPUTED = multiply(add(2, 3), 4)\n"
        "\n"
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._computed = COMPUTED\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return 0\n"
    )

    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=42,
        primitives={"check_action_collision": partial(check_action_collision, env)},
        backend=DEFAULT_BACKEND_CFG,
        load_dir=str(tmp_path),
    )
    approach.train()

    state, info = env.reset(seed=42)
    approach.reset(state, info)
    assert approach.step() == 0
    # Verify the imported modules actually ran correctly.
    assert approach._generated._computed == 20  # pylint: disable=protected-access


def test_blackbox_allows_mcp_tools():
    """blackbox and mcp_tools coexist: render tools proxy to the host server."""
    env = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
        primitives={},
        backend=DEFAULT_BACKEND_CFG,
        blackbox=True,
        env_cfg=_BLACKBOX_ENV_CFG,
        mcp_tools=("render_state",),
    )
    assert approach.total_cost_usd is None
    env.close()


def test_blackbox_requires_env_cfg():
    """blackbox needs env_cfg to start the env server."""
    env = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
    with pytest.raises(ValueError, match="env_cfg"):
        AgenticApproach(
            action_space=env.action_space,
            observation_space=env.observation_space,
            seed=123,
            primitives={},
            backend=DEFAULT_BACKEND_CFG,
            blackbox=True,
        )
    env.close()


def test_blackbox_rejects_unsupported_spaces():
    """blackbox fails loudly for spaces the protocol cannot serialize."""
    env = MazeEnv(5, 8, 5, 8)
    with pytest.raises(TypeError, match="serialize_space"):
        AgenticApproach(
            action_space=env.action_space,
            observation_space=env.observation_space,
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
        "robocode.approaches.agentic_approach.run_with_rate_limit_retry",
        fake_run,
    )
    approach = AgenticApproach(
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
    assert "BLACK BOX" in cfg.prompt
    assert "must NOT import `env_client`" in cfg.prompt
    assert "Read the environment source files" not in cfg.prompt
    assert "inspect the source code" not in cfg.prompt
    assert "black box" in cfg.system_prompt

    meta = json.loads((tmp_path / "sandbox" / "env_spaces.json").read_text())
    assert meta["host"] == "host.docker.internal"
    assert meta["max_steps"] == 50
    assert meta["observation_space"]["type"] == "Box"
    assert meta["action_space"]["type"] == "Box"


def test_load_dir_missing_file_raises(tmp_path):
    """When load_dir points to a directory without approach.py, raise
    FileNotFoundError."""
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=42,
        primitives={"check_action_collision": partial(check_action_collision, env)},
        backend=DEFAULT_BACKEND_CFG,
        load_dir=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError):
        approach.train()
