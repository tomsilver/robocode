"""Tests for the dependency-clean strict blackbox runtime."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.spaces import Box

from robocode.approaches.agentic_approach import AgenticApproach
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.docker_sandbox import _strict_docker_run_prefix
from robocode.utils.env_client import BlackboxEnv, SpaceInfo
from robocode.utils.env_server_runtime import _dispatch, _HandleRegistry

_ENV_CFG = '{"_target_": "unused.ForValidation"}'


class _FakeSocket:
    """Socket stub sufficient for constructing a client without a server."""

    def __init__(self) -> None:
        self.file = io.BytesIO()

    def makefile(self, _mode: str) -> io.BytesIO:
        """Return an inert file-like transport."""
        return self.file

    def close(self) -> None:
        """Close the inert transport."""
        self.file.close()


class _DummyEnv:
    """Tiny env surface for testing strict command dispatch without a server."""

    def reset(
        self, seed: int | None, options: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a deterministic observation for *seed*."""
        del options
        return np.array([seed or 0], dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Echo *action* as the next observation."""
        return np.asarray(action, dtype=np.float32), 1.0, False, False, {}

    def get_state(self) -> np.ndarray:
        """Return a value used to prove this method is blocked."""
        return np.zeros(1, dtype=np.float32)


def _strict_approach(**kwargs: Any) -> AgenticApproach:
    space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    return AgenticApproach(
        action_space=space,
        observation_space=space,
        seed=0,
        primitives={},
        backend=DEFAULT_BACKEND_CFG,
        container_backend="docker",
        blackbox=True,
        blackbox_runtime="strict",
        env_cfg=_ENV_CFG,
        mcp_tools=(),
        **kwargs,
    )


def test_strict_blackbox_configuration_and_prompt() -> None:
    """Strict mode advertises only its generic, containerized surface."""
    approach = _strict_approach()
    prompt, _, _ = approach._build_agentic_prompts()  # pylint: disable=protected-access
    assert approach.requires_in_process_eval
    assert "STRICT BLACK BOX" in prompt
    assert "Only `reset` and `step` are available" in prompt
    assert "NumPy, SciPy" in prompt
    assert "devectorization helpers" in prompt
    assert "/opt/robocode-strict/bin/python" in prompt


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"blackbox": False}, "requires blackbox=true"),
        ({"container_backend": "local"}, "requires container_backend=docker"),
        ({"primitives": {"helper": object()}}, "exposes no primitives"),
        ({"mcp_tools": ("render_state",)}, "exposes no MCP"),
    ],
)
def test_strict_blackbox_rejects_capability_leaks(
    overrides: dict[str, Any], message: str
) -> None:
    """Strict mode fails fast when a conflicting capability is configured."""
    space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    args: dict[str, Any] = {
        "action_space": space,
        "observation_space": space,
        "seed": 0,
        "primitives": {},
        "backend": DEFAULT_BACKEND_CFG,
        "container_backend": "docker",
        "blackbox": True,
        "blackbox_runtime": "strict",
        "env_cfg": _ENV_CFG,
        "mcp_tools": (),
    }
    args.update(overrides)
    with pytest.raises(ValueError, match=message):
        AgenticApproach(**args)


def test_strict_server_allows_only_reset_and_step(tmp_path: Path) -> None:
    """The host rejects helper commands independently of client behavior."""
    env = _DummyEnv()
    registry = _HandleRegistry()
    reset = _dispatch(env, {"cmd": "reset", "seed": 4}, tmp_path, registry, strict=True)
    assert reset["obs"]["__ndarray__"] == [4.0]
    stepped = _dispatch(
        env,
        {"cmd": "step", "action": {"__ndarray__": [0.5], "dtype": "float32"}},
        tmp_path,
        registry,
        strict=True,
    )
    assert stepped["reward"] == 1.0
    with pytest.raises(ValueError, match="only reset and step"):
        _dispatch(env, {"cmd": "get_state"}, tmp_path, registry, strict=True)


def test_strict_client_uses_plain_space_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict clients do not attach host-backed devectorization helpers."""
    monkeypatch.setattr("socket.create_connection", lambda _address: _FakeSocket())
    box = {
        "type": "Box",
        "shape": [1],
        "low": [-1.0],
        "high": [1.0],
        "dtype": "float32",
    }
    client = BlackboxEnv(
        {
            "strict": True,
            "observation_space": box,
            "action_space": box,
            "max_steps": 10,
            "primitives": [],
            "token": "unused",
            "host": "unused",
            "port": 0,
        }
    )
    assert type(client.observation_space) is SpaceInfo
    assert not hasattr(client.observation_space, "devectorize")
    client._file.close()  # pylint: disable=protected-access
    client._sock.close()  # pylint: disable=protected-access


def test_strict_docker_launch_has_no_project_mounts(tmp_path: Path) -> None:
    """The synthesis launch mounts only the agent's working directory."""
    command = _strict_docker_run_prefix(
        "strict-test",
        "strict-image",
        tmp_path,
        [],
        [],
        43210,
    )
    joined = " ".join(command)
    assert f"{tmp_path.resolve()}:/sandbox" in joined
    assert "ROBOCODE_FIREWALL_HOST_PORT=43210" in command
    assert "/robocode/src" not in joined
    assert "kindergarden" not in joined
    assert "ss-pybullet" not in joined


def test_generic_policy_worker_protocol(tmp_path: Path) -> None:
    """The standalone worker handles the lifecycle and tagged arrays."""
    (tmp_path / "approach.py").write_text(
        "import numpy as np\n"
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        assert primitives == {}\n"
        "        self.low = action_space.low\n"
        "    def reset(self, state, info):\n"
        "        self.bias = state[0]\n"
        "    def get_action(self, state):\n"
        "        return np.asarray([self.bias + state[0]], dtype=np.float32)\n",
        encoding="utf-8",
    )
    worker = (
        Path(__file__).parents[2]
        / "src"
        / "robocode"
        / "utils"
        / "strict_blackbox_runtime.py"
    )
    box = {
        "type": "Box",
        "shape": [1],
        "low": [-1.0],
        "high": [1.0],
        "dtype": "float32",
    }
    proc = subprocess.Popen(
        [sys.executable, str(worker), "--policy", str(tmp_path / "approach.py")],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def request(payload: dict[str, Any]) -> dict[str, Any]:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    assert request({"cmd": "init", "action_space": box, "observation_space": box})["ok"]
    array = {"__ndarray__": [0.25], "dtype": "float32"}
    assert request({"cmd": "reset", "state": array, "info": {}})["ok"]
    response = request({"cmd": "get_action", "state": array})
    assert response["result"]["__ndarray__"] == pytest.approx([0.5])
    assert request({"cmd": "close"})["ok"]
    proc.wait(timeout=5)
