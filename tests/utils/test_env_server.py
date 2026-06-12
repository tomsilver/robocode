"""Tests for env_server.py and env_client.py (black-box env access)."""

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.utils.env_client import BlackboxEnv
from robocode.utils.env_server import (
    decode,
    encode,
    env_server_running,
    serialize_space,
)

_ENV_CONFIG = {
    "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
    "env_id": "kinder/Motion2D-p0-v0",
}


@pytest.fixture(scope="module", name="direct_env")
def _direct_env_fixture() -> Iterator[KinderGeom2DEnv]:
    """A host-side env instance to compare against and to get spaces from."""
    env = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
    yield env
    env.close()


@pytest.fixture(scope="module", name="server")
def _server_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[int, str]]:
    """A running env server; yields (port, token)."""
    sandbox_dir = tmp_path_factory.mktemp("run") / "sandbox"
    sandbox_dir.mkdir()
    with env_server_running(json.dumps(_ENV_CONFIG), sandbox_dir) as (port, token):
        yield port, token


def _make_client(
    server: tuple[int, str], direct_env: KinderGeom2DEnv, token: str | None = None
) -> BlackboxEnv:
    """Build a client the way the agentic approach builds env_spaces.json."""
    port, real_token = server
    meta: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": port,
        "token": token if token is not None else real_token,
        "observation_space": serialize_space(direct_env.observation_space),
        "action_space": serialize_space(direct_env.action_space),
        "max_steps": 200,
    }
    return BlackboxEnv(meta)


def test_codec_roundtrip() -> None:
    """encode/decode roundtrips ndarrays, scalars, and nested containers."""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    obj = {"a": arr, "b": [np.int64(3), 1.5, "x", None, True], "c": {"d": arr}}
    wire = json.loads(json.dumps(encode(obj)))
    back = decode(wire)
    np.testing.assert_array_equal(back["a"], arr)
    assert back["a"].dtype == np.float32
    assert back["b"] == [3, 1.5, "x", None, True]
    np.testing.assert_array_equal(back["c"]["d"], arr)


def test_codec_unknown_type_raises() -> None:
    """An unregistered type fails loudly, naming the registry."""
    with pytest.raises(TypeError, match="register_codec"):
        encode(object())


def test_serialize_space(direct_env: KinderGeom2DEnv) -> None:
    """Box spaces serialize; unsupported space types fail loudly."""
    spec = serialize_space(direct_env.observation_space)
    assert spec["type"] == "Box"
    assert tuple(spec["shape"]) == direct_env.observation_space.shape
    with pytest.raises(TypeError, match="serialize_space"):
        serialize_space(Discrete(4))


def test_reset_matches_direct_env(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """Resetting through the server gives the same obs as a direct reset."""
    client = _make_client(server, direct_env)
    obs, info = client.reset(seed=123)
    direct_obs, _ = direct_env.reset(seed=123)
    np.testing.assert_array_equal(obs, direct_obs)
    assert isinstance(info, dict)
    client.close()


def test_step_and_state_roundtrip(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """step returns gymnasium-style types; set_state restores a snapshot."""
    with _make_client(server, direct_env) as client:
        obs, _ = client.reset(seed=0)
        state = client.get_state()
        np.testing.assert_array_equal(state, obs)

        action = client.action_space.sample(np.random.default_rng(0))
        next_obs, reward, terminated, truncated, info = client.step(action)
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == client.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        client.set_state(state)
        np.testing.assert_array_equal(client.get_state(), state)
        replayed, _, _, _, _ = client.step(action)
        np.testing.assert_allclose(replayed, next_obs, atol=1e-6)


def test_concurrent_clients_are_independent(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """Each connection gets its own env instance."""
    with (
        _make_client(server, direct_env) as client_a,
        _make_client(server, direct_env) as client_b,
    ):
        obs_a, _ = client_a.reset(seed=1)
        client_b.reset(seed=2)
        for _ in range(3):
            client_b.step(client_b.action_space.sample(np.random.default_rng(1)))
        np.testing.assert_array_equal(client_a.get_state(), obs_a)


def test_error_travels_over_wire_and_connection_survives(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """A server-side error raises client-side and does not kill the connection."""
    with _make_client(server, direct_env) as client:
        client.reset(seed=0)
        bad_action = np.zeros(client.action_space.shape[0] + 5, dtype=np.float32)
        with pytest.raises(RuntimeError, match="Environment server error"):
            client.step(bad_action)
        # The same connection still works afterwards.
        obs, _ = client.reset(seed=0)
        assert isinstance(obs, np.ndarray)


def test_error_reply_has_no_source_paths(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """Error replies must not leak env source via traceback frames."""
    with _make_client(server, direct_env) as client:
        client.reset(seed=0)
        bad_action = np.zeros(client.action_space.shape[0] + 5, dtype=np.float32)
        with pytest.raises(RuntimeError) as exc_info:
            client.step(bad_action)
        assert 'File "' not in str(exc_info.value)
        assert ".py" not in str(exc_info.value)


def test_wrong_token_rejected(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """A request with a bad token gets the connection closed."""
    client = _make_client(server, direct_env, token="wrong")
    with pytest.raises(RuntimeError, match="closed"):
        client.reset(seed=0)


def test_render_state_roundtrip(tmp_path: Path, direct_env: KinderGeom2DEnv) -> None:
    """render_state proxies to the host, writes a PNG, returns a sandbox path."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    with env_server_running(json.dumps(_ENV_CONFIG), sandbox_dir) as (port, token):
        meta = {
            "host": "127.0.0.1",
            "port": port,
            "token": token,
            "observation_space": serialize_space(direct_env.observation_space),
            "action_space": serialize_space(direct_env.action_space),
            "max_steps": 200,
        }
        with BlackboxEnv(meta) as client:
            rel = client.render_state(seed=7, label="probe")
        assert rel == "mcp_renders/state_seed7_probe.png"
        assert (sandbox_dir / rel).exists()


def test_render_policy_roundtrip(tmp_path: Path, direct_env: KinderGeom2DEnv) -> None:
    """render_policy loads approach.py on the host and saves frames."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "approach.py").write_text(
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
    with env_server_running(json.dumps(_ENV_CONFIG), sandbox_dir) as (port, token):
        meta = {
            "host": "127.0.0.1",
            "port": port,
            "token": token,
            "observation_space": serialize_space(direct_env.observation_space),
            "action_space": serialize_space(direct_env.action_space),
            "max_steps": 3,
        }
        with BlackboxEnv(meta) as client:
            paths = client.render_policy(seed=1, max_steps=3, max_frames=5)
    assert paths
    assert all((sandbox_dir / p).exists() for p in paths)


def test_server_lifecycle(tmp_path: Path) -> None:
    """env_server_running writes the port file and tears the server down."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    with env_server_running(json.dumps(_ENV_CONFIG), sandbox_dir) as (port, token):
        assert (tmp_path / "env_server_port").exists()
        assert int((tmp_path / "env_server_port").read_text()) == port
        assert len(token) == 32
    # After the context exits the server is gone.
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def test_server_fails_fast_on_bad_config(tmp_path: Path) -> None:
    """A bad env config makes startup fail loudly."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    bad_config = json.dumps({"_target_": "nonexistent.module.Env"})
    with pytest.raises(RuntimeError, match="env server exited at startup"):
        with env_server_running(bad_config, sandbox_dir):
            pass
