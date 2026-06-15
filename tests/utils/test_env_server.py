"""Tests for env_server.py and env_client.py (black-box env access)."""

import json
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from gymnasium.spaces import Discrete
from relational_structs.spaces import ObjectCentricBoxSpace

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.primitives import blackbox_primitive_manifest, build_primitives
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
    """Encode/decode roundtrips ndarrays, scalars, and nested containers."""
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


def test_blackbox_primitive_manifest() -> None:
    """The manifest tags env-dependent primitives and names generic sources."""
    manifest = blackbox_primitive_manifest(["check_action_collision", "csp", "BiRRT"])
    by_name = {spec["name"]: spec for spec in manifest}
    assert by_name["check_action_collision"] == {
        "name": "check_action_collision",
        "kind": "host_proxy",
    }
    assert by_name["csp"] == {
        "name": "csp",
        "kind": "generic",
        "module": "csp",
        "attr": None,
    }
    assert by_name["BiRRT"] == {
        "name": "BiRRT",
        "kind": "generic",
        "module": "motion_planning",
        "attr": "BiRRT",
    }


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
    """Step returns gymnasium-style types; set_state restores a snapshot."""
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


def test_render_state_leaves_connection_env_unchanged(
    tmp_path: Path, direct_env: KinderGeom2DEnv
) -> None:
    """render_state restores the connection's env state (no side effect)."""
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
            client.reset(seed=0)
            before = client.get_state()
            custom = np.asarray(before, dtype=np.float32) + 0.1
            client.render_state(seed=5)
            client.render_state(state=custom.tolist())
            np.testing.assert_array_equal(client.get_state(), before)


def test_check_action_collision_matches_host_primitive(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """The proxy returns the same bool as the host-side collision primitive."""
    host_primitive = build_primitives(direct_env, ["check_action_collision"])[
        "check_action_collision"
    ]
    with _make_client(server, direct_env) as client:
        client.reset(seed=0)
        state = client.get_state()
        for action in (
            np.zeros(client.action_space.shape, dtype=np.float32),
            client.action_space.high,
        ):
            got = client.check_action_collision(state, action)
            direct_env.reset(seed=0)
            expected = host_primitive(direct_env.get_state(), action)
            assert isinstance(got, bool)
            assert got == expected


def test_check_action_collision_error_has_no_source_paths(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """A bad action surfaces as a clean error, leaking no env source lines."""
    with _make_client(server, direct_env) as client:
        client.reset(seed=0)
        state = client.get_state()
        bad_action = np.zeros(client.action_space.shape[0] + 5, dtype=np.float32)
        with pytest.raises(RuntimeError) as exc_info:
            client.check_action_collision(state, bad_action)
        assert 'File "' not in str(exc_info.value)
        assert ".py" not in str(exc_info.value)


def test_make_primitives_builds_eval_dict(
    server: tuple[int, str], direct_env: KinderGeom2DEnv, tmp_path: Path
) -> None:
    """make_primitives proxies env-dependent primitives and imports generic ones.

    A toy generic source stands in for the copied primitive files so the test exercises
    the module/attr import paths without depending on real primitive internals.
    """
    sandbox = tmp_path / "sandbox"
    (sandbox / "primitives").mkdir(parents=True)
    (sandbox / "primitives" / "toy.py").write_text(
        "VALUE = 42\n\n\nclass Thing:\n    pass\n", encoding="utf-8"
    )
    manifest = [
        {"name": "check_action_collision", "kind": "host_proxy"},
        {"name": "toy", "kind": "generic", "module": "toy", "attr": None},
        {"name": "Thing", "kind": "generic", "module": "toy", "attr": "Thing"},
    ]
    port, token = server
    meta: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": port,
        "token": token,
        "observation_space": serialize_space(direct_env.observation_space),
        "action_space": serialize_space(direct_env.action_space),
        "max_steps": 200,
        "primitives": manifest,
    }
    try:
        with BlackboxEnv(meta, sandbox_root=sandbox) as client:
            primitives = client.make_primitives()
            assert set(primitives) == {"check_action_collision", "toy", "Thing"}
            assert primitives["toy"].VALUE == 42
            assert primitives["Thing"].__name__ == "Thing"
            # The host-proxy entry is the live, callable collision check.
            client.reset(seed=0)
            collision = primitives["check_action_collision"](
                client.get_state(),
                np.zeros(client.action_space.shape, dtype=np.float32),
            )
            assert isinstance(collision, bool)
    finally:
        sys.modules.pop("primitives", None)
        sys.modules.pop("primitives.toy", None)
        sys.path[:] = [p for p in sys.path if p != str(sandbox.resolve())]


def test_render_policy_roundtrip(tmp_path: Path, direct_env: KinderGeom2DEnv) -> None:
    """render_policy runs approach.py in-process with real primitives.

    The approach calls ``primitives['check_action_collision']`` each step, so
    rendering only succeeds if render_policy supplies the eval-time primitives
    (an empty dict, the old behavior, would raise a KeyError).
    """
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "approach.py").write_text(
        "import numpy as np\n"
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._action_space = action_space\n"
        "        self._collision = primitives['check_action_collision']\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        action = np.zeros(self._action_space.shape,"
        " dtype=self._action_space.dtype)\n"
        "        self._collision(state, action)\n"
        "        return action\n"
    )
    with env_server_running(json.dumps(_ENV_CONFIG), sandbox_dir) as (port, token):
        meta = {
            "host": "127.0.0.1",
            "port": port,
            "token": token,
            "observation_space": serialize_space(direct_env.observation_space),
            "action_space": serialize_space(direct_env.action_space),
            "max_steps": 3,
            "primitives": blackbox_primitive_manifest(["check_action_collision"]),
        }
        with BlackboxEnv(meta, sandbox_root=sandbox_dir) as client:
            paths = client.render_policy(seed=1, max_steps=3, max_frames=5)
    assert paths
    assert all((sandbox_dir / p).exists() for p in paths)


def test_server_rejects_render_policy_command(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """The host has no render_policy command: it never execs approach.py.

    The policy episode runs in the sandbox; the host only renders states. A
    leftover host-side render_policy would re-introduce host execution of
    agent code, so guard against it.
    """
    with _make_client(server, direct_env) as client:
        with pytest.raises(RuntimeError, match="Unknown command"):
            client._request(  # pylint: disable=protected-access
                {"cmd": "render_policy", "seed": 1}
            )


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


def test_devectorize_matches_direct_env(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """Devectorize over the wire matches the direct env's object-centric state.

    The returned handle exposes the public ObjectCentricState API, and vectorize round-
    trips back to the original observation.
    """
    with _make_client(server, direct_env) as client:
        obs, _ = client.reset(seed=0)
        ocs = client.observation_space.devectorize(obs)

        direct_env.reset(seed=0)
        direct_space = cast(ObjectCentricBoxSpace, direct_env.observation_space)
        direct_ocs = direct_space.devectorize(direct_env.get_state())
        assert ocs.get_object_names() == direct_ocs.get_object_names()

        robot = ocs.get_object_from_name("robot")
        direct_robot = direct_ocs.get_object_from_name("robot")
        for feature in ("x", "y", "theta"):
            assert ocs.get(robot, feature) == pytest.approx(
                float(direct_ocs.get(direct_robot, feature))
            )

        vec = client.observation_space.vectorize(ocs)
        assert isinstance(vec, np.ndarray)
        np.testing.assert_allclose(vec, obs, atol=1e-6)


def test_remote_handle_method_chaining(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """A handle's methods chain over the wire: ocs -> robot -> get(feature)."""
    with _make_client(server, direct_env) as client:
        obs, _ = client.reset(seed=0)
        ocs = client.observation_space.devectorize(obs)
        robot = ocs.get_object_from_name("robot")
        assert robot.name == "robot"

        direct_env.reset(seed=0)
        direct_space = cast(ObjectCentricBoxSpace, direct_env.observation_space)
        direct_ocs = direct_space.devectorize(direct_env.get_state())
        direct_robot = direct_ocs.get_object_from_name("robot")
        assert ocs.get(robot, "x") == pytest.approx(
            float(direct_ocs.get(direct_robot, "x"))
        )


def test_crv_motion_planning_through_proxy(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """plan_crv_actions runs on the host via the remote-module proxy.

    The state, the CRVConfig, and the result all cross the wire as remote handles /
    tagged values. We assert the proxy returns a valid result type (a list of action-
    shaped arrays, or None) rather than a specific plan, since planner success depends
    on the env's constant objects (out of scope here).
    """
    manifest = blackbox_primitive_manifest(["crv_motion_planning"])
    port, token = server
    meta: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": port,
        "token": token,
        "observation_space": serialize_space(direct_env.observation_space),
        "action_space": serialize_space(direct_env.action_space),
        "max_steps": 200,
        "primitives": manifest,
    }
    with BlackboxEnv(meta) as client:
        obs, _ = client.reset(seed=0)
        ocs = client.observation_space.devectorize(obs)
        crv = client.make_primitives()["crv_motion_planning"]
        robot = ocs.get_object_from_name("robot")
        cfg = crv.CRVConfig(
            x=float(ocs.get(robot, "x")),
            y=float(ocs.get(robot, "y")),
            theta=float(ocs.get(robot, "theta")),
        )
        actions = crv.plan_crv_actions(ocs, cfg, carrying=False, seed=0)
        assert actions is None or (
            isinstance(actions, list)
            and all(
                isinstance(a, np.ndarray) and a.shape == client.action_space.shape
                for a in actions
            )
        )


def test_remote_proxy_refuses_attr_outside_allowlist(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """The server allowlists handle attributes; everything else is refused.

    _RemoteHandle.__getattr__ blocks underscore names client-side, so we craft raw
    requests to exercise the SERVER guard directly (an attacker could speak the protocol
    without the client). The guard is an ALLOWLIST, not a denylist:
    obj.__class__.__bases__[0].__subclasses__() is the classic path from any object to
    os.system, and a bare "block dunders" denylist would still leak non-underscore
    members. Both a dunder and a plausible-but-unlisted public name must be refused.
    """
    with _make_client(server, direct_env) as client:
        obs, _ = client.reset(seed=0)
        ocs = client.observation_space.devectorize(obs)
        handle_id = ocs._handle_id  # pylint: disable=protected-access
        for name in ("__class__", "data", "from_vec"):
            with pytest.raises(RuntimeError, match="Refusing attribute"):
                client._request(  # pylint: disable=protected-access
                    {
                        "cmd": "getattr",
                        "target": {"__handle__": handle_id},
                        "name": name,
                    }
                )


def test_remote_proxy_refuses_module_imported_attr(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """A module proxy exposes only the public planner API, not its imports.

    The planner modules do ``import numpy as np`` and import kinder.envs types
    at module scope. Without an allowlist, ``crv_motion_planning.np`` would hand
    the agent the live numpy module (np.load is host RCE; np.fromfile/save are
    host file I/O) and ``crv_motion_planning.SE2Pose`` would leak the withheld
    env types. Both must be refused; the real planner entry point is allowed.
    """
    with _make_client(server, direct_env) as client:
        client.reset(seed=0)
        for name in ("np", "SE2Pose", "MultiBody2D", "__loader__"):
            with pytest.raises(RuntimeError, match="public planner API"):
                client._request(  # pylint: disable=protected-access
                    {
                        "cmd": "getattr",
                        "target": {"__module__": "crv_motion_planning"},
                        "name": name,
                    }
                )


def test_remote_proxy_refuses_non_whitelisted_module(
    server: tuple[int, str], direct_env: KinderGeom2DEnv
) -> None:
    """The server refuses module targets outside the CRV whitelist (e.g. os)."""
    with _make_client(server, direct_env) as client:
        client.reset(seed=0)
        with pytest.raises(RuntimeError, match="non-whitelisted module"):
            client._request(  # pylint: disable=protected-access
                {
                    "cmd": "getattr",
                    "target": {"__module__": "os"},
                    "name": "system",
                }
            )


def test_blackbox_primitive_manifest_remote_module() -> None:
    """CRV primitives are tagged as remote modules (run on the host)."""
    manifest = blackbox_primitive_manifest(
        ["crv_motion_planning", "crv_motion_planning_grasp"]
    )
    by_name = {spec["name"]: spec for spec in manifest}
    assert by_name["crv_motion_planning"] == {
        "name": "crv_motion_planning",
        "kind": "remote_module",
        "module": "crv_motion_planning",
    }
    assert by_name["crv_motion_planning_grasp"] == {
        "name": "crv_motion_planning_grasp",
        "kind": "remote_module",
        "module": "crv_motion_planning_grasp",
    }
