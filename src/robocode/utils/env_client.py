"""Client for the black-box environment server.

This file is copied into the agent sandbox. Import it in test and
exploration scripts to interact with the environment without access to its
source code::

    from env_client import make_env

    env = make_env()
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    state = env.get_state()
    env.set_state(state)
    primitives = env.make_primitives()  # same dict the eval harness passes
    env.close()

Each call to ``make_env()`` creates a fresh, independent environment
instance on the server, so parallel test scripts are safe.

NOTE: ``approach.py`` must NOT import this module. It is evaluated by a
separate harness that calls it directly with real observations; this client
is only for test and exploration scripts.

This module must stay importable with only the standard library and numpy.
"""

from __future__ import annotations

import importlib
import json
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

_METADATA_PATH = Path(__file__).resolve().parent / "env_spaces.json"
_NDARRAY_TAG = "__ndarray__"
_HANDLE_TAG = "__handle__"
_MODULE_TAG = "__module__"
_SET_TAG = "__set__"


def _encode(obj: Any) -> Any:
    """Encode values as tagged JSON (mirrors robocode.utils.env_server)."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_TAG: obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {key: _encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(value) for value in obj]
    return obj


def _decode(obj: Any) -> Any:
    """Decode tagged JSON (mirrors robocode.utils.env_server)."""
    if isinstance(obj, dict):
        if _NDARRAY_TAG in obj:
            return np.array(obj[_NDARRAY_TAG], dtype=np.dtype(obj["dtype"]))
        return {key: _decode(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_decode(value) for value in obj]
    return obj


def _encode_ref(obj: Any) -> Any:
    """Encode args/targets for the remote-object proxy commands.

    Same as ``_encode`` but ``_RemoteHandle`` objects serialize to their
    ``{"__handle__": id}`` token so the host resolves them from its registry.
    Used only by devectorize/vectorize/getattr/call requests.
    """
    if isinstance(obj, _RemoteHandle):
        return {_HANDLE_TAG: obj._handle_id}  # pylint: disable=protected-access
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_TAG: obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {key: _encode_ref(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_ref(value) for value in obj]
    if isinstance(obj, (set, frozenset)):
        return {_SET_TAG: [_encode_ref(value) for value in obj]}
    return obj


class SpaceInfo:
    """Metadata for a Box space: shape, low, high, dtype, and sample()."""

    def __init__(self, spec: dict[str, Any]) -> None:
        if spec["type"] != "Box":
            raise TypeError(
                f"Unsupported space type {spec['type']!r}; extend SpaceInfo "
                "alongside the server's serialize_space"
            )
        self.shape: tuple[int, ...] = tuple(spec["shape"])
        self.dtype: np.dtype[Any] = np.dtype(spec["dtype"])
        self.low: np.ndarray = np.array(spec["low"], dtype=self.dtype)
        self.high: np.ndarray = np.array(spec["high"], dtype=self.dtype)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample uniformly between low and high."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)


class _RemoteHandle:
    """A reference to a host-side object, used over the env server.

    Attribute access and calls are proxied to the host: ``handle.method(...)``
    issues a ``getattr`` (returning another handle for the bound method) then a
    ``call``. Returned values are decoded the same way as any response, so a
    scalar/ndarray comes back inline and another object comes back as a nested
    ``_RemoteHandle``. Private/dunder attributes are refused client-side, both
    to avoid surprise round-trips and to mirror the host security guard.
    """

    def __init__(self, client: "BlackboxEnv", handle_id: str) -> None:
        # Use object.__setattr__-free plain assignment; __getattr__ only fires
        # for names not found normally, and these two are set here.
        self._client = client
        self._handle_id = handle_id

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                f"refusing private/dunder attribute access on remote handle: "
                f"{name!r}"
            )
        response = self._client._request(  # pylint: disable=protected-access
            {
                "cmd": "getattr",
                "target": {_HANDLE_TAG: self._handle_id},
                "name": name,
            }
        )
        return self._client._decode_ref(  # pylint: disable=protected-access
            response["result"]
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        response = self._client._request(  # pylint: disable=protected-access
            {
                "cmd": "call",
                "target": {_HANDLE_TAG: self._handle_id},
                "args": _encode_ref(list(args)),
                "kwargs": _encode_ref(kwargs),
            }
        )
        return self._client._decode_ref(  # pylint: disable=protected-access
            response["result"]
        )


class _RemoteModule:
    """A reference to a whitelisted host module (e.g. crv_motion_planning).

    Attribute access proxies a ``getattr`` against the ``{"__module__": name}``
    target, so ``module.plan_crv_actions`` returns a callable handle the agent
    can invoke. Mirrors how a remote-module primitive is used at eval time,
    where ``primitives['crv_motion_planning']`` is the real module.
    """

    def __init__(self, client: "BlackboxEnv", short_name: str) -> None:
        self._client = client
        self._short_name = short_name

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                f"refusing private/dunder attribute access on remote module: "
                f"{name!r}"
            )
        response = self._client._request(  # pylint: disable=protected-access
            {
                "cmd": "getattr",
                "target": {_MODULE_TAG: self._short_name},
                "name": name,
            }
        )
        return self._client._decode_ref(  # pylint: disable=protected-access
            response["result"]
        )


class _BlackboxObservationSpace(SpaceInfo):
    """Box-space metadata plus client-backed devectorize/vectorize.

    Stands in for the eval-time ``ObjectCentricBoxSpace``: it keeps the
    shape/low/high/dtype/sample() of a plain ``SpaceInfo`` and adds
    ``devectorize(obs)`` / ``vectorize(ocs)`` that proxy to the host. The agent
    writes ``observation_space.devectorize(obs)`` exactly as it would at eval.
    """

    def __init__(self, spec: dict[str, Any], client: "BlackboxEnv") -> None:
        super().__init__(spec)
        self._client = client

    def devectorize(self, obs: Any) -> Any:
        """Return an object-centric view of *obs* as a remote handle."""
        return self._client.devectorize(obs)

    def vectorize(self, state: Any) -> np.ndarray:
        """Return the flat observation vector for an object-centric state."""
        return self._client.vectorize(state)


class BlackboxEnv:
    """A live environment instance, accessed over the wire."""

    def __init__(self, meta: dict[str, Any], sandbox_root: Any = None) -> None:
        self.observation_space = _BlackboxObservationSpace(
            meta["observation_space"], self
        )
        self.action_space = SpaceInfo(meta["action_space"])
        self.max_steps: int | None = meta["max_steps"]
        # How make_primitives rebuilds the eval-time primitives dict (see
        # robocode.primitives.blackbox_primitive_manifest). Empty when the
        # run configures no primitives.
        self._primitive_manifest: list[dict[str, Any]] = meta.get("primitives", [])
        # Where render_policy looks for approach.py and make_primitives imports
        # generic primitive sources; the sandbox dir in real runs (set by
        # make_env), the current dir for ad-hoc construction.
        self._sandbox_root = (
            Path(sandbox_root) if sandbox_root is not None else Path.cwd()
        )
        self._token: str = meta["token"]
        self._sock = socket.create_connection((meta["host"], meta["port"]))
        self._file = self._sock.makefile("rwb")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["token"] = self._token
        self._file.write(json.dumps(payload).encode("utf-8") + b"\n")
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise RuntimeError("Connection to the environment server was closed")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"Environment server error: {response['error']}")
        return _decode(response)

    def _decode_ref(self, obj: Any) -> Any:
        """Decode a remote-command result, mapping handles to _RemoteHandle.

        ``_request`` already ran ``_decode`` over the response (so ndarrays are
        arrays), but it leaves ``{"__handle__": id}`` dicts intact since it does
        not know the handle tag. Here we turn those into ``_RemoteHandle``
        objects and recurse through the remaining containers.
        """
        if isinstance(obj, dict):
            if _HANDLE_TAG in obj:
                return _RemoteHandle(self, obj[_HANDLE_TAG])
            if _SET_TAG in obj:
                return {self._decode_ref(value) for value in obj[_SET_TAG]}
            return {key: self._decode_ref(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._decode_ref(value) for value in obj]
        return obj

    def devectorize(self, obs: Any) -> _RemoteHandle:
        """Return an object-centric view of *obs* (a remote ObjectCentricState).

        Proxies ``observation_space.devectorize`` to the host. The returned
        handle exposes the public ObjectCentricState API (``get_object_names``,
        ``get_object_from_name``, ``get_objects``, ``get``, ``set``, ``copy``,
        ...) over the wire. Pass it straight to the CRV planners from
        ``make_primitives``.
        """
        response = self._request(
            {"cmd": "devectorize", "obs": _encode(np.asarray(obs))}
        )
        return self._decode_ref(response["result"])

    def vectorize(self, state: Any) -> np.ndarray:
        """Return the flat observation vector for an object-centric state handle."""
        response = self._request({"cmd": "vectorize", "state": _encode_ref(state)})
        return self._decode_ref(response["result"])

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment; returns (obs, info)."""
        response = self._request({"cmd": "reset", "seed": seed, "options": options})
        return response["obs"], response["info"]

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the environment; returns (obs, reward, terminated, truncated, info)."""
        action = np.asarray(action, dtype=self.action_space.dtype)
        response = self._request({"cmd": "step", "action": _encode(action)})
        return (
            response["obs"],
            response["reward"],
            response["terminated"],
            response["truncated"],
            response["info"],
        )

    def get_state(self) -> np.ndarray:
        """Return a snapshot of the full environment state."""
        return self._request({"cmd": "get_state"})["state"]

    def render_state(self, seed: int = 42, state: Any = None, label: str = "") -> str:
        """Render a state on the host; returns a PNG path inside the sandbox.

        Either pass ``seed`` to render the initial state after a reset, or
        ``state`` (a flat list of floats, as from ``get_state().tolist()``)
        to render an arbitrary state. The path is relative to the sandbox dir.
        """
        if state is not None and isinstance(state, np.ndarray):
            state = state.tolist()
        return self._request(
            {"cmd": "render_state", "seed": seed, "state": state, "label": label}
        )["path"]

    def render_policy(
        self,
        seed: int = 42,
        max_steps: int = 1000,
        max_frames: int = 100,
        approach_path: Any = None,
    ) -> list[str]:
        """Run an episode of approach.py here and render the visited states.

        The policy runs in this process: only ``get_action`` (which needs the
        observation alone) executes locally, while each visited state is
        rendered on the host via ``render_state``. No approach code runs on the
        host, so a black-box agent cannot reach the env source through
        rendering. Returns the PNG paths (relative to the sandbox dir).
        """
        path = (
            Path(approach_path)
            if approach_path is not None
            else self._sandbox_root / "approach.py"
        )
        approach = _load_generated_approach(
            path, self.action_space, self.observation_space, self.make_primitives()
        )
        obs, info = self.reset(seed=seed)
        approach.reset(obs, info)
        states = [self.get_state()]
        for _ in range(max_steps):
            obs, reward, terminated, truncated, info = self.step(
                approach.get_action(obs)
            )
            if hasattr(approach, "update"):
                approach.update(obs, reward, terminated or truncated, info)
            states.append(self.get_state())
            if terminated or truncated:
                break
        return [
            self.render_state(state=state, label=f"policy_seed{seed}_{i:04d}")
            for i, state in enumerate(states[:max_frames])
        ]

    def set_state(self, state: Any) -> None:
        """Restore a state snapshot previously returned by get_state()."""
        self._request({"cmd": "set_state", "state": _encode(np.asarray(state))})

    def check_action_collision(self, state: Any, action: Any) -> bool:
        """Return True if taking *action* in *state* causes a collision.

        Runs the env-dependent collision primitive on the host (the sandbox
        lacks the env source), against this connection's env. Mirrors the
        eval-time ``primitives['check_action_collision']`` callable, so test
        scripts that get it from ``make_primitives`` call it the same way.
        """
        response = self._request(
            {
                "cmd": "check_action_collision",
                "state": _encode(np.asarray(state)),
                "action": _encode(np.asarray(action, dtype=self.action_space.dtype)),
            }
        )
        return response["collision"]

    def make_primitives(self) -> dict[str, Any]:
        """Build the eval-time primitives dict for this black-box env.

        Black-box counterpart of ``robocode.primitives.build_primitives``:
        env-dependent primitives proxy to the host, while generic ones are
        imported from their copies under ``primitives/`` in the sandbox. Use
        it in test scripts so they exercise exactly the primitives the harness
        passes to ``GeneratedApproach`` at evaluation time::

            from env_client import make_env

            env = make_env()
            primitives = env.make_primitives()
        """
        sandbox_dir = str(self._sandbox_root.resolve())
        if sandbox_dir not in sys.path:
            sys.path.insert(0, sandbox_dir)
        primitives: dict[str, Any] = {}
        for spec in self._primitive_manifest:
            if spec["kind"] == "host_proxy":
                primitives[spec["name"]] = getattr(self, spec["name"])
            elif spec["kind"] == "remote_module":
                # The module source is withheld from the sandbox; reach it on
                # the host via a remote-module proxy (e.g. crv_motion_planning).
                primitives[spec["name"]] = _RemoteModule(self, spec["module"])
            else:
                module = importlib.import_module(f"primitives.{spec['module']}")
                primitives[spec["name"]] = (
                    module if spec["attr"] is None else getattr(module, spec["attr"])
                )
        return primitives

    def close(self) -> None:
        """Close the connection (and the server-side environment)."""
        self._file.write(
            json.dumps({"cmd": "close", "token": self._token}).encode("utf-8") + b"\n"
        )
        self._file.flush()
        self._sock.close()

    def __enter__(self) -> "BlackboxEnv":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _load_generated_approach(
    path: Path, action_space: Any, observation_space: Any, primitives: dict[str, Any]
) -> Any:
    """Exec ``approach.py`` and instantiate its ``GeneratedApproach``.

    Mirrors ``robocode.utils.episode.load_generated_approach``; duplicated here
    (like the codec above) because this module must stay importable with only
    the standard library and numpy, since it is copied standalone into the
    sandbox. Used only by ``render_policy`` for visual debugging.
    """
    sandbox_dir = str(path.resolve().parent)
    added = sandbox_dir not in sys.path
    if added:
        sys.path.insert(0, sandbox_dir)
    try:
        namespace: dict[str, Any] = {"__file__": str(path)}
        exec(  # pylint: disable=exec-used
            compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace
        )
    finally:
        if added:
            sys.path.remove(sandbox_dir)
    return namespace["GeneratedApproach"](
        action_space, observation_space, primitives=primitives
    )


def make_env(metadata_path: Any = None) -> BlackboxEnv:
    """Connect to the environment server; returns a fresh env instance.

    *metadata_path* defaults to ``env_spaces.json`` next to this module
    (the agent's sandbox copy); the MCP server passes an explicit path.
    """
    path = Path(metadata_path) if metadata_path is not None else _METADATA_PATH
    meta = json.loads(path.read_text(encoding="utf-8"))
    return BlackboxEnv(meta, sandbox_root=path.parent)
