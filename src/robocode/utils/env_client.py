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
    env.close()

Each call to ``make_env()`` creates a fresh, independent environment
instance on the server, so parallel test scripts are safe.

NOTE: ``approach.py`` must NOT import this module. It is evaluated by a
separate harness that calls it directly with real observations; this client
is only for test and exploration scripts.

This module must stay importable with only the standard library and numpy.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import numpy as np

_METADATA_PATH = Path(__file__).resolve().parent / "env_spaces.json"
_NDARRAY_TAG = "__ndarray__"


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


class BlackboxEnv:
    """A live environment instance, accessed over the wire."""

    def __init__(self, meta: dict[str, Any]) -> None:
        self.observation_space = SpaceInfo(meta["observation_space"])
        self.action_space = SpaceInfo(meta["action_space"])
        self.max_steps: int | None = meta["max_steps"]
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

    def set_state(self, state: Any) -> None:
        """Restore a state snapshot previously returned by get_state()."""
        self._request({"cmd": "set_state", "state": _encode(np.asarray(state))})

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


def make_env() -> BlackboxEnv:
    """Connect to the environment server; returns a fresh env instance."""
    meta = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
    return BlackboxEnv(meta)
