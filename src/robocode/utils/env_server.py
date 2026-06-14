"""Black-box environment server.

Runs on the HOST (which has the full environment source) while a sandboxed
agent runs in a container that has no environment code at all. The agent's
test scripts use the ``env_client`` module copied into its sandbox, which
connects to this server over TCP and speaks a JSON-lines protocol:

* ``{"cmd": "reset", "seed": ..., "options": ...}`` ->
  ``{"obs": ..., "info": ...}``
* ``{"cmd": "step", "action": ...}`` ->
  ``{"obs": ..., "reward": ..., "terminated": ..., "truncated": ..., "info": ...}``
* ``{"cmd": "get_state"}`` -> ``{"state": ...}``
* ``{"cmd": "set_state", "state": ...}`` -> ``{"ok": true}``
* ``{"cmd": "check_action_collision", "state": ..., "action": ...}`` ->
  ``{"collision": ...}`` (runs the env-dependent collision primitive on the
  host so the sandbox can use it without the env source)
* ``{"cmd": "close"}`` -> closes the connection

Every request must carry the per-run ``token`` so that other hosts on the
network cannot drive the environment. Each client connection gets its own
fresh environment instance, so the agent can run parallel test scripts.

Security: the protocol is JSON only, never pickle; unpickling
agent-controlled bytes on the host would be a sandbox escape. Error replies
contain only the exception type and message, not the traceback, because
traceback frames include environment source lines that the agent must not
see; full tracebacks go to the server log on the host.

This module is the import-facing API (codecs, space serialization, and
:func:`env_server_running`); it deliberately imports nothing from the
environment, primitives, or approaches so the main experiment process stays
light. The serving loop lives in
:mod:`robocode.utils.env_server_runtime`, which
:func:`env_server_running` launches as a subprocess.
"""

from __future__ import annotations

import json
import logging
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Space

logger = logging.getLogger(__name__)

# Source of the client module that approaches copy into blackbox sandboxes.
ENV_CLIENT_SRC: Path = Path(__file__).parent / "env_client.py"

_NDARRAY_TAG = "__ndarray__"

# Codec registry: the extension point for environments whose observations or
# states are not numpy arrays. Register an encoder/decoder pair here (and
# mirror the decoding in env_client.py if the agent should receive that
# type). Keys of _DECODERS are the JSON tag names used on the wire.
_ENCODERS: dict[type, tuple[str, Callable[[Any], Any]]] = {}
_DECODERS: dict[str, Callable[[Any], Any]] = {}


def register_codec(
    cls: type,
    tag: str,
    encode_fn: Callable[[Any], Any],
    decode_fn: Callable[[Any], Any],
) -> None:
    """Register a codec for a custom observation/state type.

    ``encode_fn`` must return a JSON-serializable value; on the wire the
    object becomes ``{tag: encode_fn(obj)}`` and ``decode_fn`` receives the
    tagged value back.
    """
    _ENCODERS[cls] = (tag, encode_fn)
    _DECODERS[tag] = decode_fn


def encode(obj: Any) -> Any:
    """Encode a Python object as tagged JSON for the wire."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_TAG: obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        for key in obj:
            if not isinstance(key, str):
                raise TypeError(
                    f"Cannot encode dict key {key!r}: JSON requires string keys"
                )
        return {key: encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(value) for value in obj]
    if type(obj) in _ENCODERS:
        tag, encode_fn = _ENCODERS[type(obj)]
        return {tag: encode_fn(obj)}
    raise TypeError(
        f"No codec for type {type(obj).__name__}; register one with "
        "robocode.utils.env_server.register_codec"
    )


def decode(obj: Any) -> Any:
    """Decode tagged JSON from the wire back into Python objects."""
    if isinstance(obj, dict):
        if _NDARRAY_TAG in obj:
            return np.array(obj[_NDARRAY_TAG], dtype=np.dtype(obj["dtype"]))
        for tag, decode_fn in _DECODERS.items():
            if tag in obj:
                return decode_fn(obj[tag])
        return {key: decode(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [decode(value) for value in obj]
    return obj


def serialize_space(space: Space[Any]) -> dict[str, Any]:
    """Serialize a gym space's metadata for the agent's env_spaces.json.

    The extension point for environments with non-Box spaces: add a branch
    here and mirror it in env_client.SpaceInfo.
    """
    if isinstance(space, Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "dtype": str(space.dtype),
        }
    raise TypeError(
        f"No serializer for space type {type(space).__name__}; blackbox mode "
        "currently supports Box spaces. Add a branch in "
        "robocode.utils.env_server.serialize_space and mirror it in "
        "env_client.SpaceInfo"
    )


def write_env_spaces(
    sandbox_dir: Path,
    *,
    container_backend: str,
    port: int,
    token: str,
    observation_space: Space[Any],
    action_space: Space[Any],
    max_steps: int | None,
    primitives_manifest: list[dict[str, Any]] | None = None,
) -> None:
    """Write ``env_spaces.json``, the metadata the sandbox's env_client reads.

    The host is ``host.docker.internal`` for Docker (mapped to the host
    gateway via ``--add-host``) and ``127.0.0.1`` for the apptainer and local
    backends, which share the host's loopback. *primitives_manifest* (from
    :func:`robocode.primitives.blackbox_primitive_manifest`) tells the sandbox
    how to rebuild the eval-time primitives; the caller passes it rather than
    this module importing the primitives package, keeping the host process
    free of environment imports.
    """
    host = "host.docker.internal" if container_backend == "docker" else "127.0.0.1"
    meta = {
        "host": host,
        "port": port,
        "token": token,
        "observation_space": serialize_space(observation_space),
        "action_space": serialize_space(action_space),
        "max_steps": max_steps,
        "primitives": primitives_manifest or [],
    }
    (sandbox_dir / "env_spaces.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


@contextmanager
def env_server_running(
    env_cfg_json: str, sandbox_dir: Path
) -> Iterator[tuple[int, str]]:
    """Run the env server subprocess for the duration of a sandbox run.

    Writes the config, port file, and log next to *sandbox_dir* (i.e. in
    the run's output directory), starts the server, and yields
    ``(port, token)`` once the server is listening. The server stays up
    across rate-limit retries because this wraps the whole retry loop.
    """
    parent = sandbox_dir.resolve().parent
    config_path = parent / "env_server_config.json"
    config_path.write_text(env_cfg_json, encoding="utf-8")
    port_file = parent / "env_server_port"
    port_file.unlink(missing_ok=True)
    log_path = parent / "env_server.log"
    token = secrets.token_hex(16)

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robocode.utils.env_server_runtime",
                "--env-config",
                str(config_path),
                "--token",
                token,
                "--port-file",
                str(port_file),
                "--sandbox-dir",
                str(sandbox_dir.resolve()),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 60
        while not port_file.exists():
            if proc.poll() is not None:
                raise RuntimeError(
                    f"env server exited at startup; see {log_path}:\n"
                    f"{log_path.read_text(encoding='utf-8')}"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(f"env server did not start; see {log_path}")
            time.sleep(0.1)
        port = int(port_file.read_text(encoding="utf-8"))
        logger.info("env server listening on port %d", port)
        yield port, token
        if proc.poll() is not None:
            raise RuntimeError(f"env server died during the run; see {log_path}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
