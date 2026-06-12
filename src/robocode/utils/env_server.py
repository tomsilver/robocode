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
* ``{"cmd": "close"}`` -> closes the connection

Every request must carry the per-run ``token`` so that other hosts on the
network cannot drive the environment. Each client connection gets its own
fresh environment instance, so the agent can run parallel test scripts.

Security: the protocol is JSON only, never pickle; unpickling
agent-controlled bytes on the host would be a sandbox escape. Error replies
contain only the exception type and message, not the traceback, because
traceback frames include environment source lines that the agent must not
see; full tracebacks go to the server log on the host.

Usage::

    python -m robocode.utils.env_server \
        --env-config /path/to/env_server_config.json \
        --token <hex token> \
        --port-file /path/to/env_server_port
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import socketserver
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Space
from hydra.utils import instantiate
from omegaconf import OmegaConf

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


class _EnvServer(socketserver.ThreadingTCPServer):
    """TCP server holding the env config, auth token, and sandbox dir."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self, env_config: dict[str, Any], token: str, sandbox_dir: Path
    ) -> None:
        super().__init__(("0.0.0.0", 0), _EnvRequestHandler)
        self.env_config = env_config
        self.token = token
        self.sandbox_dir = sandbox_dir


class _EnvRequestHandler(socketserver.StreamRequestHandler):
    """Serves one client connection with its own fresh env instance."""

    server: _EnvServer

    def handle(self) -> None:
        logger.info("New connection from %s", self.client_address)
        env = instantiate(OmegaConf.create(self.server.env_config))
        # Reset once so get_state/set_state/render work before the client
        # issues its own reset (mirrors the MCP server's env setup).
        env.reset(seed=0)
        try:
            for line in self.rfile:
                request = json.loads(line)
                if request.pop("token", None) != self.server.token:
                    logger.warning("Rejected request with bad token")
                    return
                if request["cmd"] == "close":
                    return
                # The one sanctioned broad except: errors triggered by
                # agent-controlled requests must be reported back over the
                # wire (and must not leak env source lines via traceback
                # frames), not kill the connection.
                try:
                    payload = _dispatch(env, request, self.server.sandbox_dir)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "Request %s failed:\n%s", request, traceback.format_exc()
                    )
                    payload = {"error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write(json.dumps(payload).encode("utf-8") + b"\n")
        finally:
            env.close()
            logger.info("Connection from %s closed", self.client_address)


def _dispatch(env: Any, request: dict[str, Any], sandbox_dir: Path) -> dict[str, Any]:
    """Execute one decoded request against the env and encode the reply."""
    cmd = request["cmd"]
    if cmd == "reset":
        obs, info = env.reset(seed=request.get("seed"), options=request.get("options"))
        return {"obs": encode(obs), "info": encode(info)}
    if cmd == "step":
        obs, reward, terminated, truncated, info = env.step(decode(request["action"]))
        return {
            "obs": encode(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": encode(info),
        }
    if cmd == "get_state":
        return {"state": encode(env.get_state())}
    if cmd == "set_state":
        env.set_state(decode(request["state"]))
        return {"ok": True}
    if cmd == "render_state":
        return {
            "path": _render_state(
                env,
                sandbox_dir,
                request.get("seed", 42),
                request.get("state"),
                request.get("label", ""),
            )
        }
    if cmd == "render_policy":
        return {
            "paths": _render_policy(
                env,
                sandbox_dir,
                request.get("seed", 42),
                request.get("max_steps", 1000),
                request.get("max_frames", 100),
            )
        }
    raise ValueError(f"Unknown command: {cmd!r}")


def _unique_render_path(directory: Path, stem: str, ext: str = ".png") -> Path:
    """Return ``directory/stem.ext``, appending _1, _2, ... if taken."""
    candidate = directory / f"{stem}{ext}"
    i = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{i}{ext}"
        i += 1
    return candidate


def _render_state(
    env: Any,
    sandbox_dir: Path,
    seed: int,
    state: list[float] | None,
    label: str,
) -> str:
    """Render a state to a PNG under ``mcp_renders/``; return the relative path.

    Mirrors the in-container render_state MCP tool, but runs on the host
    where the environment source and render code live. The PNG lands in the
    bind-mounted sandbox dir so the container sees it too. Render code is
    imported lazily to avoid a circular import with the approaches and to
    keep matplotlib/imageio off the env-server startup path.
    """
    # pylint: disable=import-outside-toplevel
    import imageio.v3 as iio

    from robocode.primitives.render_state import render_state as render_state_fn

    if state is not None:
        env_state = np.array(state, dtype=np.float32)
        env.set_state(env_state)
        stem = "state_custom"
    else:
        env.reset(seed=seed)
        env_state = env.get_state()
        stem = f"state_seed{seed}"
    if label:
        stem += f"_{label}"

    frame = render_state_fn(env, env_state)
    out_dir = sandbox_dir / "mcp_renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = _unique_render_path(out_dir, stem)
    iio.imwrite(str(out), frame)
    return str(out.relative_to(sandbox_dir))


def _render_policy(
    env: Any,
    sandbox_dir: Path,
    seed: int,
    max_steps: int,
    max_frames: int,
) -> list[str]:
    """Render a policy episode to PNGs; return paths relative to the sandbox.

    Loads ``approach.py`` from the sandbox dir and runs one episode on the
    host. Imports are lazy for the same reasons as :func:`_render_state`.
    """
    # pylint: disable=import-outside-toplevel
    from robocode.primitives import PRIMITIVE_NAME_TO_FILE, build_primitives
    from robocode.primitives.render_policy import render_policy as render_policy_fn

    primitives = build_primitives(env, list(PRIMITIVE_NAME_TO_FILE))
    out = sandbox_dir / "mcp_renders" / f"policy_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    filenames = render_policy_fn(
        env,
        primitives,
        str(sandbox_dir),
        seed,
        str(out),
        max_steps=max_steps,
        max_frames=max_frames,
    )
    return [str((out / f).relative_to(sandbox_dir)) for f in filenames]


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
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            [
                sys.executable,
                "-m",
                "robocode.utils.env_server",
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


def main() -> None:
    """Parse CLI args and serve forever."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-config", required=True, help="Path to JSON file with Hydra env config"
    )
    parser.add_argument(
        "--token", required=True, help="Auth token required on every request"
    )
    parser.add_argument(
        "--port-file", required=True, help="File to write the chosen port to"
    )
    parser.add_argument(
        "--sandbox-dir",
        required=True,
        help="Sandbox dir where render PNGs and approach.py live",
    )
    args = parser.parse_args()

    env_config = json.loads(Path(args.env_config).read_text(encoding="utf-8"))

    # Fail fast on a bad config or unserializable spaces before serving.
    env = instantiate(OmegaConf.create(env_config))
    serialize_space(env.observation_space)
    serialize_space(env.action_space)
    env.close()

    server = _EnvServer(env_config, args.token, Path(args.sandbox_dir))
    port = server.server_address[1]
    Path(args.port_file).write_text(str(port), encoding="utf-8")
    logger.info("Serving env %s on port %d", env_config.get("_target_"), port)
    server.serve_forever()


if __name__ == "__main__":
    main()
