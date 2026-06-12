"""Black-box environment server runtime (the subprocess entry point).

Launched by :func:`robocode.utils.env_server.env_server_running` as
``python -m robocode.utils.env_server_runtime``. This module holds the
serving loop and the render handlers, which depend on the environment
source, the approaches, and matplotlib/imageio. Keeping it separate from
``env_server`` (the import-facing API used by the approaches) means the
main experiment process never transitively imports any of that heavy code,
and avoids the import cycle env_server -> render_policy -> agentic_approach
-> env_server.

See :mod:`robocode.utils.env_server` for the wire protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import socketserver
import sys
import traceback
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from robocode.primitives import PRIMITIVE_NAME_TO_FILE, build_primitives
from robocode.primitives.render_policy import render_policy as render_policy_fn
from robocode.primitives.render_state import render_state as render_state_fn
from robocode.utils.env_server import decode, encode, serialize_space

logger = logging.getLogger(__name__)


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
                # Errors from agent-controlled requests are reported back
                # over the wire (without traceback frames, which would leak
                # env source lines) rather than killing the connection. The
                # broad catch is intentional: any failure the agent triggers
                # must reach its test script as a message.
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
    """Return ``directory/stem.ext``, appending _1, _2, ...

    if taken.
    """
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

    Mirrors the in-container render_state MCP tool, but runs on the host where the
    environment source and render code live. The PNG lands in the bind-mounted sandbox
    dir so the container sees it too.
    """
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
    host.
    """
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
