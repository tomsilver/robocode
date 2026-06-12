"""MCP server local-rendering variant (the normal, non-blackbox entry point).

Launched as a subprocess by the Claude CLI inside the sandbox when the
environment source IS available in the container. It instantiates the env
in-process and renders locally, which is why it depends on the environment
source, the primitives, and matplotlib/imageio. The tool registration and
logging are shared with :mod:`robocode.mcp.server` (the import-safe core);
this module only provides the local render implementations and entry point.

Usage::

    python -m robocode.mcp.local_render \
        --env-config /sandbox/.mcp/env_config.json \
        --tools render_state,render_policy \
        --log-file /path/to/mcp_server.log
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from hydra.utils import instantiate
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf

from robocode.mcp import MCP_SERVER_NAME
from robocode.mcp.server import _setup_logging, logger, register_tools
from robocode.primitives import PRIMITIVE_NAME_TO_FILE, build_primitives
from robocode.primitives.render_policy import render_policy as _render_policy_fn
from robocode.primitives.render_state import render_state as _render_state_fn


def _build_env_and_primitives(
    env_config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Instantiate the environment and build the primitives dict."""
    logger.info("Building environment and primitives from config")
    cfg = OmegaConf.create(env_config)
    env = instantiate(cfg)
    env.reset(seed=0)
    logger.info("Environment instantiated: %s", type(env).__name__)
    return env, build_primitives(env, list(PRIMITIVE_NAME_TO_FILE))


def _unique_path(directory: Path, stem: str, ext: str) -> Path:
    """Return ``directory/stem.ext``, appending _1, _2, ...

    if taken.
    """
    candidate = directory / f"{stem}{ext}"
    i = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{i}{ext}"
        i += 1
    return candidate


def build_local_server(
    tool_names: list[str],
    env_config: dict[str, Any],
    renders_dir: Path | None = None,
) -> FastMCP:
    """Create an MCP server that instantiates the env and renders locally."""
    env, primitives = _build_env_and_primitives(env_config)
    out_dir = renders_dir or Path("mcp_renders")

    def render_state_impl(seed: int, state: list[float] | None, label: str) -> str:
        if state is not None:
            env_state = np.array(state, dtype=np.float32)
            env.set_state(env_state)
        else:
            env.reset(seed=seed)
            env_state = env.get_state()

        frame = _render_state_fn(env, env_state)

        suffix = f"_{label}" if label else ""
        if state is not None:
            stem = f"state_custom{suffix}"
        else:
            stem = f"state_seed{seed}{suffix}"

        out_dir.mkdir(parents=True, exist_ok=True)
        out = _unique_path(out_dir, stem, ".png")
        iio.imwrite(str(out), frame)
        return str(out)

    def render_policy_impl(
        approach_dir: str, seed: int, max_steps: int, max_frames: int
    ) -> list[str]:
        out = out_dir / f"policy_seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        filenames = _render_policy_fn(
            env,
            primitives,
            approach_dir,
            seed,
            str(out),
            max_steps=max_steps,
            max_frames=max_frames,
        )
        return [str(out / f) for f in filenames]

    server = FastMCP(MCP_SERVER_NAME)
    register_tools(server, tool_names, render_state_impl, render_policy_impl)
    logger.info("MCP server created successfully")
    return server


def main() -> None:
    """Parse CLI args and start the local MCP server over stdio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-config",
        required=True,
        help="Path to JSON file with Hydra env config",
    )
    parser.add_argument(
        "--tools",
        required=True,
        help="Comma-separated list of tools to expose",
    )
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to write server-side log (stdout is reserved for MCP stdio)",
    )
    args = parser.parse_args()

    _setup_logging(Path(args.log_file))
    logger.info("MCP server starting (local): args=%s", vars(args))

    try:
        tool_names = [t.strip() for t in args.tools.split(",")]
        env_config_path = Path(args.env_config).resolve()
        logger.info("Loading env config from %s", env_config_path)
        env_config = json.loads(env_config_path.read_text(encoding="utf-8"))
        # Place renders next to the sandbox
        # (env_config is at <sandbox>/.mcp/env_config.json).
        sandbox_dir = env_config_path.parent.parent
        renders_dir = sandbox_dir / "mcp_renders"
        server = build_local_server(tool_names, env_config, renders_dir=renders_dir)
        logger.info("Starting stdio transport")
        server.run(transport="stdio")
        logger.info("MCP server shut down")
    except Exception:
        logger.critical("MCP server crashed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
