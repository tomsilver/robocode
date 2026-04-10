"""MCP server exposing robocode debugging tools.

Launched as a subprocess by the Claude CLI inside the sandbox.
Reads environment config from a JSON file, instantiates the env, and
serves rendering tools over stdio.

Usage::

    python -m robocode.mcp.server \
        --env-config /sandbox/.mcp/env_config.json \
        --tools render_state,render_policy \
        --log-file /path/to/mcp_server.log
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from hydra.utils import instantiate
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf

from robocode.mcp import MCP_SERVER_NAME
from robocode.primitives import PRIMITIVE_NAME_TO_FILE, build_primitives
from robocode.primitives.render_policy import render_policy as _render_policy_fn
from robocode.primitives.render_state import render_state as _render_state_fn

logger = logging.getLogger(MCP_SERVER_NAME)


def _setup_logging(log_file: Path) -> None:
    """Configure file-based logging for the MCP server.

    stdout is reserved for the MCP stdio transport, so all diagnostics go to a file.
    """
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logger.addHandler(handler)


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


def _logged_tool(fn):  # type: ignore[type-arg]
    """Decorator that logs tool calls, return values, and exceptions."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info("%s called with %s %s", fn.__name__, args, kwargs)
        try:
            result = fn(*args, **kwargs)
        except Exception:
            logger.error("%s failed:\n%s", fn.__name__, traceback.format_exc())
            raise
        logger.info("%s returned: %s", fn.__name__, result)
        return result

    return wrapper


def create_server(
    env_config: dict[str, Any],
    tool_names: list[str],
    renders_dir: Path | None = None,
) -> FastMCP:
    """Create and configure the MCP server with the requested tools."""
    logger.info("Creating MCP server, requested tools: %s", tool_names)
    server = FastMCP(MCP_SERVER_NAME)
    env, primitives = _build_env_and_primitives(env_config)
    out_dir = renders_dir or Path("mcp_renders")

    if "render_state" in tool_names:

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

        @server.tool()
        @_logged_tool
        def render_state(
            seed: int = 42,
            state: list[float] | None = None,
            label: str = "",
        ) -> str:
            """Render the environment state as a PNG image.

            There are two modes:

            1. **Reset mode** (default): pass only ``seed`` to render the
               initial state after ``env.reset(seed=seed)``.
            2. **Arbitrary state mode**: pass ``state`` as a flat list of
               floats (the same format returned by
               ``env.get_state().tolist()``). ``seed`` is ignored when
               ``state`` is provided.

            Parameters
            ----------
            seed:
                Seed for resetting the environment (used only when
                ``state`` is not provided).
            state:
                Optional flat list of floats representing an arbitrary
                environment state. When provided, renders this state
                instead of resetting.
            label:
                Optional short label used in the output filename for
                easier identification (e.g. "after_grasp", "step42").

            Returns the file path of the saved PNG image.
            """
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

        logger.info("Registered tool: render_state")

    if "render_policy" in tool_names:

        @server.tool()
        @_logged_tool
        def render_policy(
            approach_dir: str = ".",
            seed: int = 42,
            max_steps: int = 1000,
            max_frames: int = 100,
        ) -> list[str]:
            """Run a full episode of the approach and save frames as PNGs.

            Returns the list of saved PNG file paths. Use a subagent to read and analyze
            the frames — do NOT read them directly to avoid context bloat.
            """
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

        logger.info("Registered tool: render_policy")

    logger.info("MCP server created successfully")
    return server


def main() -> None:
    """Parse CLI args and start the MCP server over stdio."""
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
    logger.info("MCP server starting: args=%s", vars(args))

    try:
        env_config_path = Path(args.env_config).resolve()
        logger.info("Loading env config from %s", env_config_path)
        env_config = json.loads(env_config_path.read_text(encoding="utf-8"))
        tool_names = [t.strip() for t in args.tools.split(",")]

        # Place renders next to the sandbox
        # (env_config is at <sandbox>/.mcp/env_config.json).
        sandbox_dir = env_config_path.parent.parent
        renders_dir = sandbox_dir / "mcp_renders"

        server = create_server(env_config, tool_names, renders_dir=renders_dir)
        logger.info("Starting stdio transport")
        server.run(transport="stdio")
        logger.info("MCP server shut down")
    except Exception:
        logger.critical("MCP server crashed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
