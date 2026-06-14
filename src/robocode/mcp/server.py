"""MCP server exposing robocode debugging tools (black-box entry point).

Launched as a subprocess by the Claude CLI inside the sandbox.

This module is the import-facing core: it registers the render tools and
serves the BLACK-BOX variant, where the environment source is absent from
the container so the tools proxy to the host-side env server via
``env_client``. It deliberately imports nothing from the environment,
primitives, or approaches, so it stays importable in a stripped blackbox
container. The local-rendering variant (which needs the env source,
primitives, and matplotlib/imageio) lives in
:mod:`robocode.mcp.local_render`, which reuses :func:`register_tools` from
here. This mirrors the env_server / env_server_runtime split.

Usage (blackbox)::

    python -m robocode.mcp.server \
        --env-spaces /sandbox/env_spaces.json \
        --tools render_state,render_policy \
        --log-file /path/to/mcp_server.log
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from robocode.mcp import MCP_SERVER_NAME
from robocode.utils.env_client import BlackboxEnv

logger = logging.getLogger(MCP_SERVER_NAME)

RenderStateImpl = Callable[[int, "list[float] | None", str], str]
RenderPolicyImpl = Callable[[str, int, int, int], "list[str]"]


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


def register_tools(
    server: FastMCP,
    tool_names: list[str],
    render_state_impl: RenderStateImpl,
    render_policy_impl: RenderPolicyImpl,
) -> None:
    """Register the requested render tools, delegating to the given impls.

    The tool signatures and docstrings (what the agent sees) live here and are shared by
    the local and blackbox variants; only the implementations differ.
    """
    logger.info("Registering tools: %s", tool_names)

    if "render_state" in tool_names:

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
            return render_state_impl(seed, state, label)

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
            return render_policy_impl(approach_dir, seed, max_steps, max_frames)

        logger.info("Registered tool: render_policy")


def build_blackbox_server(tool_names: list[str], env_spaces_path: Path) -> FastMCP:
    """Create an MCP server whose render tools proxy to the host env server.

    The environment source is absent from this container, so rendering runs
    on the host (via ``env_client``), which writes PNGs into the shared
    sandbox mount. The returned sandbox-relative paths are translated back to
    absolute paths in this container.
    """
    meta = json.loads(env_spaces_path.read_text(encoding="utf-8"))
    sandbox_root = env_spaces_path.resolve().parent
    client = BlackboxEnv(meta, sandbox_root=sandbox_root)
    logger.info("MCP server in blackbox mode, proxying to host env server")

    def render_state_impl(seed: int, state: list[float] | None, label: str) -> str:
        rel = client.render_state(seed=seed, state=state, label=label)
        return str(sandbox_root / rel)

    def render_policy_impl(
        _approach_dir: str, seed: int, max_steps: int, max_frames: int
    ) -> list[str]:
        # render_policy runs the sandbox's own approach.py in this container
        # (only the per-state render is proxied to the host), so the
        # agent-supplied approach_dir does not apply in blackbox mode.
        rels = client.render_policy(
            seed=seed, max_steps=max_steps, max_frames=max_frames
        )
        return [str(sandbox_root / r) for r in rels]

    server = FastMCP(MCP_SERVER_NAME)
    register_tools(server, tool_names, render_state_impl, render_policy_impl)
    logger.info("MCP server created successfully")
    return server


def main() -> None:
    """Parse CLI args and start the blackbox MCP server over stdio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-spaces",
        required=True,
        help="Path to env_spaces.json with the host env server connection info",
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
    logger.info("MCP server starting (blackbox): args=%s", vars(args))

    try:
        tool_names = [t.strip() for t in args.tools.split(",")]
        env_spaces_path = Path(args.env_spaces).resolve()
        logger.info("Loading env server connection info from %s", env_spaces_path)
        server = build_blackbox_server(tool_names, env_spaces_path)
        logger.info("Starting stdio transport")
        server.run(transport="stdio")
        logger.info("MCP server shut down")
    except Exception:
        logger.critical("MCP server crashed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
