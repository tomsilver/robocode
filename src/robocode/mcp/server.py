"""MCP server exposing robocode debugging tools.

Launched as a subprocess by the Claude CLI inside the sandbox.
Reads environment config from a JSON file, instantiates the env, and
serves rendering tools over stdio.

Usage::

    python -m robocode.mcp.server \
        --env-config /sandbox/.mcp/env_config.json \
        --tools render_state,render_policy
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import imageio.v3 as iio
from hydra.utils import instantiate
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf

from robocode.primitives import csp as csp_module
from robocode.primitives.check_action_collision import check_action_collision
from robocode.primitives.motion_planning import BiRRT
from robocode.primitives.render_policy import render_policy as _render_policy_fn
from robocode.primitives.render_state import render_state as _render_state_fn


def _build_env_and_primitives(
    env_config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Instantiate the environment and build the primitives dict."""
    cfg = OmegaConf.create(env_config)
    env = instantiate(cfg)
    env.reset(seed=0)

    primitives = {
        "check_action_collision": partial(check_action_collision, env),
        "render_state": partial(_render_state_fn, env),
        "csp": csp_module,
        "BiRRT": BiRRT,
    }
    return env, primitives


def create_server(
    env_config: dict[str, Any],
    tool_names: list[str],
    renders_dir: Path | None = None,
) -> FastMCP:
    """Create and configure the MCP server with the requested tools."""
    server = FastMCP("robocode-tools")
    env, primitives = _build_env_and_primitives(env_config)
    out_dir = renders_dir or Path("mcp_renders")

    if "render_state" in tool_names:

        @server.tool()
        def render_state(seed: int = 42) -> str:
            """Render the environment's initial state for a given seed.

            Returns the file path of the saved PNG image.
            """
            env.reset(seed=seed)
            state = env.get_state()
            frame = _render_state_fn(env, state)

            out = out_dir / f"state_seed{seed}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(str(out), frame)
            return str(out)

    if "render_policy" in tool_names:

        @server.tool()
        def render_policy(
            approach_dir: str = ".",
            seed: int = 42,
            max_steps: int = 1000,
            max_frames: int = 100,
        ) -> list[str]:
            """Run a full episode of the approach and save frames as PNGs.

            Returns the list of saved PNG file paths. Use a Task subagent to read and
            analyze the frames — do NOT read them directly to avoid context bloat.
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
    args = parser.parse_args()

    env_config_path = Path(args.env_config).resolve()
    env_config = json.loads(env_config_path.read_text(encoding="utf-8"))
    tool_names = [t.strip() for t in args.tools.split(",")]

    # Place renders next to the sandbox
    # (env_config is at <sandbox>/.mcp/env_config.json).
    sandbox_dir = env_config_path.parent.parent
    renders_dir = sandbox_dir / "mcp_renders"

    server = create_server(env_config, tool_names, renders_dir=renders_dir)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
