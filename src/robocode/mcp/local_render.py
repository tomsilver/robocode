"""MCP server local-rendering variant (the normal, non-blackbox entry point).

Launched as a subprocess by the Claude CLI inside the sandbox when the
environment source IS available in the container. It instantiates the env
in-process and renders locally, which is why it depends on the environment
source and matplotlib/imageio. It does NOT import ``robocode.primitives`` (the
agentic mount strips that package); render code comes from the
``robocode.rendering`` package, source-free metadata from
``robocode.primitive_specs``, and the primitives dict is built from the
in-sandbox copied ``primitives/`` package. The tool registration and logging
are shared with :mod:`robocode.mcp.server` (the import-safe core); this module
only provides the local render implementations and entry point.

Usage::

    python -m robocode.mcp.local_render \
        --env-config /sandbox/.mcp/env_config.json \
        --tools render_state,render_policy \
        --log-file /path/to/mcp_server.log
"""

from __future__ import annotations

import argparse
import functools
import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from hydra.utils import instantiate
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf

from robocode.mcp import MCP_SERVER_NAME
from robocode.mcp.server import (
    _setup_logging,
    add_transport_args,
    logger,
    register_tools,
    run_server,
)
from robocode.primitive_specs import (
    ENV_DEPENDENT_ATTR,
    ENV_DEPENDENT_PRIMITIVES,
    GENERIC_PRIMITIVE_ATTR,
    PRIMITIVE_NAME_TO_FILE,
)
from robocode.rendering.render_policy import render_policy as _render_policy_fn
from robocode.rendering.render_state import render_state as _render_state_fn
from robocode.utils.render_paths import safe_label, unique_path


def _build_env(env_config: dict[str, Any]) -> Any:
    """Instantiate the environment and reset it once."""
    logger.info("Building environment from config")
    cfg = OmegaConf.create(env_config)
    env = instantiate(cfg)
    env.reset(seed=0)
    logger.info("Environment instantiated: %s", type(env).__name__)
    return env


def _build_sandbox_primitives(env: Any, primitives_dir: Path | None) -> dict[str, Any]:
    """Build the primitives dict from the copied in-sandbox `primitives/` package.

    Mirrors robocode.primitives.build_primitives but imports the generic
    primitive modules from the sandbox's top-level `primitives` package (the
    subset copied by the sandbox setup) instead of the stripped robocode.primitives.
    Returns an empty dict when no primitives were copied.
    """
    if primitives_dir is None or not primitives_dir.exists():
        return {}
    sandbox_root = str(primitives_dir.parent.resolve())
    if sandbox_root not in sys.path:
        sys.path.insert(0, sandbox_root)
    file_to_names: dict[str, list[str]] = {}
    for prim_name, file_stem in PRIMITIVE_NAME_TO_FILE.items():
        file_to_names.setdefault(file_stem, []).append(prim_name)
    out: dict[str, Any] = {}
    for py in sorted(primitives_dir.glob("*.py")):
        for prim_name in file_to_names.get(py.stem, []):
            module = importlib.import_module(f"primitives.{py.stem}")
            if prim_name in ENV_DEPENDENT_PRIMITIVES:
                obj = getattr(module, ENV_DEPENDENT_ATTR[prim_name])
                # A class binds the env by instantiation (BilevelModels(env)); a
                # function binds it by partial (check_action_collision).
                out[prim_name] = (
                    obj(env) if inspect.isclass(obj) else functools.partial(obj, env)
                )
            else:
                attr = GENERIC_PRIMITIVE_ATTR[prim_name]
                out[prim_name] = module if attr is None else getattr(module, attr)
    return out


def build_local_server(
    tool_names: list[str],
    env_config: dict[str, Any],
    renders_dir: Path | None = None,
    primitives_dir: Path | None = None,
) -> FastMCP:
    """Create an MCP server that instantiates the env and renders locally."""
    env = _build_env(env_config)
    primitives = _build_sandbox_primitives(env, primitives_dir)
    out_dir = renders_dir or Path("mcp_renders")

    def render_state_impl(seed: int, state: list[float] | None, label: str) -> str:
        if state is not None:
            env_state = np.array(state, dtype=np.float32)
            env.set_state(env_state)
        else:
            env.reset(seed=seed)
            env_state = env.get_state()

        frame = _render_state_fn(env, env_state)

        safe = safe_label(label)
        suffix = f"_{safe}" if safe else ""
        if state is not None:
            stem = f"state_custom{suffix}"
        else:
            stem = f"state_seed{seed}{suffix}"

        out_dir.mkdir(parents=True, exist_ok=True)
        out = unique_path(out_dir, stem, ".png")
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
    add_transport_args(parser)
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
        primitives_dir = sandbox_dir / "primitives"
        server = build_local_server(
            tool_names,
            env_config,
            renders_dir=renders_dir,
            primitives_dir=primitives_dir,
        )
        run_server(server, args.transport, args.host, args.port)
        logger.info("MCP server shut down")
    except Exception:
        logger.critical("MCP server crashed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
