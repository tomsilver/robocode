"""MCP server for robocode debugging tools."""

import json
import shutil
from pathlib import Path

# Server name used by FastMCP and to build Claude CLI tool names
# (e.g. ``mcp__robocode-tools__render_state``).
MCP_SERVER_NAME = "robocode-tools"

# Descriptions shown to the Claude agent so it knows how to call each MCP tool.
MCP_TOOL_DESCRIPTIONS: dict[str, str] = {
    "render_state": (
        f"`mcp__{MCP_SERVER_NAME}__render_state(seed=42)` \u2014 renders the "
        "environment's initial state for a given seed and returns the path to "
        "a PNG file. Use this to visually understand the spatial layout, "
        "obstacle placement, and goal positions for a specific seed.\n"
        "  IMPORTANT: You must call this MCP tool DIRECTLY — MCP tools are "
        "NOT available inside Task subagents. Call it yourself, then delegate "
        "image reading to a Task subagent: have it Read the PNG, describe the "
        "scene, and return a concise summary. Delete the file when done."
    ),
    "render_policy": (
        f'`mcp__{MCP_SERVER_NAME}__render_policy(approach_dir=".", seed=42, '
        "max_steps=1000, max_frames=100)` \u2014 runs a full episode of the "
        "approach in `approach_dir/approach.py` on the given seed and saves "
        "each frame as a PNG. Returns a list of file paths. Use this to "
        "visually debug policy failures: see where the agent gets stuck, "
        "overshoots, or collides.\n"
        "  IMPORTANT: You must call this MCP tool DIRECTLY — MCP tools are "
        "NOT available inside Task subagents. Call it yourself to generate "
        "frames, then delegate reading to a Task subagent. The subagent "
        "should Read a sample of frames (e.g. first, middle, last, and any "
        "where behavior changes), describe the trajectory, identify failure "
        "modes, and return a concise text summary. Delete the output "
        "directory when done.\n"
        "  Typical workflow:\n"
        f"  1. Call mcp__{MCP_SERVER_NAME}__render_policy yourself to "
        "generate frames\n"
        '  2. Spawn a Task subagent: "Read these frame PNGs and describe '
        "the agent's trajectory. What goes wrong? Return a short summary.\"\n"
        "  3. Use the summary to fix your approach\n"
        "  4. Delete the frames directory with Bash"
    ),
}

# List of available MCP tool names.
MCP_TOOL_NAMES: tuple[str, ...] = tuple(MCP_TOOL_DESCRIPTIONS)


def mcp_tool_cli_names(tool_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return Claude CLI tool names (e.g. ``mcp__robocode-tools__render_state``)."""
    return tuple(f"mcp__{MCP_SERVER_NAME}__{t}" for t in tool_names)


def setup_mcp_config(
    sandbox_dir: Path,
    tool_names: tuple[str, ...],
    python_cmd: str,
    env_config_path: str,
    log_file_path: str,
) -> Path:
    """Write MCP server config into ``sandbox_dir/.mcp/``.

    Copies ``env_config.json`` from *sandbox_dir*'s parent into ``.mcp/``
    and writes ``mcp_config.json``.  Returns the path to ``mcp_config.json``.
    """
    mcp_dir = sandbox_dir / ".mcp"
    mcp_dir.mkdir(exist_ok=True)

    shutil.copy2(
        sandbox_dir.parent / "env_config.json",
        mcp_dir / "env_config.json",
    )

    # Use a shell wrapper so that stderr (import errors, tracebacks) is also
    # captured in the log file even if the Python process never reaches main().
    stderr_log_path = str(Path(log_file_path).with_suffix(".stderr.log"))
    server_cmd = (
        f"{python_cmd} -m robocode.mcp.server"
        f" --env-config {env_config_path}"
        f" --tools {','.join(tool_names)}"
        f" --log-file {log_file_path}"
        f" 2>>{stderr_log_path}"
    )
    mcp_config = {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "command": "bash",
                "args": ["-c", server_cmd],
            }
        }
    }
    config_path = mcp_dir / "mcp_config.json"
    config_path.write_text(json.dumps(mcp_config, indent=2))
    return config_path
