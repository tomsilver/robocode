"""MCP server for robocode debugging tools."""

# Server name used by FastMCP and to build Claude CLI tool names
# (e.g. ``mcp__robocode-tools__render_state``).
MCP_SERVER_NAME = "robocode-tools"

# Descriptions shown to the Claude agent so it knows how to call each MCP tool.
MCP_TOOL_DESCRIPTIONS: dict[str, str] = {
    "render_state": (
        f"`mcp__{MCP_SERVER_NAME}__render_state(seed=42, state=None, "
        'label="")` \u2014 renders an environment state as a PNG and returns '
        "the file path.\n"
        "  Two modes:\n"
        "  1. **Reset mode** (default): pass `seed` to render the initial "
        "state after `env.reset(seed=seed)`.\n"
        "  2. **Arbitrary state mode**: pass `state` as a flat list of floats "
        "to render any state you want. `seed` is ignored when `state` is "
        "provided.\n"
        "  The optional `label` parameter is included in the output filename "
        'for easier identification (e.g. label="after_grasp").\n'
        "  Use reset mode to visually understand the spatial layout, obstacle "
        "placement, and goal positions. Use arbitrary state mode to visualize "
        "intermediate states during debugging, e.g. after applying actions or "
        "to verify a planned trajectory.\n"
        "  How to get a state list:\n"
        "  - From an existing observation: `obs.tolist()`\n"
        "  - To inspect/modify named features: use "
        "`env.observation_space.devectorize(obs)` to get an "
        "`ObjectCentricState` (a dict of `{Object: feature_array}`), "
        "modify it, then `env.observation_space.vectorize(ocs).tolist()` "
        "to convert back.\n"
        "  - To build a state from scratch: construct an "
        "`ObjectCentricState(data={obj: np.array([...]), ...}, "
        "type_features=env.observation_space.type_features)` using the "
        "objects from `env.observation_space.constant_objects`, then "
        "vectorize it.\n"
        "  IMPORTANT: You must call this MCP tool DIRECTLY \u2014 MCP tools are "
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

# System prompt suffix appended when MCP tools are available.
MCP_TOOLS_SYSTEM_PROMPT_SUFFIX = (
    " IMPORTANT: You have visual debugging tools (render_state, render_policy). "
    "Start by calling render_state to see the environment before writing code. "
    "When your approach fails, call render_policy to visually diagnose the "
    "failure BEFORE guessing at fixes. You can also render arbitrary states by "
    "passing a flat list of floats to render_state's `state` parameter; use "
    "devectorize/vectorize on env.observation_space to construct or modify "
    "states with named features. "
    "CRITICAL: MCP tools are only available to YOU directly, they CANNOT be "
    "called from inside Task subagents. Always call MCP tools yourself, then "
    "delegate image reading to a Task subagent."
)


def mcp_tool_cli_names(tool_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return Claude CLI tool names (e.g. ``mcp__robocode-tools__render_state``)."""
    return tuple(f"mcp__{MCP_SERVER_NAME}__{t}" for t in tool_names)
