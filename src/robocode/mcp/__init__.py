"""MCP server for robocode debugging tools."""

import json
import shutil
from pathlib import Path
from typing import Any

# Server name used by FastMCP and to build Claude CLI tool names
# (e.g. ``mcp__robocode-tools__render_state``).
MCP_SERVER_NAME = "robocode-tools"

# Startup wait (milliseconds) the agent CLI is given for the render MCP server
# to connect, passed as the ``MCP_TIMEOUT`` env var by every backend. The server
# imports the MCP framework plus the env/render stack (~2s) before it can answer
# the stdio handshake; if the CLI snapshots its tool list before that, the render
# tools are silently absent for the whole run ("No such tool available"). A
# generous timeout makes the CLI wait for the server instead of racing it; the
# CLI still proceeds as soon as the server connects, so this only adds latency
# when the server is genuinely slow.
MCP_STARTUP_TIMEOUT_MS = 60000

# HTTP transport. Instead of letting the agent CLI spawn the render server over
# stdio (which can still be importing when the CLI snapshots its tools, so the
# render tools are missing on the first turn), the launch flow starts a
# standalone streamable-http server and health-checks it BEFORE the CLI, so its
# tools are connected on turn 1. A fixed loopback port is safe: the server is the
# only one in its isolated sandbox. The launch flow runs MCP_START_SCRIPT (under
# <sandbox>/.mcp/), waits for the port, then execs the CLI.
MCP_HTTP_HOST = "127.0.0.1"
MCP_HTTP_PORT = 8765
MCP_START_SCRIPT = "start_server.sh"

# Tool description templates. Use {render_state} and {render_policy}
# placeholders for the backend-specific tool names (Claude uses
# ``mcp__robocode-tools__render_state``, OpenCode uses
# ``robocode-tools_render_state``).
_TOOL_DESC_TEMPLATES: dict[str, str] = {
    "render_state": (
        "`{render_state}(seed=42, state=None, "
        'label="")`: renders an environment state as a PNG and returns '
        "the file path.\n"
        "  Two modes:\n"
        "  1. **Reset mode** (default): pass `seed` to render the initial "
        "state after `env.reset(seed=seed)`.\n"
        "  2. **Arbitrary state mode**: pass `state` as a flat list of floats "
        "to render any state you want. `seed` is ignored when `state` is "
        "provided.\n"
        "  The optional `label` parameter is included in the output filename "
        'for easier identification (e.g. label="after_grasp").\n'
        "  How to get a state list:\n"
        "  - From an existing observation: `obs.tolist()`\n"
        "  - To inspect/modify named features: use "
        "`env.observation_space.devectorize(obs)` to get an "
        "`ObjectCentricState` (a dict of `{{Object: feature_array}}`), "
        "modify it, then `env.observation_space.vectorize(ocs).tolist()` "
        "to convert back.\n"
        "  - To build a state from scratch: construct an "
        "`ObjectCentricState(data={{obj: np.array([...]), ...}}, "
        "type_features=env.observation_space.type_features)` using the "
        "objects from `env.observation_space.constant_objects`, then "
        "vectorize it.\n"
        "  IMPORTANT: You must call this MCP tool DIRECTLY; MCP tools are "
        "NOT available inside subagents. Call it yourself, then delegate "
        "image reading to a subagent: have it Read the PNG, describe the "
        "scene, and return a concise summary. Delete the file when done."
    ),
    "render_policy": (
        '`{render_policy}(approach_dir=".", seed=42, '
        "max_steps=1000, max_frames=100)`: runs a full episode of the "
        "approach in `approach_dir/approach.py` on the given seed and saves "
        "each frame as a PNG. Returns a list of file paths.\n"
        "  IMPORTANT: You must call this MCP tool DIRECTLY; MCP tools are "
        "NOT available inside subagents. Call it yourself to generate "
        "frames, then delegate reading to a subagent. The subagent "
        "should Read a sample of frames (e.g. first, middle, last, and any "
        "where behavior changes), describe the trajectory, identify failure "
        "modes, and return a concise text summary. Delete the output "
        "directory when done.\n"
        "  Typical workflow:\n"
        "  1. Call {render_policy} yourself to generate frames\n"
        '  2. Spawn a subagent: "Read these frame PNGs and describe '
        "the agent's trajectory. What goes wrong? Return a short summary.\"\n"
        "  3. Delete the frames directory"
    ),
}

# Per-tool blackbox overrides, merged over _TOOL_DESC_TEMPLATES. In blackbox
# mode the agent has no env source, but env_client still proxies
# observation_space.devectorize/vectorize to the host: devectorize returns a
# remote ObjectCentricState handle accessed via methods (get_object_names,
# get_objects, get_object_from_name, get, set), not the in-process
# dict-of-arrays form, and there are no constant_objects/type_features to build
# a state from scratch. The override describes that handle API. render_policy
# needs no override (it never references devectorize), so it stays shared.
_TOOL_DESC_TEMPLATES_BLACKBOX: dict[str, str] = {
    "render_state": (
        "`{render_state}(seed=42, state=None, "
        'label="")`: renders an environment state as a PNG and returns '
        "the file path.\n"
        "  Two modes:\n"
        "  1. **Reset mode** (default): pass `seed` to render the initial "
        "state after `env.reset(seed=seed)`.\n"
        "  2. **Arbitrary state mode**: pass `state` as a flat list of floats "
        "to render any state you want. `seed` is ignored when `state` is "
        "provided.\n"
        "  The optional `label` parameter is included in the output filename "
        'for easier identification (e.g. label="after_grasp").\n'
        "  How to get a state list (a flat list of floats):\n"
        "  - From an existing observation: `obs.tolist()`\n"
        "  - From the live env: `env.get_state().tolist()`\n"
        "  - To inspect/modify named features: "
        "`env.observation_space.devectorize(obs)` returns an "
        "`ObjectCentricState` handle; read it with `get_object_names()`, "
        "`get_objects(type)`, `get_object_from_name(name)`, and "
        "`get(obj, feature)`, modify it with `set(obj, feature, value)`, then "
        "`env.observation_space.vectorize(ocs).tolist()` to convert back.\n"
        "  The features are named, but you must still discover what each one "
        "means empirically by driving the env.\n"
        "  IMPORTANT: You must call this MCP tool DIRECTLY; MCP tools are "
        "NOT available inside subagents. Call it yourself, then delegate "
        "image reading to a subagent: have it Read the PNG, describe the "
        "scene, and return a concise summary. Delete the file when done."
    ),
}


# Object-centric override (variable object count). The observation is an
# ObjectCentricState with a varying number of objects, so there is no flat state
# vector: render_state's arbitrary-state mode (a list of floats) does not apply, and
# the devectorize/vectorize/constant_objects guidance is meaningless. Only seed mode
# and render_policy are useful. render_policy needs no per-tool override (it never
# references a vector); its object_count argument is advertised in the object-centric
# system-prompt suffix instead.
_TOOL_DESC_TEMPLATES_OBJECT_CENTRIC: dict[str, str] = {
    "render_state": (
        '`{render_state}(seed=42, object_count=None, label="")`: renders the '
        "environment's initial state after `env.reset(seed=seed)` as a PNG and returns "
        "the file path.\n"
        "  This environment's observations are object-centric states with a VARIABLE "
        "number of objects, so there is no flat state vector: the arbitrary-state mode "
        "(passing a list of floats) does NOT apply here.\n"
        "  Pass `object_count` to pin the number of objects for this render; omit "
        "it to sample the count as `env.reset(seed=seed)` does.\n"
        "  The optional `label` parameter is included in the output filename.\n"
        "  IMPORTANT: You must call this MCP tool DIRECTLY; MCP tools are NOT available "
        "inside subagents. Call it yourself, then delegate image reading to a subagent: "
        "have it Read the PNG, describe the scene, and return a concise summary. Delete "
        "the file when done."
    ),
}


def mcp_tool_name_claude(tool: str) -> str:
    """Claude Code MCP tool name: ``mcp__robocode-tools__<tool>``."""
    return f"mcp__{MCP_SERVER_NAME}__{tool}"


def mcp_tool_name_opencode(tool: str) -> str:
    """OpenCode MCP tool name: ``robocode-tools_<tool>``."""
    return f"{MCP_SERVER_NAME}_{tool}"


def mcp_tool_descriptions(
    backend_name: str, blackbox: bool = False, object_centric: bool = False
) -> dict[str, str]:
    """Return MCP tool descriptions with backend-specific tool names.

    In blackbox mode the render_state description swaps the in-process
    devectorize/vectorize/ObjectCentricState guidance for the host-proxied handle API
    (no constant_objects/type_features), matching what env_client exposes when the
    sandbox has no env source. For a variable-count (object-centric) env the
    render_state description drops the flat-vector arbitrary-state mode entirely -- it
    does not apply to a state with a varying number of objects -- taking precedence over
    the blackbox variant.
    """
    if backend_name == "opencode":
        namer = mcp_tool_name_opencode
    else:
        namer = mcp_tool_name_claude
    templates = dict(_TOOL_DESC_TEMPLATES)
    if blackbox:
        templates.update(_TOOL_DESC_TEMPLATES_BLACKBOX)
    if object_centric:
        templates.update(_TOOL_DESC_TEMPLATES_OBJECT_CENTRIC)
    names = {tool: namer(tool) for tool in templates}
    return {tool: template.format(**names) for tool, template in templates.items()}


# Backward compat: descriptions with Claude naming (used by existing code
# that doesn't pass a backend name).
MCP_TOOL_DESCRIPTIONS: dict[str, str] = mcp_tool_descriptions("claude")

# List of available MCP tool names.
MCP_TOOL_NAMES: tuple[str, ...] = tuple(MCP_TOOL_DESCRIPTIONS)

# System prompt suffix appended when MCP tools are available.
MCP_TOOLS_SYSTEM_PROMPT_SUFFIX = (
    " IMPORTANT: You have visual debugging tools (render_state, render_policy). "
    "You can render arbitrary states by "
    "passing a flat list of floats to render_state's `state` parameter; use "
    "devectorize/vectorize on env.observation_space to construct or modify "
    "states with named features. "
    "CRITICAL: MCP tools are only available to YOU directly, they CANNOT be "
    "called from inside subagents. Always call MCP tools yourself, then "
    "delegate image reading to a subagent."
)

# Object-centric variant (variable object count): observations are ObjectCentricStates,
# not flat vectors, so render_state's arbitrary-state mode does not apply -- only seed
# mode and render_policy are useful.
MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_OBJECT_CENTRIC = (
    " IMPORTANT: You have visual debugging tools (render_state, render_policy). "
    "This environment's observations are object-centric states (a set of typed "
    "objects), not flat vectors, so render_state's arbitrary-state mode does not "
    "apply. Both render_state and render_policy take an optional object_count to "
    "pin the number of objects. "
    "CRITICAL: MCP tools are only available to YOU directly, they CANNOT be "
    "called from inside subagents. Always call MCP tools yourself, then "
    "delegate image reading to a subagent."
)

# Blackbox variant: same workflow and subagent guidance, but devectorize/
# vectorize are proxied to the host env server and return a remote
# ObjectCentricState handle (read via get/get_objects, write via set) rather
# than the in-process dict-of-arrays form.
MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX = (
    " IMPORTANT: You have visual debugging tools (render_state, render_policy). "
    "You can render arbitrary states by "
    "passing a flat list of floats (e.g. from obs.tolist() or "
    "env.get_state().tolist()) to render_state's `state` parameter, or use "
    "env.observation_space.devectorize/vectorize to inspect or modify states "
    "by named features. "
    "CRITICAL: MCP tools are only available to YOU directly, they CANNOT be "
    "called from inside subagents. Always call MCP tools yourself, then "
    "delegate image reading to a subagent."
)


def mcp_tool_cli_names(tool_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return Claude CLI tool names (e.g. ``mcp__robocode-tools__render_state``)."""
    return tuple(f"mcp__{MCP_SERVER_NAME}__{t}" for t in tool_names)


def setup_mcp_config(
    sandbox_dir: Path,
    tool_names: tuple[str, ...],
    python_cmd: str,
    env_config_path: str,
    log_file_path: str,
    blackbox: bool = False,
    transport: str = "stdio",
    port: int = MCP_HTTP_PORT,
) -> Path:
    """Write MCP server config into ``sandbox_dir/.mcp/``.

    In normal mode, copies ``env_config.json`` from *sandbox_dir*'s parent
    into ``.mcp/`` so the server can instantiate the env locally. In blackbox
    mode the env source is absent from the container, so the server instead
    proxies render tools to the host env server using the connection info in
    ``env_spaces.json`` (written by the approach at the sandbox root).

    With ``transport="http"`` the agent CLI is pointed at a standalone
    streamable-http server on ``127.0.0.1:MCP_HTTP_PORT`` and the server-start
    command is written to ``.mcp/MCP_START_SCRIPT`` for the launch flow to start
    and health-check BEFORE the CLI (so the render tools are connected on the
    agent's first turn). With the default ``"stdio"`` the agent CLI spawns the
    server itself. Returns the path to ``mcp_config.json``.
    """
    mcp_dir = sandbox_dir / ".mcp"
    mcp_dir.mkdir(exist_ok=True)

    # Use a shell wrapper so that stderr (import errors, tracebacks) is also
    # captured in the log file even if the Python process never reaches main().
    stderr_log_path = str(Path(log_file_path).with_suffix(".stderr.log"))
    transport_args = (
        f" --transport http --host {MCP_HTTP_HOST} --port {port}"
        if transport == "http"
        else ""
    )
    if blackbox:
        # The blackbox MCP server (robocode.mcp.server) imports only
        # env_client, so it stays importable in a container with the env
        # source stripped. env_config_path is
        # <container_sandbox>/.mcp/env_config.json; the env_spaces.json the
        # approach wrote sits at the sandbox root.
        env_spaces_path = Path(env_config_path).parent.parent / "env_spaces.json"
        server_cmd = (
            f"{python_cmd} -m robocode.mcp.server"
            f" --env-spaces {env_spaces_path}"
            f" --tools {','.join(tool_names)}"
            f" --log-file {log_file_path}"
            f"{transport_args}"
            f" 2>>{stderr_log_path}"
        )
    else:
        shutil.copy2(
            sandbox_dir.parent / "env_config.json",
            mcp_dir / "env_config.json",
        )
        # The local variant (robocode.mcp.local_render) instantiates the env
        # and renders in-process, so it needs the env source and primitives.
        server_cmd = (
            f"{python_cmd} -m robocode.mcp.local_render"
            f" --env-config {env_config_path}"
            f" --tools {','.join(tool_names)}"
            f" --log-file {log_file_path}"
            f"{transport_args}"
            f" 2>>{stderr_log_path}"
        )
    if transport == "http":
        # The launch flow starts this and waits for the port before the CLI.
        # ``exec`` so the script's pid IS the server, letting the wrapper kill
        # it by pid when the CLI exits (apptainer shares the host pid namespace,
        # so a backgrounded server would otherwise leak after the run).
        (mcp_dir / MCP_START_SCRIPT).write_text(
            f"exec {server_cmd}\n", encoding="utf-8"
        )
        mcp_config: dict[str, Any] = {
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "type": "http",
                    "url": f"http://{MCP_HTTP_HOST}:{port}/mcp",
                }
            }
        }
    else:
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
