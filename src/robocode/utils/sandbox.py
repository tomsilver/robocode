"""Sandboxed Claude agent runner using the Claude Code CLI.

Runs the `claude` CLI as a subprocess in a restricted working directory.
The CLI's --dangerously-skip-permissions flag enables OS-level sandboxing
(macOS Seatbelt / Linux bubblewrap) which restricts filesystem writes to
the working directory.

WARNING: The OS-level sandbox restricts filesystem *writes* to the sandbox
directory, but allows *reads* of the entire filesystem. Bash commands like
`cat /etc/passwd` or `python -c "open('/etc/hosts').read()"` will succeed.

Set the ROBOCODE_CLAUDE_CMD environment variable to override the default
`claude` binary (e.g. to point to a specific installation).

TODO: Transition to Docker-based sandboxing for full filesystem isolation.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_RATE_LIMIT_RE = re.compile(
    r"(?:out of extra usage|hit your limit).*resets\s+(\d{1,2}(?:am|pm))",
    re.IGNORECASE,
)

_WRITE_TOOLS: set[str] = {"Write", "Edit"}

_PATH_KEYS: dict[str, str] = {
    "Write": "file_path",
    "Edit": "file_path",
}

_VALIDATE_SANDBOX_SCRIPT = """\
#!/usr/bin/env python3
import json
import os
import sys

data = json.load(sys.stdin)
tool_name = data.get("tool_name", "")
tool_input = data.get("tool_input", {})

if tool_name not in ("Write", "Edit"):
    sys.exit(0)

file_path = tool_input.get("file_path", "")
if not file_path:
    sys.exit(0)

sandbox = os.path.realpath(os.getcwd())
resolved = os.path.realpath(file_path)

if resolved == sandbox or resolved.startswith(sandbox + os.sep):
    sys.exit(0)

json.dump({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            f"Blocked: {file_path} resolves outside the sandbox directory"
        ),
    }
}, sys.stdout)
"""

_SANDBOX_SETTINGS: dict = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "Write|Edit",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python3 .claude/validate_sandbox.py",
                    }
                ],
            }
        ]
    }
}


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for a sandboxed Claude agent run."""

    sandbox_dir: Path
    init_files: dict[str, Path] = field(default_factory=dict)
    prompt: str = ""
    output_filename: str = ""
    model: str = "sonnet"
    max_budget_usd: float = 5.0
    system_prompt: str = ""
    mcp_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class SandboxResult:
    """Result from a sandboxed Claude agent run."""

    success: bool
    output_file: Path | None
    error: str | None
    total_cost_usd: float | None = None
    rate_limit_reset: str | None = None  # e.g. "3am" from usage limit message


@dataclass(frozen=True)
class _StreamParseResult:
    """Intermediate result from parsing Claude CLI stream-json output."""

    is_error: bool
    error_text: str | None
    num_turns: int
    total_cost: float | None
    rate_limit_reset: str | None = None  # e.g. "3am" parsed from usage message


# ---------------------------------------------------------------------------
# Shared helpers (used by both sandbox.py and docker_sandbox.py)
# ---------------------------------------------------------------------------


def _setup_sandbox_common(sandbox_dir: Path, init_files: dict[str, Path]) -> None:
    """Create sandbox directory, copy init files, git init, and install hooks.

    Does NOT write ``CLAUDE.md`` — callers should write their own
    variant after calling this function.
    """
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    for dest_name, source_path in init_files.items():
        dest = sandbox_dir / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)

    if not (sandbox_dir / ".git" / "HEAD").exists():
        subprocess.run(
            ["git", "init"],
            cwd=str(sandbox_dir),
            check=True,
            capture_output=True,
        )

    claude_dir = sandbox_dir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "settings.json").write_text(
        json.dumps(_SANDBOX_SETTINGS, indent=2) + "\n"
    )
    (claude_dir / "validate_sandbox.py").write_text(_VALIDATE_SANDBOX_SCRIPT)


_MCP_SERVER_NAME = "robocode-tools"


def _mcp_tool_names(tool_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return Claude CLI tool names for MCP tools (e.g. ``mcp__robocode-
    tools__render_state``)."""
    return tuple(f"mcp__{_MCP_SERVER_NAME}__{t}" for t in tool_names)


def _setup_mcp_config(
    sandbox_dir: Path,
    tool_names: tuple[str, ...],
    python_cmd: str,
    env_config_path: str,
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

    mcp_config = {
        "mcpServers": {
            _MCP_SERVER_NAME: {
                "command": python_cmd,
                "args": [
                    "-m",
                    "robocode.mcp.server",
                    "--env-config",
                    env_config_path,
                    "--tools",
                    ",".join(tool_names),
                ],
            }
        }
    }
    config_path = mcp_dir / "mcp_config.json"
    config_path.write_text(json.dumps(mcp_config, indent=2))
    return config_path


def _build_claude_cli_args(
    prompt: str,
    model: str,
    system_prompt: str,
    max_budget_usd: float,
    extra_tools: tuple[str, ...] = (),
) -> list[str]:
    """Build the common Claude CLI arguments (excluding the binary itself)."""
    tools = "Bash,Read,Write,Edit,Glob,Grep,Task"
    if extra_tools:
        tools += "," + ",".join(extra_tools)
    logger.info("Enabled tools: %s", tools)
    args = [
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--tools",
        tools,
        "--setting-sources",
        "project",
    ]
    if system_prompt:
        args += ["--system-prompt", system_prompt]
    if max_budget_usd > 0:
        args += ["--max-budget-usd", str(max_budget_usd)]
    return args


def _build_sandbox_env(
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a clean environment dict, stripping ``CLAUDECODE*`` vars."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
    env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "128000")
    if extra:
        env.update(extra)
    return env


def _parse_claude_stream(
    proc: subprocess.Popen[str],
) -> _StreamParseResult:
    """Parse ``stream-json`` stdout from a Claude CLI process and wait for exit.

    Logs assistant messages, tool calls, and tool results as they arrive. After the
    process exits, checks stderr for errors.
    """
    is_error = False
    error_text: str | None = None
    num_turns = 0
    total_cost: float | None = None
    rate_limit_reset: str | None = None

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Non-JSON output: %s", line[:200])
            continue

        msg_type = msg.get("type", "")

        if msg_type == "system" and msg.get("subtype") == "init":
            for srv in msg.get("mcp_servers", []):
                status = srv.get("status")
                name = srv.get("name")
                if status != "connected":
                    logger.warning("MCP server %s: %s", name, status)
                else:
                    logger.info("MCP server %s: %s", name, status)

        if msg_type == "assistant":
            for block in msg.get("message", {}).get("content", []):
                if block.get("type") == "thinking":
                    logger.info("Thinking: %s", block.get("thinking", ""))
                elif block.get("type") == "text":
                    text = block["text"]
                    logger.info("Agent: %s", text)
                    m = _RATE_LIMIT_RE.search(text)
                    if m:
                        rate_limit_reset = m.group(1)  # e.g. "3am"
                elif block.get("type") == "tool_use":
                    input_str = json.dumps(block.get("input", {}))
                    if len(input_str) > 300:
                        input_str = input_str[:300] + "..."
                    logger.info("Tool call: %s(%s)", block.get("name"), input_str)

        elif msg_type in ("tool_result", "user"):
            # Log only MCP tool results (contain "mcp_renders") and errors.
            tool_use_result = msg.get("tool_use_result")
            if tool_use_result is not None:
                if len(tool_use_result) > 500:
                    tool_use_result = tool_use_result[:500] + "..."
                if "Error" in tool_use_result:
                    logger.warning("Tool result: %s", tool_use_result)
                elif "mcp_renders" in tool_use_result:
                    logger.info("Tool result: %s", tool_use_result)
                else:
                    logger.debug("Tool result: %s", tool_use_result)

        elif msg_type == "result":
            is_error = msg.get("is_error", False)
            num_turns = msg.get("num_turns", 0)
            total_cost = msg.get("total_cost_usd")
            if is_error:
                error_text = msg.get("result", "Unknown error")
                # Also check the result text for rate-limit info.
                if not rate_limit_reset:
                    m = _RATE_LIMIT_RE.search(error_text)
                    if m:
                        rate_limit_reset = m.group(1)

    proc.wait()

    assert proc.stderr is not None
    stderr_output = proc.stderr.read()
    if proc.returncode != 0 and not is_error:
        is_error = True
        error_text = (
            stderr_output[:1000]
            if stderr_output
            else f"Process exited with code {proc.returncode}"
        )
        # Check stderr for rate-limit info too.
        if not rate_limit_reset and stderr_output:
            m = _RATE_LIMIT_RE.search(stderr_output)
            if m:
                rate_limit_reset = m.group(1)

    # If we detected a rate-limit message but no error was flagged, mark it.
    if rate_limit_reset and not is_error:
        is_error = True
        error_text = f"Rate-limited: resets {rate_limit_reset}"

    return _StreamParseResult(
        is_error=is_error,
        error_text=error_text,
        num_turns=num_turns,
        total_cost=total_cost,
        rate_limit_reset=rate_limit_reset,
    )


def _stream_result_to_sandbox_result(
    stream: _StreamParseResult,
    sandbox_dir: Path,
    output_filename: str,
) -> SandboxResult:
    """Convert a :class:`_StreamParseResult` to a :class:`SandboxResult`."""
    cost = stream.total_cost

    if stream.is_error:
        return SandboxResult(
            success=False,
            output_file=None,
            error=stream.error_text,
            total_cost_usd=cost,
            rate_limit_reset=stream.rate_limit_reset,
        )

    output_path = sandbox_dir / output_filename
    if output_path.exists():
        return SandboxResult(
            success=True,
            output_file=output_path,
            error=None,
            total_cost_usd=cost,
        )
    return SandboxResult(
        success=False,
        output_file=None,
        error=f"Output file not found: {output_filename}",
        total_cost_usd=cost,
    )


def _is_path_within_sandbox(path_str: str, sandbox_dir: Path) -> bool:
    """Check whether a resolved path is within the sandbox directory."""
    try:
        resolved = Path(path_str).resolve()
        sandbox_resolved = sandbox_dir.resolve()
        return resolved == sandbox_resolved or sandbox_resolved in resolved.parents
    except (OSError, ValueError):
        return False


def _get_claude_cmd() -> str:
    """Return the claude CLI command, respecting ROBOCODE_CLAUDE_CMD."""
    return os.environ.get("ROBOCODE_CLAUDE_CMD", "claude")


async def run_agent_in_sandbox(config: SandboxConfig) -> SandboxResult:
    """Run a Claude agent in a sandboxed working directory.

    Initializes the sandbox with files, gives the agent a prompt via the Claude Code
    CLI, and retrieves the specified output file.
    """
    _setup_sandbox_common(config.sandbox_dir, config.init_files)

    # Write a CLAUDE.md that instructs the agent to keep files in the sandbox.
    claude_md = config.sandbox_dir / "CLAUDE.md"
    if not claude_md.exists():
        claude_md.write_text(
            "All files you create MUST use relative paths so they "
            "stay in the current working directory. Never write files "
            "using absolute paths.\n"
        )

    # MCP server config (written when mcp_tools is non-empty).
    # Expects env_config.json in sandbox_dir's parent (the output dir).
    if config.mcp_tools:
        mcp_config_path = _setup_mcp_config(
            config.sandbox_dir,
            config.mcp_tools,
            python_cmd=sys.executable,
            env_config_path=str(
                (config.sandbox_dir / ".mcp" / "env_config.json").resolve()
            ),
        )

    claude_cmd = _get_claude_cmd()
    cmd = [claude_cmd] + _build_claude_cli_args(
        config.prompt,
        config.model,
        config.system_prompt,
        config.max_budget_usd,
        extra_tools=_mcp_tool_names(config.mcp_tools),
    )
    if config.mcp_tools:
        cmd += ["--mcp-config", str(mcp_config_path.resolve())]
    env = _build_sandbox_env()
    sandbox_abs = str(config.sandbox_dir.resolve())

    logger.info("Running: %s (cwd=%s)", " ".join(cmd[:6]) + " ...", sandbox_abs)
    logger.info("System prompt:\n%s", config.system_prompt)
    logger.info("Prompt:\n%s", config.prompt)

    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd,
        cwd=sandbox_abs,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stream = _parse_claude_stream(proc)

    logger.info(
        "Session done: turns=%d, cost=$%s, error=%s",
        stream.num_turns,
        stream.total_cost,
        stream.is_error,
    )

    return _stream_result_to_sandbox_result(
        stream, config.sandbox_dir, config.output_filename
    )
