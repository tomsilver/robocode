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
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_WRITE_TOOLS: set[str] = {"Write", "Edit"}

_PATH_KEYS: dict[str, str] = {
    "Write": "file_path",
    "Edit": "file_path",
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


@dataclass(frozen=True)
class SandboxResult:
    """Result from a sandboxed Claude agent run."""

    success: bool
    output_file: Path | None
    error: str | None


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

    Initializes the sandbox with files, gives the agent a prompt via the
    Claude Code CLI, and retrieves the specified output file.
    """
    config.sandbox_dir.mkdir(parents=True, exist_ok=True)

    for dest_name, source_path in config.init_files.items():
        dest = config.sandbox_dir / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)

    # Initialize a git repo so the CLI treats the sandbox as the project
    # root and doesn't walk up to the real repo.
    if not (config.sandbox_dir / ".git" / "HEAD").exists():
        subprocess.run(
            ["git", "init"],
            cwd=str(config.sandbox_dir),
            check=True,
            capture_output=True,
        )

    # Write a CLAUDE.md that instructs the agent to keep files in the
    # sandbox.  The CLI automatically reads this from the project root.
    claude_md = config.sandbox_dir / "CLAUDE.md"
    if not claude_md.exists():
        claude_md.write_text(
            "All files you create MUST use relative paths so they "
            "stay in the current working directory. Never write files "
            "using absolute paths.\n"
        )

    claude_cmd = _get_claude_cmd()
    sandbox_abs = str(config.sandbox_dir.resolve())
    cmd = [
        claude_cmd,
        "-p",
        config.prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        config.model,
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--tools",
        "Bash,Read,Write,Edit,Glob,Grep",
        "--setting-sources",
        "",
    ]
    if config.system_prompt:
        cmd += ["--system-prompt", config.system_prompt]
    if config.max_budget_usd > 0:
        cmd += ["--max-budget-usd", str(config.max_budget_usd)]

    # Strip CLAUDECODE env vars so the subprocess doesn't inherit parent
    # session state.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}

    logger.info("Running: %s (cwd=%s)", " ".join(cmd[:6]) + " ...", sandbox_abs)

    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd,
        cwd=sandbox_abs,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    is_error = False
    error_text: str | None = None
    num_turns = 0
    total_cost: float | None = None

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

        if msg_type == "assistant":
            # Log text content from assistant messages.
            for block in msg.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    logger.info("Agent: %s", block["text"])
                elif block.get("type") == "tool_use":
                    input_str = json.dumps(block.get("input", {}))
                    if len(input_str) > 300:
                        input_str = input_str[:300] + "..."
                    logger.info("Tool call: %s(%s)", block.get("name"), input_str)

        elif msg_type == "tool_result":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "..."
            logger.info("Tool result: %s", content)

        elif msg_type == "result":
            is_error = msg.get("is_error", False)
            num_turns = msg.get("num_turns", 0)
            total_cost = msg.get("total_cost_usd")
            if is_error:
                error_text = msg.get("result", "Unknown error")

    proc.wait()

    # Check stderr for process-level errors.
    assert proc.stderr is not None
    stderr_output = proc.stderr.read()
    if proc.returncode != 0 and not is_error:
        is_error = True
        error_text = (
            stderr_output[:1000]
            if stderr_output
            else f"CLI exited with code {proc.returncode}"
        )

    logger.info(
        "Session done: turns=%d, cost=$%s, error=%s",
        num_turns,
        total_cost,
        is_error,
    )

    if is_error:
        return SandboxResult(
            success=False,
            output_file=None,
            error=error_text,
        )

    output_path = config.sandbox_dir / config.output_filename
    if output_path.exists():
        return SandboxResult(success=True, output_file=output_path, error=None)
    return SandboxResult(
        success=False,
        output_file=None,
        error=f"Output file not found: {config.output_filename}",
    )
