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

_PLAN_TOOLS = "Bash,Read,Glob,Grep"
_EXECUTE_TOOLS = "Bash,Read,Write,Edit,Glob,Grep"


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
    plan_mode: bool = True
    plan_budget_fraction: float = 0.2


@dataclass(frozen=True)
class SandboxResult:
    """Result from a sandboxed Claude agent run."""

    success: bool
    output_file: Path | None
    error: str | None
    assistant_text: str = ""


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


def _setup_sandbox(config: SandboxConfig) -> None:
    """Initialize the sandbox directory with files, git repo, and hooks."""
    config.sandbox_dir.mkdir(parents=True, exist_ok=True)

    for dest_name, source_path in config.init_files.items():
        dest = config.sandbox_dir / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)

    if not (config.sandbox_dir / ".git" / "HEAD").exists():
        subprocess.run(
            ["git", "init"],
            cwd=str(config.sandbox_dir),
            check=True,
            capture_output=True,
        )

    claude_md = config.sandbox_dir / "CLAUDE.md"
    if not claude_md.exists():
        claude_md.write_text(
            "All files you create MUST use relative paths so they "
            "stay in the current working directory. Never write files "
            "using absolute paths.\n"
        )

    claude_dir = config.sandbox_dir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "settings.json").write_text(
        json.dumps(_SANDBOX_SETTINGS, indent=2) + "\n"
    )
    (claude_dir / "validate_sandbox.py").write_text(_VALIDATE_SANDBOX_SCRIPT)


def _build_planning_prompt(original_prompt: str) -> str:
    """Wrap the original prompt with instructions to produce a plan only."""
    return (
        "You are in a PLANNING phase. Read the environment source code "
        "and produce TWO things: a strategy and a test curriculum. "
        "Do NOT write any files.\n\n"
        "First, read all relevant source files to understand the state "
        "type, action space, reward structure, and dynamics. Pay "
        "special attention to the state class — understand every field "
        "and how to construct a state object directly.\n\n"
        "Then output:\n\n"
        "## Strategy\n"
        "Briefly describe the algorithm you would implement and why.\n\n"
        "## Curriculum\n"
        "This is the most important part. Design 3-5 environment "
        "states ordered from simple to complex. Each state is "
        "constructed directly using the environment's state class "
        "(the state object encodes both the configuration AND the "
        "goal). For EACH state provide:\n"
        "1. The exact Python code to construct the state object.\n"
        "2. What the goal is in this state.\n"
        "3. What the correct behavior / expected outcome is.\n"
        "4. What aspect of the approach this tests.\n\n"
        "Start with the simplest possible state (e.g., agent already "
        "at the goal, or one step away with no obstacles) and "
        "progressively increase difficulty.\n\n"
        "IMPORTANT: Do NOT write any files. Only read and analyze, "
        "then output your strategy and curriculum as text.\n\n"
        "The environment you are analyzing is described by the "
        "following task:\n\n"
        f"{original_prompt}"
    )


def _build_execution_prompt(original_prompt: str, plan_text: str) -> str:
    """Wrap the original prompt with the plan for execution."""
    return (
        "A planning agent analyzed the environment and produced a "
        "strategy and test curriculum. Your job is to implement the "
        "approach.\n\n"
        "--- Plan ---\n"
        f"{plan_text}\n\n"
        "--- Task ---\n"
        f"{original_prompt}\n\n"
        "IMPORTANT: Start by writing the curriculum tests from the "
        "plan above. Create a separate file for each curriculum "
        "state: `test_curriculum_1.py`, `test_curriculum_2.py`, etc. "
        "Each file constructs the concrete state object, runs the "
        "approach on it, and checks the expected behavior. "
        "Then implement the approach incrementally, getting each "
        "curriculum test to pass in order (simplest first) before "
        "moving to the next."
    )


@dataclass
class _CliRunResult:
    """Internal result from a single CLI invocation."""

    is_error: bool
    error_text: str | None
    result_text: str
    num_turns: int
    total_cost: float | None


def _run_cli_in_sandbox(
    config: SandboxConfig,
    prompt: str,
    max_budget_usd: float,
    tools: str,
) -> _CliRunResult:
    """Run a single CLI invocation inside the sandbox directory."""
    claude_cmd = _get_claude_cmd()
    sandbox_abs = str(config.sandbox_dir.resolve())

    cmd = [
        claude_cmd,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        config.model,
        "--no-session-persistence",
        "--tools",
        tools,
        "--setting-sources",
        "project",
    ]

    cmd.append("--dangerously-skip-permissions")

    if config.system_prompt:
        cmd += ["--system-prompt", config.system_prompt]
    if max_budget_usd > 0:
        cmd += ["--max-budget-usd", str(max_budget_usd)]

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
    result_text = ""
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
            result_text = msg.get("result", "")
            if is_error:
                error_text = result_text or "Unknown error"

    proc.wait()

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

    return _CliRunResult(
        is_error=is_error,
        error_text=error_text,
        result_text=result_text,
        num_turns=num_turns,
        total_cost=total_cost,
    )


async def run_agent_in_sandbox(config: SandboxConfig) -> SandboxResult:
    """Run a Claude agent in a sandboxed working directory.

    Initializes the sandbox with files, gives the agent a prompt via the Claude Code
    CLI, and retrieves the specified output file.

    When plan_mode is enabled, runs two phases:
    1. Plan phase (read-only): analyzes the environment and outputs a plan.
    2. Execute phase (full tools): follows the plan to write the output file.

    If the plan phase fails, falls back to single-phase execution.
    """
    _setup_sandbox(config)

    if config.plan_mode:
        plan_budget = config.max_budget_usd * config.plan_budget_fraction
        execute_budget = config.max_budget_usd - plan_budget

        logger.info(
            "Plan mode: plan_budget=$%.2f, execute_budget=$%.2f",
            plan_budget,
            execute_budget,
        )

        plan_prompt = _build_planning_prompt(config.prompt)
        plan_result = _run_cli_in_sandbox(
            config,
            plan_prompt,
            plan_budget,
            _PLAN_TOOLS,
        )

        if plan_result.is_error or not plan_result.result_text.strip():
            logger.warning(
                "Plan phase failed (error=%s), falling back to single-phase",
                plan_result.error_text,
            )
            cli_result = _run_cli_in_sandbox(
                config,
                config.prompt,
                execute_budget,
                _EXECUTE_TOOLS,
            )
        else:
            logger.info("Plan phase succeeded, proceeding to execution")
            exec_prompt = _build_execution_prompt(
                config.prompt, plan_result.result_text
            )
            cli_result = _run_cli_in_sandbox(
                config,
                exec_prompt,
                execute_budget,
                _EXECUTE_TOOLS,
            )
    else:
        cli_result = _run_cli_in_sandbox(
            config,
            config.prompt,
            config.max_budget_usd,
            _EXECUTE_TOOLS,
        )

    if cli_result.is_error:
        return SandboxResult(
            success=False,
            output_file=None,
            error=cli_result.error_text,
            assistant_text=cli_result.result_text,
        )

    output_path = config.sandbox_dir / config.output_filename
    if output_path.exists():
        return SandboxResult(
            success=True,
            output_file=output_path,
            error=None,
            assistant_text=cli_result.result_text,
        )
    return SandboxResult(
        success=False,
        output_file=None,
        error=f"Output file not found: {config.output_filename}",
        assistant_text=cli_result.result_text,
    )
