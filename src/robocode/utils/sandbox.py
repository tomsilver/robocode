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


@dataclass
class _CliRunResult:
    """Internal result from a single CLI invocation."""

    is_error: bool
    error_text: str | None
    result_text: str
    num_turns: int
    total_cost: float | None


class _ProgressTracker:
    """Tracks agent progress through the 3-step workflow and logs updates.

    Monitors the stream-json output for tool_use and tool_result events to detect when
    curriculum test files are written, when tests pass, and when the integration test
    runs.
    """

    def __init__(self) -> None:
        self._current_stage = ""
        self._plan_written = False
        self._curriculum_files: list[str] = []
        self._tests_passing: set[str] = set()
        self._approach_written = False
        self._integration_passed = False
        # Map tool_use id -> (tool_name, tool_input) for correlating results.
        self._pending: dict[str, tuple[str, dict]] = {}

    def on_tool_use(self, block: dict) -> None:
        """Process a tool_use block from an assistant message."""
        tool_id = block.get("id", "")
        name = block.get("name", "")
        input_data = block.get("input", {})

        if tool_id:
            self._pending[tool_id] = (name, input_data)

        if name in ("Write", "Edit"):
            self._handle_write(input_data)

    def on_tool_result(self, msg: dict) -> None:
        """Process a tool_result message to detect test outcomes."""
        tool_use_id = msg.get("tool_use_id", "")
        tool_info = self._pending.pop(tool_use_id, None)
        if tool_info is None:
            return
        name, input_data = tool_info

        if name != "Bash":
            return

        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        is_error = msg.get("is_error", False)
        self._handle_bash_result(input_data.get("command", ""), str(content), is_error)

    def _handle_write(self, input_data: dict) -> None:
        file_path = input_data.get("file_path", "")
        basename = Path(file_path).name
        if basename == "CLAUDE.md" and not self._plan_written:
            self._plan_written = True
            self._enter_stage("PLAN")
            logger.info("  + Plan written to CLAUDE.md")
        elif re.match(r"test_curriculum_\d+\.py", basename):
            if basename not in self._curriculum_files:
                self._curriculum_files.append(basename)
                self._enter_stage("CURRICULUM")
                logger.info("  + %s", basename)
        elif basename == "approach.py" and not self._approach_written:
            self._approach_written = True
            self._enter_stage("IMPLEMENT")
            logger.info("  + approach.py created")

    def _handle_bash_result(self, cmd: str, content: str, is_error: bool) -> None:
        if is_error:
            return
        has_failure = any(marker in content for marker in ("Traceback", "FAILED"))
        if has_failure:
            return

        match = re.search(r"test_curriculum_(\d+)\.py", cmd)
        if match:
            test_file = f"test_curriculum_{match.group(1)}.py"
            if test_file not in self._tests_passing:
                self._tests_passing.add(test_file)
                self._enter_stage("IMPLEMENT")
                logger.info("  + %s passing", test_file)
            return

        if "integration" in cmd.lower():
            if not self._integration_passed:
                self._integration_passed = True
                self._enter_stage("INTEGRATION")
                logger.info("  + Integration test passing")

    def _enter_stage(self, stage: str) -> None:
        if stage != self._current_stage:
            self._current_stage = stage
            labels = {
                "PLAN": "[STEP 0/3] Plan approach",
                "CURRICULUM": "[STEP 1/3] Design test curriculum",
                "IMPLEMENT": "[STEP 2/3] Implement approach",
                "INTEGRATION": "[STEP 3/3] Integration test",
            }
            logger.info(labels[stage])


def _run_cli_in_sandbox(config: SandboxConfig) -> _CliRunResult:
    """Run a single CLI invocation inside the sandbox directory."""
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
        "project",
    ]

    if config.system_prompt:
        cmd += ["--system-prompt", config.system_prompt]
    if config.max_budget_usd > 0:
        cmd += ["--max-budget-usd", str(config.max_budget_usd)]

    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}

    logger.info("Running: %s (cwd=%s)", " ".join(cmd[:6]) + " ...", sandbox_abs)

    stderr_path = config.sandbox_dir / "agent_stderr.log"
    stderr_file = open(stderr_path, "w", encoding="utf-8")  # noqa: SIM115

    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd,
        cwd=sandbox_abs,
        env=env,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        text=True,
    )

    is_error = False
    error_text: str | None = None
    result_text = ""
    num_turns = 0
    total_cost: float | None = None
    tracker = _ProgressTracker()

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
                    tracker.on_tool_use(block)

        elif msg_type == "tool_result":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "..."
            logger.info("Tool result: %s", content)
            tracker.on_tool_result(msg)

        elif msg_type == "result":
            is_error = msg.get("is_error", False)
            num_turns = msg.get("num_turns", 0)
            total_cost = msg.get("total_cost_usd")
            result_text = msg.get("result", "")
            if is_error:
                error_text = result_text or "Unknown error"

    proc.wait()
    stderr_file.close()

    if proc.returncode != 0 and not is_error:
        is_error = True
        stderr_output = stderr_path.read_text(encoding="utf-8")
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
    """
    _setup_sandbox(config)

    cli_result = _run_cli_in_sandbox(config)

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
