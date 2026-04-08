"""Sandboxed agent runner.

Runs a coding agent CLI as a subprocess in a restricted working directory.
The backend (Claude Code CLI, OpenCode, etc.) is selected via
:class:`~robocode.utils.backends.base.AgentBackend`.

The CLI's --dangerously-skip-permissions flag (Claude) or permission config
(OpenCode) enables OS-level sandboxing which restricts filesystem writes to
the working directory.

WARNING: The OS-level sandbox restricts filesystem *writes* to the sandbox
directory, but allows *reads* of the entire filesystem. Bash commands like
``cat /etc/passwd`` or ``python -c "open('/etc/hosts').read()"`` will succeed.
Use Docker-based sandboxing (``use_docker: true``) for full isolation.

Set ROBOCODE_CLAUDE_CMD or ROBOCODE_OPENCODE_CMD environment variables
to override the default binary paths.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from robocode.utils.backends import AgentBackend
from robocode.utils.sandbox_types import (
    SandboxConfig,
    SandboxResult,
    _StreamParseResult,
)

logger = logging.getLogger(__name__)


_SANDBOX_GITIGNORE = """\
__pycache__/
*.pyc
*.png
*.gif
*.jpg
*.jpeg
*.mp4
agent_log.txt
.mcp/
mcp_renders/
"""


# ---------------------------------------------------------------------------
# Shared helpers (used by both sandbox.py and docker_sandbox.py)
# ---------------------------------------------------------------------------


def _setup_sandbox_common(sandbox_dir: Path, init_files: dict[str, Path]) -> None:
    """Create sandbox directory, copy init files, and git init.

    Does NOT write backend-specific config files (CLAUDE.md, settings.json,
    opencode.json, AGENTS.md, etc.). Callers must invoke
    ``backend.setup_sandbox_files()`` after this function.
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
        for key, val in [("user.email", "agent@robocode"), ("user.name", "agent")]:
            subprocess.run(
                ["git", "config", key, val],
                cwd=str(sandbox_dir),
                capture_output=True,
                check=False,
            )

    gitignore = sandbox_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_SANDBOX_GITIGNORE)


def _initial_commit(sandbox_dir: Path) -> None:
    """Make an initial commit so git log works and the agent has a baseline."""
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(sandbox_dir),
        capture_output=True,
        check=False,
    )
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "initial sandbox setup",
            "--author",
            "robocode <noreply@robocode>",
        ],
        cwd=str(sandbox_dir),
        capture_output=True,
        check=False,
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


def _final_commit(sandbox_dir: Path) -> None:
    """Commit any uncommitted changes in the sandbox so nothing is lost."""
    sandbox = str(sandbox_dir)
    subprocess.run(["git", "add", "-A"], cwd=sandbox, capture_output=True, check=False)
    # --porcelain check avoids a no-op commit.
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=sandbox,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.stdout.strip():
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "auto-commit uncommitted changes",
                "--author",
                "robocode <noreply@robocode>",
            ],
            cwd=sandbox,
            capture_output=True,
            check=False,
        )
        logger.info("Auto-committed uncommitted changes in sandbox:")
        logger.info(status.stdout.strip())
    else:
        logger.info("No uncommitted changes to commit in sandbox.")


async def run_agent_in_sandbox(
    config: SandboxConfig,
    backend: AgentBackend,
) -> SandboxResult:
    """Run an agent in a sandboxed working directory.

    Initializes the sandbox with files, gives the agent a prompt via the
    configured backend CLI, and retrieves the specified output file.

    Parameters
    ----------
    config:
        Sandbox configuration.
    backend:
        Agent backend to use.
    """

    _setup_sandbox_common(config.sandbox_dir, config.init_files)

    # build_cli_cmd must run before setup_sandbox_files because it writes
    # .mcp/mcp_config.json which setup_sandbox_files reads to convert MCP
    # config for OpenCode's opencode.json.
    cmd = backend.build_cli_cmd(
        config,
        mcp_python_cmd=sys.executable,
        mcp_env_config_path=str(
            (config.sandbox_dir / ".mcp" / "env_config.json").resolve()
        ),
    )
    backend.setup_sandbox_files(config)
    _initial_commit(config.sandbox_dir)

    env = backend.build_env(config)
    sandbox_abs = str(config.sandbox_dir.resolve())

    logger.info("Running: %s (cwd=%s)", " ".join(cmd[:6]) + " ...", sandbox_abs)
    logger.info("System prompt:\n%s", config.system_prompt)
    logger.info("Prompt:\n%s", config.prompt)

    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd,
        cwd=sandbox_abs,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stream = backend.parse_stream(
        proc,
        stream_log_path=config.sandbox_dir.parent / "stream.jsonl",
    )

    logger.info(
        "Session done: turns=%d, cost=$%s, error=%s",
        stream.num_turns,
        stream.total_cost,
        stream.is_error,
    )

    # Auto-commit any uncommitted changes so nothing is lost.
    _final_commit(config.sandbox_dir)

    return _stream_result_to_sandbox_result(
        stream, config.sandbox_dir, config.output_filename
    )
