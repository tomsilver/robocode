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
Use container sandboxing (``container_backend: docker``) for full isolation.

Where the kernel allows an unprivileged user namespace, :func:`pid_isolate`
gives the agent its own PID namespace so it cannot signal anything it did not
start. Where it does not (macOS, and Linux hosts that forbid unprivileged user
namespaces such as CI runners), the agent can signal any process the operator
owns; that is another reason to prefer a container backend.

Set ROBOCODE_CLAUDE_CMD or ROBOCODE_OPENCODE_CMD environment variables
to override the default binary paths.
"""

from __future__ import annotations

import functools
import logging
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import IO

from robocode.mcp import MCP_START_SCRIPT
from robocode.primitive_specs import (
    ENV_DEPENDENT_PRIMITIVES,
    PRIMITIVE_NAME_TO_FILE,
    REMOTE_MODULE_PRIMITIVES,
)
from robocode.utils.backends import AgentBackend
from robocode.utils.claude_auth import sandbox_claude_config
from robocode.utils.sandbox_types import (
    GenerationMetrics,
    SandboxConfig,
    SandboxResult,
    _StreamParseResult,
)
from robocode.utils.telemetry import SINK_FILENAME, host_launch_env

logger = logging.getLogger(__name__)

_PRIMITIVES_SRC: Path = Path(__file__).parent.parent / "primitives"


# Docker and apptainer give the agent its own PID namespace; the local backend runs
# the CLI as an ordinary host process, so without this a `pkill -f <pattern>` from the
# agent's Bash tool reaches every process the operator owns: the harness supervising
# the run, concurrent runs, and the agent's own CLI. unshare puts the CLI in its own
# PID namespace with its own /proc, so it can only see and signal what it started.
# --user is what makes this available unprivileged; --map-current-user keeps the uid
# unchanged so the CLI reads the same config and writes files with the same ownership.
_PID_ISOLATE_PREFIX = (
    "unshare",
    "--user",
    "--map-current-user",
    "--pid",
    "--fork",
    "--mount-proc",
)


@functools.cache
def _userns_available() -> bool:
    """Whether an unprivileged process can create a user+PID namespace here.

    :func:`pid_isolate` calls this only on Linux. Some Linux hosts still forbid
    the unprivileged user namespace the PID namespace rides on: CI runners and
    hardened kernels (e.g. Ubuntu's ``apparmor_restrict_unprivileged_userns``)
    make the ``uid_map`` write return EPERM; ``unshare`` may also be missing.
    Probe by running the real prefix on ``true`` rather than reading one
    distro-specific sysctl, since the block has several sources. Cached to avoid
    re-probing (and re-warning) on every launch in a batch run; a probe that
    later succeeds intermittently is not worth chasing here.
    """
    reason: str
    try:
        probe = subprocess.run(
            [*_PID_ISOLATE_PREFIX, "true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        reason = str(exc) or exc.__class__.__name__
    else:
        if probe.returncode == 0:
            return True
        reason = probe.stderr.decode(errors="replace").strip() or "unshare failed"
    logger.warning(
        "Unprivileged user namespaces are unavailable (%s); the local backend "
        "will run the agent CLI in the host PID namespace, so a `pkill` from the "
        "agent's Bash tool can reach other processes the operator owns. Use a "
        "container backend (docker/apptainer) for process isolation.",
        reason,
    )
    return False


def pid_isolate(cmd: list[str]) -> list[str]:
    """Return *cmd* prefixed so it runs in its own PID namespace, when it can be.

    Requires an unprivileged user namespace, so it is a no-op on macOS (no
    equivalent) and on Linux hosts that forbid them (see :func:`_userns_available`);
    there the local backend keeps the agent in the host process namespace, which is
    one more reason to prefer a container backend (see the module docstring).
    """
    if sys.platform != "linux" or not _userns_available():
        return cmd
    return [*_PID_ISOLATE_PREFIX, *cmd]


def _free_port() -> int:
    """Return a currently-free loopback TCP port.

    Shared by the sandbox backends that pre-start the render http server on the host
    network namespace (local here, apptainer) so it cannot collide with the host or a
    concurrent run. Lives here (not in docker_sandbox) because that module imports from
    this one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_local_http_mcp(
    sandbox_dir: Path, port: int, env: dict[str, str]
) -> subprocess.Popen[bytes]:
    """Start the http render MCP server as a host process; wait until it serves.

    Runs ``.mcp/<MCP_START_SCRIPT>`` (which ``exec``s the server, so the Popen
    pid IS the server) with its output off the agent's pipes, then polls *port*
    until it accepts a connection. Raises if the server exits or never binds.
    The caller terminates the returned process when the agent run ends.
    """
    script = sandbox_dir / ".mcp" / MCP_START_SCRIPT
    # Keep telemetry off the render MCP server: it does its own env resets, so
    # instrumenting it would pollute the agent's stream.
    mcp_env = {k: v for k, v in env.items() if k != "ROBOCODE_TELEMETRY"}
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        ["bash", str(script)],
        env=mcp_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"render MCP server exited at startup; see "
                f"{sandbox_dir / '.mcp' / 'mcp_server.stderr.log'}"
            )
        try:
            socket.create_connection(("127.0.0.1", port), 0.3).close()
            return proc
        except OSError:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(f"render MCP server did not bind port {port} within 30s")


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
.agent_sessions/
.agent_home/
.agent_prompts/
mcp_renders/
"""


@contextmanager
def agent_stdin(backend: AgentBackend, config: SandboxConfig) -> Iterator[IO[str]]:
    """Yield a file holding the CLI's stdin: its prompt, or empty.

    A file rather than a pipe so a prompt larger than the pipe buffer cannot deadlock
    against the CLI's stdout, and so the same fd works whether the CLI runs on the host
    or inside a container.
    """
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as handle:
        handle.write(backend.stdin_text(config))
        handle.seek(0)
        yield handle


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


def _setup_sandbox_dir(config: SandboxConfig) -> None:
    """Populate ``config.sandbox_dir`` with the standard sandbox scaffolding.

    Shared by the docker and apptainer backends. Via
    :func:`_setup_sandbox_common`, creates (idempotently) the sandbox directory,
    any ``config.init_files``, a ``.git`` repo (so the agent CLI treats it as the
    project root), and a ``.gitignore``. Then copies the requested primitive
    source files into ``sandbox_dir/primitives/``.

    In black-box mode, env-dependent and remote-module primitives are skipped:
    their source imports the hidden env (so it would not import in the sandbox)
    and would leak its structure; the sandbox reaches them via
    env_client.make_primitives instead (per-callable host proxies and
    whole-module remote proxies).

    Backend-specific config files (CLAUDE.md, settings.json, AGENTS.md,
    opencode.json) are NOT written here; callers invoke
    ``backend.setup_sandbox_files()`` afterward, since it needs files written
    later in the launch (e.g. ``.mcp/mcp_config.json`` from build_cli_cmd).
    """
    _setup_sandbox_common(config.sandbox_dir, config.init_files)
    if not config.primitive_names:
        return
    primitives_dest = config.sandbox_dir / "primitives"
    primitives_dest.mkdir(exist_ok=True)
    for name in config.primitive_names:
        if config.blackbox and name in (
            ENV_DEPENDENT_PRIMITIVES | REMOTE_MODULE_PRIMITIVES
        ):
            continue
        file_stem = PRIMITIVE_NAME_TO_FILE.get(name)
        if file_stem is None:
            logger.warning("No source file mapping for primitive %r", name)
            continue
        src_file = _PRIMITIVES_SRC / f"{file_stem}.py"
        if src_file.exists():
            shutil.copy2(src_file, primitives_dest / src_file.name)
        else:
            raise RuntimeError(f"Primitive source file not found: {src_file}")


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
    wall_time_s: float | None = None,
) -> SandboxResult:
    """Convert a :class:`_StreamParseResult` to a :class:`SandboxResult`."""
    cost = stream.total_cost
    metrics = GenerationMetrics(
        wall_time_s=wall_time_s,
        cli_duration_ms=stream.cli_duration_ms,
        cli_duration_api_ms=stream.cli_duration_api_ms,
        num_turns=stream.num_turns,
        input_tokens=stream.input_tokens,
        output_tokens=stream.output_tokens,
        cache_read_tokens=stream.cache_read_tokens,
        cache_creation_tokens=stream.cache_creation_tokens,
        num_tool_calls=stream.num_tool_calls,
        num_autocompactions=stream.num_autocompactions,
        num_permission_denials=stream.num_permission_denials,
        turn_limit_hit=stream.turn_limit_hit,
        output_token_limit_hit=stream.output_token_limit_hit,
        prompt_too_long_hit=stream.prompt_too_long_hit,
        stop_reason=stream.stop_reason,
        model_usage=stream.model_usage,
    )

    # Retryable interruptions must propagate before considering a partial output.
    # The retry loop resumes the same conversation with the remaining budget.
    if (
        stream.rate_limit_reset is not None
        or stream.output_token_limit_hit
        or stream.prompt_too_long_hit
    ):
        return SandboxResult(
            success=False,
            output_file=None,
            error=stream.error_text,
            total_cost_usd=cost,
            rate_limit_reset=stream.rate_limit_reset,
            output_token_limit_hit=stream.output_token_limit_hit,
            prompt_too_long_hit=stream.prompt_too_long_hit,
            generation_metrics=metrics,
        )

    # The agent commits its best-effort approach.py as it iterates, so if that
    # file exists we evaluate it even when the run ended on an error such as the
    # budget or turn cap being hit. Those are normal stopping conditions, not a
    # reason to throw away a working approach and fall back to a random policy.
    output_path = sandbox_dir / output_filename
    if output_path.exists():
        return SandboxResult(
            success=True,
            output_file=output_path,
            error=stream.error_text,
            total_cost_usd=cost,
            generation_metrics=metrics,
        )
    return SandboxResult(
        success=False,
        output_file=None,
        error=stream.error_text or f"Output file not found: {output_filename}",
        total_cost_usd=cost,
        generation_metrics=metrics,
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
    # Pre-start the render MCP server over http and point the CLI at it (see
    # docker_sandbox for why: a CLI-spawned stdio server can still be importing
    # when the CLI snapshots its tools). The server runs on the host, so use a
    # free port to avoid colliding with the host or a concurrent run.
    mcp_port = _free_port()
    cmd = backend.build_cli_cmd(
        config,
        mcp_python_cmd=sys.executable,
        mcp_env_config_path=str(
            (config.sandbox_dir / ".mcp" / "env_config.json").resolve()
        ),
        mcp_transport="http",
        mcp_port=mcp_port,
    )
    backend.setup_sandbox_files(config)
    _initial_commit(config.sandbox_dir)

    env = backend.build_env(config)
    if config.telemetry and not config.blackbox:
        # Run id = the run's output dir name (sandbox_dir is always ".../sandbox").
        run_dir = config.sandbox_dir.resolve().parent
        sink_dir = run_dir / "telemetry"
        sink_dir.mkdir(parents=True, exist_ok=True)
        env.update(
            host_launch_env(sink_dir / SINK_FILENAME, run_dir.name, with_hook=True)
        )
    else:
        # Make the config flag authoritative: an inherited ROBOCODE_TELEMETRY must
        # not enable telemetry for the agent when this run (or blackbox) did not.
        env.pop("ROBOCODE_TELEMETRY", None)
        env.pop("ROBOCODE_RUN_ID", None)
    sandbox_abs = str(config.sandbox_dir.resolve())

    logger.info("Running: %s (cwd=%s)", " ".join(cmd[:6]) + " ...", sandbox_abs)
    logger.info("System prompt:\n%s", config.system_prompt)
    logger.info("Prompt:\n%s", config.prompt)

    server_proc = (
        _start_local_http_mcp(config.sandbox_dir, mcp_port, env)
        if config.mcp_tools
        else None
    )
    wall_start = time.monotonic()
    claude_config = (
        sandbox_claude_config(config.sandbox_dir)
        if backend.name == "claude"
        else nullcontext(None)
    )
    try:
        with claude_config as config_dir:
            if config_dir is not None:
                env["CLAUDE_CONFIG_DIR"] = str(config_dir)
            with (
                tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file,
                agent_stdin(backend, config) as stdin_file,
            ):
                proc = subprocess.Popen(  # pylint: disable=consider-using-with
                    pid_isolate(cmd),
                    cwd=sandbox_abs,
                    env=env,
                    stdin=stdin_file,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True,
                    start_new_session=True,
                )

                stream = backend.parse_stream(
                    proc,
                    stream_log_path=config.sandbox_dir.parent / "stream.jsonl",
                    stderr_file=stderr_file,
                )
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    logger.info(
        "Session done: turns=%d, cost=$%s, error=%s",
        stream.num_turns,
        stream.total_cost,
        stream.is_error,
    )

    wall_time_s = time.monotonic() - wall_start

    # Auto-commit any uncommitted changes so nothing is lost.
    _final_commit(config.sandbox_dir)

    return _stream_result_to_sandbox_result(
        stream,
        config.sandbox_dir,
        config.output_filename,
        wall_time_s=wall_time_s,
    )
