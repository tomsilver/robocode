"""Docker-based sandboxed Claude agent runner.

Runs the `claude` CLI inside a ``robocode-sandbox`` Docker container that
provides:

* **Filesystem isolation** — the container can only *write* to ``/sandbox``
  (the bind-mounted output directory); host paths are unreachable.
* **Network restriction** — ``init-firewall.sh`` whitelists only
  ``api.anthropic.com``, GitHub, and Claude telemetry endpoints.
* **Claude Code CLI auth** — on macOS the OAuth access token is extracted
  from the Keychain (service ``"Claude Code-credentials"``) and forwarded
  as ``CLAUDE_CODE_OAUTH_TOKEN``; on other platforms the host ``~/.claude``
  directory is bind-mounted so the CLI authenticates via the existing
  ``claude login`` session.
* **Reproducible Python environment** — all robocode dependencies are
  pre-installed in ``/robocode/.venv`` via ``uv sync --frozen``.
* **Primitive source files** — ``src/robocode/primitives/*.py`` are copied
  into ``/sandbox/primitives/`` so the agent can read their API.
* **prpl-mono bind-mount** — the current submodule is mounted read-only at
  ``/robocode/prpl-mono/``, overriding the stale in-image copy without
  requiring an image rebuild.

Usage
-----
Build the image once (from the repo root)::

    bash docker/build.sh

Then use :func:`run_agent_in_docker_sandbox` in place of
:func:`~robocode.utils.sandbox.run_agent_in_sandbox`.

The Python interpreter inside the container is :data:`DOCKER_PYTHON`.

See Also
--------
docker/Dockerfile, docker/build.sh, docker/init-firewall.sh
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from robocode.utils.sandbox import _SANDBOX_SETTINGS  # pylint: disable=protected-access
from robocode.utils.sandbox import (  # pylint: disable=protected-access
    _VALIDATE_SANDBOX_SCRIPT,
    SandboxResult,
)

logger = logging.getLogger(__name__)

# Path to the primitive source files that are copied into the sandbox.
_PRIMITIVES_SRC: Path = Path(__file__).parent.parent / "primitives"

# Python interpreter inside the Docker container.
DOCKER_PYTHON: str = "/robocode/.venv/bin/python"

# Default Docker image name.
_DEFAULT_IMAGE: str = "robocode-sandbox"


def _get_claude_oauth_token() -> str | None:
    """Extract the Claude Code OAuth access token from the macOS Keychain.

    Returns ``None`` on non-macOS platforms or when the token cannot be
    found or parsed.  On macOS, ``claude login`` stores credentials in the
    Keychain under the service name ``"Claude Code-credentials"`` as a JSON
    object with key ``claudeAiOauth.accessToken``.

    The returned token is forwarded to the container via the
    ``CLAUDE_CODE_OAUTH_TOKEN`` environment variable, which the Claude CLI
    checks before any filesystem credential store.
    """
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        creds = json.loads(result.stdout.strip())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError):
        return None


def _find_repo_root() -> Path:
    """Return the repository root by locating ``pyproject.toml`` upward."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        "Could not find repository root: no pyproject.toml found in any "
        f"parent of {__file__}"
    )


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Configuration for a Docker-sandboxed Claude agent run.

    Mirrors :class:`~robocode.utils.sandbox.SandboxConfig` but adds
    ``docker_image`` and automatically copies primitive source files into the
    sandbox directory.
    """

    sandbox_dir: Path
    init_files: dict[str, Path] = field(default_factory=dict)
    prompt: str = ""
    output_filename: str = ""
    model: str = "sonnet"
    max_budget_usd: float = 5.0
    system_prompt: str = ""
    docker_image: str = _DEFAULT_IMAGE


def _setup_sandbox_dir(config: DockerSandboxConfig) -> None:
    """Populate *config.sandbox_dir* with the standard sandbox scaffolding.

    Creates (idempotently):

    * ``primitives/*.py`` — copied from ``src/robocode/primitives/``
    * ``CLAUDE.md`` — instructs the agent to use relative paths and points to
      the Docker Python interpreter
    * ``.claude/settings.json`` — PreToolUse hook that blocks writes outside
      the sandbox
    * ``.claude/validate_sandbox.py`` — the hook implementation
    * ``.git/`` — so that ``claude`` treats ``/sandbox`` as the project root

    Also copies any files listed in ``config.init_files``.
    """
    config.sandbox_dir.mkdir(parents=True, exist_ok=True)

    # Copy caller-supplied init files.
    for dest_name, source_path in config.init_files.items():
        dest = config.sandbox_dir / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)

    # Copy primitive source files (always overwrite to stay current).
    primitives_dest = config.sandbox_dir / "primitives"
    primitives_dest.mkdir(exist_ok=True)
    for py_file in _PRIMITIVES_SRC.glob("*.py"):
        if py_file.name != "__init__.py":
            shutil.copy2(py_file, primitives_dest / py_file.name)

    # Initialise a git repo so claude treats /sandbox as the project root.
    if not (config.sandbox_dir / ".git" / "HEAD").exists():
        subprocess.run(
            ["git", "init"],
            cwd=str(config.sandbox_dir),
            check=True,
            capture_output=True,
        )

    # CLAUDE.md — written once; describes the Docker environment.
    claude_md = config.sandbox_dir / "CLAUDE.md"
    if not claude_md.exists():
        claude_md.write_text(
            "All files you create MUST use relative paths so they stay in "
            "the current working directory (/sandbox). Never write files "
            "using absolute paths.\n\n"
            f"The Python interpreter is at {DOCKER_PYTHON}\n"
            "Run test scripts with:\n"
            f"    {DOCKER_PYTHON} test_approach.py\n\n"
            "Primitive source files (for reference) are in ./primitives/\n"
        )

    # .claude/ directory with the PreToolUse hook that blocks writes outside
    # /sandbox (belt-and-suspenders on top of Docker's filesystem isolation).
    claude_dir = config.sandbox_dir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "settings.json").write_text(
        json.dumps(_SANDBOX_SETTINGS, indent=2) + "\n"
    )
    (claude_dir / "validate_sandbox.py").write_text(_VALIDATE_SANDBOX_SCRIPT)


async def run_agent_in_docker_sandbox(
    config: DockerSandboxConfig,
) -> SandboxResult:
    """Run a Claude agent inside the ``robocode-sandbox`` Docker container.

    Steps
    -----
    1. Calls :func:`_setup_sandbox_dir` to populate the sandbox directory.
    2. Resolves the ``prpl-mono`` submodule path from the repo root.
    3. Launches ``docker run`` with:

       * ``--cap-add=NET_ADMIN --cap-add=NET_RAW`` for the iptables firewall
       * ``-e CLAUDE_CODE_OAUTH_TOKEN`` — OAuth access token extracted from
         the macOS Keychain (``claude login``) so the CLI authenticates
         without an ``ANTHROPIC_API_KEY``; on non-macOS platforms the host
         ``~/.claude`` directory is bind-mounted as a fallback
       * ``-v <sandbox_dir>:/sandbox`` (writable bind-mount)
       * ``-v <prpl_mono>:/robocode/prpl-mono:ro`` (read-only bind-mount)
       * ``-w /sandbox`` as the working directory
       * The container's entrypoint runs ``init-firewall.sh`` then ``claude``

    4. Streams and parses the JSON output from ``claude --output-format
       stream-json``, logging assistant messages and tool calls.
    5. Returns a :class:`~robocode.utils.sandbox.SandboxResult`.

    Parameters
    ----------
    config:
        Sandbox configuration including the output directory and prompt.

    Returns
    -------
    SandboxResult
        ``success=True`` with ``output_file`` set when the agent writes the
        requested output file; ``success=False`` with an ``error`` otherwise.
    """
    _setup_sandbox_dir(config)

    repo_root = _find_repo_root()
    prpl_mono = repo_root / "prpl-mono"
    if not prpl_mono.exists():
        logger.warning(
            "prpl-mono not found at %s; "
            "run: git submodule update --init --recursive",
            prpl_mono,
        )

    sandbox_abs = str(config.sandbox_dir.resolve())
    prpl_mono_abs = str(prpl_mono.resolve())
    container_name = f"robocode-sandbox-{uuid.uuid4().hex[:8]}"

    # --- Authentication ---
    # On macOS, claude login stores credentials in the Keychain (not as files).
    # Extract the OAuth access token and pass it via CLAUDE_CODE_OAUTH_TOKEN,
    # which the CLI checks before any filesystem credential store.
    # On other platforms, bind-mount ~/.claude as a fallback.
    oauth_token = _get_claude_oauth_token()
    if oauth_token:
        logger.debug("Using OAuth token from macOS Keychain for container auth.")
    else:
        logger.warning(
            "No Claude OAuth token found in Keychain; "
            "falling back to ~/.claude bind-mount. "
            "Run `claude login` on the host if the container cannot authenticate."
        )

    host_claude_cfg = Path(
        os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))
    )

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--cap-add=NET_ADMIN",
        "--cap-add=NET_RAW",
        "-e",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000",
    ]

    if oauth_token:
        # Pass by env-var name so the token value is not exposed in the
        # process list (Docker reads it from the inherited environment).
        docker_cmd += ["-e", "CLAUDE_CODE_OAUTH_TOKEN"]
    else:
        # Non-macOS fallback: bind-mount the host ~/.claude directory.
        docker_cmd += ["-v", f"{host_claude_cfg}:/home/node/.claude"]

    docker_cmd += [
        "-v",
        f"{sandbox_abs}:/sandbox",
        "-v",
        f"{prpl_mono_abs}:/robocode/prpl-mono:ro",
        "-w",
        "/sandbox",
        config.docker_image,
        # Args forwarded to `claude` by entrypoint.sh.
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
        "Bash,Read,Write,Edit,Glob,Grep,Task",
        "--setting-sources",
        "project",
    ]
    if config.system_prompt:
        docker_cmd += ["--system-prompt", config.system_prompt]
    if config.max_budget_usd > 0:
        docker_cmd += ["--max-budget-usd", str(config.max_budget_usd)]

    # Strip CLAUDECODE* vars so the container does not inherit parent-session
    # state.  If we extracted an OAuth token above, inject it so Docker can
    # pick it up via `-e CLAUDE_CODE_OAUTH_TOKEN` (pass-by-name).
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
    if oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token

    logger.info(
        "Starting Docker sandbox: container=%s image=%s sandbox=%s",
        container_name,
        config.docker_image,
        sandbox_abs,
    )

    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        docker_cmd,
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
            for block in msg.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    logger.info("Agent: %s", block["text"])
                elif block.get("type") == "tool_use":
                    input_str = json.dumps(block.get("input", {}))
                    if len(input_str) > 300:
                        input_str = input_str[:300] + "..."
                    logger.info("Tool call: %s(%s)", block.get("name"), input_str)

        elif msg_type == "user":
            # Tool results come back as user-type messages in stream-json.
            for item in msg.get("message", {}).get("content", []):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "tool_result":
                    content = item.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
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

    assert proc.stderr is not None
    stderr_output = proc.stderr.read()
    if proc.returncode != 0 and not is_error:
        is_error = True
        error_text = (
            stderr_output[:1000]
            if stderr_output
            else f"docker run exited with code {proc.returncode}"
        )

    logger.info(
        "Docker session done: container=%s turns=%d cost=$%s error=%s",
        container_name,
        num_turns,
        total_cost,
        is_error,
    )

    if is_error:
        return SandboxResult(success=False, output_file=None, error=error_text)

    output_path = config.sandbox_dir / config.output_filename
    if output_path.exists():
        return SandboxResult(success=True, output_file=output_path, error=None)
    return SandboxResult(
        success=False,
        output_file=None,
        error=f"Output file not found: {config.output_filename}",
    )
