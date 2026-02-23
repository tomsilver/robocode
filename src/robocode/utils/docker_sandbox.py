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
from dataclasses import dataclass
from pathlib import Path

from robocode.primitives import PRIMITIVE_NAME_TO_FILE
from robocode.utils.sandbox import (
    SandboxConfig,
    SandboxResult,
    _build_claude_cli_args,
    _build_sandbox_env,
    _parse_claude_stream,
    _setup_sandbox_common,
    _stream_result_to_sandbox_result,
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
    platform: str = sys.platform
    if platform != "darwin":
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
class DockerSandboxConfig(SandboxConfig):
    """Configuration for a Docker-sandboxed Claude agent run.

    Extends :class:`~robocode.utils.sandbox.SandboxConfig` with
    ``docker_image`` for Docker-based sandboxing and ``primitive_names``
    to control which primitive source files are copied into the sandbox.
    """

    docker_image: str = _DEFAULT_IMAGE
    primitive_names: tuple[str, ...] = ()


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
    _setup_sandbox_common(config.sandbox_dir, config.init_files)

    # Copy only the primitive source files that were provided.
    if config.primitive_names:
        primitives_dest = config.sandbox_dir / "primitives"
        primitives_dest.mkdir(exist_ok=True)
        for name in config.primitive_names:
            file_stem = PRIMITIVE_NAME_TO_FILE.get(name)
            if file_stem is None:
                logger.warning("No source file mapping for primitive %r", name)
                continue
            src_file = _PRIMITIVES_SRC / f"{file_stem}.py"
            if src_file.exists():
                shutil.copy2(src_file, primitives_dest / src_file.name)
            else:
                raise RuntimeError(f"Primitive source file not found: {src_file}")

    # CLAUDE.md — written once; describes the Docker environment.
    claude_md = config.sandbox_dir / "CLAUDE.md"
    if not claude_md.exists():
        claude_md_text = (
            "All files you create MUST use relative paths so they stay in "
            "the current working directory (/sandbox). Never write files "
            "using absolute paths.\n\n"
            f"The Python interpreter is at {DOCKER_PYTHON}\n"
            "Run test scripts with:\n"
            f"    {DOCKER_PYTHON} test_approach.py\n"
        )
        if config.primitive_names:
            claude_md_text += (
                "\nPrimitive source files (for reference) are in " "./primitives/\n"
            )
        claude_md.write_text(claude_md_text)


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
        raise RuntimeError(
            f"prpl-mono not found at {prpl_mono}; "
            "run: git submodule update --init --recursive"
        )

    sandbox_abs = str(config.sandbox_dir.resolve())
    prpl_mono_abs = str(prpl_mono.resolve())
    container_name = f"robocode-sandbox-{uuid.uuid4().hex[:8]}"

    # --- Authentication ---
    oauth_token = _get_claude_oauth_token()
    if not oauth_token:
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
        docker_cmd += ["-e", "CLAUDE_CODE_OAUTH_TOKEN"]
    else:
        docker_cmd += ["-v", f"{host_claude_cfg}:/home/node/.claude"]

    docker_cmd += [
        "-v",
        f"{sandbox_abs}:/sandbox",
        "-v",
        f"{prpl_mono_abs}:/robocode/prpl-mono:ro",
        "-w",
        "/sandbox",
        config.docker_image,
    ]
    docker_cmd += _build_claude_cli_args(
        config.prompt, config.model, config.system_prompt, config.max_budget_usd
    )

    extra_env: dict[str, str] = {}
    if oauth_token:
        extra_env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
    env = _build_sandbox_env(extra_env if extra_env else None)

    logger.info(
        "Starting Docker sandbox: container=%s image=%s sandbox=%s",
        container_name,
        config.docker_image,
        sandbox_abs,
    )
    logger.info("System prompt:\n%s", config.system_prompt)
    logger.info("Prompt:\n%s", config.prompt)

    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        docker_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stream = _parse_claude_stream(proc)

    logger.info(
        "Docker session done: container=%s turns=%d cost=$%s error=%s",
        container_name,
        stream.num_turns,
        stream.total_cost,
        stream.is_error,
    )

    return _stream_result_to_sandbox_result(
        stream, config.sandbox_dir, config.output_filename
    )
