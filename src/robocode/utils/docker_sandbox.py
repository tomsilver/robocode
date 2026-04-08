"""Docker-based sandboxed agent runner.

Runs an agent CLI inside a ``robocode-sandbox`` Docker container that
provides:

* **Filesystem isolation** -- the container can only *write* to ``/sandbox``
  (the bind-mounted output directory); host paths are unreachable.
* **Network restriction** -- ``init-firewall.sh`` whitelists only the API
  endpoints needed by the configured backend/provider, plus GitHub and
  telemetry endpoints.
* **Auth forwarding** -- for Claude: OAuth token or ``~/.claude`` bind-mount;
  for OpenCode: ``~/.local/share/opencode`` bind-mount or API key env vars.
* **Reproducible Python environment** -- all robocode dependencies are
  pre-installed in ``/robocode/.venv`` via ``uv sync --frozen``.
* **Primitive source files** -- ``src/robocode/primitives/*.py`` are copied
  into ``/sandbox/primitives/`` so the agent can read their API.
* **prpl-mono bind-mount** -- the current submodule is mounted read-only at
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

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from robocode.primitives import PRIMITIVE_NAME_TO_FILE
from robocode.utils.backends import PROVIDERS, AgentBackend
from robocode.utils.backends.opencode import firewall_domains_for_model
from robocode.utils.sandbox import (
    SandboxConfig,
    SandboxResult,
    _final_commit,
    _initial_commit,
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


def _copy_prpl_mono_without_tests(prpl_mono: Path, dest: Path) -> None:
    """Copy ``prpl-mono`` to *dest*, skipping ``tests/`` and ``docs/``."""
    shutil.copytree(
        prpl_mono,
        dest,
        ignore=shutil.ignore_patterns("tests", "docs"),
    )


def _copy_src_without_oracles(src: Path, dest: Path) -> None:
    """Copy ``src/`` to *dest*, skipping ``oracles/`` and ``primitives/``.

    Both directories contain solution code that must not be exposed to
    the agent.  Primitive source files are selectively copied into the
    sandbox via :func:`_setup_sandbox_dir` instead.
    """
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns("oracles", "primitives"),
    )


@dataclass(frozen=True)
class DockerSandboxConfig(SandboxConfig):
    """Configuration for a Docker-sandboxed agent run.

    Extends :class:`~robocode.utils.sandbox.SandboxConfig` with
    ``docker_image`` for Docker-based sandboxing and ``primitive_names``
    to control which primitive source files are copied into the sandbox.
    """

    docker_image: str = _DEFAULT_IMAGE
    primitive_names: tuple[str, ...] = ()
    mcp_tools: tuple[str, ...] = ()


def _setup_sandbox_dir(
    config: DockerSandboxConfig,
    backend: AgentBackend,
) -> None:
    """Populate *config.sandbox_dir* with the standard sandbox scaffolding.

    Creates (idempotently):

    * ``primitives/*.py`` -- copied from ``src/robocode/primitives/``
    * Backend-specific config files (CLAUDE.md + .claude/, or AGENTS.md +
      opencode.json)
    * ``.git/`` -- so the agent CLI treats ``/sandbox`` as the project root

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

    # Write backend-specific config and instruction files.
    backend.setup_sandbox_files(
        config,
        docker_python=DOCKER_PYTHON,
        primitive_names=config.primitive_names,
    )

    _initial_commit(config.sandbox_dir)


def _build_docker_auth_args(
    backend_name: str,
) -> tuple[list[str], dict[str, str]]:
    """Return Docker CLI args and env vars for backend authentication.

    For Claude: OAuth token or ~/.claude bind-mount.
    For OpenCode: ~/.local/share/opencode bind-mount + API key passthrough.
    """
    docker_args: list[str] = []
    extra_env: dict[str, str] = {}

    if backend_name == "claude":
        oauth_token = _get_claude_oauth_token()
        if oauth_token:
            docker_args += ["-e", "CLAUDE_CODE_OAUTH_TOKEN"]
            extra_env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        else:
            logger.warning(
                "No Claude OAuth token found in Keychain; "
                "falling back to ~/.claude bind-mount. "
                "Run `claude login` on the host if the container "
                "cannot authenticate."
            )
            host_claude_cfg = Path(
                os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))
            )
            docker_args += ["-v", f"{host_claude_cfg}:/home/node/.claude"]
    else:
        # OpenCode: bind-mount auth store and pass through API key env vars.
        opencode_data = Path.home() / ".local" / "share" / "opencode"
        if opencode_data.exists():
            docker_args += [
                "-v",
                f"{opencode_data}:/home/node/.local/share/opencode",
            ]

        # Pass through provider API key env vars if set.
        for info in PROVIDERS.values():
            if info.api_key_env:
                val = os.environ.get(info.api_key_env)
                if val:
                    docker_args += ["-e", info.api_key_env]
                    extra_env[info.api_key_env] = val

    return docker_args, extra_env


async def run_agent_in_docker_sandbox(
    config: DockerSandboxConfig,
    backend: AgentBackend,
) -> SandboxResult:
    """Run an agent inside the ``robocode-sandbox`` Docker container.

    Steps
    -----
    1. Calls :func:`_setup_sandbox_dir` to populate the sandbox directory.
    2. Resolves the ``prpl-mono`` submodule path from the repo root.
    3. Launches ``docker run`` with:

       * ``--cap-add=NET_ADMIN --cap-add=NET_RAW`` for the iptables firewall
       * Backend-specific auth forwarding (Claude OAuth token / ``~/.claude``
         bind-mount, or OpenCode ``~/.local/share/opencode`` / API key env
         vars)
       * ``-v <sandbox_dir>:/sandbox`` (writable bind-mount)
       * ``-v <prpl_mono>:/robocode/prpl-mono:ro`` (read-only bind-mount)
       * ``-w /sandbox`` as the working directory
       * The container's entrypoint runs ``init-firewall.sh`` then the
         agent CLI command

    4. Streams and parses the JSON output, logging assistant messages and
       tool calls.
    5. Returns a :class:`~robocode.utils.sandbox.SandboxResult`.

    Parameters
    ----------
    config:
        Sandbox configuration including the output directory and prompt.
    backend:
        Agent backend to use.

    Returns
    -------
    SandboxResult
        ``success=True`` with ``output_file`` set when the agent writes the
        requested output file; ``success=False`` with an ``error`` otherwise.
    """
    backend_name = backend.name

    _setup_sandbox_dir(config, backend)

    repo_root = _find_repo_root()
    prpl_mono = repo_root / "prpl-mono"
    if not prpl_mono.exists():
        raise RuntimeError(
            f"prpl-mono not found at {prpl_mono}; "
            "run: git submodule update --init --recursive"
        )

    src_dir = repo_root / "src"
    sandbox_abs = str(config.sandbox_dir.resolve())
    container_name = f"robocode-sandbox-{uuid.uuid4().hex[:8]}"

    # Create filtered copies: prpl-mono without tests, src without oracles.
    tmp_dir = tempfile.mkdtemp(prefix="robocode-mount-")
    filtered_prpl_mono = Path(tmp_dir) / "prpl-mono"
    filtered_src = Path(tmp_dir) / "src"
    try:
        _copy_prpl_mono_without_tests(prpl_mono, filtered_prpl_mono)
        _copy_src_without_oracles(src_dir, filtered_src)
        prpl_mono_abs = str(filtered_prpl_mono.resolve())
        src_abs = str(filtered_src.resolve())

        # --- Authentication ---
        auth_args, auth_env = _build_docker_auth_args(backend_name)

        # --- Firewall extra domains ---
        firewall_domains: list[str] = []
        if backend_name == "opencode":
            firewall_domains = firewall_domains_for_model(config.model)
            # OpenCode also needs access to its own telemetry/update servers.
            # firewall_domains.append("registry.npmjs.org")

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--cap-add=NET_ADMIN",
            "--cap-add=NET_RAW",
            "-e",
            f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={config.max_output_tokens}",
            "-e",
            f"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE={config.autocompact_pct}",
        ]

        if firewall_domains:
            docker_cmd += [
                "-e",
                f"ROBOCODE_FIREWALL_EXTRA_DOMAINS={','.join(firewall_domains)}",
            ]

        docker_cmd += auth_args

        docker_cmd += [
            "-v",
            f"{sandbox_abs}:/sandbox",
            "-v",
            f"{src_abs}:/robocode/src",
            "-v",
            f"{prpl_mono_abs}:/robocode/prpl-mono",
            "-w",
            "/sandbox",
            config.docker_image,
        ]

        # Build the agent CLI command.
        agent_cmd = backend.build_cli_cmd(
            config,
            mcp_python_cmd=DOCKER_PYTHON,
            mcp_env_config_path="/sandbox/.mcp/env_config.json",
            mcp_config_cli_path="/sandbox/.mcp/mcp_config.json",
            mcp_log_file_path="/sandbox/.mcp/mcp_server.log",
        )
        docker_cmd += agent_cmd

        env = backend.build_env(config, auth_env if auth_env else None)

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
            "Docker session done: container=%s turns=%d cost=$%s error=%s",
            container_name,
            stream.num_turns,
            stream.total_cost,
            stream.is_error,
        )

        _final_commit(config.sandbox_dir)

        return _stream_result_to_sandbox_result(
            stream, config.sandbox_dir, config.output_filename
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
