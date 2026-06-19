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
* **kindergarden bind-mount** -- the ``third-party/kindergarden`` submodule
  is mounted read-only at ``/robocode/third-party/kindergarden/``, overriding
  the stale in-image copy without requiring an image rebuild.

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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robocode.mcp import (
    MCP_HTTP_HOST,
    MCP_HTTP_PORT,
    MCP_RENDER_BIND_TIMEOUT_S,
    MCP_START_SCRIPT,
    MCP_STARTUP_TIMEOUT_MS,
)
from robocode.utils.backends import (
    PROVIDERS,
    AgentBackend,
    firewall_domains_for_provider,
    provider_from_model,
)
from robocode.utils.sandbox import (
    SandboxConfig,
    SandboxResult,
    _final_commit,
    _initial_commit,
    _setup_sandbox_dir,
    _stream_result_to_sandbox_result,
)

logger = logging.getLogger(__name__)

# Python interpreter inside the Docker container.
DOCKER_PYTHON: str = "/robocode/.venv/bin/python"

# Default Docker image name.
_DEFAULT_IMAGE: str = "robocode-sandbox"


def _get_claude_oauth_token() -> str | None:
    """Resolve the Claude Code OAuth access token.

    Checked in order:
      1. ``CLAUDE_CODE_OAUTH_TOKEN`` env var (works on every platform).
      2. macOS Keychain (service ``"Claude Code-credentials"``, key
         ``claudeAiOauth.accessToken``) populated by ``claude login``.

    Returns ``None`` when neither is available. The returned token is
    forwarded to the container via the ``CLAUDE_CODE_OAUTH_TOKEN`` env var,
    which the Claude CLI checks before any filesystem credential store.
    """
    env_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if env_token:
        return env_token
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


def _copy_kindergarden_without_tests(
    kindergarden: Path, dest: Path, *, blackbox: bool = False
) -> None:
    """Copy ``kindergarden`` to *dest*, skipping ``tests/`` and ``docs/``.

    With *blackbox*, also skips ``kinder/envs/`` (all environment dynamics,
    rewards, and scene generation) and ``demos/`` (recorded solution
    trajectories) so the agent cannot read them. The package skeleton
    (pyproject.toml, ``kinder/core.py`` etc.) stays so the entrypoint's
    ``uv sync --frozen`` still succeeds.
    """
    skip = ("tests", "docs", "envs", "demos") if blackbox else ("tests", "docs")
    shutil.copytree(
        kindergarden,
        dest,
        ignore=shutil.ignore_patterns(*skip),
    )


@contextmanager
def _filtered_repo_mounts(
    *, keep_primitives: bool = False, blackbox: bool = False
) -> Iterator[tuple[Path, Path]]:
    """Yield filtered copies of ``src/`` and ``kindergarden/`` to bind-mount.

    ``src`` is copied without ``oracles/`` (and without ``primitives/`` unless
    *keep_primitives*); ``kindergarden`` without tests/docs. With *blackbox*,
    environment source (``robocode/environments/``, ``kinder/envs/``, demos)
    is also excluded so the agent can only interact with the env through the
    host-side env server. The copies live in a temp dir that is removed on
    exit.
    """
    repo_root = _find_repo_root()
    kindergarden = repo_root / "third-party" / "kindergarden"
    if not kindergarden.exists():
        raise RuntimeError(
            f"kindergarden not found at {kindergarden}; "
            "run: git submodule update --init --recursive"
        )
    tmp_dir = tempfile.mkdtemp(prefix="robocode-mount-")
    filtered_src = Path(tmp_dir) / "src"
    filtered_kindergarden = Path(tmp_dir) / "kindergarden"
    try:
        _copy_src(
            repo_root / "src",
            filtered_src,
            keep_primitives=keep_primitives,
            blackbox=blackbox,
        )
        _copy_kindergarden_without_tests(
            kindergarden, filtered_kindergarden, blackbox=blackbox
        )
        yield filtered_src, filtered_kindergarden
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _is_local_model(model: str) -> bool:
    """True when *model* is served by a local server on the host (ollama/vllm).

    Such runs need the container to reach a host-loopback model server, so the
    docker launch maps ``host.docker.internal`` to the host gateway (like
    blackbox does for the env server).
    """
    return model.split("/", 1)[0] in ("ollama", "vllm")


def _docker_run_prefix(
    container_name: str,
    image: str,
    sandbox_dir: Path,
    filtered_src: Path,
    filtered_kindergarden: Path,
    auth_args: list[str],
    firewall_domains: list[str],
    env_args: list[str] | None = None,
    map_host_gateway: bool = False,
) -> list[str]:
    """Build the shared ``docker run`` prefix: caps, env, auth, mounts, image.

    Callers append the in-container command (agent CLI or genplan driver).
    With *map_host_gateway*, ``host.docker.internal`` is mapped to the host
    gateway so the container can reach host-loopback services: the blackbox
    env server and/or a local model server (ollama/vLLM). The name is built
    into Docker Desktop but needs ``--add-host`` on Linux.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--cap-add=NET_ADMIN",
        "--cap-add=NET_RAW",
        *(env_args or []),
    ]
    if map_host_gateway:
        # Reaching a host service also depends on the unconditional host /24
        # allow rule in docker/init-firewall.sh (default-deny otherwise).
        cmd += ["--add-host", "host.docker.internal:host-gateway"]
    if firewall_domains:
        cmd += [
            "-e",
            f"ROBOCODE_FIREWALL_EXTRA_DOMAINS={','.join(firewall_domains)}",
        ]
    cmd += auth_args
    cmd += [
        "-v",
        f"{sandbox_dir.resolve()}:/sandbox",
        "-v",
        f"{filtered_src.resolve()}:/robocode/src",
        "-v",
        f"{filtered_kindergarden.resolve()}:/robocode/third-party/kindergarden",
        "-w",
        "/sandbox",
        image,
    ]
    return cmd


def run_genplan_in_docker(
    sandbox_dir: Path,
    completion_cfg: dict[str, Any],
    image: str = _DEFAULT_IMAGE,
    timeout: float = 3600.0,
) -> None:
    """Run the whole LLM-GenPlan loop inside one sandbox container.

    Mirrors :func:`run_agent_in_docker_sandbox` (single ``docker run`` through
    the firewall + ``uv sync`` entrypoint) but the command is the genplan driver
    instead of the agent CLI. The driver reads ``sandbox_dir/genplan_config.json``
    and writes ``sandbox_dir/approach.py``. ``src`` keeps ``primitives`` (unlike
    the agent mount) so the policy can build/use them as it does at eval.
    """
    container_name = f"robocode-genplan-{uuid.uuid4().hex[:8]}"
    with _filtered_repo_mounts(keep_primitives=True) as (
        filtered_src,
        filtered_kindergarden,
    ):
        # Reuse the agent auth: the cli provider needs Claude OAuth/~/.claude;
        # the SDK providers just need their API key forwarded (the non-claude
        # branch forwards ANTHROPIC_API_KEY/OPENAI_API_KEY).
        auth_backend = "claude" if completion_cfg["provider"] == "cli" else "opencode"
        auth_args, auth_env = _build_docker_auth_args(auth_backend)
        # Whitelist the provider's API endpoint in the firewall (the default
        # whitelist only covers Anthropic). The cli provider needs nothing
        # extra; self-hosted endpoints are covered via base_url.
        firewall_domains = firewall_domains_for_provider(
            completion_cfg["provider"], completion_cfg.get("base_url", "")
        )
        docker_cmd = _docker_run_prefix(
            container_name,
            image,
            sandbox_dir,
            filtered_src,
            filtered_kindergarden,
            auth_args,
            firewall_domains,
        ) + [DOCKER_PYTHON, "-m", "robocode.approaches.genplan_driver"]
        logger.info("Starting genplan Docker container %s", container_name)
        subprocess.run(
            docker_cmd,
            env={**os.environ, **auth_env},
            stdin=subprocess.DEVNULL,
            check=True,
            timeout=timeout,
        )


def _copy_src(
    src: Path, dest: Path, *, keep_primitives: bool = False, blackbox: bool = False
) -> None:
    """Copy ``src/`` to *dest*, always skipping ``oracles/`` (solution code).

    The agent mount also skips ``primitives/`` so the author cannot read all
    primitive solutions (it gets only the requested files via
    :func:`_setup_sandbox_dir`). The genplan container sets ``keep_primitives``
    so its driver can call ``build_primitives`` in-container exactly as eval does
    on the host -- it is non-agentic, so there is no author to expose them to.
    With *blackbox*, ``environments/`` is also skipped so the agent cannot
    read the environment source.
    """
    skip: tuple[str, ...] = (
        ("oracles",) if keep_primitives else ("oracles", "primitives")
    )
    if blackbox:
        skip += ("environments",)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(*skip))


@dataclass(frozen=True)
class DockerSandboxConfig(SandboxConfig):
    """Configuration for a Docker-sandboxed agent run.

    Extends :class:`~robocode.utils.sandbox.SandboxConfig` with
    ``docker_image`` for Docker-based sandboxing.
    """

    docker_image: str = _DEFAULT_IMAGE
    # The container reaches host-loopback services (env server, local model
    # server) via the gateway, mapped to this name by --add-host.
    local_model_host: str = "host.docker.internal"


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


def _mcp_prestart_wrapper(agent_cmd: list[str], port: int = MCP_HTTP_PORT) -> list[str]:
    """Wrap *agent_cmd* so the http render server starts and is healthy first.

    Returns a container command that starts ``.mcp/<MCP_START_SCRIPT>`` in the
    background (its output redirected to a file so it cannot hold the CLI's
    stdout pipe), waits (up to MCP_RENDER_BIND_TIMEOUT_S) for *port* to accept
    connections, then runs
    the agent CLI in the foreground and kills the server when it exits. Starting
    and health-checking the server before the CLI is what makes the render tools
    connected on the agent's first turn. If the server dies or never binds the
    port, the wrapper kills it and exits non-zero instead of launching the CLI
    anyway (which would silently reintroduce the first-turn tool race). The CLI
    argv is passed positionally (``"$@"``) so it needs no quoting. Shared by
    docker and apptainer (same in-container python path and ``/sandbox`` bind);
    the explicit kill matters for apptainer, which shares the host pid namespace
    (no container teardown to reap the server).
    """
    start_script = f"/sandbox/.mcp/{MCP_START_SCRIPT}"
    server_log = "/sandbox/.mcp/mcp_server.boot.log"
    probe = (
        f"{DOCKER_PYTHON} -c "
        f'"import socket; socket.create_connection('
        f"('{MCP_HTTP_HOST}', {port}), 0.3).close()\""
    )
    script = (
        f"bash {start_script} >>{server_log} 2>&1 & srv=$!; ok=0; "
        f"for _ in $(seq 1 {int(MCP_RENDER_BIND_TIMEOUT_S * 10)}); do "
        f'kill -0 "$srv" 2>/dev/null || break; '
        f"{probe} 2>/dev/null && {{ ok=1; break; }}; sleep 0.1; "
        f"done; "
        f'if [ "$ok" -ne 1 ]; then '
        f'echo "render MCP server did not bind port {port}; see {server_log}" >&2; '
        f'kill "$srv" 2>/dev/null; exit 1; fi; '
        '"$@"; rc=$?; kill "$srv" 2>/dev/null; exit "$rc"'
    )
    return ["bash", "-c", script, "bash", *agent_cmd]


async def run_agent_in_docker_sandbox(
    config: DockerSandboxConfig,
    backend: AgentBackend,
) -> SandboxResult:
    """Run an agent inside the ``robocode-sandbox`` Docker container.

    Steps
    -----
    1. Calls :func:`_setup_sandbox_dir` to populate the sandbox directory.
    2. Resolves the ``third-party/kindergarden`` submodule path from the repo
       root.
    3. Launches ``docker run`` with:

       * ``--cap-add=NET_ADMIN --cap-add=NET_RAW`` for the iptables firewall
       * Backend-specific auth forwarding (Claude OAuth token / ``~/.claude``
         bind-mount, or OpenCode ``~/.local/share/opencode`` / API key env
         vars)
       * ``-v <sandbox_dir>:/sandbox`` (writable bind-mount)
       * ``-v <kindergarden>:/robocode/third-party/kindergarden:ro`` (read-only)
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

    _setup_sandbox_dir(config)

    sandbox_abs = str(config.sandbox_dir.resolve())
    container_name = f"robocode-sandbox-{uuid.uuid4().hex[:8]}"

    with _filtered_repo_mounts(blackbox=config.blackbox) as (
        filtered_src,
        filtered_kindergarden,
    ):
        # --- Authentication ---
        auth_args, auth_env = _build_docker_auth_args(backend_name)

        # --- Firewall extra domains ---
        firewall_domains: list[str] = []
        if backend_name == "opencode":
            firewall_domains = firewall_domains_for_provider(
                provider_from_model(config.model)
            )
            # OpenCode also needs access to its own telemetry/update servers.
            # firewall_domains.append("registry.npmjs.org")

        docker_cmd = _docker_run_prefix(
            container_name,
            config.docker_image,
            config.sandbox_dir,
            filtered_src,
            filtered_kindergarden,
            auth_args,
            firewall_domains,
            env_args=[
                "-e",
                f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={config.max_output_tokens}",
                "-e",
                f"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE={config.autocompact_pct}",
                # Wait for the render MCP server to connect before the CLI
                # snapshots its tools, else render tools race and can vanish.
                "-e",
                f"MCP_TIMEOUT={MCP_STARTUP_TIMEOUT_MS}",
            ],
            # Map the host gateway for blackbox (env server) and local model
            # runs (ollama/vLLM on the host), which both reach host loopback.
            map_host_gateway=config.blackbox or _is_local_model(config.model),
        )

        # Build the agent CLI command. Use the http MCP transport: the render
        # server is started and health-checked BELOW, before the CLI, so its
        # tools are connected on the agent's first turn (a CLI-spawned stdio
        # server can still be importing then and its tools register too late).
        agent_cmd = backend.build_cli_cmd(
            config,
            mcp_python_cmd=DOCKER_PYTHON,
            mcp_env_config_path="/sandbox/.mcp/env_config.json",
            mcp_config_cli_path="/sandbox/.mcp/mcp_config.json",
            mcp_log_file_path="/sandbox/.mcp/mcp_server.log",
            mcp_transport="http",
        )
        if config.mcp_tools:
            docker_cmd += _mcp_prestart_wrapper(agent_cmd)
        else:
            docker_cmd += agent_cmd

        # Write backend config files (opencode.json, AGENTS.md, etc.)
        # AFTER build_cli_cmd so .mcp/mcp_config.json exists for conversion.
        backend.setup_sandbox_files(
            config,
            docker_python=DOCKER_PYTHON,
            primitive_names=config.primitive_names,
        )
        _initial_commit(config.sandbox_dir)

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
