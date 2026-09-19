"""Apptainer agent runner with an isolated network and host inference broker.

Agents run without root in a fresh user/network/PID/IPC namespace using
``--net --network none``, filtered mounts, a clean environment, and no-new-privileges.
A supervisor verifies the network and capability boundary before starting the CLI.
The host broker accepts only validated model inference over a mounted Unix socket;
provider credentials remain on the host. A separate socket relays to one pinned
experiment environment server. Neither relay provides general internet access.

Regular Python dependencies are prepared in a trusted installer phase and mounted
read-only; the agent phase never runs the network-dependent image entrypoint.
Strict runs use the dependency-clean strict SIF. Unsupported backend/GenPlan paths
fail closed. See ``docs/apptainer-network-isolation.md`` for evidence and limits.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from robocode.mcp import MCP_STARTUP_TIMEOUT_MS
from robocode.utils.apptainer_environment import (
    clean_apptainer_env,
    prepared_environment,
)
from robocode.utils.backends import AgentBackend
from robocode.utils.claude_auth import sandbox_claude_session_store
from robocode.utils.codex_auth import sandbox_codex_sessions
from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    _filtered_repo_mounts,
    _find_repo_root,
    _mcp_prestart_wrapper,
    container_python,
)
from robocode.utils.isolated_transport import UnixRelay
from robocode.utils.model_broker import (
    BROKER_DIR,
    MODEL_PORT,
    BrokerUpstream,
    load_broker_upstream,
    model_broker,
)
from robocode.utils.sandbox import (
    SandboxConfig,
    SandboxResult,
    _final_commit,
    _initial_commit,
    _setup_sandbox_dir,
    _stream_result_to_sandbox_result,
    agent_stdin,
)
from robocode.utils.telemetry import container_launch

logger = logging.getLogger(__name__)

# Python interpreter inside the SIF (same path as in the Docker image).
APPTAINER_PYTHON: str = DOCKER_PYTHON

# Default SIF paths: <repo_root>/robocode-sandbox.sif and, for strict blackbox
# runs, <repo_root>/robocode-strict-blackbox.sif.
_DEFAULT_SIF: Path = _find_repo_root() / "robocode-sandbox.sif"
_DEFAULT_STRICT_SIF: Path = _find_repo_root() / "robocode-strict-blackbox.sif"


def _telemetry_apptainer(config: SandboxConfig) -> tuple[list[str], dict[str, str]]:
    """(extra binds, ``APPTAINERENV_`` vars) enabling telemetry for a whitebox run.

    Empty when telemetry is off or for blackbox runs (the host env server is
    instrumented instead). The env is passed with Apptainer's ``APPTAINERENV_``
    prefix so it lands inside the container.
    """
    if not config.telemetry or config.blackbox:
        return [], {}
    # Run id = the run's output dir name (sandbox_dir is always ".../sandbox").
    run_dir = config.sandbox_dir.resolve().parent
    sink_dir = run_dir / "telemetry"
    sink_dir.mkdir(parents=True, exist_ok=True)
    mounts, env = container_launch(sink_dir, run_dir.name)
    binds = [f"{host}:{cont}{':ro' if ro else ''}" for host, cont, ro in mounts]
    return binds, {f"APPTAINERENV_{key}": val for key, val in env.items()}


@dataclass(frozen=True)
class ApptainerSandboxConfig(SandboxConfig):
    """Configuration for an Apptainer-sandboxed agent run.

    Extends :class:`~robocode.utils.sandbox.SandboxConfig` with ``sif_path``
    for the SIF image and ``strict_sif_path`` for the dependency-clean image a
    strict blackbox run executes in instead.
    """

    sif_path: Path = _DEFAULT_SIF
    blackbox_strict: bool = False
    strict_sif_path: Path = _DEFAULT_STRICT_SIF
    # Trusted host destination. Never inferred from agent-writable metadata.
    env_server_port: int | None = None


def sif_path_for(config: ApptainerSandboxConfig) -> Path:
    """The image a run executes in: the dependency-clean one under strict."""
    return config.strict_sif_path if config.blackbox_strict else config.sif_path


def _apptainer_exec_prefix() -> list[str]:
    """Return the filesystem/process isolation shared by all Apptainer runs."""
    # --no-home alone does not reliably suppress administrator-configured host
    # binds. --containall drops default home, tmp, and cwd binds in every mode,
    # leaving only the explicit mounts added by each caller.
    return [
        "apptainer",
        "exec",
        "--userns",
        "--net",
        "--network",
        "none",
        "--containall",
        # Apptainer shares the host PID namespace by default, so a `pkill -f`
        # inside the container could otherwise reach the harness, concurrent
        # runs, and unrelated user processes. --containall implies --pid, but
        # keeping it explicit documents this boundary.
        "--pid",
        "--writable-tmpfs",
        "--no-home",
        "--cleanenv",
        "--pwd",
        "/sandbox",
    ]


def _build_apptainer_cmd(
    config: ApptainerSandboxConfig,
    sandbox_abs: str,
    src_abs: str | None,
    kindergarden_abs: str | None,
    kinder_baselines_abs: str | None,
    agent_cmd: list[str],
    extra_binds: list[str] | None = None,
    ss_pybullet_abs: str | None = None,
) -> list[str]:
    """Assemble the full ``apptainer exec`` command line.

    Split out from :func:`run_agent_in_apptainer_sandbox` so unit tests
    can inspect the constructed command without running anything.

    Strict blackbox launches omit project source mounts. The high-level runner
    adds the broker and session mounts. This builder never installs dependencies
    or forwards provider credentials and firewall settings.
    """
    cmd = _apptainer_exec_prefix()
    cmd += [
        "--env",
        f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={config.max_output_tokens}",
        "--env",
        f"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE={config.autocompact_pct}",
        # Wait for the render MCP server to connect before the CLI snapshots its
        # tools (--containall drops the host env, so this must be explicit).
        "--env",
        f"MCP_TIMEOUT={MCP_STARTUP_TIMEOUT_MS}",
        # Headless container has no GPU, so mujoco's Dynamic3D offscreen renderer
        # must use OSMesa (software); EGL device displays fail without a GPU.
        "--env",
        "MUJOCO_GL=osmesa",
        "--env",
        "PYOPENGL_PLATFORM=osmesa",
    ]

    cmd += ["--bind", f"{sandbox_abs}:/sandbox"]
    if src_abs is not None:
        cmd += ["--bind", f"{src_abs}:/robocode/src"]
    if kindergarden_abs is not None:
        cmd += ["--bind", f"{kindergarden_abs}:/robocode/third-party/kindergarden"]
    if kinder_baselines_abs is not None:
        cmd += [
            "--bind",
            f"{kinder_baselines_abs}:/robocode/third-party/kinder-baselines",
        ]
    if ss_pybullet_abs is not None:
        cmd += ["--bind", f"{ss_pybullet_abs}:/robocode/third-party/ss-pybullet:ro"]
    for bind in extra_binds or []:
        cmd += ["--bind", bind]
    cmd += [
        str(sif_path_for(config)),
        "/usr/bin/setpriv",
        "--no-new-privs",
        "--",
    ]
    cmd += agent_cmd
    return cmd


@contextmanager
def _isolated_transport(
    config: ApptainerSandboxConfig, provider: BrokerUpstream
) -> Iterator[Path]:
    """Own the broker and optional pinned environment relay for one agent run."""
    with ExitStack() as isolation:
        bridge = Path(
            isolation.enter_context(tempfile.TemporaryDirectory(prefix="robocode-net-"))
        )
        isolation.enter_context(
            model_broker(bridge, provider, config.sandbox_dir.parent / "broker.jsonl")
        )
        shutil.copyfile(
            Path(__file__).with_name("isolated_transport.py"), bridge / "transport.py"
        )
        listeners = [{"port": MODEL_PORT, "socket": f"{BROKER_DIR}/model.sock"}]
        # Only immutable host configuration selects a destination. Sandbox metadata
        # describes the client view and never authorizes a host connection.
        metadata_path = config.sandbox_dir / "env_spaces.json"
        port = config.env_server_port
        if metadata_path.exists() and port is None:
            raise RuntimeError(
                "env_spaces.json requires an explicit trusted env_server_port"
            )
        if port is not None:
            if (
                isinstance(port, bool)
                or not isinstance(port, int)
                or not 1 <= port <= 65535
            ):
                raise RuntimeError("Invalid trusted environment server port")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            relay = isolation.enter_context(
                UnixRelay(str(bridge / "environment.sock"), ("127.0.0.1", port))
            )
            thread = threading.Thread(target=relay.serve_forever, daemon=True)
            thread.start()
            isolation.callback(thread.join, 5)
            isolation.callback(relay.shutdown)
            listeners.append(
                {"port": MODEL_PORT + 1, "socket": f"{BROKER_DIR}/environment.sock"}
            )
            metadata.update(host="127.0.0.1", port=MODEL_PORT + 1)
            (config.sandbox_dir / "env_spaces.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
        (bridge / "transport.json").write_text(
            json.dumps(
                {"listeners": listeners, "strict_blackbox": config.blackbox_strict}
            ),
            encoding="utf-8",
        )
        yield bridge


def _model_client(
    backend_name: str, agent_cmd: list[str]
) -> tuple[list[str], dict[str, str]]:
    """Point a CLI at the local broker using inert tokens; never load real auth."""
    agent_cmd = list(agent_cmd)
    local_token = "local-broker-no-provider-secret"
    client_env = {
        "APPTAINERENV_ROBOCODE_MODEL_TOKEN": local_token,
        "APPTAINERENV_UV_OFFLINE": "1",
        "APPTAINERENV_PIP_NO_INDEX": "1",
    }
    if backend_name == "claude":
        client_env.update(
            {
                "APPTAINERENV_ANTHROPIC_BASE_URL": f"http://127.0.0.1:{MODEL_PORT}",
                "APPTAINERENV_ANTHROPIC_AUTH_TOKEN": local_token,
                "APPTAINERENV_CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        )
    if backend_name == "codex":
        # Explicit custom provider avoids giving the CLI any real credential.
        # Transport is SSE; websocket upgrades are rejected at the broker.
        overrides = {
            "model_provider": "robocode",
            "model_providers.robocode.name": "Robocode isolated broker",
            "model_providers.robocode.base_url": f"http://127.0.0.1:{MODEL_PORT}/v1",
            "model_providers.robocode.wire_api": "responses",
            "model_providers.robocode.env_key": "ROBOCODE_MODEL_TOKEN",
            "model_providers.robocode.supports_websockets": False,
            "features.responses_websockets": False,
            "features.responses_websockets_v2": False,
        }
        for key, value in overrides.items():
            agent_cmd[-1:-1] = ["--config", f"{key}={json.dumps(value)}"]
    return agent_cmd, client_env


async def run_agent_in_apptainer_sandbox(
    config: ApptainerSandboxConfig,
    backend: AgentBackend,
) -> SandboxResult:
    """Run a supported agent with isolated networking and validated inference."""
    backend_name = backend.name
    if getattr(backend, "_base_url", ""):
        raise RuntimeError(
            "Custom model endpoints are not supported by the isolated broker"
        )
    strict_blackbox = config.blackbox_strict

    sif_path = sif_path_for(config)
    if not sif_path.exists():
        build_script = (
            "docker/build_strict_blackbox_sif.sh"
            if strict_blackbox
            else "docker/build_sif.sh"
        )
        raise RuntimeError(
            f"SIF image not found at {sif_path}; build it with: bash {build_script}"
        )

    provider = load_broker_upstream(backend_name)
    _setup_sandbox_dir(config)

    sandbox_abs = str(config.sandbox_dir.resolve())
    run_id = f"apptainer-sandbox-{uuid.uuid4().hex[:8]}"

    # The strict image needs no project source mounts.
    mounts = (
        nullcontext((None, None, None, None))
        if strict_blackbox
        else _filtered_repo_mounts(
            blackbox=config.blackbox,
            include_bilevel="bilevel_models" in config.primitive_names,
        )
    )
    with (
        mounts as (
            filtered_src,
            filtered_kindergarden,
            filtered_kinder_baselines,
            ss_pybullet,
        ),
        _isolated_transport(config, provider) as bridge,
    ):
        transport_binds = [f"{bridge}:{BROKER_DIR}:ro"]
        if not strict_blackbox:
            preparation_binds = [
                f"{filtered_src}:/robocode/src",
                f"{filtered_kindergarden}:/robocode/third-party/kindergarden",
                f"{_find_repo_root() / 'pyproject.toml'}:/robocode/pyproject.toml:ro",
                f"{_find_repo_root() / 'uv.lock'}:/robocode/uv.lock:ro",
            ]
            if filtered_kinder_baselines is not None:
                preparation_binds.append(
                    f"{filtered_kinder_baselines}:/robocode/third-party/kinder-baselines"
                )
            venv = prepared_environment(
                sif_path,
                preparation_binds,
                include_bilevel=filtered_kinder_baselines is not None,
            )
            transport_binds += [
                f"{venv}:/robocode/.venv:ro",
                f"{venv}:/prepared/venv:ro",
            ]
        mcp_port = MODEL_PORT + 2
        # Strict rendering uses the same dependency-clean interpreter as agents.
        agent_python = container_python(strict_blackbox)
        mcp_python = agent_python
        agent_cmd = backend.build_cli_cmd(
            config,
            mcp_python_cmd=mcp_python,
            mcp_env_config_path="/sandbox/.mcp/env_config.json",
            mcp_config_cli_path="/sandbox/.mcp/mcp_config.json",
            mcp_log_file_path="/sandbox/.mcp/mcp_server.log",
            mcp_transport="http",
            mcp_port=mcp_port,
        )
        agent_cmd, client_env = _model_client(backend_name, agent_cmd)
        # Start and health-check the render server before the CLI (same wrapper
        # as docker) so its tools are connected on the agent's first turn.
        if config.mcp_tools:
            agent_cmd = _mcp_prestart_wrapper(
                agent_cmd, port=mcp_port, python_cmd=agent_python
            )

        # Persist the CLI session store under the sandbox dir (survives the
        # ephemeral container) so a rate-limited run can be resumed via
        # the backend resume command in a fresh retry container.
        session_binds: list[str] = []
        if backend_name == "claude":
            sessions_dir = sandbox_claude_session_store(config.sandbox_dir)
            session_binds = [f"{sessions_dir.resolve()}:/home/node/.claude/projects"]
        elif backend_name == "codex":
            sessions_dir = sandbox_codex_sessions(config.sandbox_dir)
            session_binds = [f"{sessions_dir.resolve()}:/home/node/.codex/sessions"]

        tel_binds, tel_env = _telemetry_apptainer(config)
        apptainer_cmd = _build_apptainer_cmd(
            config,
            sandbox_abs=sandbox_abs,
            src_abs=str(filtered_src.resolve()) if filtered_src is not None else None,
            kindergarden_abs=(
                str(filtered_kindergarden.resolve())
                if filtered_kindergarden is not None
                else None
            ),
            ss_pybullet_abs=(
                str(ss_pybullet.resolve()) if ss_pybullet is not None else None
            ),
            kinder_baselines_abs=(
                str(filtered_kinder_baselines.resolve())
                if filtered_kinder_baselines is not None
                else None
            ),
            agent_cmd=[
                agent_python,
                f"{BROKER_DIR}/transport.py",
                f"{BROKER_DIR}/transport.json",
                *agent_cmd,
            ],
            extra_binds=session_binds + tel_binds + transport_binds,
        )

        backend.setup_sandbox_files(
            config,
            docker_python=agent_python,
            primitive_names=config.primitive_names,
        )
        _initial_commit(config.sandbox_dir)

        env = clean_apptainer_env()
        env.update(client_env)
        env.update(tel_env)

        logger.info(
            "Starting Apptainer sandbox: run_id=%s sif=%s sandbox=%s",
            run_id,
            sif_path,
            sandbox_abs,
        )
        logger.info("System prompt:\n%s", config.system_prompt)
        logger.info("Prompt:\n%s", config.prompt)

        wall_start = time.monotonic()
        with (
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file,
            agent_stdin(backend, config) as stdin_file,
        ):
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                apptainer_cmd,
                env=env,
                stdin=stdin_file,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                # Claude stops capped runs via killpg(proc.pid); own the group.
                start_new_session=True,
            )

            stream = backend.parse_stream(
                proc,
                stream_log_path=config.sandbox_dir.parent / "stream.jsonl",
                stderr_file=stderr_file,
            )
            stderr_file.seek(0)
            (config.sandbox_dir.parent / "container.stderr").write_text(
                stderr_file.read(), encoding="utf-8"
            )
        wall_time_s = time.monotonic() - wall_start

        logger.info(
            "Apptainer session done: run_id=%s turns=%d cost=$%s error=%s",
            run_id,
            stream.num_turns,
            stream.total_cost,
            stream.is_error,
        )

        _final_commit(config.sandbox_dir)

        return _stream_result_to_sandbox_result(
            stream,
            config.sandbox_dir,
            config.output_filename,
            wall_time_s=wall_time_s,
        )
