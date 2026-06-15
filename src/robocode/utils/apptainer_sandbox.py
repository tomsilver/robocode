"""Apptainer/Singularity-based sandboxed agent runner.

Mirror of :mod:`robocode.utils.docker_sandbox` for environments where the
Docker daemon is unavailable (typical on HPC clusters). The SIF image is
built from the existing ``docker/Dockerfile`` via ``docker/build_sif.sh``
(podman build + apptainer build) -- no separate definition file.

The container interior (entrypoint, firewall script, /robocode/.venv,
bind-mount layout) is byte-for-byte identical to the Docker image. The
only differences are at the host invocation layer:

* ``--bind`` instead of ``-v``
* ``--env KEY=val`` instead of ``-e KEY=val``
* ``--pwd`` instead of ``-w``
* ``--writable-tmpfs`` so the entrypoint's ``uv sync`` can write to
  ``/robocode/.venv`` (the SIF rootfs is read-only)
* ``--no-home`` so the host home doesn't shadow ``/home/node``
* ``--cleanenv`` so the host env doesn't leak in

``init-firewall.sh`` is skipped via ``ROBOCODE_SKIP_FIREWALL=1``: the
unprivileged apptainer install on the target cluster can't grant real
``CAP_NET_ADMIN``, so iptables would fail.

The image ENTRYPOINT is invoked explicitly rather than via
``apptainer run`` so behaviour does not depend on Apptainer's runscript
translation of Docker images.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robocode.primitives import (
    ENV_DEPENDENT_PRIMITIVES,
    PRIMITIVE_NAME_TO_FILE,
    REMOTE_MODULE_PRIMITIVES,
)
from robocode.utils.backends import (
    PROVIDERS,
    AgentBackend,
    firewall_domains_for_provider,
    provider_from_model,
)
from robocode.utils.docker_sandbox import (
    _PRIMITIVES_SRC,
    DOCKER_PYTHON,
    _filtered_repo_mounts,
    _find_repo_root,
    _get_claude_oauth_token,
)
from robocode.utils.sandbox import (
    SandboxConfig,
    SandboxResult,
    _final_commit,
    _initial_commit,
    _setup_sandbox_common,
    _stream_result_to_sandbox_result,
)

logger = logging.getLogger(__name__)

# Python interpreter inside the SIF (same path as in the Docker image).
APPTAINER_PYTHON: str = DOCKER_PYTHON

# Default SIF path: <repo_root>/robocode-sandbox.sif.
_DEFAULT_SIF: Path = _find_repo_root() / "robocode-sandbox.sif"


@dataclass(frozen=True)
class ApptainerSandboxConfig(SandboxConfig):
    """Configuration for an Apptainer-sandboxed agent run.

    Extends :class:`~robocode.utils.sandbox.SandboxConfig` with ``sif_path``
    for the SIF image and ``primitive_names`` to control which primitive
    source files are copied into the sandbox.
    """

    sif_path: Path = _DEFAULT_SIF
    primitive_names: tuple[str, ...] = ()
    mcp_tools: tuple[str, ...] = ()


def _setup_sandbox_dir(config: ApptainerSandboxConfig) -> None:
    """Populate ``config.sandbox_dir`` with the standard sandbox scaffolding.

    Mirrors :func:`robocode.utils.docker_sandbox._setup_sandbox_dir`; kept
    separate so it accepts :class:`ApptainerSandboxConfig` without making
    docker_sandbox.py aware of this module.
    """
    _setup_sandbox_common(config.sandbox_dir, config.init_files)

    if config.primitive_names:
        primitives_dest = config.sandbox_dir / "primitives"
        primitives_dest.mkdir(exist_ok=True)
        for name in config.primitive_names:
            # Skip env-dependent and remote-module primitives in black-box mode
            # (see the docker_sandbox counterpart): unimportable here and they
            # leak the env structure; env_client.make_primitives proxies them
            # instead (host proxies and remote-module proxies).
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


def _build_apptainer_auth_args(
    backend_name: str,
) -> tuple[list[str], dict[str, str]]:
    """Return Apptainer CLI args and env vars for backend authentication.

    Mirrors :func:`docker_sandbox._build_docker_auth_args`. Secrets (the
    Claude OAuth token, provider API keys) are returned as host env vars
    with Apptainer's ``APPTAINERENV_`` prefix rather than inline ``--env``
    flags: Apptainer injects ``APPTAINERENV_*`` into the container even
    under ``--cleanenv``, and the value never reaches argv (world-readable
    via ``ps`` / ``/proc/<pid>/cmdline`` on shared nodes). Only non-secret
    bind mounts are returned as CLI args.

    The ``~/.claude`` bind-mount fallback relies on the image being built
    with build.sh / build_sif.sh (which pass USER_UID/USER_GID build args
    so the in-container node user matches the host's UID and can read host
    files). Without that, the in-container CLI would see the host config
    as inaccessible and overwrite it.
    """
    apptainer_args: list[str] = []
    extra_env: dict[str, str] = {}

    if backend_name == "claude":
        oauth_token = _get_claude_oauth_token()
        if oauth_token:
            # APPTAINERENV_ prefix, not an inline --env flag, so the secret is
            # injected into the container (surviving --cleanenv) without ever
            # appearing on the command line.
            extra_env["APPTAINERENV_CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        else:
            logger.warning(
                "No Claude OAuth token found; falling back to ~/.claude "
                "bind-mount. Requires the image to be built via build_sif.sh "
                "(matching host UID/GID). Run `claude login` on the host "
                "if the container cannot authenticate."
            )
            host_claude_cfg = Path(
                os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))
            )
            apptainer_args += ["--bind", f"{host_claude_cfg}:/home/node/.claude"]
    else:
        opencode_data = Path.home() / ".local" / "share" / "opencode"
        if opencode_data.exists():
            apptainer_args += [
                "--bind",
                f"{opencode_data}:/home/node/.local/share/opencode",
            ]

        for info in PROVIDERS.values():
            if info.api_key_env:
                val = os.environ.get(info.api_key_env)
                if val:
                    # APPTAINERENV_ prefix keeps the key off argv (see the
                    # OAuth token note above).
                    extra_env[f"APPTAINERENV_{info.api_key_env}"] = val

    return apptainer_args, extra_env


def _build_apptainer_cmd(
    config: ApptainerSandboxConfig,
    sandbox_abs: str,
    src_abs: str,
    kindergarden_abs: str,
    auth_args: list[str],
    firewall_domains: list[str],
    agent_cmd: list[str],
) -> list[str]:
    """Assemble the full ``apptainer exec`` command line.

    Split out from :func:`run_agent_in_apptainer_sandbox` so unit tests
    can inspect the constructed command without running anything.
    """
    cmd: list[str] = [
        "apptainer",
        "exec",
        "--writable-tmpfs",
        "--no-home",
        "--cleanenv",
        "--pwd",
        "/sandbox",
        "--env",
        f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={config.max_output_tokens}",
        "--env",
        f"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE={config.autocompact_pct}",
        "--env",
        "ROBOCODE_SKIP_FIREWALL=1",
    ]

    if firewall_domains:
        cmd += [
            "--env",
            f"ROBOCODE_FIREWALL_EXTRA_DOMAINS={','.join(firewall_domains)}",
        ]

    cmd += auth_args

    cmd += [
        "--bind",
        f"{sandbox_abs}:/sandbox",
        "--bind",
        f"{src_abs}:/robocode/src",
        "--bind",
        f"{kindergarden_abs}:/robocode/third-party/kindergarden",
        str(config.sif_path),
        "/usr/local/bin/entrypoint.sh",
    ]
    cmd += agent_cmd
    return cmd


async def run_agent_in_apptainer_sandbox(
    config: ApptainerSandboxConfig,
    backend: AgentBackend,
) -> SandboxResult:
    """Run an agent inside the ``robocode-sandbox`` SIF via apptainer.

    Step-for-step parallel of
    :func:`~robocode.utils.docker_sandbox.run_agent_in_docker_sandbox`.
    See the module docstring for the docker -> apptainer flag mapping.
    """
    backend_name = backend.name

    if not config.sif_path.exists():
        raise RuntimeError(
            f"SIF image not found at {config.sif_path}; "
            "build it with: bash docker/build_sif.sh"
        )

    _setup_sandbox_dir(config)

    sandbox_abs = str(config.sandbox_dir.resolve())
    run_id = f"apptainer-sandbox-{uuid.uuid4().hex[:8]}"

    with _filtered_repo_mounts(blackbox=config.blackbox) as (
        filtered_src,
        filtered_kindergarden,
    ):
        auth_args, auth_env = _build_apptainer_auth_args(backend_name)

        firewall_domains: list[str] = []
        if backend_name == "opencode":
            firewall_domains = firewall_domains_for_provider(
                provider_from_model(config.model)
            )

        agent_cmd = backend.build_cli_cmd(
            config,
            mcp_python_cmd=APPTAINER_PYTHON,
            mcp_env_config_path="/sandbox/.mcp/env_config.json",
            mcp_config_cli_path="/sandbox/.mcp/mcp_config.json",
            mcp_log_file_path="/sandbox/.mcp/mcp_server.log",
        )

        apptainer_cmd = _build_apptainer_cmd(
            config,
            sandbox_abs=sandbox_abs,
            src_abs=str(filtered_src.resolve()),
            kindergarden_abs=str(filtered_kindergarden.resolve()),
            auth_args=auth_args,
            firewall_domains=firewall_domains,
            agent_cmd=agent_cmd,
        )

        backend.setup_sandbox_files(
            config,
            docker_python=APPTAINER_PYTHON,
            primitive_names=config.primitive_names,
        )
        _initial_commit(config.sandbox_dir)

        env = backend.build_env(config, auth_env if auth_env else None)

        logger.info(
            "Starting Apptainer sandbox: run_id=%s sif=%s sandbox=%s",
            run_id,
            config.sif_path,
            sandbox_abs,
        )
        logger.info("System prompt:\n%s", config.system_prompt)
        logger.info("Prompt:\n%s", config.prompt)

        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            apptainer_cmd,
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
            "Apptainer session done: run_id=%s turns=%d cost=$%s error=%s",
            run_id,
            stream.num_turns,
            stream.total_cost,
            stream.is_error,
        )

        _final_commit(config.sandbox_dir)

        return _stream_result_to_sandbox_result(
            stream, config.sandbox_dir, config.output_filename
        )


def run_genplan_in_apptainer(
    sandbox_dir: Path,
    completion_cfg: dict[str, Any],
    sif_path: Path = _DEFAULT_SIF,
    timeout: float = 3600.0,
) -> None:
    """Apptainer analog of :func:`docker_sandbox.run_genplan_in_docker`.

    Mirrors the docker function: runs the whole LLM-GenPlan loop inside one
    sandbox container via the genplan driver, which reads
    ``sandbox_dir/genplan_config.json`` and writes ``sandbox_dir/approach.py``
    and ``sandbox_dir/cost.json``. Keeps ``primitives`` in the source mount so
    the policy can build/use them as eval does on the host.
    """
    if not sif_path.exists():
        raise RuntimeError(
            f"SIF image not found at {sif_path}; build it with: bash docker/build_sif.sh"
        )
    run_id = f"apptainer-genplan-{uuid.uuid4().hex[:8]}"
    with _filtered_repo_mounts(keep_primitives=True) as (
        filtered_src,
        filtered_kindergarden,
    ):
        auth_backend = "claude" if completion_cfg["provider"] == "cli" else "opencode"
        auth_args, auth_env = _build_apptainer_auth_args(auth_backend)
        firewall_domains = firewall_domains_for_provider(
            completion_cfg["provider"], completion_cfg.get("base_url", "")
        )
        firewall_env: list[str] = []
        if firewall_domains:
            firewall_env = [
                "--env",
                f"ROBOCODE_FIREWALL_EXTRA_DOMAINS={','.join(firewall_domains)}",
            ]
        apptainer_cmd = [
            "apptainer",
            "exec",
            "--writable-tmpfs",
            "--no-home",
            "--cleanenv",
            "--pwd",
            "/sandbox",
            "--env",
            "ROBOCODE_SKIP_FIREWALL=1",
            *firewall_env,
            *auth_args,
            "--bind",
            f"{sandbox_dir.resolve()}:/sandbox",
            "--bind",
            f"{filtered_src.resolve()}:/robocode/src",
            "--bind",
            f"{filtered_kindergarden.resolve()}:/robocode/third-party/kindergarden",
            str(sif_path),
            "/usr/local/bin/entrypoint.sh",
            APPTAINER_PYTHON,
            "-m",
            "robocode.approaches.genplan_driver",
        ]
        logger.info("Starting genplan Apptainer run %s sif=%s", run_id, sif_path)
        subprocess.run(
            apptainer_cmd,
            env={**os.environ, **auth_env},
            stdin=subprocess.DEVNULL,
            check=True,
            timeout=timeout,
        )
