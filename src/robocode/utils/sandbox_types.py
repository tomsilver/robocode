"""Shared data types for the sandbox runner and agent backends.

Kept separate from ``sandbox.py`` to avoid circular imports between
the sandbox module and the backend implementations.
"""

from dataclasses import dataclass, field
from pathlib import Path

CONTAINER_BACKENDS = ("docker", "apptainer", "local")


def resolve_container_backend(container_backend: str | None, use_docker: bool) -> str:
    """Resolve and validate an approach's sandbox backend selection.

    ``container_backend`` takes precedence; ``use_docker`` is the legacy
    boolean kept for back-compat (True -> docker, False -> local).
    """
    if container_backend is None:
        container_backend = "docker" if use_docker else "local"
    if container_backend not in CONTAINER_BACKENDS:
        raise ValueError(
            f"Invalid container_backend {container_backend!r}; "
            f"expected one of {CONTAINER_BACKENDS}"
        )
    return container_backend


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for a sandboxed agent run."""

    sandbox_dir: Path
    init_files: dict[str, Path] = field(default_factory=dict)
    prompt: str = ""
    output_filename: str = ""
    model: str = "sonnet"
    max_budget_usd: float = 5.0
    max_turns: int = 0  # 0 = unlimited
    system_prompt: str = ""
    mcp_tools: tuple[str, ...] = ()
    max_output_tokens: int = 16384
    autocompact_pct: int = 80
    blackbox: bool = False
    # Address the sandbox uses to reach a service on the host loopback (a local
    # model server, ollama/vLLM). 127.0.0.1 for the local and apptainer backends
    # (they share the host network namespace); DockerSandboxConfig overrides this
    # to host.docker.internal (mapped to the host gateway via --add-host).
    local_model_host: str = "127.0.0.1"


@dataclass(frozen=True)
class SandboxResult:
    """Result from a sandboxed agent run."""

    success: bool
    output_file: Path | None
    error: str | None
    total_cost_usd: float | None = None
    rate_limit_reset: str | None = None  # e.g. "3am" from usage limit message


@dataclass(frozen=True)
class _StreamParseResult:
    """Intermediate result from parsing agent CLI stream output."""

    is_error: bool
    error_text: str | None
    num_turns: int
    total_cost: float | None
    rate_limit_reset: str | None = None  # e.g. "3am" parsed from usage message
