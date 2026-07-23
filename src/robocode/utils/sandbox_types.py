"""Shared data types for the sandbox runner and agent backends.

Kept separate from ``sandbox.py`` to avoid circular imports between
the sandbox module and the backend implementations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    # Which primitive source files to copy into the sandbox so the agent can
    # read their API. Empty for the local backend (the host venv already exposes
    # robocode.primitives); the docker/apptainer backends strip that package and
    # copy the listed sources instead.
    primitive_names: tuple[str, ...] = ()
    # Address the sandbox uses to reach a service on the host loopback (a local
    # model server, ollama/vLLM). 127.0.0.1 for the local and apptainer backends
    # (they share the host network namespace); DockerSandboxConfig overrides this
    # to host.docker.internal (mapped to the host gateway via --add-host).
    local_model_host: str = "127.0.0.1"


@dataclass(frozen=True)
class GenerationMetrics:
    """Runtime metrics from one agent generation run.

    These exist only while the generation subprocess is alive (the coding agent
    writing approach.py); they are parsed from its CLI stream and persisted next
    to the eval results. ``wall_time_s`` is our own end-to-end timer around the
    subprocess; ``cli_duration_ms`` is the agent-session time the CLI reports.
    """

    wall_time_s: float | None = None
    cli_duration_ms: int | None = None
    cli_duration_api_ms: int | None = None
    model_wait_time_s: float | None = None
    experiment_time_s: float | None = None
    other_tool_time_s: float | None = None
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    num_tool_calls: int = 0
    num_autocompactions: int = 0
    num_permission_denials: int = 0
    turn_limit_hit: bool = False
    stop_reason: str | None = None
    model_usage: dict[str, Any] = field(default_factory=dict)
    # Rate-limit interruptions. The retry loop sleeps until the usage window
    # resets and reruns the whole sandbox; the interrupted attempts are dropped
    # from evaluation but their count and spend are recorded here so runs that
    # straddled a reset can be identified and excluded in analysis. The main
    # fields above describe the final, successful attempt only.
    rate_limit_retries: int = 0
    aborted_tokens: int = 0
    aborted_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Sum of input, output, and cache read/creation tokens."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        """Flat ``gen_*`` keys for results.json (one analysis column each)."""
        return {
            "gen_wall_time_s": self.wall_time_s,
            "gen_cli_duration_ms": self.cli_duration_ms,
            "gen_cli_duration_api_ms": self.cli_duration_api_ms,
            "gen_model_wait_time_s": self.model_wait_time_s,
            "gen_experiment_time_s": self.experiment_time_s,
            "gen_other_tool_time_s": self.other_tool_time_s,
            "gen_num_turns": self.num_turns,
            "gen_input_tokens": self.input_tokens,
            "gen_output_tokens": self.output_tokens,
            "gen_cache_read_tokens": self.cache_read_tokens,
            "gen_cache_creation_tokens": self.cache_creation_tokens,
            "gen_total_tokens": self.total_tokens,
            "gen_num_tool_calls": self.num_tool_calls,
            "gen_num_autocompactions": self.num_autocompactions,
            "gen_num_permission_denials": self.num_permission_denials,
            "gen_turn_limit_hit": self.turn_limit_hit,
            "gen_stop_reason": self.stop_reason,
            "gen_model_usage": self.model_usage,
            "gen_rate_limit_retries": self.rate_limit_retries,
            "gen_aborted_tokens": self.aborted_tokens,
            "gen_aborted_cost_usd": self.aborted_cost_usd,
        }


@dataclass(frozen=True)
class SandboxResult:
    """Result from a sandboxed agent run."""

    success: bool
    output_file: Path | None
    error: str | None
    total_cost_usd: float | None = None
    rate_limit_reset: str | None = None  # e.g. "3am" from usage limit message
    generation_metrics: GenerationMetrics | None = None


@dataclass(frozen=True)
class _StreamParseResult:
    """Intermediate result from parsing agent CLI stream output."""

    is_error: bool
    error_text: str | None
    num_turns: int
    total_cost: float | None
    rate_limit_reset: str | None = None  # e.g. "3am" parsed from usage message
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    num_tool_calls: int = 0
    num_autocompactions: int = 0
    num_permission_denials: int = 0
    turn_limit_hit: bool = False
    cli_duration_ms: int | None = None
    cli_duration_api_ms: int | None = None
    model_wait_time_s: float | None = None
    experiment_time_s: float | None = None
    other_tool_time_s: float | None = None
    stop_reason: str | None = None
    model_usage: dict[str, Any] = field(default_factory=dict)
