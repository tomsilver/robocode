"""Agent backend protocol and implementations.

Each backend adapts a specific coding agent CLI (Claude Code, OpenCode,
etc.) to the shared sandbox runner.

To add a new provider, add an entry to :data:`PROVIDERS` below. The
``domains`` list is used by the Docker firewall whitelist, and
``api_key_env`` is the environment variable forwarded into Docker
containers for authentication.
"""

from dataclasses import dataclass, field

from omegaconf import DictConfig

from robocode.utils.backends.base import AgentBackend
from robocode.utils.backends.claude import ClaudeBackend
from robocode.utils.backends.opencode import OpenCodeBackend

__all__ = [
    "AgentBackend",
    "DEFAULT_BACKEND",
    "DEFAULT_BACKEND_CFG",
    "DEFAULT_OPENCODE_CFG",
    "PROVIDERS",
    "CLAUDE_PROMPT_SUFFIX",
    "OPENCODE_PROMPT_SUFFIX",
    "create_backend",
    "firewall_domains_for_model",
    "provider_from_model",
]


def create_backend(backend_cfg: DictConfig) -> AgentBackend:
    """Instantiate an agent backend from a config mapping.

    Parameters
    ----------
    backend_cfg:
        Backend config (DictConfig or dict) with at least a ``backend``
        key (``"claude"`` or ``"opencode"``). Additional keys like
        ``base_url`` or ``auth_token`` are passed through.
    """
    name = backend_cfg["backend"]
    if name == "claude":
        return ClaudeBackend(backend_cfg)
    if name == "opencode":
        return OpenCodeBackend(backend_cfg)
    raise ValueError(f"Unknown backend {name!r}. Expected 'claude' or 'opencode'.")


# Convenience defaults for tests and callers that don't use Hydra configs.
DEFAULT_BACKEND_CFG = DictConfig({"backend": "claude", "model": "sonnet"})
DEFAULT_OPENCODE_CFG = DictConfig({"backend": "opencode", "model": "openai/gpt-4o"})
DEFAULT_BACKEND = create_backend(DEFAULT_BACKEND_CFG)


@dataclass(frozen=True)
class ProviderInfo:
    """Metadata for an LLM provider."""

    # API domains to whitelist in the Docker firewall.
    domains: list[str] = field(default_factory=list)
    # Environment variable holding the API key (forwarded into Docker).
    api_key_env: str = ""


# ---- Provider registry ----
# Add new providers here. The key is the provider prefix used in model
# strings (e.g. "openai" in "openai/gpt-4o").
PROVIDERS: dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        domains=["api.openai.com"],
        api_key_env="OPENAI_API_KEY",
    ),
    "anthropic": ProviderInfo(
        domains=["api.anthropic.com"],
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "google": ProviderInfo(
        domains=["generativelanguage.googleapis.com"],
        api_key_env="GOOGLE_API_KEY",
    ),
}


def provider_from_model(model: str) -> str:
    """Extract the provider prefix from a model string.

    E.g. ``"openai/gpt-4o"`` -> ``"openai"``.
    Returns ``""`` if no slash is present.
    """
    if "/" in model:
        return model.split("/", 1)[0]
    return ""


def firewall_domains_for_model(model: str) -> list[str]:
    """Return the API domains that must be whitelisted for *model*."""
    provider = provider_from_model(model)
    info = PROVIDERS.get(provider)
    return list(info.domains) if info else []


# Backend-specific system prompt suffixes.
# Always appended to guide tool usage and subagent delegation.

CLAUDE_PROMPT_SUFFIX = (
    " CRITICAL: Tool names are case-sensitive and MUST be capitalized exactly: "
    "Bash, Read, Write, Edit, Glob, Grep, Task. "
    "Using lowercase (e.g. 'bash' instead of 'Bash') will fail. "
    "Always use the exact capitalized names."
    " SUBAGENTS: Use the Task tool to delegate work to subagents. "
    "Available subagent types: "
    "'general-purpose' for research, multi-step tasks, and broad exploration; "
    "'Explore' for fast read-only codebase searches (grep, glob, read); "
    "'Plan' for designing an implementation strategy before coding. "
    "Start complex tasks by spawning a Plan subagent to outline your approach, "
    "then use Explore subagents to read source code in parallel, "
    "and general-purpose subagents for deeper analysis."
)

OPENCODE_PROMPT_SUFFIX = (
    " Your tools use lowercase names: bash, read, write, edit, glob, grep, task."
    " SUBAGENTS: Use the task tool to delegate work to subagents. "
    "Available subagent_type values: "
    "'general' for research, multi-step tasks, and broad exploration; "
    "'explore' for fast read-only codebase searches (grep, glob, read); "
    "'plan' for designing an implementation strategy before coding. "
    "Start complex tasks by spawning a plan subagent to outline your approach, "
    "then use explore subagents to read source code in parallel, "
    "and general subagents for deeper analysis."
)
