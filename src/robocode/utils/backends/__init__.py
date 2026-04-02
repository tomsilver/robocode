"""Agent backend protocol and implementations.

Each backend adapts a specific coding agent CLI (Claude Code, OpenCode,
etc.) to the shared sandbox runner.

To add a new provider, add an entry to :data:`PROVIDERS` below. The
``domains`` list is used by the Docker firewall whitelist, and
``api_key_env`` is the environment variable forwarded into Docker
containers for authentication.
"""

from dataclasses import dataclass, field

from robocode.utils.backends.base import AgentBackend, create_backend

__all__ = [
    "AgentBackend",
    "PROVIDERS",
    "TOOL_NAMES_PROMPT_SUFFIX",
    "create_backend",
    "provider_from_model",
]


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


# Appended to the system prompt when using non-Anthropic models via
# Claude Code CLI (base_url set), since smaller models often get tool
# name casing wrong.
TOOL_NAMES_PROMPT_SUFFIX = (
    " CRITICAL: Tool names are case-sensitive and MUST be capitalized exactly: "
    "Bash, Read, Write, Edit, Glob, Grep, Task. "
    "Using lowercase (e.g. 'bash' instead of 'Bash') will fail. "
    "Always use the exact capitalized names."
)
