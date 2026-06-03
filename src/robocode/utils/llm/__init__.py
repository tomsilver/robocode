"""Pure LLM completion clients.

A completion client is a single model API call (messages in, one text response
out), with no tools and no agent loop -- the caller drives the conversation.
This is distinct from ``robocode.utils.backends``, which spawns an autonomous
coding agent (Claude Code / OpenCode CLI) that reads, writes, and runs code.

Providers (selected by ``cfg.provider``):
    - ``anthropic``: Anthropic Messages API (faithful plain LLM)
    - ``openai_compatible``: OpenAI ``/v1`` API, incl. vLLM and Ollama
    - ``cli``: Claude CLI single-shot, no tools (experimental, see cli_client)
"""

from omegaconf import DictConfig

from robocode.utils.llm.base import LLMClient, LLMResponse

__all__ = ["LLMClient", "LLMResponse", "create_llm_client"]


def create_llm_client(cfg: DictConfig) -> LLMClient:
    """Instantiate a completion client from a config mapping."""
    # Imports are local so optional SDK dependencies load only when used.
    provider = cfg["provider"]
    if provider == "anthropic":
        from robocode.utils.llm.anthropic_client import (  # pylint: disable=import-outside-toplevel
            AnthropicClient,
        )

        return AnthropicClient(cfg)
    if provider == "openai_compatible":
        from robocode.utils.llm.openai_client import (  # pylint: disable=import-outside-toplevel
            OpenAICompatibleClient,
        )

        return OpenAICompatibleClient(cfg)
    if provider == "cli":
        from robocode.utils.llm.cli_client import (  # pylint: disable=import-outside-toplevel
            ClaudeCLIClient,
        )

        return ClaudeCLIClient(cfg)
    raise ValueError(
        f"Unknown completion provider {provider!r}. Expected "
        "'anthropic', 'openai_compatible', or 'cli'."
    )
