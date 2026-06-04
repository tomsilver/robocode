"""Pure LLM completion clients: one model API call (no tools, no agent loop),
unlike ``robocode.utils.backends`` which drives an autonomous coding agent.

Providers (``cfg.provider``): ``anthropic``, ``openai_compatible`` (vLLM/Ollama),
``cli`` (experimental).
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
