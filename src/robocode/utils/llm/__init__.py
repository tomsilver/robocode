"""Pure LLM completion clients: one model API call (no tools, no agent loop),
unlike ``robocode.utils.backends`` which drives an autonomous coding agent.

Providers (``cfg.provider``): ``anthropic``, ``openai_compatible`` (vLLM/Ollama),
``cli`` (experimental).
"""

from collections.abc import Callable

from omegaconf import DictConfig

from robocode.utils.llm.anthropic_client import AnthropicClient
from robocode.utils.llm.base import LLMClient, LLMResponse
from robocode.utils.llm.cli_client import ClaudeCLIClient
from robocode.utils.llm.openai_client import OpenAICompatibleClient

__all__ = ["LLMClient", "LLMResponse", "create_llm_client"]

_CLIENTS: dict[str, Callable[[DictConfig], LLMClient]] = {
    "anthropic": AnthropicClient,
    "openai_compatible": OpenAICompatibleClient,
    "cli": ClaudeCLIClient,
}


def create_llm_client(cfg: DictConfig) -> LLMClient:
    """Instantiate a completion client from a config mapping."""
    provider = cfg["provider"]
    if provider not in _CLIENTS:
        raise ValueError(
            f"Unknown completion provider {provider!r}. "
            f"Expected one of {sorted(_CLIENTS)}."
        )
    return _CLIENTS[provider](cfg)
