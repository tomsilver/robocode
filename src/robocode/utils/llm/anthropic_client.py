"""Anthropic SDK completion client (a pure API call, not an agent)."""

from __future__ import annotations

import os

from omegaconf import DictConfig

from robocode.utils.llm.base import LLMResponse

# Anthropic recommends an explicit max_tokens; generated policies are short.
_MAX_TOKENS = 8192


class AnthropicClient:
    """Single-shot completions via the Anthropic Messages API.

    ``model`` must be a full API id (e.g. ``claude-sonnet-4-6``). An optional
    ``base_url`` points at an Anthropic-compatible endpoint; ``api_key_env``
    names the environment variable holding the API key.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._model = cfg["model"]
        self._base_url = cfg.get("base_url", "") or None
        self._api_key = os.environ[cfg.get("api_key_env", "ANTHROPIC_API_KEY")]
        self._max_tokens = cfg.get("max_tokens", _MAX_TOKENS)

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return the model's reply to a message list."""
        import anthropic  # pylint: disable=import-outside-toplevel,import-error

        client = anthropic.Anthropic(api_key=self._api_key, base_url=self._base_url)
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,  # type: ignore[arg-type]
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(text=text, cost_usd=None)
