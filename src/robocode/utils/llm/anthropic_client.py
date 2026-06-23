"""Anthropic SDK completion client (a pure API call, not an agent)."""

from __future__ import annotations

import os

import anthropic
from omegaconf import DictConfig

from robocode.utils.llm.base import LLMResponse, pricing_from_cfg, usage_cost


class AnthropicClient:
    """Single-shot completions via the Anthropic Messages API.

    ``model`` is a full API id (e.g. ``claude-sonnet-4-6``). The Messages API
    requires an explicit ``max_tokens``; set it via ``max_tokens`` in the config.
    The API returns token usage but no dollar amount, so ``cost_usd`` is an
    estimate from the ``input_cost_per_mtok`` / ``output_cost_per_mtok`` list
    prices in the config (left ``None`` when those are unset).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._model = cfg["model"]
        self._base_url = cfg.get("base_url", "") or None
        api_key = os.environ[cfg.get("api_key_env", "ANTHROPIC_API_KEY")]
        self._max_tokens = cfg.get("max_tokens", 32000)
        self._pricing = pricing_from_cfg(cfg)
        self._client = anthropic.Anthropic(api_key=api_key, base_url=self._base_url)

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return the model's reply to a message list."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,  # type: ignore[arg-type]
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        cost = None
        if self._pricing is not None:
            cost = usage_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                *self._pricing,
            )
        return LLMResponse(text=text, cost_usd=cost)
