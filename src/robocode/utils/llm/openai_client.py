"""OpenAI-compatible completion client (a pure API call, not an agent).

Covers any server exposing the OpenAI ``/v1/chat/completions`` API, including
vLLM and Ollama. Point it at the server with ``base_url``.
"""

from __future__ import annotations

import os

import openai
from omegaconf import DictConfig

from robocode.utils.backends.ollama_server import ensure_ollama
from robocode.utils.llm.base import LLMResponse, pricing_from_cfg, usage_cost


class OpenAICompatibleClient:
    """Single-shot completions via an OpenAI-compatible chat endpoint.

    The API returns token usage but no dollar amount, so ``cost_usd`` is an
    estimate from the ``input_cost_per_mtok`` / ``output_cost_per_mtok`` list
    prices in the config (left ``None`` when those are unset, as for local
    vLLM/Ollama servers).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._model = cfg["model"]
        self._base_url = cfg.get("base_url", "") or None
        self._pricing = pricing_from_cfg(cfg)
        # Local servers (vLLM/Ollama) usually need no real key.
        api_key_env = cfg.get("api_key_env", "")
        if api_key_env:
            api_key = os.environ.get(api_key_env, "")
            if not api_key:
                raise ValueError(
                    f"Config sets api_key_env={api_key_env} but that variable is "
                    "not set; export it, or clear api_key_env for keyless local "
                    "servers."
                )
        else:
            api_key = "EMPTY"
        if self._base_url and "11434" in self._base_url:
            ensure_ollama(keep_alive=cfg.get("ollama_keep_alive", "") or "5m")
        self._client = openai.OpenAI(api_key=api_key, base_url=self._base_url)

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return the model's reply to a message list."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        cost = None
        if self._pricing is not None:
            usage = response.usage
            assert usage is not None
            cost = usage_cost(
                usage.prompt_tokens, usage.completion_tokens, *self._pricing
            )
        return LLMResponse(
            text=response.choices[0].message.content or "", cost_usd=cost
        )
