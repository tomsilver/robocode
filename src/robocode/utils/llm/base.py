"""Pure LLM completion interface: one model API call (messages in, text out),
no tools and no agent loop. Contrast with ``robocode.utils.backends``, which
drives an autonomous coding agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from omegaconf import DictConfig


@dataclass(frozen=True)
class LLMResponse:
    """A single completion result."""

    text: str
    cost_usd: float | None = None


def usage_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_mtok: float,
    output_cost_per_mtok: float,
) -> float:
    """Estimate the dollar cost of a completion from token usage and list prices.

    The Anthropic and OpenAI-compatible APIs report token counts but no dollar amount,
    so we estimate from per-million-token list prices supplied in the completion config.
    This is an estimate: it uses standard list prices and does not account for prompt-
    caching reads/writes, batch, or priority discounts.
    """
    return (
        input_tokens * input_cost_per_mtok + output_tokens * output_cost_per_mtok
    ) / 1_000_000


def pricing_from_cfg(cfg: DictConfig) -> tuple[float, float] | None:
    """``(input, output)`` per-million-token list prices from config, or None.

    Returns None when neither ``input_cost_per_mtok`` nor ``output_cost_per_mtok``
    is set (cost stays unknown for that backend, e.g. local vLLM/Ollama); a
    missing side defaults to 0.0.
    """
    in_price = cfg.get("input_cost_per_mtok", None)
    out_price = cfg.get("output_cost_per_mtok", None)
    if in_price is None and out_price is None:
        return None
    return float(in_price or 0.0), float(out_price or 0.0)


class LLMClient(Protocol):
    """A provider that turns a message list into one text response."""

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return the model's reply to a ``[{"role", "content"}, ...]`` list."""
