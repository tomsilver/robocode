"""Tests for the shared LLM client helpers (cost estimation)."""

from omegaconf import DictConfig

from robocode.utils.llm.base import pricing_from_cfg, usage_cost


def test_usage_cost_from_list_prices():
    """Cost is tokens-in/out times per-million-token prices."""
    # 1M input @ $3 + 1M output @ $15 = $18.
    assert usage_cost(1_000_000, 1_000_000, 3.0, 15.0) == 18.0
    # Partial millions scale linearly: 0.5M @ $5 + 0.2M @ $25 = $2.5 + $5 = $7.5.
    assert usage_cost(500_000, 200_000, 5.0, 25.0) == 7.5


def test_pricing_from_cfg():
    """Prices are read when present; absent on either side defaults to 0.0."""
    assert pricing_from_cfg(
        DictConfig({"input_cost_per_mtok": 3.0, "output_cost_per_mtok": 15.0})
    ) == (3.0, 15.0)
    # One side only -> the other defaults to 0.0 (still "priced").
    assert pricing_from_cfg(DictConfig({"output_cost_per_mtok": 25.0})) == (0.0, 25.0)
    # Neither set -> unpriced (cost stays unknown, e.g. local vLLM/Ollama).
    assert pricing_from_cfg(DictConfig({"model": "x"})) is None
