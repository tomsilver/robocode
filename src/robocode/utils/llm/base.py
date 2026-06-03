"""Pure LLM completion interface.

This is a single model API call (messages in, one text response out) -- NOT
an agent. There are no tools, no file access, and no multi-turn tool-use loop;
the caller owns the conversation. Contrast with ``robocode.utils.backends``,
which drives an autonomous coding agent (Claude Code / OpenCode CLI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    """A single completion result."""

    text: str
    cost_usd: float | None = None


class LLMClient(Protocol):
    """A provider that turns a message list into one text response."""

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return the model's reply to a ``[{"role", "content"}, ...]`` list."""
