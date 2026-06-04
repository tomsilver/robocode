"""Claude CLI driven as a plain LLM (experimental, best-effort non-agentic).

Runs ``claude -p`` single-shot with tools and system prompt stripped. The CLI
keeps an irreducible ~2k-token system prompt, so prefer the SDK clients
(anthropic/openai) when a faithful plain LLM is needed.
"""

from __future__ import annotations

import json
import os
import subprocess

from omegaconf import DictConfig

from robocode.utils.backends.claude import anthropic_compatible_env, get_claude_cmd
from robocode.utils.llm.base import LLMResponse


class ClaudeCLIClient:
    """Single-shot completions via the Claude Code CLI (no tools)."""

    def __init__(self, cfg: DictConfig) -> None:
        self._model = cfg["model"]
        self._base_url = cfg.get("base_url", "")
        self._auth_token = cfg.get("auth_token", "ollama")
        self._ollama_keep_alive = cfg.get("ollama_keep_alive", "")
        self._timeout_s = cfg.get("request_timeout_s", 1200.0)
        self._max_thinking_tokens = cfg.get("max_thinking_tokens", 0)

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return the model's reply to a message list."""
        prompt = _flatten(messages)
        args = [
            get_claude_cmd(),
            "-p",
            "--output-format",
            "json",
            "--model",
            self._model,
            "--tools",
            "",
            "--system-prompt",
            "",
            "--exclude-dynamic-system-prompt-sections",
            "--max-thinking-tokens",
            str(self._max_thinking_tokens),
        ]
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
        env.update(
            anthropic_compatible_env(
                self._base_url, self._auth_token, self._ollama_keep_alive
            )
        )
        # Prompt via stdin, not argv: it can exceed the OS per-arg limit (128KB).
        result = subprocess.run(
            args,
            env=env,
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            timeout=self._timeout_s,
        )
        data = json.loads(result.stdout)
        return LLMResponse(text=data["result"], cost_usd=data.get("total_cost_usd"))


def _flatten(messages: list[dict[str, str]]) -> str:
    """Collapse the multi-turn history into one prompt for the stateless CLI."""
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
