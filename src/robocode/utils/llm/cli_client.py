"""Claude CLI driven as a plain LLM (experimental, best-effort non-agentic).

Runs ``claude -p`` single-shot with tools disabled and the system prompt
minimized, so it behaves as closely as possible to a normal completion API.

Caveat: the Claude CLI keeps an irreducible baseline system prompt (~2k
tokens) that cannot be removed via flags, so this is NOT a perfectly faithful
plain-LLM path. The SDK clients (anthropic/openai) inject zero system prompt
and are the faithful default; prefer those when possible.
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
        # Generous: the subscription CLI throttles batched calls, so allow long
        # waits, but still fail loudly rather than hang forever.
        self._timeout_s = cfg.get("request_timeout_s", 1200.0)

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
            # Disable extended thinking: it adds latency/tokens we don't need for
            # one-shot code generation and worsens throttling.
            "--max-thinking-tokens",
            "0",
        ]
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
        env.update(
            anthropic_compatible_env(
                self._base_url, self._auth_token, self._ollama_keep_alive
            )
        )
        # Feed the prompt via stdin, NOT as a CLI argument: with the env source
        # and debug history it can exceed the OS per-argument limit
        # (MAX_ARG_STRLEN, 128KB -> "Argument list too long"). `claude -p` reads
        # the query from stdin, which also avoids the stdin-wait hang.
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
    """Collapse a multi-turn history into one prompt string.

    The CLI ``-p`` mode is stateless, so the framework-owned conversation is
    replayed as labeled turns each call.
    """
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
