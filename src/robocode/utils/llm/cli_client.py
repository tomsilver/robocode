"""Claude CLI driven as a plain LLM: the default completion backend.

Runs ``claude -p`` with tools and system prompt stripped, resuming matching
conversations by session ID to preserve message boundaries and cache reuse. Billing
follows the authenticated CLI (no API key needed), and the same CLI serves the
agentic backend, so completion and agentic runs share one auth path and one set
of model ids.

The CLI keeps an irreducible ~2k-token system prompt that ``--system-prompt ""``
does not remove. The plain-LLM baselines are therefore prompted with that
preamble present, which is worth stating whenever their prompts are described.
"""

from __future__ import annotations

import json
import os
import subprocess

from omegaconf import DictConfig

from robocode.utils.backends.claude import anthropic_compatible_env, get_claude_cmd
from robocode.utils.llm.base import LLMResponse


class ClaudeCLIClient:
    """Text completions via the Claude Code CLI with per-client session reuse."""

    def __init__(self, cfg: DictConfig) -> None:
        self._model = cfg["model"]
        self._base_url = cfg.get("base_url", "")
        self._auth_token = cfg.get("auth_token", "ollama")
        self._ollama_keep_alive = cfg.get("ollama_keep_alive", "")
        self._timeout_s = cfg.get("request_timeout_s", 1200.0)
        self._max_thinking_tokens = cfg.get("max_thinking_tokens", 0)
        self._session_id: str | None = None
        self._history: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return the model's reply to a message list."""
        # Only resume an exact continuation. Callers may reuse this client for
        # independent samples or edit history, which must start fresh sessions.
        resume = (
            self._session_id is not None
            and len(messages) == len(self._history) + 1
            and messages[:-1] == self._history
            and messages[-1]["role"] == "user"
        )
        session_id = self._session_id if resume else None
        prompt = messages[-1]["content"] if resume else _flatten(messages)
        # A failed request may have advanced the remote session. Do not resume
        # that uncertain state if the caller retries.
        self._session_id = None
        self._history = []
        args = [
            get_claude_cmd(),
            "-p",
            "--output-format",
            "json",
            "--model",
            self._model,
            "--tools",
            "",
            # --tools only controls built-ins; deny MCP tools as well.
            "--disallowedTools",
            "*",
            "--system-prompt",
            "",
            "--exclude-dynamic-system-prompt-sections",
            "--max-thinking-tokens",
            str(self._max_thinking_tokens),
        ]
        if session_id is not None:
            args += ["--resume", session_id]
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
        env.update(
            anthropic_compatible_env(
                self._base_url, self._auth_token, self._ollama_keep_alive
            )
        )
        # Prompt via stdin, not argv: it can exceed the OS per-arg limit (128KB).
        try:
            result = subprocess.run(
                args,
                env=env,
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=self._timeout_s,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Claude exited with status {exc.returncode}.\n"
                f"stdout: {exc.stdout or '(empty)'}\n"
                f"stderr: {exc.stderr or '(empty)'}"
            ) from exc
        data = json.loads(result.stdout)
        if data.get("is_error"):
            raise RuntimeError(f"Claude completion failed: {data.get('result', data)}")
        response = LLMResponse(text=data["result"], cost_usd=data.get("total_cost_usd"))
        self._session_id = data.get("session_id")
        self._history = [dict(message) for message in messages] + [
            {"role": "assistant", "content": response.text}
        ]
        return response


def _flatten(messages: list[dict[str, str]]) -> str:
    """Serialize initial or replaced history when starting a fresh CLI session."""
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
