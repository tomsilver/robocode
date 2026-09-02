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
import logging
import os
import subprocess
from typing import Any

from omegaconf import DictConfig
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from robocode.utils.backends.claude import anthropic_compatible_env, get_claude_cmd
from robocode.utils.llm.base import LLMResponse

logger = logging.getLogger(__name__)


class ClaudeCLIError(RuntimeError):
    """A failed Claude CLI invocation, including its structured error output."""

    def __init__(self, returncode: int, stdout: str | None, stderr: str | None) -> None:
        self.returncode = returncode
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        super().__init__(
            f"Claude exited with status {returncode}.\n"
            f"stdout: {self.stdout or '(empty)'}\n"
            f"stderr: {self.stderr or '(empty)'}"
        )


def _is_transient_claude_error(exc: BaseException) -> bool:
    """Return whether Claude reported a retryable server-side API failure."""
    if not isinstance(exc, ClaudeCLIError):
        return False
    try:
        data: Any = json.loads(exc.stdout)
    except (json.JSONDecodeError, TypeError):
        return False
    status = data.get("api_error_status") if isinstance(data, dict) else None
    cost = data.get("total_cost_usd") if isinstance(data, dict) else None
    return isinstance(status, int) and 500 <= status < 600 and cost in (None, 0)


def _session_limit_reset(exc: BaseException) -> str | None:
    """Return Claude's reset message for a safe-to-repeat usage-limit failure."""
    if not isinstance(exc, ClaudeCLIError):
        return None
    try:
        data: Any = json.loads(exc.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("api_error_status") != 429:
        return None
    message = data.get("result")
    cost = data.get("total_cost_usd")
    if not isinstance(message, str) or cost not in (None, 0):
        return None
    normalized = message.lower()
    if "limit" not in normalized or "reset" not in normalized:
        return None
    return message


class ClaudeCLIClient:
    """Text completions via the Claude Code CLI with per-client session reuse."""

    def __init__(self, cfg: DictConfig) -> None:
        self._model = cfg["model"]
        self._base_url = cfg.get("base_url", "")
        self._auth_token = cfg.get("auth_token", "ollama")
        self._ollama_keep_alive = cfg.get("ollama_keep_alive", "")
        self._timeout_s = cfg.get("request_timeout_s", 1200.0)
        self._max_thinking_tokens = cfg.get("max_thinking_tokens", 0)
        self._retry_attempts = cfg.get("retry_attempts", 3)
        self._retry_wait_min_s = cfg.get("retry_wait_min_s", 2.0)
        self._retry_wait_max_s = cfg.get("retry_wait_max_s", 30.0)
        self._session_id: str | None = None
        self._history: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return the model's reply to a message list."""
        retrying = Retrying(
            retry=retry_if_exception(_is_transient_claude_error),
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_random_exponential(
                min=self._retry_wait_min_s, max=self._retry_wait_max_s
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        while True:
            try:
                return retrying(self._complete_once, messages)
            except ClaudeCLIError as exc:
                reset_message = _session_limit_reset(exc)
                if reset_message is None:
                    raise
                # Import lazily to keep the low-level completion client independent
                # of sandbox orchestration during module initialization.
                from robocode.utils.rate_limit import wait_for_rate_limit_reset

                wait_for_rate_limit_reset(reset_message)

    def _complete_once(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Make one attempt, retaining session state only for a safe reset retry."""
        # Only resume an exact continuation. Callers may reuse this client for
        # independent samples or edit history, which must start fresh sessions.
        resume = (
            self._session_id is not None
            and len(messages) == len(self._history) + 1
            and messages[:-1] == self._history
            and messages[-1]["role"] == "user"
        )
        session_id = self._session_id if resume else None
        previous_history = self._history
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
            error = ClaudeCLIError(exc.returncode, exc.stdout, exc.stderr)
            if _session_limit_reset(error) is not None:
                self._session_id = session_id
                self._history = previous_history
            raise error from exc
        data = json.loads(result.stdout)
        if data.get("is_error"):
            error = ClaudeCLIError(result.returncode, result.stdout, result.stderr)
            if _session_limit_reset(error) is not None:
                self._session_id = session_id
                self._history = previous_history
            raise error
        response = LLMResponse(text=data["result"], cost_usd=data.get("total_cost_usd"))
        self._session_id = data.get("session_id")
        self._history = [dict(message) for message in messages] + [
            {"role": "assistant", "content": response.text}
        ]
        return response


def _flatten(messages: list[dict[str, str]]) -> str:
    """Serialize initial or replaced history when starting a fresh CLI session."""
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
