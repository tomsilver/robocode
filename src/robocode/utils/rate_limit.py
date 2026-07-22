"""Helpers for resuming sandboxed agents after retryable interruptions."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from robocode.utils.apptainer_sandbox import (
    ApptainerSandboxConfig,
    run_agent_in_apptainer_sandbox,
)
from robocode.utils.backends import AgentBackend
from robocode.utils.docker_sandbox import (
    DockerSandboxConfig,
    run_agent_in_docker_sandbox,
)
from robocode.utils.sandbox import SandboxConfig, SandboxResult, run_agent_in_sandbox
from robocode.utils.sandbox_types import GenerationMetrics

logger = logging.getLogger(__name__)

_DEFAULT_RESET_HOUR = 3  # fallback hour if we can't parse the reset time
# Subscription usage caps reset on a 5-hour window, so a wait longer than this
# means we mis-parsed the reset time. Cap the sleep; the retry loop re-checks.
_MAX_WAIT_SECS = 5.5 * 3600
# Avoid an infinite resume loop when a model repeatedly produces an oversized
# response, especially for configurations where both cost and turns are unlimited.
_MAX_OUTPUT_TOKEN_RETRIES = 2

_OUTPUT_TOKEN_RESUME_PROMPT = (
    "Continue the task from where the previous response was interrupted because it "
    "exceeded the output-token limit. Be concise: do not repeat prior reasoning or "
    "emit a long explanation. Use tools directly and finish the implementation."
)


def run_async(make_coro: Callable[[], Any]) -> SandboxResult:
    """Run an async sandbox call, handling an already-running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())

    def _run() -> SandboxResult:
        return asyncio.run(make_coro())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


def parse_reset_hour(reset_str: str) -> tuple[int, bool]:
    """Parse a reset message into (hour_24, is_utc).

    Handles bare times like '3am' as well as full messages such as "You've hit your
    session limit . resets 11pm (UTC)". The timezone flag matters because the box may
    not be in UTC.
    """
    text = reset_str.strip().lower()
    match = re.search(r"(\d{1,2})\s*(am|pm)", text)
    if not match:
        return _DEFAULT_RESET_HOUR, False
    hour = int(match.group(1))
    period = match.group(2)
    if period == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = hour if hour == 12 else hour + 12
    return hour, "utc" in text


def seconds_until_reset(reset_hour: int, is_utc: bool = False) -> float:
    """Return seconds until the given reset hour, capped at the window length.

    When ``is_utc`` the hour is interpreted in UTC (and compared against UTC
    now); otherwise it is local time. The result is clamped to _MAX_WAIT_SECS
    so a misparsed time cannot cause a multi-hour oversleep.
    """
    now = datetime.now(timezone.utc) if is_utc else datetime.now().astimezone()
    target = now.replace(hour=reset_hour, minute=5, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    wait = (target - now).total_seconds()
    return min(wait, _MAX_WAIT_SECS)


def _fold_retry_metrics(
    result: SandboxResult,
    aborted_tokens: int,
    aborted_cost: float,
    rate_limit_retries: int,
    output_token_retries: int = 0,
) -> SandboxResult:
    """Record discarded retryable attempts' counts and spend on *result*."""
    if (
        rate_limit_retries == 0
        and output_token_retries == 0
        and aborted_tokens == 0
        and aborted_cost == 0.0
    ):
        return result
    base = result.generation_metrics or GenerationMetrics()
    metrics = replace(
        base,
        rate_limit_retries=rate_limit_retries,
        output_token_retries=output_token_retries,
        aborted_tokens=aborted_tokens,
        aborted_cost_usd=aborted_cost,
    )
    return replace(result, generation_metrics=metrics)


def _run_active_sandbox(config: SandboxConfig, backend: AgentBackend) -> SandboxResult:
    """Dispatch to the sandbox runner matching the config's backend type."""
    if isinstance(config, ApptainerSandboxConfig):
        return run_async(lambda: run_agent_in_apptainer_sandbox(config, backend))
    if isinstance(config, DockerSandboxConfig):
        return run_async(lambda: run_agent_in_docker_sandbox(config, backend))
    return run_async(lambda: run_agent_in_sandbox(config, backend))


def run_with_rate_limit_retry(
    docker_config: DockerSandboxConfig | None,
    local_config: SandboxConfig | None,
    backend: AgentBackend,
    apptainer_config: ApptainerSandboxConfig | None = None,
) -> SandboxResult:
    """Run the sandbox, resuming after retryable Claude interruptions.

    Each retry resumes the interrupted conversation (--continue) with the
    budget the run had left, so an agent that hit the usage cap continues its
    work rather than restarting cold without receiving a fresh experimental
    budget. Rehydrating the transcript may itself consume uncached input tokens,
    so a resumed run is recorded explicitly and is not claimed to be identical
    to an uninterrupted run.
    """
    active: SandboxConfig | None = apptainer_config or docker_config or local_config
    assert active is not None
    original_budget = active.max_budget_usd
    original_turns = active.max_turns

    aborted_tokens = 0
    aborted_turns = 0
    aborted_cost = 0.0
    rate_limit_retries = 0
    output_token_retries = 0
    while True:
        result = _run_active_sandbox(active, backend)

        rate_limited = result.rate_limit_reset is not None
        output_token_limited = result.output_token_limit_hit
        if not rate_limited and not output_token_limited:
            return _fold_retry_metrics(
                result,
                aborted_tokens,
                aborted_cost,
                rate_limit_retries,
                output_token_retries,
            )

        if rate_limited:
            rate_limit_retries += 1
        assert result.generation_metrics is not None
        aborted_tokens += result.generation_metrics.total_tokens
        aborted_turns += result.generation_metrics.num_turns
        aborted_cost += result.total_cost_usd or 0.0

        # Both CLI limits use zero to mean unlimited.  Never turn an exhausted
        # positive limit into zero on a retry, which would silently grant an
        # unbounded second run.  Likewise, without a reported cost we cannot
        # carry a positive dollar budget forward without changing the
        # experimental condition.
        missing_cost = original_budget > 0 and result.total_cost_usd is None
        budget_exhausted = 0 < original_budget <= aborted_cost
        turns_exhausted = 0 < original_turns <= aborted_turns
        if missing_cost or budget_exhausted or turns_exhausted:
            reason = (
                "the interrupted attempt did not report its cost"
                if missing_cost
                else (
                    "the dollar budget is exhausted"
                    if budget_exhausted
                    else "the turn budget is exhausted"
                )
            )
            logger.warning("Not resuming interrupted run because %s.", reason)
            return _fold_retry_metrics(
                result,
                aborted_tokens,
                aborted_cost,
                rate_limit_retries,
                output_token_retries,
            )

        if output_token_limited and output_token_retries >= _MAX_OUTPUT_TOKEN_RETRIES:
            logger.warning(
                "Not resuming after output-token overflow: retry limit (%d) reached.",
                _MAX_OUTPUT_TOKEN_RETRIES,
            )
            return _fold_retry_metrics(
                result,
                aborted_tokens,
                aborted_cost,
                rate_limit_retries,
                output_token_retries,
            )

        if rate_limited:
            assert result.rate_limit_reset is not None
            reset_hour, is_utc = parse_reset_hour(result.rate_limit_reset)
            wait_secs = seconds_until_reset(reset_hour, is_utc)
            hours = wait_secs / 3600
            logger.warning(
                "Rate-limited (%s). Sleeping %.1f hours until %d:05 %s ...",
                result.error,
                hours,
                reset_hour,
                "UTC" if is_utc else "local",
            )
            time.sleep(wait_secs)
            logger.info(
                "Woke up after rate-limit sleep, resuming with remaining budget..."
            )
            resume_prompt = active.prompt
        else:
            output_token_retries += 1
            logger.warning(
                "Claude exceeded its output-token maximum; resuming with remaining "
                "budget (attempt %d/%d).",
                output_token_retries,
                _MAX_OUTPUT_TOKEN_RETRIES,
            )
            resume_prompt = _OUTPUT_TOKEN_RESUME_PROMPT

        # Zero means unlimited for both fields; exhausted positive limits were
        # handled above, so every finite remainder here is strictly positive.
        remaining_turns = original_turns - aborted_turns if original_turns else 0
        active = replace(
            active,
            resume_previous_session=True,
            prompt=resume_prompt,
            max_budget_usd=original_budget - aborted_cost if original_budget else 0.0,
            max_turns=remaining_turns,
        )
