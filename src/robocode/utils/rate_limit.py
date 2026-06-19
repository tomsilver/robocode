"""Helpers for running sandboxed agents with rate-limit retry."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import time
from collections.abc import Callable
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

logger = logging.getLogger(__name__)

_DEFAULT_RESET_HOUR = 3  # fallback hour if we can't parse the reset time
# Subscription usage caps reset on a 5-hour window, so a wait longer than this
# means we mis-parsed the reset time. Cap the sleep; the retry loop re-checks.
_MAX_WAIT_SECS = 5.5 * 3600


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


def run_with_rate_limit_retry(
    docker_config: DockerSandboxConfig | None,
    local_config: SandboxConfig | None,
    backend: AgentBackend,
    apptainer_config: ApptainerSandboxConfig | None = None,
) -> SandboxResult:
    """Run the sandbox, retrying on rate-limit by sleeping until reset."""
    while True:
        if apptainer_config is not None:
            result = run_async(
                lambda: run_agent_in_apptainer_sandbox(apptainer_config, backend)
            )
        elif docker_config is not None:
            result = run_async(
                lambda: run_agent_in_docker_sandbox(docker_config, backend)
            )
        else:
            assert local_config is not None
            result = run_async(lambda: run_agent_in_sandbox(local_config, backend))

        if result.rate_limit_reset is None:
            return result

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
        logger.info("Woke up after rate-limit sleep, retrying...")
