"""Helpers for running sandboxed agents with rate-limit retry."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from robocode.utils.backends import AgentBackend
from robocode.utils.docker_sandbox import (
    DockerSandboxConfig,
    run_agent_in_docker_sandbox,
)
from robocode.utils.sandbox import SandboxConfig, SandboxResult, run_agent_in_sandbox

logger = logging.getLogger(__name__)

_DEFAULT_RESET_HOUR = 3  # fallback hour if we can't parse the reset time


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


def parse_reset_hour(reset_str: str) -> int:
    """Parse a reset time like '3am' or '11pm' into a 24-hour int."""
    reset_str = reset_str.strip().lower()
    match = re.match(r"(\d{1,2})(am|pm)", reset_str)
    if not match:
        return _DEFAULT_RESET_HOUR
    hour = int(match.group(1))
    period = match.group(2)
    if period == "am":
        return 0 if hour == 12 else hour
    return hour if hour == 12 else hour + 12


def seconds_until_reset(reset_hour: int) -> float:
    """Return seconds until the given hour (local time), plus a small buffer."""
    now = datetime.now()
    target = now.replace(hour=reset_hour, minute=5, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def run_with_rate_limit_retry(
    docker_config: DockerSandboxConfig | None,
    local_config: SandboxConfig | None,
    backend: AgentBackend,
) -> SandboxResult:
    """Run the sandbox, retrying on rate-limit by sleeping until reset."""
    while True:
        if docker_config is not None:
            result = run_async(
                lambda: run_agent_in_docker_sandbox(docker_config, backend)
            )
        else:
            assert local_config is not None
            result = run_async(lambda: run_agent_in_sandbox(local_config, backend))

        if result.rate_limit_reset is None:
            return result

        reset_hour = parse_reset_hour(result.rate_limit_reset)
        wait_secs = seconds_until_reset(reset_hour)
        hours = wait_secs / 3600
        logger.warning(
            "Rate-limited (%s). Sleeping %.1f hours until %d:05 ...",
            result.error,
            hours,
            reset_hour,
        )
        time.sleep(wait_secs)
        logger.info("Woke up after rate-limit sleep, retrying...")
