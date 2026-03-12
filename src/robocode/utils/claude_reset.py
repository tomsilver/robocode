"""Shared utilities for running async sandbox calls and handling rate limits."""

import asyncio
import concurrent.futures
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from robocode.utils.sandbox import SandboxResult

DEFAULT_RESET_HOUR = 3  # fallback hour if we can't parse the reset time


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
        return DEFAULT_RESET_HOUR
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
