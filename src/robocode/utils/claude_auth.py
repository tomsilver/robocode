"""Isolated Claude configuration used by experiment CLI processes."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_CREDENTIALS = ".credentials.json"


def host_claude_config_dir() -> Path:
    """Return the operator's Claude configuration directory."""
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))


def _copy_credentials(destination: Path) -> Path | None:
    """Copy host credentials into *destination*, returning the copied path.

    Token-based authentication does not need a credentials file.  In that case
    there may be nothing to copy, which is intentionally not an error here.
    """
    destination.mkdir(parents=True, exist_ok=True)
    copied = destination / _CREDENTIALS
    # Never follow a file or symlink left by the previous agent process.
    copied.unlink(missing_ok=True)
    credentials = host_claude_config_dir() / _CREDENTIALS
    if not credentials.is_file():
        return None
    shutil.copy2(credentials, copied)
    return copied


@contextmanager
def throwaway_claude_config() -> Iterator[Path]:
    """Yield a writable config containing only a copy of host credentials.

    The operator's transcripts, history, memory, settings, and job artifacts
    never enter the sandbox, and writes or token refreshes cannot reach the
    operator's live Claude directory.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="robocode-claude-"))
    try:
        config_dir = tmp_dir / ".claude"
        config_dir.mkdir()
        copied = _copy_credentials(config_dir)
        if copied is None:
            raise RuntimeError(
                f"No Claude credentials at {host_claude_config_dir() / _CREDENTIALS}. "
                "Run `claude login` on the host, or set "
                "CLAUDE_CODE_OAUTH_TOKEN."
            )
        yield config_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@contextmanager
def sandbox_claude_config(sandbox_dir: Path) -> Iterator[Path]:
    """Yield a resumable sandbox-local config without retaining credentials.

    Session state remains under ``.agent_home`` between rate-limit attempts,
    while the copied credential is removed after every CLI process.  This
    prevents either reads from or writes to the operator's live config and
    avoids leaving an authentication secret in experiment output.
    """
    config_dir = sandbox_dir / ".agent_home"
    config_dir.mkdir(exist_ok=True)
    copied = _copy_credentials(config_dir)
    try:
        yield config_dir
    finally:
        if copied is not None:
            copied.unlink(missing_ok=True)
