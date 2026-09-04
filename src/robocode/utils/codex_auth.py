"""Isolated Codex authentication used by experiment CLI processes."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def host_codex_home() -> Path:
    """Return the operator's Codex home."""
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()


def sandbox_codex_sessions(sandbox_dir: Path) -> Path:
    """Return the persistent session directory for one experiment sandbox."""
    sessions = sandbox_dir / ".agent_sessions" / "codex"
    sessions.mkdir(parents=True, exist_ok=True)
    return sessions


@contextmanager
def throwaway_codex_home() -> Iterator[Path]:
    """Yield a writable Codex home containing only copied authentication."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="robocode-codex-"))
    try:
        auth = host_codex_home() / "auth.json"
        if not auth.is_file():
            raise RuntimeError(
                f"No Codex credentials at {auth}. Run `codex login` on the host, "
                "or set CODEX_API_KEY."
            )
        copied = tmp_dir / "auth.json"
        shutil.copy2(auth, copied)
        copied.chmod(0o600)
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@contextmanager
def sandbox_codex_home(sandbox_dir: Path) -> Iterator[Path]:
    """Yield a resumable local Codex home without retaining credentials."""
    codex_home = sandbox_dir / ".agent_home" / "codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    sessions = codex_home / "sessions"
    if not sessions.exists():
        sessions.symlink_to(
            sandbox_codex_sessions(sandbox_dir), target_is_directory=True
        )
    if os.environ.get("CODEX_API_KEY"):
        yield codex_home
        return
    auth = codex_home / "auth.json"
    source = host_codex_home() / "auth.json"
    if not source.is_file():
        raise RuntimeError(
            f"No Codex credentials at {source}. Run `codex login` on the host, "
            "or set CODEX_API_KEY."
        )
    shutil.copy2(source, auth)
    auth.chmod(0o600)
    try:
        yield codex_home
    finally:
        auth.unlink(missing_ok=True)
