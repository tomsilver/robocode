"""Tests for claude_auth.py."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from robocode.utils.claude_auth import (
    host_claude_config_dir,
    sandbox_claude_config,
    throwaway_claude_config,
)


def _fake_host_config(tmp_path: Path) -> Path:
    """A host Claude config dir holding credentials plus the usual session state."""
    config_dir = tmp_path / ".claude"
    (config_dir / "jobs" / "abc123" / "tmp").mkdir(parents=True)
    (config_dir / "projects").mkdir()
    (config_dir / "jobs" / "abc123" / "tmp" / "results.json").write_text(
        json.dumps({"per_episode": [{"object_count": 7}]}), encoding="utf-8"
    )
    (config_dir / "projects" / "transcript.jsonl").write_text("{}", encoding="utf-8")
    (config_dir / "CLAUDE.md").write_text("Operator instructions", encoding="utf-8")
    (config_dir / "history.jsonl").write_text("{}", encoding="utf-8")
    (config_dir / "settings.json").write_text("{}", encoding="utf-8")
    (config_dir / ".credentials.json").write_text('{"token": "x"}', encoding="utf-8")
    return config_dir


def test_config_dir_follows_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE_CONFIG_DIR overrides the default location."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "elsewhere"))
    assert host_claude_config_dir() == tmp_path / "elsewhere"


def test_copy_carries_credentials_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the credentials file is copied; the operator's own state stays behind."""
    config_dir = _fake_host_config(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    with throwaway_claude_config() as copy_dir:
        assert sorted(p.name for p in copy_dir.iterdir()) == [".credentials.json"]
        assert (copy_dir / ".credentials.json").read_text(encoding="utf-8") == (
            '{"token": "x"}'
        )
        copied = copy_dir

    assert not copied.exists()


def test_copy_is_writable_and_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token refresh inside the copy never reaches the host config."""
    config_dir = _fake_host_config(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    with throwaway_claude_config() as copy_dir:
        (copy_dir / ".credentials.json").write_text('{"token": "y"}', encoding="utf-8")
        (copy_dir / "memory.md").write_text("written in the sandbox", encoding="utf-8")

    assert (config_dir / ".credentials.json").read_text(encoding="utf-8") == (
        '{"token": "x"}'
    )
    assert not (config_dir / "memory.md").exists()


def test_refreshed_credentials_reach_the_next_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rotated refresh token survives without modifying host credentials."""
    config_dir = _fake_host_config(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    with throwaway_claude_config() as first:
        (first / ".credentials.json").write_text(
            '{"token": "refreshed"}', encoding="utf-8"
        )

    with throwaway_claude_config() as second:
        assert (second / ".credentials.json").read_text(encoding="utf-8") == (
            '{"token": "refreshed"}'
        )

    assert (config_dir / ".credentials.json").read_text(encoding="utf-8") == (
        '{"token": "x"}'
    )


def test_malformed_broker_is_replaced_by_valid_host_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fake or corrupted newer broker cannot shadow a valid host login."""
    config_dir = _fake_host_config(tmp_path)
    credentials = '{"claudeAiOauth":{"accessToken":"access","refreshToken":"refresh"}}'
    (config_dir / ".credentials.json").write_text(credentials, encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    broker_dir = Path(os.environ["ROBOCODE_CLAUDE_AUTH_DIR"])
    broker_dir.mkdir()
    (broker_dir / ".credentials.json").write_text("not-json", encoding="utf-8")

    with throwaway_claude_config() as copied:
        assert (copied / ".credentials.json").read_text(encoding="utf-8") == credentials


def test_cleanup_handles_credentials_replaced_with_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent cannot crash cleanup by replacing its credential with a directory."""
    config_dir = _fake_host_config(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with sandbox_claude_config(sandbox) as agent_config:
        copied = agent_config / ".credentials.json"
        copied.unlink()
        copied.mkdir()
        (copied / "agent-file").write_text("not credentials", encoding="utf-8")

    assert not copied.exists()
    assert (config_dir / ".credentials.json").read_text(encoding="utf-8") == (
        '{"token": "x"}'
    )


def test_persistence_rejects_credentials_replaced_with_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced credential cannot make the host broker follow an agent symlink."""
    config_dir = _fake_host_config(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.write_text("must not enter broker", encoding="utf-8")

    with sandbox_claude_config(sandbox) as agent_config:
        copied = agent_config / ".credentials.json"
        copied.unlink()
        copied.symlink_to(unrelated)

    assert not copied.exists()
    assert unrelated.read_text(encoding="utf-8") == "must not enter broker"
    with sandbox_claude_config(sandbox) as next_config:
        assert (next_config / ".credentials.json").read_text(encoding="utf-8") == (
            '{"token": "x"}'
        )


def test_missing_credentials_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No credentials to copy is reported here, not as an auth failure in-container."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(empty))

    with pytest.raises(RuntimeError, match="No Claude credentials"):
        with throwaway_claude_config():
            pass


def test_file_credentials_serialize_parallel_cli_lifetimes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copied refresh tokens cannot be used by concurrent CLI processes."""
    config_dir = _fake_host_config(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def use_credentials() -> None:
        nonlocal active, max_active
        with throwaway_claude_config():
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(use_credentials) for _ in range(2)]
        for future in futures:
            future.result(timeout=2)

    assert max_active == 1


def test_file_credentials_parallel_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit sweep override permits concurrent copied-credential lifetimes."""
    config_dir = _fake_host_config(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ROBOCODE_CLAUDE_AUTH_ALLOW_PARALLEL", "1")
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def use_credentials() -> None:
        nonlocal active, max_active
        with throwaway_claude_config():
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(use_credentials) for _ in range(2)]
        for future in futures:
            future.result(timeout=2)

    assert max_active == 2
