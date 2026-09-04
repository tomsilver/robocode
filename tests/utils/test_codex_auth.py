"""Tests for isolated Codex credentials and sessions."""

import json
from pathlib import Path

from robocode.utils.codex_auth import sandbox_codex_home, throwaway_codex_home


def test_throwaway_codex_home_copies_only_auth(tmp_path: Path, monkeypatch) -> None:
    host_home = tmp_path / "host"
    host_home.mkdir()
    (host_home / "auth.json").write_text(json.dumps({"tokens": {}}))
    (host_home / "config.toml").write_text("model = 'private'")
    (host_home / "sessions").mkdir()
    monkeypatch.setenv("CODEX_HOME", str(host_home))

    with throwaway_codex_home() as staged:
        assert {path.name for path in staged.iterdir()} == {"auth.json"}


def test_local_codex_home_retains_sessions_not_auth(
    tmp_path: Path, monkeypatch
) -> None:
    host_home = tmp_path / "host"
    host_home.mkdir()
    (host_home / "auth.json").write_text(json.dumps({"tokens": {}}))
    monkeypatch.setenv("CODEX_HOME", str(host_home))
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with sandbox_codex_home(sandbox) as codex_home:
        assert (codex_home / "auth.json").is_file()
        assert (codex_home / "sessions").is_symlink()

    assert not (codex_home / "auth.json").exists()
    assert (sandbox / ".agent_sessions" / "codex").is_dir()
