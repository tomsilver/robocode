"""Tests for the live Internet-access probe's offline transcript analysis."""

import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

from integration_tests import check_agent_internet_access as probe
from integration_tests.check_agent_internet_access import (
    _stream_used_mcp,
    _stream_used_web_search,
)


def test_stream_used_web_search_tolerates_non_json_lines(tmp_path: Path) -> None:
    """Blank lines and CLI noise do not prevent detecting a later event."""
    stream = tmp_path / "stream.jsonl"
    stream.write_text(
        '\nnot json\n{"type":"item.completed","item":{"type":"web_search"}}\n',
        encoding="utf-8",
    )
    assert _stream_used_web_search(stream)


def test_stream_used_web_search_handles_absent_event(tmp_path: Path) -> None:
    """A missing transcript and an event-free transcript both return false."""
    stream = tmp_path / "stream.jsonl"
    assert not _stream_used_web_search(stream)
    stream.write_text('\n{"type":"command_execution"}\nnoise\n', encoding="utf-8")
    assert not _stream_used_web_search(stream)


def test_stream_used_mcp_handles_codex_and_claude_events(tmp_path: Path) -> None:
    """Recognize MCP calls in the JSON formats emitted by both supported CLIs."""
    stream = tmp_path / "stream.jsonl"
    stream.write_text('{"item":{"type":"mcp_tool_call"}}\n', encoding="utf-8")
    assert _stream_used_mcp(stream)
    stream.write_text(
        '{"content":[{"type":"tool_use","name":"mcp__robocode__render_state"}]}\n',
        encoding="utf-8",
    )
    assert _stream_used_mcp(stream)


def test_web_search_mention_is_not_an_invocation(tmp_path: Path) -> None:
    """An explicit red-team request or refusal must not count as a tool call."""
    stream = tmp_path / "stream.jsonl"
    stream.write_text(
        '{"item":{"type":"agent_message","text":"web_search is disabled"}}\n',
        encoding="utf-8",
    )
    assert not _stream_used_web_search(stream)


def test_claude_web_fetch_is_an_invocation(tmp_path: Path) -> None:
    """Claude web fetch events must fail the same policy as Codex searches."""
    stream = tmp_path / "stream.jsonl"
    stream.write_text(
        '{"content":[{"type":"tool_use","name":"WebFetch"}]}\n', encoding="utf-8"
    )
    assert _stream_used_web_search(stream)


def test_blocked_self_report_is_inconclusive(tmp_path: Path, monkeypatch) -> None:
    """A model's BLOCKED claim cannot certify an operating-system boundary."""

    async def fake_run(config, _backend):
        config.sandbox_dir.mkdir()
        (config.sandbox_dir / "site_text.txt").write_text("BLOCKED", encoding="utf-8")
        return SimpleNamespace(success=True, error=None)

    monkeypatch.setattr(probe, "run_agent_in_apptainer_sandbox", fake_run)
    args = argparse.Namespace(
        results_dir=tmp_path,
        container="apptainer",
        backend="codex",
        model=None,
        max_budget_usd=1.0,
        strict_sif_path=None,
    )
    assert asyncio.run(probe._run(args)) == 2  # pylint: disable=protected-access
