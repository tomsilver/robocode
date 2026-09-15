"""Tests for the live Internet-access probe's offline transcript analysis."""

from pathlib import Path

from integration_tests.check_agent_internet_access import _stream_used_web_search


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
