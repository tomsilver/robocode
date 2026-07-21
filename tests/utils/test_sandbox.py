"""Tests for sandbox.py."""

from pathlib import Path

from robocode.utils.backends.claude import _RATE_LIMIT_RE
from robocode.utils.claude_auth import sandbox_claude_config
from robocode.utils.sandbox import (
    SandboxConfig,
    SandboxResult,
    _is_path_within_sandbox,
    _stream_result_to_sandbox_result,
)
from robocode.utils.sandbox_types import GenerationMetrics, _StreamParseResult


def test_generation_metrics_to_dict_flat_keys() -> None:
    """to_dict() emits flat gen_* keys and sums total_tokens."""
    m = GenerationMetrics(
        wall_time_s=12.5,
        num_turns=7,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=2000,
        cache_creation_tokens=10,
        num_tool_calls=9,
        rate_limit_retries=2,
        aborted_tokens=300,
        aborted_cost_usd=1.5,
    )
    d = m.to_dict()
    assert d["gen_wall_time_s"] == 12.5
    assert d["gen_num_turns"] == 7
    assert d["gen_num_tool_calls"] == 9
    assert d["gen_total_tokens"] == 2160
    assert d["gen_rate_limit_retries"] == 2
    assert d["gen_aborted_tokens"] == 300
    assert d["gen_aborted_cost_usd"] == 1.5
    assert all(k.startswith("gen_") for k in d)


def test_stream_result_carries_generation_metrics(tmp_path: Path) -> None:
    """The stream->SandboxResult conversion attaches metrics and wall time."""
    (tmp_path / "approach.py").write_text("x = 1\n")
    stream = _StreamParseResult(
        is_error=False,
        error_text=None,
        num_turns=4,
        total_cost=0.42,
        input_tokens=100,
        output_tokens=50,
        num_tool_calls=8,
        num_autocompactions=2,
        cli_duration_ms=1234,
        stop_reason="end_turn",
    )
    result = _stream_result_to_sandbox_result(
        stream, tmp_path, "approach.py", wall_time_s=9.0
    )
    assert result.success
    assert result.generation_metrics is not None
    assert result.generation_metrics.wall_time_s == 9.0
    assert result.generation_metrics.num_turns == 4
    assert result.generation_metrics.num_autocompactions == 2
    assert result.generation_metrics.stop_reason == "end_turn"


def test_sandbox_claude_config_copies_then_removes_creds(
    tmp_path: Path, monkeypatch
) -> None:
    """Host credentials cannot be modified and do not remain in run output."""
    host = tmp_path / "host_claude"
    host.mkdir()
    (host / ".credentials.json").write_text("{}")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(host))
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with sandbox_claude_config(sandbox) as agent_home:
        copied = agent_home / ".credentials.json"
        assert copied.read_text() == "{}"
        assert not copied.is_symlink()
        copied.write_text('{"refreshed": true}')
        (agent_home / "projects").mkdir()

    assert (host / ".credentials.json").read_text() == "{}"
    assert not copied.exists()
    assert (agent_home / "projects").is_dir()


def test_redirect_claude_config_without_creds_file(tmp_path: Path, monkeypatch) -> None:
    """No symlink when the host has no credentials file (token-env auth)."""
    host = tmp_path / "host_claude"
    host.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(host))
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with sandbox_claude_config(sandbox) as agent_home:
        assert agent_home.is_dir()
        assert not (agent_home / ".credentials.json").exists()


def test_path_within_sandbox(tmp_path: Path) -> None:
    """Paths inside sandbox are accepted."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    assert _is_path_within_sandbox(str(sandbox / "file.py"), sandbox)
    assert _is_path_within_sandbox(str(sandbox / "sub" / "file.py"), sandbox)
    assert _is_path_within_sandbox(str(sandbox), sandbox)


def test_path_outside_sandbox(tmp_path: Path) -> None:
    """Paths outside sandbox are rejected."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    assert not _is_path_within_sandbox(str(tmp_path / "outside.py"), sandbox)
    assert not _is_path_within_sandbox("/etc/passwd", sandbox)
    assert not _is_path_within_sandbox("/", sandbox)


def test_path_traversal_blocked(tmp_path: Path) -> None:
    """Directory traversal attacks are caught."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    assert not _is_path_within_sandbox(str(sandbox / ".." / "outside.py"), sandbox)
    assert not _is_path_within_sandbox(
        str(sandbox / "sub" / ".." / ".." / "outside.py"), sandbox
    )


def test_sandbox_config_defaults() -> None:
    """SandboxConfig has expected defaults."""
    config = SandboxConfig(sandbox_dir=Path("/tmp/test"))
    assert config.sandbox_dir == Path("/tmp/test")
    assert not config.init_files
    assert config.prompt == ""
    assert config.output_filename == ""
    assert config.model == "sonnet"
    assert config.max_budget_usd == 5.0
    assert config.system_prompt == ""


def test_sandbox_result_success() -> None:
    """SandboxResult can represent success."""
    result = SandboxResult(success=True, output_file=Path("/tmp/out.txt"), error=None)
    assert result.success
    assert result.output_file == Path("/tmp/out.txt")
    assert result.error is None


def test_sandbox_result_failure() -> None:
    """SandboxResult can represent failure."""
    result = SandboxResult(success=False, output_file=None, error="file not found")
    assert not result.success
    assert result.output_file is None
    assert result.error == "file not found"


class TestRateLimitRegex:
    """Tests for _RATE_LIMIT_RE matching rate-limit messages."""

    def test_old_format(self) -> None:
        """Matches the old 'out of extra usage' message."""
        msg = "You are out of extra usage. Your limit resets 3am"
        m = _RATE_LIMIT_RE.search(msg)
        assert m is not None
        assert m.group(1) == "3am"

    def test_new_format(self) -> None:
        """Matches the new 'hit your limit' message."""
        msg = "You've hit your limit \u00b7 resets 2pm (Etc/Unknown)"
        m = _RATE_LIMIT_RE.search(msg)
        assert m is not None
        assert m.group(1) == "2pm"

    def test_new_format_12hr(self) -> None:
        """Matches 12-hour times like 11pm."""
        msg = "You've hit your limit \u00b7 resets 11pm (Etc/Unknown)"
        m = _RATE_LIMIT_RE.search(msg)
        assert m is not None
        assert m.group(1) == "11pm"

    def test_no_match(self) -> None:
        """Does not match unrelated messages."""
        assert _RATE_LIMIT_RE.search("Everything is fine") is None
