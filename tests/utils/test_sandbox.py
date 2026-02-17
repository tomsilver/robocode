"""Tests for sandbox.py."""

from pathlib import Path

from robocode.utils.sandbox import (
    SandboxConfig,
    SandboxResult,
    _is_path_within_sandbox,
)


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
