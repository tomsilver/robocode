"""Tests for sandbox.py."""

import asyncio
from pathlib import Path
from typing import Any

from robocode.utils.sandbox import (
    SandboxConfig,
    SandboxResult,
    _is_path_within_sandbox,
    _make_pre_tool_use_hook,
)


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_hook_input(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Create a minimal PreToolUse hook input dict."""
    return {"tool_name": tool_name, "tool_input": tool_input}


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


def test_hook_allows_read_inside_sandbox(tmp_path: Path) -> None:
    """Read tool targeting sandbox paths is allowed."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input("Read", {"file_path": str(sandbox / "file.py")}),
            None,
            {},
        )
    )
    assert result == {}


def test_hook_allows_write_inside_sandbox(tmp_path: Path) -> None:
    """Write tool targeting sandbox paths is allowed."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input(
                "Write",
                {"file_path": str(sandbox / "out.txt"), "content": "hello"},
            ),
            None,
            {},
        )
    )
    assert result == {}


def test_hook_blocks_read_outside_sandbox(tmp_path: Path) -> None:
    """Read tool targeting paths outside sandbox is denied."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input("Read", {"file_path": "/etc/passwd"}),
            None,
            {},
        )
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_blocks_write_outside_sandbox(tmp_path: Path) -> None:
    """Write tool targeting paths outside sandbox is denied."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input("Write", {"file_path": "/tmp/evil.txt"}),
            None,
            {},
        )
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_blocks_edit_outside_sandbox(tmp_path: Path) -> None:
    """Edit tool targeting paths outside sandbox is denied."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input(
                "Edit",
                {
                    "file_path": str(tmp_path / "other.py"),
                    "old_string": "a",
                    "new_string": "b",
                },
            ),
            None,
            {},
        )
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_allows_relative_path_inside_sandbox(tmp_path: Path) -> None:
    """Relative paths are resolved against sandbox_dir."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input("Read", {"file_path": "subdir/file.py"}),
            None,
            {},
        )
    )
    assert result == {}


def test_hook_blocks_glob_outside_sandbox(tmp_path: Path) -> None:
    """Glob tool targeting paths outside sandbox is denied."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input("Glob", {"path": "/etc", "pattern": "*.conf"}),
            None,
            {},
        )
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_allows_glob_without_path(tmp_path: Path) -> None:
    """Glob tool without explicit path is allowed (uses cwd)."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input("Glob", {"pattern": "*.py"}),
            None,
            {},
        )
    )
    assert result == {}


def test_hook_passes_through_bash(tmp_path: Path) -> None:
    """Bash commands are not filtered (OS sandbox handles restrictions)."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input("Bash", {"command": "python script.py"}),
            None,
            {},
        )
    )
    assert result == {}


def test_hook_blocks_dangerously_disable_sandbox(tmp_path: Path) -> None:
    """Bash with dangerouslyDisableSandbox is denied."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input(
                "Bash",
                {
                    "command": "cat /etc/passwd",
                    "dangerouslyDisableSandbox": True,
                },
            ),
            None,
            {},
        )
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_allows_unknown_tool(tmp_path: Path) -> None:
    """Unknown tools are allowed (not restricted)."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    hook = _make_pre_tool_use_hook(sandbox)
    result = _run(
        hook(
            _make_hook_input("SomeOtherTool", {"arg": "value"}),
            None,
            {},
        )
    )
    assert result == {}


def test_sandbox_config_defaults() -> None:
    """SandboxConfig has expected defaults."""
    config = SandboxConfig(sandbox_dir=Path("/tmp/test"))
    assert config.sandbox_dir == Path("/tmp/test")
    assert not config.init_files
    assert config.prompt == ""
    assert config.output_filename == ""
    assert config.max_turns == 50


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
