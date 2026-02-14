"""Sandboxed Claude agent runner."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (  # type: ignore[import-untyped]
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
)

_FILE_TOOLS: set[str] = {"Read", "Write", "Edit", "Glob", "Grep"}

_PATH_KEYS: dict[str, str] = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Glob": "path",
    "Grep": "path",
}


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for a sandboxed Claude agent run."""

    sandbox_dir: Path
    init_files: dict[str, Path] = field(default_factory=dict)
    prompt: str = ""
    output_filename: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 50
    system_prompt: str = ""


@dataclass(frozen=True)
class SandboxResult:
    """Result from a sandboxed Claude agent run."""

    success: bool
    output_file: Path | None
    error: str | None


def _is_path_within_sandbox(path_str: str, sandbox_dir: Path) -> bool:
    """Check whether a resolved path is within the sandbox directory."""
    try:
        resolved = Path(path_str).resolve()
        sandbox_resolved = sandbox_dir.resolve()
        return resolved == sandbox_resolved or sandbox_resolved in resolved.parents
    except (OSError, ValueError):
        return False


def _deny(reason: str) -> dict[str, Any]:
    """Return a hook response that denies tool use."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _make_pre_tool_use_hook(
    sandbox_dir: Path,
) -> Any:
    """Create a PreToolUse hook restricting operations to sandbox_dir."""

    async def _hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,  # pylint: disable=unused-argument
        context: Any,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        tool_name: str = input_data.get("tool_name", "")
        tool_input: dict[str, Any] = input_data.get("tool_input", {})

        if tool_name in _FILE_TOOLS:
            path_key = _PATH_KEYS.get(tool_name)
            if path_key and path_key in tool_input:
                path_str = tool_input[path_key]
                if not Path(path_str).is_absolute():
                    path_str = str(sandbox_dir / path_str)
                if not _is_path_within_sandbox(path_str, sandbox_dir):
                    return _deny(f"Path {path_str} is outside the sandbox")
            return {}

        return {}

    return _hook


async def run_agent_in_sandbox(config: SandboxConfig) -> SandboxResult:
    """Run a Claude agent in a sandboxed working directory.

    Initializes the sandbox with files, gives the agent a prompt, and retrieves the
    specified output file.
    """
    config.sandbox_dir.mkdir(parents=True, exist_ok=True)

    for dest_name, source_path in config.init_files.items():
        dest = config.sandbox_dir / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)

    hook_fn = _make_pre_tool_use_hook(config.sandbox_dir)

    options = ClaudeAgentOptions(
        cwd=str(config.sandbox_dir),
        model=config.model,
        sandbox={"enabled": True, "autoAllowBashIfSandboxed": True},
        hooks={"PreToolUse": [HookMatcher(hooks=[hook_fn])]},
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        max_turns=config.max_turns,
        system_prompt=config.system_prompt,
        setting_sources=[],
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(config.prompt)
        async for message in client.receive_response():
            if isinstance(message, (AssistantMessage, ResultMessage)):
                pass

    output_path = config.sandbox_dir / config.output_filename
    if output_path.exists():
        return SandboxResult(success=True, output_file=output_path, error=None)
    return SandboxResult(
        success=False,
        output_file=None,
        error=f"Output file not found: {config.output_filename}",
    )
