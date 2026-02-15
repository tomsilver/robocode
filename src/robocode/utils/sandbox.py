"""Sandboxed Claude agent runner.

WARNING: The current sandbox is NOT fully secure. The OS-level sandbox
(macOS Seatbelt / Linux bubblewrap) restricts filesystem *writes* to the
sandbox directory, but allows *reads* of the entire filesystem. This means
bash commands like `cat /etc/passwd` or `python -c "open('/etc/hosts').read()"`
will succeed. The PreToolUse hook restricts Write/Edit to the sandbox
directory; Read/Glob/Grep are unrestricted (matching the OS-level policy).

TODO: Transition to Docker-based sandboxing for full filesystem isolation.
TODO: Consider using the Anthropic API directly instead of the Claude Agent
SDK, which would give us full control over tool execution.
"""

import logging
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
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)

_WRITE_TOOLS: set[str] = {"Write", "Edit"}

_PATH_KEYS: dict[str, str] = {
    "Write": "file_path",
    "Edit": "file_path",
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


def _deny(tool_name: str, reason: str) -> dict[str, Any]:
    """Return a hook response that denies tool use."""
    logger.warning("SANDBOX DENIED %s: %s", tool_name, reason)
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

        if tool_name == "Bash":
            if tool_input.get("dangerouslyDisableSandbox"):
                return _deny(tool_name, "dangerouslyDisableSandbox is not allowed")
            return {}

        if tool_name in _WRITE_TOOLS:
            path_key = _PATH_KEYS.get(tool_name)
            if path_key and path_key in tool_input:
                path_str = tool_input[path_key]
                if not Path(path_str).is_absolute():
                    path_str = str(sandbox_dir / path_str)
                if not _is_path_within_sandbox(path_str, sandbox_dir):
                    return _deny(tool_name, f"Path {path_str} is outside the sandbox")
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

    result_message: ResultMessage | None = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(config.prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        logger.info("Agent: %s", block.text)
                    elif isinstance(block, ToolUseBlock):
                        logger.info("Tool call: %s(%s)", block.name, block.input)
                    elif isinstance(block, ToolResultBlock):
                        content = block.content
                        if isinstance(content, str) and len(content) > 500:
                            content = content[:500] + "..."
                        logger.info(
                            "Tool result (error=%s): %s",
                            block.is_error,
                            content,
                        )
            elif isinstance(message, ResultMessage):
                result_message = message

    if result_message is not None:
        logger.info(
            "Session done: turns=%d, cost=$%s, error=%s",
            result_message.num_turns,
            result_message.total_cost_usd,
            result_message.is_error,
        )
        if result_message.is_error:
            return SandboxResult(
                success=False,
                output_file=None,
                error=result_message.result,
            )

    output_path = config.sandbox_dir / config.output_filename
    if output_path.exists():
        return SandboxResult(success=True, output_file=output_path, error=None)
    return SandboxResult(
        success=False,
        output_file=None,
        error=f"Output file not found: {config.output_filename}",
    )
