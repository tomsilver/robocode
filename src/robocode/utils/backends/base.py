"""Agent backend protocol for running coding agents in sandboxes.

Defines the interface that all agent backends (Claude Code CLI, OpenCode, etc.) must
implement. Each backend handles CLI invocation, environment setup, sandbox
configuration, and output stream parsing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol

from robocode.mcp import MCP_HTTP_PORT
from robocode.utils.sandbox_types import SandboxConfig, _StreamParseResult


class AgentBackend(Protocol):
    """Protocol for agent backends that can run in a sandbox."""

    @property
    def name(self) -> str:
        """Short identifier, e.g. ``"claude"`` or ``"opencode"``."""

    def build_cli_cmd(
        self,
        config: SandboxConfig,
        *,
        mcp_python_cmd: str = "",
        mcp_env_config_path: str = "",
        mcp_config_cli_path: str | None = None,
        mcp_log_file_path: str = "",
        mcp_transport: str = "stdio",
        mcp_port: int = MCP_HTTP_PORT,
    ) -> list[str]:
        """Return the full CLI command (binary + args) to run the agent."""

    def build_env(
        self,
        config: SandboxConfig,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Return the subprocess environment dict."""

    def setup_sandbox_files(
        self,
        config: SandboxConfig,
        *,
        docker_python: str = "",
        primitive_names: tuple[str, ...] = (),
    ) -> None:
        """Write backend-specific config files into the sandbox dir.

        Called after the shared sandbox setup (git init, init_files copy)
        but before the agent process is launched.

        For Claude: .claude/settings.json (hooks), CLAUDE.md
        For OpenCode: opencode.json (permissions, MCP, tools), AGENTS.md
        """

    def parse_stream(
        self,
        proc: subprocess.Popen[str],
        stream_log_path: Path | None = None,
    ) -> _StreamParseResult:
        """Parse the CLI's streaming output and return a unified result."""
