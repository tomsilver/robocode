"""Claude Code CLI agent backend.

Extracts all Claude-specific logic (CLI arg building, stream parsing, sandbox file
setup) from the shared sandbox module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

from omegaconf import DictConfig

from robocode.mcp import MCP_SERVER_NAME, mcp_tool_cli_names, setup_mcp_config
from robocode.utils.sandbox_types import SandboxConfig, _StreamParseResult

logger = logging.getLogger(__name__)

_RATE_LIMIT_RE = re.compile(
    r"(?:out of extra usage|hit your limit).*resets\s+(\d{1,2}(?:am|pm))",
    re.IGNORECASE,
)

_VALIDATE_SANDBOX_SCRIPT = """\
#!/usr/bin/env python3
import json
import os
import sys

data = json.load(sys.stdin)
tool_name = data.get("tool_name", "")
tool_input = data.get("tool_input", {})

if tool_name not in ("Write", "Edit"):
    sys.exit(0)

file_path = tool_input.get("file_path", "")
if not file_path:
    sys.exit(0)

sandbox = os.path.realpath(os.getcwd())
resolved = os.path.realpath(file_path)

if resolved == sandbox or resolved.startswith(sandbox + os.sep):
    sys.exit(0)

json.dump({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            f"Blocked: {file_path} resolves outside the sandbox directory"
        ),
    }
}, sys.stdout)
"""

_SANDBOX_SETTINGS: dict = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "Write|Edit",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python3 .claude/validate_sandbox.py",
                    }
                ],
            }
        ],
    }
}


class ClaudeBackend:
    """Claude Code CLI agent backend.

    When ``base_url`` is set in the backend config, ``ANTHROPIC_BASE_URL``
    and ``ANTHROPIC_AUTH_TOKEN`` are injected into the process environment,
    allowing Claude Code to talk to Ollama, liteLLM, vLLM, or any other
    Anthropic-API-compatible endpoint.
    """

    def __init__(self, backend_cfg: DictConfig) -> None:
        self._base_url = backend_cfg.get("base_url", "")
        self._auth_token = backend_cfg.get("auth_token", "ollama")
        self._max_turns: int = 0

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "claude"

    def build_cli_cmd(
        self,
        config: SandboxConfig,
        *,
        mcp_python_cmd: str = "",
        mcp_env_config_path: str = "",
        mcp_config_cli_path: str | None = None,
        mcp_log_file_path: str = "",
    ) -> list[str]:
        """Build the Claude CLI command."""
        self._max_turns = config.max_turns
        claude_cmd = _get_claude_cmd()
        tools = "Bash,Read,Write,Edit,Glob,Grep,Task"
        if config.mcp_tools:
            tools += "," + ",".join(mcp_tool_cli_names(config.mcp_tools))
        logger.info("Enabled tools: %s", tools)
        args = [
            claude_cmd,
            "-p",
            config.prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            config.model,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--tools",
            tools,
            "--setting-sources",
            "project",
        ]
        if config.system_prompt:
            args += ["--system-prompt", config.system_prompt]
        if config.max_budget_usd > 0:
            args += ["--max-budget-usd", str(config.max_budget_usd)]
        if config.mcp_tools:
            log_path = mcp_log_file_path or str(
                (config.sandbox_dir / ".mcp" / "mcp_server.log").resolve()
            )
            config_path = setup_mcp_config(
                config.sandbox_dir,
                config.mcp_tools,
                mcp_python_cmd,
                mcp_env_config_path,
                log_path,
            )
            cli_path = mcp_config_cli_path or str(config_path.resolve())
            args += ["--mcp-config", cli_path, "--strict-mcp-config"]
        return args

    def build_env(
        self,
        config: SandboxConfig,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build a clean environment dict, stripping ``CLAUDECODE*`` vars."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(config.max_output_tokens)
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(config.autocompact_pct)
        if self._base_url:
            env["ANTHROPIC_BASE_URL"] = self._base_url
            env["ANTHROPIC_AUTH_TOKEN"] = self._auth_token
        if extra:
            env.update(extra)
        return env

    def setup_sandbox_files(
        self,
        config: SandboxConfig,
        *,
        docker_python: str = "",
        primitive_names: tuple[str, ...] = (),
    ) -> None:
        """Write ``.claude/settings.json``, validation hook, and ``CLAUDE.md``."""
        claude_dir = config.sandbox_dir / ".claude"
        claude_dir.mkdir(exist_ok=True)
        (claude_dir / "settings.json").write_text(
            json.dumps(_SANDBOX_SETTINGS, indent=2) + "\n"
        )
        (claude_dir / "validate_sandbox.py").write_text(_VALIDATE_SANDBOX_SCRIPT)

        claude_md = config.sandbox_dir / "CLAUDE.md"
        if not claude_md.exists():
            if docker_python:
                text = (
                    "All files you create MUST use relative paths so they "
                    "stay in the current working directory (/sandbox). Never "
                    "write files using absolute paths.\n\n"
                    "CONTEXT MANAGEMENT: your context window is limited and "
                    "you MUST protect it aggressively: "
                    "(1) Delegate ALL source code reading, exploration, and "
                    "deep reasoning to Task subagents: have them return only "
                    "concise summaries and actionable suggestions. "
                    "(2) Never read large files directly; spawn a subagent to "
                    "read and summarize them. "
                    "(3) When running Bash commands, pipe output through "
                    "`head` or `tail` to limit verbosity if possible. "
                    "(4) Keep your thinking brief, do not write long reasoning "
                    "traces. If you need to reason deeply about a design "
                    "decision, delegate that to a subagent and have it return "
                    "the conclusion. "
                    "(5) Keep your main conversation focused on writing and "
                    "testing code, not on reading source or reasoning at "
                    "length."
                    f"The Python interpreter is at {docker_python}\n"
                    "Run test scripts with:\n"
                    f"    {docker_python} test_approach.py\n"
                )
                if primitive_names:
                    text += (
                        "\nPrimitive source files (for reference) are in "
                        "./primitives/\n"
                    )
            else:
                text = (
                    "All files you create MUST use relative paths so they "
                    "stay in the current working directory. Never write files "
                    "using absolute paths.\n"
                )
            claude_md.write_text(text)

    def parse_stream(
        self,
        proc: subprocess.Popen[str],
        stream_log_path: Path | None = None,
    ) -> _StreamParseResult:
        """Parse ``stream-json`` stdout from a Claude CLI process."""
        is_error = False
        error_text: str | None = None
        num_turns = 0
        total_cost: float | None = None
        rate_limit_reset: str | None = None
        mcp_log: Path | None = None

        stream_log_fh = (
            open(stream_log_path, "a", encoding="utf-8")  # noqa: SIM115
            if stream_log_path
            else None
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if stream_log_fh is not None:
                stream_log_fh.write(line + "\n")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Non-JSON output: %s", line[:200])
                continue

            msg_type = msg.get("type", "")

            if msg_type == "system":
                subtype = msg.get("subtype", "")
                if subtype == "init":
                    mcp_servers = msg.get("mcp_servers", [])
                    if mcp_servers:
                        assert stream_log_path is not None, (
                            "stream_log_path must be set when MCP servers "
                            "are configured"
                        )
                        mcp_log = (
                            stream_log_path.parent
                            / "sandbox"
                            / ".mcp"
                            / "mcp_server.log"
                        )
                    for srv in mcp_servers:
                        status = srv.get("status")
                        name = srv.get("name")
                        if name != MCP_SERVER_NAME:
                            continue
                        if status != "connected":
                            logger.warning("MCP server %s: %s", name, status)
                            logger.warning("MCP server %s full status: %s", name, srv)
                            logger.warning(
                                "Check the MCP server log for details: %s",
                                mcp_log,
                            )
                            logger.warning(
                                "If running in Docker, try rebuilding the "
                                "image with `bash docker/build.sh`, as the "
                                "robocode package (including MCP server code) "
                                "is baked into the image at build time."
                            )
                        else:
                            logger.info("MCP server %s: %s", name, status)
                elif subtype == "compact_boundary":
                    meta = msg.get("compact_metadata", {})
                    logger.info(
                        "Context compaction: trigger=%s, pre_tokens=%s",
                        meta.get("trigger"),
                        meta.get("pre_tokens"),
                    )
                elif subtype != "status":
                    logger.info("System event: subtype=%s", subtype)

            if msg_type == "assistant":
                num_turns += 1
                if self._max_turns > 0 and num_turns > self._max_turns:
                    logger.warning(
                        "Turn limit reached: %d >= %d, terminating",
                        num_turns,
                        self._max_turns,
                    )
                    proc.kill()
                    is_error = True
                    error_text = (
                        f"Turn limit reached: {num_turns} >= " f"{self._max_turns}"
                    )
                    break
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") == "thinking":
                        logger.info("Thinking: %s", block.get("thinking", ""))
                    elif block.get("type") == "text":
                        text = block["text"]
                        logger.info("Agent: %s", text)
                        m = _RATE_LIMIT_RE.search(text)
                        if m:
                            rate_limit_reset = m.group(1)
                    elif block.get("type") == "tool_use":
                        input_str = json.dumps(block.get("input", {}))
                        if len(input_str) > 300:
                            input_str = input_str[:300] + "..."
                        logger.info(
                            "Tool call: %s(%s)",
                            block.get("name"),
                            input_str,
                        )

            elif msg_type in ("tool_result", "user"):
                tool_use_result = msg.get("tool_use_result")
                if tool_use_result is not None:
                    if len(tool_use_result) > 500:
                        tool_use_result = tool_use_result[:500] + "..."
                    if "Error" in tool_use_result:
                        logger.warning("Tool result: %s", tool_use_result)
                        if "No such tool" in tool_use_result:
                            logger.warning(
                                "Tool not found, if running in Docker, try "
                                "rebuilding the image with "
                                "`bash docker/build.sh`. "
                                "Check %s for server-side details.",
                                mcp_log,
                            )
                    elif "mcp_renders" in tool_use_result:
                        logger.info("Tool result: %s", tool_use_result)
                    else:
                        logger.debug("Tool result: %s", tool_use_result)

            elif msg_type == "result":
                is_error = msg.get("is_error", False)
                num_turns = msg.get("num_turns", 0)
                total_cost = msg.get("total_cost_usd")
                if is_error:
                    error_text = msg.get("result", "Unknown error")
                    if not rate_limit_reset:
                        m = _RATE_LIMIT_RE.search(error_text)
                        if m:
                            rate_limit_reset = m.group(1)

        proc.wait()

        if stream_log_fh is not None:
            stream_log_fh.close()

        assert proc.stderr is not None
        stderr_output = proc.stderr.read()
        if proc.returncode != 0 and not is_error:
            is_error = True
            error_text = (
                stderr_output[:1000]
                if stderr_output
                else f"Process exited with code {proc.returncode}"
            )
            if not rate_limit_reset and stderr_output:
                m = _RATE_LIMIT_RE.search(stderr_output)
                if m:
                    rate_limit_reset = m.group(1)

        if rate_limit_reset and not is_error:
            is_error = True
            error_text = f"Rate-limited: resets {rate_limit_reset}"

        return _StreamParseResult(
            is_error=is_error,
            error_text=error_text,
            num_turns=num_turns,
            total_cost=total_cost,
            rate_limit_reset=rate_limit_reset,
        )


def _get_claude_cmd() -> str:
    """Return the claude CLI command, respecting ROBOCODE_CLAUDE_CMD."""
    return os.environ.get("ROBOCODE_CLAUDE_CMD", "claude")
