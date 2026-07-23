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
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from robocode.mcp import (
    MCP_HTTP_PORT,
    MCP_SERVER_NAME,
    MCP_STARTUP_TIMEOUT_MS,
    mcp_tool_cli_names,
    setup_mcp_config,
)
from robocode.utils.backends.agent_files import build_claude_md
from robocode.utils.backends.base import AgentBackend
from robocode.utils.backends.ollama_server import ensure_ollama
from robocode.utils.sandbox_types import SandboxConfig, _StreamParseResult

logger = logging.getLogger(__name__)

_RATE_LIMIT_RE = re.compile(
    r"(?:out of extra usage|hit your (?:\w+\s+)*limit)"
    r".*?resets\s+(\d{1,2}(?:am|pm))(?:\s*\(?([A-Za-z][A-Za-z/_]*)\)?)?",
    re.IGNORECASE,
)


def _tool_timing_category(block: dict[str, Any]) -> str:
    """Classify a Claude tool call for non-invasive wall-time accounting."""
    name = block.get("name", "")
    if name in ("Task", "Agent"):
        return "model"
    if "render_policy" in name or "render_state" in name:
        return "experiment"
    if name != "Bash":
        return "other"
    command = str((block.get("input") or {}).get("command", ""))
    has_python = bool(
        re.search(
            r"(?:^|[;&|]\s*)(?:(?:uv\s+run\s+)?(?:/\S+/)?)python(?:3)?\s",
            command,
            re.MULTILINE,
        )
        or re.search(r"(?:^|[;&|]\s*)pytest\s", command, re.MULTILINE)
    )
    if not has_python:
        return "other"
    # A tiny ``python -c 'import ...; print(...)'`` is usually source discovery,
    # while these markers identify environment rollouts and policy diagnostics.
    experiment_markers = (
        "test",
        "debug",
        "approach",
        "run_episode",
        "env.reset",
        "render",
        "evaluate",
    )
    is_inspection = "python -c" in command and not any(
        marker in command.lower() for marker in experiment_markers
    )
    return "other" if is_inspection else "experiment"


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


class ClaudeBackend(AgentBackend):
    """Claude Code CLI agent backend.

    When ``base_url`` is set in the backend config, ``ANTHROPIC_BASE_URL``
    and ``ANTHROPIC_AUTH_TOKEN`` are injected into the process environment,
    allowing Claude Code to talk to Ollama, liteLLM, vLLM, or any other
    Anthropic-API-compatible endpoint.
    """

    def __init__(self, backend_cfg: DictConfig) -> None:
        self._base_url = backend_cfg.get("base_url", "")
        self._auth_token = backend_cfg.get("auth_token", "ollama")
        self._ollama_keep_alive = backend_cfg.get("ollama_keep_alive", "")
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
        mcp_transport: str = "stdio",
        mcp_port: int = MCP_HTTP_PORT,
    ) -> list[str]:
        """Build the Claude CLI command."""
        self._max_turns = config.max_turns
        claude_cmd = get_claude_cmd()
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
                blackbox=config.blackbox,
                transport=mcp_transport,
                port=mcp_port,
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
        # Give the render MCP server time to connect before the CLI snapshots
        # its tool list (local backend; docker/apptainer pass this via -e/--env).
        env["MCP_TIMEOUT"] = str(MCP_STARTUP_TIMEOUT_MS)
        env.update(
            anthropic_compatible_env(
                self._base_url, self._auth_token, self._ollama_keep_alive
            )
        )
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
            claude_md.write_text(build_claude_md(docker_python, primitive_names))

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
        num_tool_calls = 0
        num_autocompactions = 0
        num_permission_denials = 0
        turn_limit_hit = False
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0
        cli_duration_ms: int | None = None
        cli_duration_api_ms: int | None = None
        stop_reason: str | None = None
        model_usage: dict[str, Any] = {}
        pending_tools: dict[str, str] = {}
        model_wait_time_s = 0.0
        experiment_time_s = 0.0
        other_tool_time_s = 0.0
        last_event_time = time.monotonic()

        stream_log_fh = (
            open(stream_log_path, "a", encoding="utf-8")  # noqa: SIM115
            if stream_log_path
            else None
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            now = time.monotonic()
            elapsed = max(0.0, now - last_event_time)
            categories = set(pending_tools.values())
            if "experiment" in categories:
                experiment_time_s += elapsed
            elif "other" in categories:
                other_tool_time_s += elapsed
            else:
                # Includes direct Claude API waits and Task/Agent subagent
                # thinking. Parser/logging overhead here is negligible.
                model_wait_time_s += elapsed
            last_event_time = now
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
                        if status == "connected":
                            logger.info("MCP server %s: %s", name, status)
                        elif status == "pending":
                            # Expected transient: the render MCP server needs a
                            # couple of seconds to import and answer the stdio
                            # handshake; the CLI waits up to MCP_TIMEOUT and
                            # registers its tools once it connects. A genuine
                            # failure surfaces later as "No such tool".
                            logger.info(
                                "MCP server %s still connecting (pending); the "
                                "CLI waits up to MCP_TIMEOUT for it.",
                                name,
                            )
                        else:
                            logger.warning("MCP server %s: %s", name, status)
                            logger.warning("MCP server %s full status: %s", name, srv)
                            logger.warning(
                                "Check the MCP server log for details: %s",
                                mcp_log,
                            )
                            logger.warning(
                                "Possible causes: the server did not connect "
                                "before the CLI snapshotted its tools (raise "
                                "MCP_TIMEOUT), or, if running in Docker, the "
                                "image is stale -- rebuild with "
                                "`bash docker/build.sh`, as the robocode package "
                                "(including MCP server code) is baked into the "
                                "image at build time."
                            )
                elif subtype == "compact_boundary":
                    num_autocompactions += 1
                    meta = msg.get("compact_metadata", {})
                    logger.info(
                        "Context compaction: trigger=%s, pre_tokens=%s",
                        meta.get("trigger"),
                        meta.get("pre_tokens"),
                    )
                elif subtype not in ("status", "thinking_tokens"):
                    # thinking_tokens fires on every reasoning chunk and floods
                    # the logs; other system events are rare diagnostics, so keep
                    # them at debug rather than info.
                    logger.debug("System event: subtype=%s", subtype)

            if msg_type == "assistant":
                num_turns += 1
                if self._max_turns > 0 and num_turns > self._max_turns:
                    logger.warning(
                        "Turn limit reached: %d >= %d, terminating",
                        num_turns,
                        self._max_turns,
                    )
                    os.killpg(proc.pid, 9)
                    is_error = True
                    turn_limit_hit = True
                    error_text = f"Turn limit reached: {num_turns} >= {self._max_turns}"
                    break
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") == "thinking":
                        logger.info("Thinking: %s", block.get("thinking", ""))
                    elif block.get("type") == "text":
                        text = block["text"]
                        logger.info("Agent: %s", text)
                        m = _RATE_LIMIT_RE.search(text)
                        if m:
                            rate_limit_reset = m.group(0)
                    elif block.get("type") == "tool_use":
                        num_tool_calls += 1
                        tool_id = block.get("id")
                        if tool_id:
                            pending_tools[tool_id] = _tool_timing_category(block)
                        input_str = json.dumps(block.get("input", {}))
                        if len(input_str) > 300:
                            input_str = input_str[:300] + "..."
                        logger.info(
                            "Tool call: %s(%s)",
                            block.get("name"),
                            input_str,
                        )

            elif msg_type in ("tool_result", "user"):
                content = (msg.get("message") or {}).get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_result":
                            pending_tools.pop(block.get("tool_use_id", ""), None)
                pending_tools.pop(msg.get("tool_use_id", ""), None)
                tool_use_result = msg.get("tool_use_result")
                if tool_use_result is not None:
                    if len(tool_use_result) > 500:
                        tool_use_result = tool_use_result[:500] + "..."
                    if "Error" in tool_use_result:
                        logger.warning("Tool result: %s", tool_use_result)
                        if "No such tool" in tool_use_result:
                            logger.warning(
                                "MCP render tool was unavailable when called. "
                                "Likely the server had not finished connecting "
                                "before the CLI snapshotted its tools (raise "
                                "MCP_TIMEOUT); or, if running in Docker, the "
                                "image may be stale -- rebuild with "
                                "`bash docker/build.sh`. Check the server log "
                                "and its .stderr.log: %s",
                                mcp_log,
                            )
                    elif "mcp_renders" in tool_use_result:
                        logger.info("Tool result: %s", tool_use_result)
                    else:
                        logger.debug("Tool result: %s", tool_use_result)

            elif msg_type == "result":
                # A single generation can span several CLI sessions (autocompaction
                # or budget-continue re-inits the CLI, each emitting its own result).
                # total_cost_usd and modelUsage are cumulative, so keep the latest;
                # per-session fields accumulate. num_turns is counted from assistant
                # messages above, not read here (the final session's value is stale).
                is_error = msg.get("is_error", False)
                total_cost = msg.get("total_cost_usd", total_cost)
                model_usage = msg.get("modelUsage") or model_usage
                stop_reason = msg.get("subtype") or stop_reason
                num_permission_denials += len(msg.get("permission_denials", []))
                cli_duration_ms = (cli_duration_ms or 0) + (msg.get("duration_ms") or 0)
                cli_duration_api_ms = max(
                    cli_duration_api_ms or 0, msg.get("duration_api_ms") or 0
                )
                if is_error:
                    error_text = msg.get("result", "Unknown error")
                    if not rate_limit_reset:
                        m = _RATE_LIMIT_RE.search(error_text)
                        if m:
                            rate_limit_reset = m.group(0)

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
                    rate_limit_reset = m.group(0)

        if rate_limit_reset and not is_error:
            is_error = True
            error_text = f"Rate-limited: resets {rate_limit_reset}"

        # Token counts come from the cumulative per-model usage, not the top-level
        # ``usage`` (which is empty on a final budget-error session).
        for usage in model_usage.values():
            input_tokens += usage.get("inputTokens", 0)
            output_tokens += usage.get("outputTokens", 0)
            cache_read_tokens += usage.get("cacheReadInputTokens", 0)
            cache_creation_tokens += usage.get("cacheCreationInputTokens", 0)

        return _StreamParseResult(
            is_error=is_error,
            error_text=error_text,
            num_turns=num_turns,
            total_cost=total_cost,
            rate_limit_reset=rate_limit_reset,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            num_tool_calls=num_tool_calls,
            num_autocompactions=num_autocompactions,
            num_permission_denials=num_permission_denials,
            turn_limit_hit=turn_limit_hit,
            cli_duration_ms=cli_duration_ms,
            cli_duration_api_ms=cli_duration_api_ms,
            model_wait_time_s=model_wait_time_s,
            experiment_time_s=experiment_time_s,
            other_tool_time_s=other_tool_time_s,
            stop_reason=stop_reason,
            model_usage=model_usage,
        )


def get_claude_cmd() -> str:
    """Return the claude CLI command, respecting ROBOCODE_CLAUDE_CMD."""
    return os.environ.get("ROBOCODE_CLAUDE_CMD", "claude")


def anthropic_compatible_env(
    base_url: str,
    auth_token: str = "ollama",
    ollama_keep_alive: str = "",
) -> dict[str, str]:
    """Env vars to point an Anthropic-API client at a custom endpoint.

    Shared by the Claude agent backend and the CLI completion client so both
    can talk to Ollama, vLLM, liteLLM, or any Anthropic-compatible server.
    Returns an empty dict when ``base_url`` is unset. Starts Ollama if the
    URL targets the default Ollama port.
    """
    if not base_url:
        return {}
    if "11434" in base_url:
        ensure_ollama(keep_alive=ollama_keep_alive or "5m")
    return {"ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_AUTH_TOKEN": auth_token}
