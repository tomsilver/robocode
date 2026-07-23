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
    r".*?resets\s+(\d{1,2}(?::\d{2})?(?:am|pm))"
    r"(?:\s*\(?([A-Za-z][A-Za-z/_]*)\)?)?",
    re.IGNORECASE,
)
_OUTPUT_TOKEN_LIMIT_RE = re.compile(
    r"response exceeded (?:the )?\d+ output token maximum", re.IGNORECASE
)
_PROMPT_TOO_LONG_RE = re.compile(r"\bprompt is too long\b", re.IGNORECASE)

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
            "--tools",
            tools,
            "--setting-sources",
            "project",
        ]
        # Persist the session (no --no-session-persistence) so a run interrupted
        # by the usage cap can be continued. --continue resumes the most recent
        # conversation in the working directory, which is unique to this
        # generation, so it reattaches to exactly this run's context.
        if config.resume_previous_session:
            args.append("--continue")
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
        output_token_limit_hit = False
        prompt_too_long_hit = False
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
                            rate_limit_reset = m.group(0)
                        if _OUTPUT_TOKEN_LIMIT_RE.search(text):
                            output_token_limit_hit = True
                        if _PROMPT_TOO_LONG_RE.search(text):
                            prompt_too_long_hit = True
                    elif block.get("type") == "tool_use":
                        num_tool_calls += 1
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
                    error_text = str(msg.get("result") or "Unknown error")
                    if not rate_limit_reset:
                        m = _RATE_LIMIT_RE.search(error_text)
                        if m:
                            rate_limit_reset = m.group(0)
                    if _OUTPUT_TOKEN_LIMIT_RE.search(error_text or ""):
                        output_token_limit_hit = True
                    if _PROMPT_TOO_LONG_RE.search(error_text):
                        prompt_too_long_hit = True

        proc.wait()

        if stream_log_fh is not None:
            stream_log_fh.close()

        assert proc.stderr is not None
        stderr_output = proc.stderr.read()
        if not rate_limit_reset and stderr_output:
            m = _RATE_LIMIT_RE.search(stderr_output)
            if m:
                rate_limit_reset = m.group(0)
        if stderr_output and _OUTPUT_TOKEN_LIMIT_RE.search(stderr_output):
            output_token_limit_hit = True
        if stderr_output and _PROMPT_TOO_LONG_RE.search(stderr_output):
            prompt_too_long_hit = True
        if proc.returncode != 0 and not is_error:
            is_error = True
            error_text = (
                stderr_output[:1000]
                if stderr_output
                else f"Process exited with code {proc.returncode}"
            )

        if rate_limit_reset and not is_error:
            is_error = True
            error_text = f"Rate-limited: resets {rate_limit_reset}"
        if output_token_limit_hit and not is_error:
            is_error = True
            error_text = "Claude response exceeded the output token maximum"
        if prompt_too_long_hit and not is_error:
            is_error = True
            error_text = "Claude prompt is too long"

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
            output_token_limit_hit=output_token_limit_hit,
            prompt_too_long_hit=prompt_too_long_hit,
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
