"""Codex CLI agent backend."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import TextIO

from omegaconf import DictConfig

from robocode.mcp import MCP_HTTP_PORT, MCP_SERVER_NAME, setup_mcp_config
from robocode.utils.backends.agent_files import build_agents_md
from robocode.utils.backends.base import AgentBackend, read_stderr
from robocode.utils.codex_auth import sandbox_codex_sessions
from robocode.utils.sandbox_types import SandboxConfig, _StreamParseResult

logger = logging.getLogger(__name__)
_RATE_LIMIT_RE = re.compile(
    r"(?:rate.?limit|usage limit|too many requests|quota exceeded|429).*?"
    r"(?:resets?\s+)(\d{1,2}(?::\d{2})?(?:am|pm)(?:\s+UTC)?)",
    re.IGNORECASE,
)
_OUTPUT_TOKEN_LIMIT_RE = re.compile(r"(?:maximum output|max output|output token)", re.I)
_PROMPT_TOO_LONG_RE = re.compile(r"(?:prompt is too long|context length)", re.I)


class CodexBackend(AgentBackend):
    """Run Codex non-interactively inside the shared sandbox."""

    def __init__(self, backend_cfg: DictConfig) -> None:
        self._reasoning_effort = backend_cfg.get("reasoning_effort", "medium")
        self._input_rate = float(backend_cfg.get("input_usd_per_mtok", 4.0))
        self._cached_rate = float(backend_cfg.get("cached_input_usd_per_mtok", 0.4))
        self._output_rate = float(backend_cfg.get("output_usd_per_mtok", 20.0))
        self._max_budget_usd = 0.0
        self._max_turns = 0
        self._session_root: Path | None = None
        self._session_path: Path | None = None
        self._session_offset = 0
        self._latest_session_usage: dict[str, int] = {}
        self._previous_session_usage: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "codex"

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
        self._max_budget_usd = config.max_budget_usd
        self._max_turns = config.max_turns
        self._session_root = sandbox_codex_sessions(config.sandbox_dir)
        command = os.environ.get("ROBOCODE_CODEX_CMD", "codex")
        if config.resume_previous_session:
            self._session_path = None
            self._session_offset = 0
            self._latest_session_usage = {}
            self._previous_session_usage = self._read_session_usage()
            args = [command, "exec", "resume", "--last", "--all"]
        else:
            self._previous_session_usage = {}
            self._session_path = None
            self._session_offset = 0
            self._latest_session_usage = {}
            args = [command, "exec"]
        args += [
            "--json",
            "--model",
            config.model,
            "--config",
            f"model_reasoning_effort={json.dumps(self._reasoning_effort)}",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
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
            server = json.loads(config_path.read_text())["mcpServers"][MCP_SERVER_NAME]
            if server.get("type") == "http":
                args += [
                    "--config",
                    f"mcp_servers.{MCP_SERVER_NAME}.url={json.dumps(server['url'])}",
                ]
            else:
                args += [
                    "--config",
                    f"mcp_servers.{MCP_SERVER_NAME}.command="
                    f"{json.dumps(server['command'])}",
                    "--config",
                    f"mcp_servers.{MCP_SERVER_NAME}.args="
                    f"{json.dumps(server.get('args', []))}",
                ]
        args.append("-")
        return args

    def stdin_text(self, config: SandboxConfig) -> str:
        return config.prompt

    def build_env(
        self, config: SandboxConfig, extra: dict[str, str] | None = None
    ) -> dict[str, str]:
        del config
        env = os.environ.copy()
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
        template = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
        instructions = template.format(
            system_prompt=config.system_prompt,
            sandbox_instructions=build_agents_md(docker_python, primitive_names),
        ).strip()
        (config.sandbox_dir / "AGENTS.md").write_text(instructions + "\n")

    def parse_stream(
        self,
        proc: subprocess.Popen[str],
        stream_log_path: Path | None = None,
        stderr_file: TextIO | None = None,
    ) -> _StreamParseResult:
        is_error = False
        error_text: str | None = None
        rate_limit_reset: str | None = None
        num_turns = 0
        num_tool_calls = 0
        turn_limit_hit = False
        output_token_limit_hit = False
        prompt_too_long_hit = False
        latest_usage: dict[str, int] = {}
        stream_log = (
            open(stream_log_path, "a", encoding="utf-8") if stream_log_path else None
        )
        started = time.monotonic()
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            if stream_log:
                stream_log.write(line + "\n")
                stream_log.flush()
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            message_type = message.get("type", "")
            item = message.get("item") or {}
            if message_type == "item.completed" and item.get("type") == "agent_message":
                num_turns += 1
                logger.info("Agent: %s", item.get("text", ""))
                if self._max_turns > 0 and num_turns >= self._max_turns:
                    proc.kill()
                    is_error = True
                    turn_limit_hit = True
                    error_text = f"Turn limit reached: {num_turns} >= {self._max_turns}"
                    break
            elif message_type == "item.started" and item.get("type") in {
                "command_execution",
                "mcp_tool_call",
            }:
                num_tool_calls += 1
            elif message_type in {"turn.failed", "error", "fatal"} or (
                message_type == "item.completed" and item.get("type") == "error"
            ):
                error = message.get("error") or item.get("message") or message
                error_text = (
                    str(error.get("message")) if isinstance(error, dict) else str(error)
                )
                if "Reconnecting..." in error_text or error_text.startswith(
                    "Falling back from WebSockets"
                ):
                    error_text = None
                else:
                    is_error = True
                    match = _RATE_LIMIT_RE.search(error_text)
                    if match:
                        rate_limit_reset = match.group(1)
                    output_token_limit_hit = bool(
                        _OUTPUT_TOKEN_LIMIT_RE.search(error_text)
                    )
                    prompt_too_long_hit = bool(_PROMPT_TOO_LONG_RE.search(error_text))
            usage = message.get("usage")
            if isinstance(usage, dict):
                latest_usage = {key: int(value or 0) for key, value in usage.items()}
            session_usage = self._read_session_usage()
            if session_usage:
                latest_usage = session_usage
            run_cost = self._usage_cost(latest_usage) - self._usage_cost(
                self._previous_session_usage
            )
            if self._max_budget_usd > 0 and run_cost >= self._max_budget_usd:
                proc.kill()
                is_error = True
                error_text = f"Codex budget reached: ${run_cost:.4f}"
                break
        proc.wait()
        if stream_log:
            stream_log.close()
        stderr = read_stderr(proc, stderr_file)
        if proc.returncode and not is_error:
            is_error = True
            error_text = stderr[:1000] or f"Process exited with {proc.returncode}"
        previous = self._previous_session_usage
        input_tokens = max(
            latest_usage.get("input_tokens", 0) - previous.get("input_tokens", 0), 0
        )
        output_tokens = max(
            latest_usage.get("output_tokens", 0) - previous.get("output_tokens", 0), 0
        )
        cached_tokens = max(
            latest_usage.get("cached_input_tokens", 0)
            - previous.get("cached_input_tokens", 0),
            0,
        )
        self._previous_session_usage = latest_usage
        run_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_tokens,
        }
        return _StreamParseResult(
            is_error=is_error,
            error_text=error_text,
            num_turns=num_turns,
            total_cost=self._usage_cost(run_usage),
            rate_limit_reset=rate_limit_reset,
            output_token_limit_hit=output_token_limit_hit,
            prompt_too_long_hit=prompt_too_long_hit,
            input_tokens=max(input_tokens - cached_tokens, 0),
            output_tokens=output_tokens,
            cache_read_tokens=cached_tokens,
            num_tool_calls=num_tool_calls,
            turn_limit_hit=turn_limit_hit,
            cli_duration_ms=int((time.monotonic() - started) * 1000),
            model_usage=dict(run_usage),
        )

    def _read_session_usage(self) -> dict[str, int]:
        if self._session_root is None:
            return {}
        if self._session_path is None:
            sessions = sorted(
                self._session_root.rglob("*.jsonl"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if not sessions:
                return {}
            self._session_path = sessions[0]
        with self._session_path.open(encoding="utf-8") as session:
            session.seek(self._session_offset)
            lines = session.readlines()
            self._session_offset = session.tell()
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = event.get("payload") or {}
            if (
                event.get("type") == "event_msg"
                and payload.get("type") == "token_count"
            ):
                usage = (payload.get("info") or {}).get("total_token_usage")
                if isinstance(usage, dict):
                    self._latest_session_usage = {
                        key: int(value or 0) for key, value in usage.items()
                    }
        return self._latest_session_usage

    def _usage_cost(self, usage: dict[str, int]) -> float:
        cached = usage.get("cached_input_tokens", 0)
        uncached = max(usage.get("input_tokens", 0) - cached, 0)
        return (
            uncached * self._input_rate
            + cached * self._cached_rate
            + usage.get("output_tokens", 0) * self._output_rate
        ) / 1_000_000
