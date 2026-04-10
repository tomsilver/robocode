"""OpenCode CLI agent backend.

Implements the AgentBackend protocol for the OpenCode CLI
(https://opencode.ai), enabling use of OpenAI, Google, Anthropic,
and local (Ollama, vLLM) models.

OpenCode JSON event types (from ``--format json``):
  - ``step_start``: new inference step begins
  - ``text``: completed text response part
  - ``reasoning``: completed thinking/reasoning block
  - ``tool_use``: tool invocation completed or errored
  - ``step_finish``: inference step done (has cost/tokens)
  - ``error``: session-level error (auth, API, output length, etc.)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

from omegaconf import DictConfig

from robocode.mcp import setup_mcp_config
from robocode.utils.backends.ollama_server import ensure_ollama
from robocode.utils.sandbox_types import SandboxConfig, _StreamParseResult

logger = logging.getLogger(__name__)

_RATE_LIMIT_RE = re.compile(
    r"(?:rate.?limit|too many requests|quota exceeded|429)"
    r".*?(?:retry.?after|resets?\s+)(\d{1,2}(?:am|pm)|\d+\s*s(?:ec)?)",
    re.IGNORECASE,
)


class OpenCodeBackend:
    """OpenCode CLI agent backend."""

    def __init__(self, backend_cfg: DictConfig) -> None:
        self._variant = backend_cfg.get("variant", "")
        self._ollama_keep_alive = backend_cfg.get("ollama_keep_alive", "")
        self._max_budget_usd: float = 0.0
        self._max_turns: int = 0

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "opencode"

    def build_cli_cmd(  # pylint: disable=unused-argument
        self,
        config: SandboxConfig,
        *,
        mcp_python_cmd: str = "",
        mcp_env_config_path: str = "",
        mcp_config_cli_path: str | None = None,
        mcp_log_file_path: str = "",
    ) -> list[str]:
        """Build the OpenCode CLI command."""
        self._max_budget_usd = config.max_budget_usd
        self._max_turns = config.max_turns
        opencode_cmd = _get_opencode_cmd()
        args = [
            opencode_cmd,
            "run",
            config.prompt,
            "--format",
            "json",
            "--print-logs",
            "--log-level",
            "INFO",
        ]
        if config.model:
            args += ["--model", config.model]
        if self._variant:
            args += ["--variant", self._variant]
        # MCP config is written into opencode.json by setup_sandbox_files,
        # not passed via CLI flags. But we still need to set up the .mcp/
        # directory with env_config.json and mcp_config.json for the MCP
        # server process itself.
        if config.mcp_tools:
            log_path = mcp_log_file_path or str(
                (config.sandbox_dir / ".mcp" / "mcp_server.log").resolve()
            )
            setup_mcp_config(
                config.sandbox_dir,
                config.mcp_tools,
                mcp_python_cmd,
                mcp_env_config_path,
                log_path,
            )
        return args

    def build_env(  # pylint: disable=unused-argument
        self,
        config: SandboxConfig,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build a clean environment dict for OpenCode."""
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("CLAUDECODE") and not k.startswith("OPENCODE")
        }
        # Prevent OpenCode from reading the host's CLAUDE.md as a fallback
        # instruction file, since we write our own AGENTS.md.
        env["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
        if config.model.startswith("ollama/"):
            ensure_ollama(keep_alive=self._ollama_keep_alive or "5m")
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
        """Write ``opencode.json`` and ``AGENTS.md`` into the sandbox dir."""
        # --- opencode.json ---
        oc_config: dict = {
            "$schema": "https://opencode.ai/config.json",
            "model": config.model,
            "permission": "allow",
            "compaction": {"auto": True, "prune": True},
        }

        # Auto-configure Ollama provider when model uses ollama/ prefix.
        if config.model.startswith("ollama/"):
            model_name = config.model.split("/", 1)[1]
            oc_config["provider"] = {
                "ollama": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {"baseURL": "http://localhost:11434/v1"},
                    "models": {model_name: {"name": model_name}},
                },
            }

        # MCP server config (if tools are configured).
        if config.mcp_tools:
            mcp_config_path = config.sandbox_dir / ".mcp" / "mcp_config.json"
            if mcp_config_path.exists():
                claude_mcp = json.loads(mcp_config_path.read_text())
                servers = claude_mcp.get("mcpServers", {})
                # Convert Claude MCP format to OpenCode format.
                oc_mcp: dict = {}
                for name, srv in servers.items():
                    oc_mcp[name] = {
                        "type": "local",
                        "command": [srv["command"]] + srv.get("args", []),
                        "enabled": True,
                    }
                oc_config["mcp"] = oc_mcp

        (config.sandbox_dir / "opencode.json").write_text(
            json.dumps(oc_config, indent=2) + "\n"
        )

        # --- AGENTS.md ---
        agents_md = config.sandbox_dir / "AGENTS.md"
        if not agents_md.exists():
            parts = []
            # Include the system prompt as instructions.
            if config.system_prompt:
                parts.append(config.system_prompt)

            if docker_python:
                parts.append(
                    "\nAll files you create MUST use relative paths so they "
                    "stay in the current working directory (/sandbox). Never "
                    "write files using absolute paths.\n\n"
                    "CONTEXT MANAGEMENT: your context window is limited and "
                    "you MUST protect it aggressively: "
                    "(1) When running Bash commands, pipe output through "
                    "`head` or `tail` to limit verbosity if possible. "
                    "(2) Keep your thinking brief, do not write long "
                    "reasoning traces. "
                    "(3) Keep your main conversation focused on writing and "
                    "testing code, not on reading source or reasoning at "
                    "length.\n"
                    f"The Python interpreter is at {docker_python}\n"
                    "Run test scripts with:\n"
                    f"    {docker_python} test_approach.py\n"
                )
                if primitive_names:
                    parts.append(
                        "\nPrimitive source files (for reference) are in "
                        "./primitives/\n"
                    )
            else:
                parts.append(
                    "\nAll files you create MUST use relative paths so they "
                    "stay in the current working directory. Never write files "
                    "using absolute paths.\n"
                )

            agents_md.write_text("\n".join(parts))

    def parse_stream(
        self,
        proc: subprocess.Popen[str],
        stream_log_path: Path | None = None,
    ) -> _StreamParseResult:
        """Parse OpenCode's ``--format json`` output.

        OpenCode emits one JSON object per line to stdout when using
        ``--format json``.
        Event types: step_start, text, reasoning, tool_use, step_finish,
        error.
        """
        is_error = False
        error_text: str | None = None
        num_turns = 0
        total_cost: float = 0.0
        rate_limit_reset: str | None = None

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
            part = msg.get("part", {})

            if msg_type == "step_start":
                num_turns += 1
                logger.info("Step %d started", num_turns)
                if self._max_turns > 0 and num_turns > self._max_turns:
                    logger.warning(
                        "Turn limit reached: %d >= %d, terminating",
                        num_turns,
                        self._max_turns,
                    )
                    os.killpg(proc.pid, 9)
                    is_error = True
                    error_text = (
                        f"Turn limit reached: {num_turns} >= " f"{self._max_turns}"
                    )
                    break

            elif msg_type == "text":
                text = part.get("text", "")
                logger.info("Agent: %s", text)
                m = _RATE_LIMIT_RE.search(text)
                if m:
                    rate_limit_reset = m.group(1)

            elif msg_type == "reasoning":
                text = part.get("text", "")
                logger.info("Thinking: %s", text)

            elif msg_type == "tool_use":
                tool_name = part.get("tool", "unknown")
                state = part.get("state", {})
                status = state.get("status", "")
                if status == "completed":
                    output = state.get("output", "")
                    if len(output) > 300:
                        output = output[:300] + "..."
                    logger.info("Tool call: %s -> %s", tool_name, output)
                elif status == "error":
                    err = state.get("error", "unknown error")
                    logger.warning("Tool error: %s -> %s", tool_name, err)

            elif msg_type == "step_finish":
                step_cost = part.get("cost", 0) or 0
                total_cost += step_cost
                tokens = part.get("tokens", {})
                logger.info(
                    "Step done: reason=%s, cost=$%.4f, tokens_in=%s, tokens_out=%s",
                    part.get("reason", "unknown"),
                    step_cost,
                    tokens.get("input", 0),
                    tokens.get("output", 0),
                )
                # Enforce budget: kill the process if cost exceeds limit.
                if self._max_budget_usd > 0 and total_cost >= self._max_budget_usd:
                    logger.warning(
                        "Budget exceeded: $%.4f >= $%.2f, terminating",
                        total_cost,
                        self._max_budget_usd,
                    )
                    os.killpg(proc.pid, 9)
                    is_error = True
                    error_text = (
                        f"Budget exceeded: ${total_cost:.4f} "
                        f">= ${self._max_budget_usd:.2f}"
                    )
                    break

            elif msg_type == "error":
                is_error = True
                err_data = msg.get("error", {})
                error_name = err_data.get("name", "UnknownError")
                err_msg = err_data.get("data", {}).get("message", str(err_data))
                error_text = f"{error_name}: {err_msg}"
                logger.error("Session error: %s", error_text)
                m = _RATE_LIMIT_RE.search(error_text)
                if m:
                    rate_limit_reset = m.group(1)

        proc.wait()

        # Read stderr: contains --print-logs debug output and possibly
        # error JSON events. Write to a separate log file.
        assert proc.stderr is not None
        stderr_output = proc.stderr.read()
        if stderr_output.strip():
            # Write debug logs to a separate file (not stream.jsonl).
            if stream_log_path is not None:
                debug_log_path = stream_log_path.with_name("opencode_debug.log")
                debug_log_path.write_text(stderr_output)

            # Still check for JSON error events in stderr.
            for line in stderr_output.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    msg_type = msg.get("type", "")
                    if msg_type == "error":
                        is_error = True
                        err_data = msg.get("error", {})
                        error_name = err_data.get("name", "UnknownError")
                        err_msg = err_data.get("data", {}).get("message", str(err_data))
                        error_text = f"{error_name}: {err_msg}"
                        logger.error("Session error (stderr): %s", error_text)
                except json.JSONDecodeError:
                    pass

        if proc.returncode != 0 and not is_error:
            is_error = True
            error_text = f"opencode exited with code {proc.returncode}"

        return _StreamParseResult(
            is_error=is_error,
            error_text=error_text,
            num_turns=num_turns,
            total_cost=total_cost if total_cost > 0 else None,
            rate_limit_reset=rate_limit_reset,
        )


def _get_opencode_cmd() -> str:
    """Return the opencode CLI command, respecting ROBOCODE_OPENCODE_CMD."""
    return os.environ.get("ROBOCODE_OPENCODE_CMD", "opencode")
