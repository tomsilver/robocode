"""Codex CLI agent backend."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
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
        self._input_rate = float(backend_cfg.get("input_usd_per_mtok", 10.0))
        self._cached_rate = float(backend_cfg.get("cached_input_usd_per_mtok", 1.0))
        self._output_rate = float(backend_cfg.get("output_usd_per_mtok", 50.0))
        self._max_budget_usd = 0.0
        self._max_turns = 0
        self._session_root: Path | None = None
        self._solution_confident_path: Path | None = None
        self._session_offsets: dict[Path, int] = {}
        self._response_usages: dict[str, dict[str, int]] = {}
        self._usage_lock = threading.Lock()
        self._previous_session_usage: dict[str, int] = {}
        self._usage_start_timeout = float(
            backend_cfg.get("usage_start_timeout_s", 120.0)
        )
        self._model: str | None = None

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
        self._model = config.model
        self._session_root = sandbox_codex_sessions(config.sandbox_dir)
        self._solution_confident_path = self._session_root / "solution_confident"
        command = os.environ.get("ROBOCODE_CODEX_CMD", "codex")
        if config.resume_previous_session:
            self._previous_session_usage = self._read_session_usage()
            if not self._previous_session_usage:
                raise ValueError(
                    "Cannot resume Codex without authoritative usage history"
                )
            args = [command, "exec", "resume", "--last", "--all"]
        else:
            self._response_usages = {}
            shutil.rmtree(self._session_root)
            self._session_root.mkdir(parents=True)
            self._previous_session_usage = {}
            self._session_offsets = {}
            args = [command, "exec"]
        args += [
            "--json",
            "--model",
            config.model,
            "--config",
            f"model_reasoning_effort={json.dumps(self._reasoning_effort)}",
            "--ignore-user-config",
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
        if config.resume_previous_session and config.max_budget_usd > 0:
            return (
                f"{config.prompt}\n\nRemaining shared budget: "
                f"${config.max_budget_usd:.2f}. Do not restart from scratch. "
                "Read .agent_sessions/codex/budget_status.json before more work. "
                "This continuation authorizes useful finishing work even when "
                "wrap_up is true. Warnings change priorities, not permission to work. "
                "Identify a remaining uncertainty, run a focused check using your "
                "tools, and correct any issue the evidence reveals. Validation "
                "that leaves the policy unchanged is useful work too. If still "
                "uncertain, continue with the next useful check while budget "
                "remains; do not merely "
                "repeat that the policy is saved or that the budget is low. "
                "Before stopping, create .agent_sessions/codex/solution_confident "
                "only if extremely confident; otherwise leave it absent."
            )
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
            max_budget_usd=config.max_budget_usd,
            original_budget_usd=(
                config.max_budget_usd + self._usage_cost(self._previous_session_usage)
            ),
            sandbox_instructions=build_agents_md(docker_python, primitive_names),
        ).strip()
        (config.sandbox_dir / "AGENTS.md").write_text(instructions + "\n")
        if config.max_budget_usd > 0:
            self._write_budget_status(0.0)

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
        stop_reason: str | None = None
        latest_usage: dict[str, int] = {}
        stdout_usage: dict[str, int] = {}
        accounting_error: list[str] = []
        budget_reached = threading.Event()
        monitor_ready = threading.Event()
        monitor_stop = threading.Event()
        monitor_started = time.monotonic()
        initial_responses = len(self._response_usages)
        kill_lock = threading.Lock()
        killed = False

        def stop_agent() -> None:
            nonlocal killed
            with kill_lock:
                if not killed:
                    killed = True
                    self._kill_process_tree(proc)

        def monitor_budget() -> None:
            nonlocal latest_usage
            try:
                while not monitor_stop.is_set():
                    latest_usage = self._read_session_usage()
                    run_cost = self._usage_cost(latest_usage) - self._usage_cost(
                        self._previous_session_usage
                    )
                    self._write_budget_status(run_cost)
                    if (
                        len(self._response_usages) == initial_responses
                        and time.monotonic() - monitor_started
                        >= self._usage_start_timeout
                    ):
                        raise ValueError("Timed out waiting for Codex response usage")
                    if run_cost >= self._max_budget_usd:
                        budget_reached.set()
                        stop_agent()
                        return
                    monitor_ready.set()
                    monitor_stop.wait(0.1)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # No monitor failure may leave generation running unmetered.
                accounting_error.append(str(exc))
                stop_agent()
            finally:
                monitor_ready.set()

        monitor = None
        if self._max_budget_usd > 0:
            monitor = threading.Thread(target=monitor_budget, daemon=True)
            monitor.start()
            monitor_ready.wait()
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
                    stop_agent()
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
            if isinstance(usage, dict) and self._max_budget_usd <= 0:
                # stdout is turn-scoped, never overwrite the lifetime ledger.
                stdout_usage = {key: int(value or 0) for key, value in usage.items()}
            run_cost = self._usage_cost(latest_usage) - self._usage_cost(
                self._previous_session_usage
            )
            if self._max_budget_usd > 0 and run_cost >= self._max_budget_usd:
                if not budget_reached.is_set():
                    stop_agent()
                is_error = True
                stop_reason = "error_max_budget_usd"
                error_text = f"Codex budget reached: ${run_cost:.4f}"
                break
        monitor_stop.set()
        if monitor is not None:
            monitor.join()
        proc.wait()
        try:
            session_usage = self._read_session_usage()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            accounting_error.append(str(exc))
            session_usage = latest_usage
        if session_usage:
            latest_usage = session_usage
        elif self._max_budget_usd <= 0:
            latest_usage = stdout_usage
        if self._max_budget_usd > 0 and not session_usage:
            accounting_error.append(
                "No authoritative token_usage_record usage available"
            )
        run_cost = self._usage_cost(latest_usage) - self._usage_cost(
            self._previous_session_usage
        )
        if budget_reached.is_set() or (
            self._max_budget_usd > 0 and run_cost >= self._max_budget_usd
        ):
            is_error = True
            stop_reason = "error_max_budget_usd"
            error_text = f"Codex budget reached: ${run_cost:.4f}"
            rate_limit_reset = None
            output_token_limit_hit = False
            prompt_too_long_hit = False
        if accounting_error:
            is_error = True
            stop_reason = "error_budget_accounting"
            error_text = "; ".join(accounting_error)
            rate_limit_reset = None
            output_token_limit_hit = False
            prompt_too_long_hit = False
        if (
            not is_error
            and not proc.returncode
            and 0 < self._max_budget_usd
            and self._max_budget_usd - run_cost <= 0.50 + 1e-9
        ):
            # A normal early exit with a tiny remainder is ready for evaluation,
            # even without confidence. Reuse the accepted budget stop category.
            stop_reason = "error_max_budget_usd"
            is_error = True
            error_text = "Codex remaining budget <= $0.50; not resuming"
        unconfirmed_solution = bool(
            not is_error
            and not proc.returncode
            and self._max_budget_usd > 0
            and run_cost < self._max_budget_usd
            and self._solution_confident_path is not None
            and not self._solution_confident_path.is_file()
        )
        if unconfirmed_solution:
            is_error = True
            stop_reason = "unconfirmed_solution"
            error_text = (
                "Codex stopped without confirming high confidence in the solution"
            )
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
            total_cost=None if accounting_error else self._usage_cost(run_usage),
            rate_limit_reset=rate_limit_reset,
            output_token_limit_hit=output_token_limit_hit,
            prompt_too_long_hit=prompt_too_long_hit,
            unconfirmed_solution=unconfirmed_solution,
            input_tokens=max(input_tokens - cached_tokens, 0),
            output_tokens=output_tokens,
            cache_read_tokens=cached_tokens,
            num_tool_calls=num_tool_calls,
            turn_limit_hit=turn_limit_hit,
            cli_duration_ms=int((time.monotonic() - started) * 1000),
            stop_reason=stop_reason,
            model_usage=dict(run_usage),
        )

    def _read_session_usage(self) -> dict[str, int]:
        """Sum unique response usage, never resettable token_count snapshots."""
        with self._usage_lock:
            return self._read_response_usage()

    def _read_response_usage(self) -> dict[str, int]:
        if self._session_root is None:
            return {}
        if any(not path.is_file() for path in self._session_offsets):
            raise ValueError("Codex usage log disappeared during the experiment")
        for path in self._session_root.rglob("*.jsonl"):
            if path.stat().st_size < self._session_offsets.get(path, 0):
                raise ValueError(f"Codex usage log truncated: {path.name}")
            with path.open(encoding="utf-8") as session:
                session.seek(self._session_offsets.get(path, 0))
                while line := session.readline():
                    if not line.endswith("\n"):
                        break
                    self._session_offsets[path] = session.tell()
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Malformed Codex session record: {path.name}"
                        ) from exc
                    payload = event.get("payload") or {}
                    if (
                        event.get("type") == "turn_context"
                        and self._model
                        and payload.get("model") != self._model
                    ):
                        raise ValueError(
                            "Codex model differs from configured pricing model"
                        )
                    if event.get("type") != "token_usage_record":
                        continue
                    response_id = payload["response_id"]
                    usage = payload["usage"]
                    if not isinstance(response_id, str) or not response_id:
                        raise ValueError("Missing response ID in Codex usage ledger")
                    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                        value = usage[key]
                        # Reject bool too: it is an int subclass, not a token count.
                        # pylint: disable-next=unidiomatic-typecheck
                        if type(value) is not int or value < 0:
                            raise ValueError(f"Invalid Codex {key}: {value!r}")
                    if usage["cached_input_tokens"] > usage["input_tokens"]:
                        raise ValueError("Cached input exceeds total input")
                    previous = self._response_usages.get(response_id)
                    if previous is not None and previous != usage:
                        raise ValueError("Conflicting duplicate Codex response usage")
                    self._response_usages[response_id] = dict(usage)
        totals: dict[str, int] = {}
        for usage in self._response_usages.values():
            for key, value in usage.items():
                totals[key] = totals.get(key, 0) + value
        return totals

    @staticmethod
    def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
        """Stop this invocation's descendants, including new-session tool hosts.

        A process-group kill alone misses Codex's setsid children. Never target
        the harness's shared process group or other experiment sessions.
        Docker additionally removes its named container in the runner's finally.
        """
        pid = getattr(proc, "pid", None)
        # Reject bool/mock PIDs rather than signaling an unrelated process.
        # pylint: disable-next=unidiomatic-typecheck
        if type(pid) is not int or not Path("/proc").is_dir():
            proc.kill()
            return
        targets = {pid}
        try:
            os.kill(pid, signal.SIGSTOP)
            while True:
                rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
                parents = {
                    int(p): int(parent)
                    for p, parent in (line.split() for line in rows.splitlines())
                }
                discovered = {
                    p for p, parent in parents.items() if parent in targets
                } - targets
                if not discovered:
                    break
                for child in discovered:
                    try:
                        os.kill(child, signal.SIGSTOP)
                    except ProcessLookupError:
                        pass
                targets.update(discovered)
        except ProcessLookupError:
            pass
        finally:
            for child in targets - {pid}:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.kill()

    def _write_budget_status(self, run_cost: float) -> None:
        """Best-effort agent-readable warning; not an injected CLI message."""
        if self._session_root is None:
            return
        remaining = max(self._max_budget_usd - run_cost, 0.0)
        previous_cost = self._usage_cost(self._previous_session_usage)
        warning_threshold = (
            1.0 if remaining <= 1.0 else 2.0 if remaining <= 2.0 else None
        )
        status = {
            "max_budget_usd": previous_cost + self._max_budget_usd,
            "spent_usd": previous_cost + run_cost,
            "remaining_usd": remaining,
            "attempt_budget_usd": self._max_budget_usd,
            "attempt_spent_usd": run_cost,
            "wrap_up": remaining <= 2.0,
            "warning_threshold_usd": warning_threshold,
            "message": (
                "At most $1 remains: prioritize essential quick checks, minimal "
                "corrections, and saving/committing your final approach. No new "
                "subagents. This warning is not an instruction to exit; continue "
                "useful finishing work if not confident."
                if remaining <= 1.0
                else (
                    "At most $2 remains: save and commit your best approach; focus on "
                    "targeted validation and small fixes, not broad exploration or "
                    "new subagents. This warning is not an instruction to exit."
                    if remaining <= 2.0
                    else "Continue within the shared budget."
                )
            ),
        }
        path = self._session_root / "budget_status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _usage_cost(self, usage: dict[str, int]) -> float:
        cached = usage.get("cached_input_tokens", 0)
        uncached = max(usage.get("input_tokens", 0) - cached, 0)
        return (
            uncached * self._input_rate
            + cached * self._cached_rate
            + usage.get("output_tokens", 0) * self._output_rate
        ) / 1_000_000
