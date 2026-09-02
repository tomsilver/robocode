"""LLM helpers and opt-in Docker Claude tool-isolation tests.

Run with pytest tests/utils/test_llm.py -m integration. These tests make paid
model calls using existing Claude authentication and a harmless MCP server.
"""

import json
import shutil
import subprocess
import uuid
from contextlib import nullcontext
from pathlib import Path

import pytest
from omegaconf import DictConfig

from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    GENPLAN_CONTAINER_TIMEOUT_S,
    run_genplan_in_docker,
)
from robocode.utils.llm.base import pricing_from_cfg, usage_cost
from robocode.utils.llm.cli_client import ClaudeCLIClient


def test_usage_cost_from_list_prices():
    """Cost is tokens-in/out times per-million-token prices."""
    # 1M input @ $3 + 1M output @ $15 = $18.
    assert usage_cost(1_000_000, 1_000_000, 3.0, 15.0) == 18.0
    # Partial millions scale linearly: 0.5M @ $5 + 0.2M @ $25 = $2.5 + $5 = $7.5.
    assert usage_cost(500_000, 200_000, 5.0, 25.0) == 7.5


def test_pricing_from_cfg():
    """Prices are read when present; absent on either side defaults to 0.0."""
    assert pricing_from_cfg(
        DictConfig({"input_cost_per_mtok": 3.0, "output_cost_per_mtok": 15.0})
    ) == (3.0, 15.0)
    # One side only -> the other defaults to 0.0 (still "priced").
    assert pricing_from_cfg(DictConfig({"output_cost_per_mtok": 25.0})) == (0.0, 25.0)
    # Neither set -> unpriced (cost stays unknown, e.g. local vLLM/Ollama).
    assert pricing_from_cfg(DictConfig({"model": "x"})) is None


def test_cli_denies_all_tools(monkeypatch):
    """The plain completion client blocks built-in and MCP tools explicitly."""

    def fake_run(args, **kwargs):
        assert args[args.index("--tools") + 1] == ""
        assert args[args.index("--disallowedTools") + 1] == "*"
        assert kwargs["input"] == "USER: hello"
        return subprocess.CompletedProcess(
            args, 0, json.dumps({"result": "world", "total_cost_usd": 0.01}), ""
        )

    monkeypatch.setattr("robocode.utils.llm.cli_client.subprocess.run", fake_run)
    reply = ClaudeCLIClient(DictConfig({"model": "test-model"})).complete(
        [{"role": "user", "content": "hello"}]
    )
    assert reply.text == "world"
    assert reply.cost_usd == 0.01


def test_genplan_docker_timeout_removes_container(tmp_path, monkeypatch):
    """A timed-out Docker client does not leave a paid container running."""
    filtered_src = tmp_path / "src"
    filtered_kindergarden = tmp_path / "kindergarden"
    filtered_src.mkdir()
    filtered_kindergarden.mkdir()
    monkeypatch.setattr(
        "robocode.utils.docker_sandbox._filtered_repo_mounts",
        lambda **_kwargs: nullcontext(
            (filtered_src, filtered_kindergarden, None, None)
        ),
    )
    monkeypatch.setattr(
        "robocode.utils.docker_sandbox._build_docker_auth_args",
        lambda _backend: nullcontext(([], {})),
    )
    monkeypatch.setattr(
        "robocode.utils.docker_sandbox.firewall_domains_for_provider",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        "robocode.utils.docker_sandbox._docker_run_prefix",
        lambda name, *_args, **_kwargs: ["docker", "run", "--name", name],
    )
    calls: list[list[str]] = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(args, GENPLAN_CONTAINER_TIMEOUT_S)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("robocode.utils.docker_sandbox.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired):
        run_genplan_in_docker(tmp_path, {"provider": "cli"})

    container_name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "rm", "-f", container_name]


def test_cli_failure_exposes_diagnostics(monkeypatch):
    """Captured CLI errors include the actual reason rather than only exit 1."""

    def fail(args, **kwargs):
        raise subprocess.CalledProcessError(
            1, args, output="Session limit reached", stderr="resets 4:50am (UTC)"
        )

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Session limit reached") as error:
        ClaudeCLIClient(DictConfig({"model": "test"})).complete([])
    assert "resets 4:50am" in str(error.value)


@pytest.fixture(name="cli_calls")
def _record_cli_calls(monkeypatch):
    """Record requests and return a distinct session ID for each fake call."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs["input"]))
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps(
                {
                    "result": "reply",
                    "session_id": f"session-{len(calls)}",
                    "total_cost_usd": 0.01,
                }
            ),
            "",
        )

    monkeypatch.setattr("robocode.utils.llm.cli_client.subprocess.run", fake_run)
    return calls


def test_cli_resumes_matching_history(cli_calls):
    """Follow-ups send only the new user turn to the explicit previous session."""
    client = ClaudeCLIClient(DictConfig({"model": "test"}))
    messages = [{"role": "user", "content": "hello"}]
    reply = client.complete(messages)
    messages += [
        {"role": "assistant", "content": reply.text},
        {"role": "user", "content": "continue"},
    ]
    followup = client.complete(messages)
    args, prompt = cli_calls[-1]
    assert args == cli_calls[0][0] + ["--resume", "session-1"]
    assert prompt == "continue"
    assert followup.cost_usd == 0.01


def test_cli_repeated_sample_starts_fresh(cli_calls):
    """Independent samples of the same prompt must not continue each other."""
    client = ClaudeCLIClient(DictConfig({"model": "test"}))
    messages = [{"role": "user", "content": "hello"}]
    client.complete(messages)
    client.complete(messages)
    assert "--resume" not in cli_calls[-1][0]
    assert cli_calls[-1][1] == "USER: hello"


def test_cli_edited_history_starts_fresh(cli_calls):
    """Mutating the caller's history cannot silently change session identity."""
    client = ClaudeCLIClient(DictConfig({"model": "test"}))
    messages = [{"role": "user", "content": "hello"}]
    client.complete(messages)
    messages[0]["content"] = "changed"
    messages += [
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "continue"},
    ]
    client.complete(messages)
    assert "--resume" not in cli_calls[-1][0]
    assert cli_calls[-1][1] == "USER: changed\n\nASSISTANT: reply\n\nUSER: continue"


def test_cli_failed_resume_discards_session(cli_calls, monkeypatch):
    """A retry rebuilds history rather than replaying a possibly accepted turn."""
    client = ClaudeCLIClient(DictConfig({"model": "test"}))
    messages = [{"role": "user", "content": "hello"}]
    client.complete(messages)
    messages += [
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "continue"},
    ]
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise subprocess.TimeoutExpired("claude", 1)

        patch.setattr("robocode.utils.llm.cli_client.subprocess.run", fail)
        with pytest.raises(subprocess.TimeoutExpired):
            client.complete(messages)
    client.complete(messages)
    assert "--resume" not in cli_calls[-1][0]


def test_cli_error_result_not_saved_as_session(monkeypatch):
    """An error envelope is not a successful model response."""

    def fake_run(args, **_kwargs):
        return subprocess.CompletedProcess(
            args, 0, json.dumps({"is_error": True, "result": "limit reached"}), ""
        )

    monkeypatch.setattr("robocode.utils.llm.cli_client.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="limit reached"):
        ClaudeCLIClient(DictConfig({"model": "test"})).complete([])


def _create_mcp_tool(sandbox_dir: Path) -> str:
    """Write the Docker MCP server/config and return its secret test token."""
    # A harmless tool returns a token the model cannot know without invoking it.
    token = uuid.uuid4().hex
    shutil.copyfile(
        Path(__file__).parent / "fixtures" / "claude_mcp_server.py",
        sandbox_dir / "server.py",
    )
    (sandbox_dir / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "probe": {
                        "command": DOCKER_PYTHON,
                        "args": ["/sandbox/server.py", token],
                    }
                }
            }
        )
    )

    return token


def _parse_claude_output(transcript_path: Path) -> tuple[dict, dict]:
    """Return the CLI initialization and final result from a JSONL transcript."""
    events = [json.loads(line) for line in transcript_path.read_text().splitlines()]
    init = next(
        e for e in events if e.get("type") == "system" and e.get("subtype") == "init"
    )
    final = next(e for e in reversed(events) if e.get("type") == "result")
    return init, final


@pytest.mark.integration
def test_docker_claude_calls_mcp_without_deny_flag(tmp_path, monkeypatch):
    """Without deny-all, sandboxed Claude invokes the configured MCP tool."""
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")

    token = _create_mcp_tool(tmp_path)

    # Execute the real completion wrapper inside the image, recording CLI events.
    shutil.copyfile(
        Path(__file__).parent / "fixtures" / "claude_mcp_probe.py",
        tmp_path / "probe.py",
    )

    # Keep the production auth, mounts, entrypoint, firewall and privilege drop.
    # Replace only the training driver with the diagnostic script.
    tmp_path.chmod(0o777)  # The container's unprivileged node user writes artifacts.
    real_run = subprocess.run

    def launch_probe(args, **kwargs):
        if args[:2] == ["docker", "run"]:
            assert args[-3:] == [
                DOCKER_PYTHON,
                "-m",
                "robocode.approaches.genplan_driver",
            ]
            args = [
                *args[:-3],
                DOCKER_PYTHON,
                "/sandbox/probe.py",
                "--without-deny-flag",
            ]
        return real_run(args, check=kwargs.pop("check", True), **kwargs)

    monkeypatch.setattr("robocode.utils.docker_sandbox.subprocess.run", launch_probe)
    run_genplan_in_docker(tmp_path, {"provider": "cli"}, timeout=600)

    init, final = _parse_claude_output(tmp_path / "transcript.jsonl")
    assert "mcp__probe__probe_token" in init["tools"]
    assert (tmp_path / "calls.txt").read_text().splitlines()
    assert token in final["result"]


@pytest.mark.integration
def test_docker_claude_blocks_mcp_with_deny_flag(tmp_path, monkeypatch):
    """With deny-all, sandboxed Claude cannot invoke the configured MCP tool."""
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")

    token = _create_mcp_tool(tmp_path)

    # Execute the real completion wrapper inside the image, recording CLI events.
    shutil.copyfile(
        Path(__file__).parent / "fixtures" / "claude_mcp_probe.py",
        tmp_path / "probe.py",
    )

    # Keep the production auth, mounts, entrypoint, firewall and privilege drop.
    # Replace only the training driver with the diagnostic script.
    tmp_path.chmod(0o777)  # The container's unprivileged node user writes artifacts.
    real_run = subprocess.run

    def launch_probe(args, **kwargs):
        if args[:2] == ["docker", "run"]:
            assert args[-3:] == [
                DOCKER_PYTHON,
                "-m",
                "robocode.approaches.genplan_driver",
            ]
            args = [*args[:-3], DOCKER_PYTHON, "/sandbox/probe.py"]
        return real_run(args, check=kwargs.pop("check", True), **kwargs)

    monkeypatch.setattr("robocode.utils.docker_sandbox.subprocess.run", launch_probe)
    run_genplan_in_docker(tmp_path, {"provider": "cli"}, timeout=600)

    init, final = _parse_claude_output(tmp_path / "transcript.jsonl")
    assert init["tools"] == []
    assert not (tmp_path / "calls.txt").exists()
    assert token not in final["result"]
