"""Tests for the live Internet-access probe's offline transcript analysis."""

import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from integration_tests import check_agent_internet_access as probe
from integration_tests.check_agent_internet_access import (
    _stream_used_mcp,
    _stream_used_web_search,
    assess,
)


def test_stream_used_web_search_tolerates_non_json_lines(tmp_path: Path) -> None:
    """Blank lines and CLI noise do not prevent detecting a later event."""
    stream = tmp_path / "stream.jsonl"
    stream.write_text(
        '\nnot json\n{"type":"item.completed","item":{"type":"web_search"}}\n',
        encoding="utf-8",
    )
    assert _stream_used_web_search(stream)


def test_stream_used_web_search_handles_absent_event(tmp_path: Path) -> None:
    """A missing transcript and an event-free transcript both return false."""
    stream = tmp_path / "stream.jsonl"
    assert not _stream_used_web_search(stream)
    stream.write_text('\n{"type":"command_execution"}\nnoise\n', encoding="utf-8")
    assert not _stream_used_web_search(stream)


def test_stream_used_mcp_handles_codex_and_claude_events(tmp_path: Path) -> None:
    """Recognize MCP calls in the JSON formats emitted by both supported CLIs."""
    stream = tmp_path / "stream.jsonl"
    stream.write_text('{"item":{"type":"mcp_tool_call"}}\n', encoding="utf-8")
    assert _stream_used_mcp(stream)
    stream.write_text(
        '{"content":[{"type":"tool_use","name":"mcp__robocode__render_state"}]}\n',
        encoding="utf-8",
    )
    assert _stream_used_mcp(stream)


def test_web_search_mention_is_not_an_invocation(tmp_path: Path) -> None:
    """An explicit red-team request or refusal must not count as a tool call."""
    stream = tmp_path / "stream.jsonl"
    stream.write_text(
        '{"item":{"type":"agent_message","text":"web_search is disabled"}}\n',
        encoding="utf-8",
    )
    assert not _stream_used_web_search(stream)


def test_claude_web_fetch_is_an_invocation(tmp_path: Path) -> None:
    """Claude web fetch events must fail the same policy as Codex searches."""
    stream = tmp_path / "stream.jsonl"
    stream.write_text(
        '{"content":[{"type":"tool_use","name":"WebFetch"}]}\n', encoding="utf-8"
    )
    assert _stream_used_web_search(stream)


def test_blocked_self_report_is_inconclusive(tmp_path: Path, monkeypatch) -> None:
    """A model's BLOCKED claim cannot certify an operating-system boundary."""

    async def fake_run(config, _backend):
        config.sandbox_dir.mkdir()
        (config.sandbox_dir / "site_text.txt").write_text("BLOCKED", encoding="utf-8")
        return SimpleNamespace(success=True, error=None)

    monkeypatch.setattr(probe, "run_agent_in_apptainer_sandbox", fake_run)
    args = argparse.Namespace(
        results_dir=tmp_path,
        container="apptainer",
        backend="codex",
        model=None,
        max_budget_usd=1.0,
        strict_sif_path=None,
        probe="webpage",
        no_agent=False,
    )
    assert asyncio.run(probe._run(args)) == 2  # pylint: disable=protected-access


_NAMES = (
    "explicit_host_proxy",
    "tcp_host_ipv4",
    "tcp_host_ipv6",
    "udp_host",
    "unix_abstract_host",
    "http_host",
    "https_public",
    "tcp_public_ipv4",
    "tcp_public_ipv6",
    "dns_udp",
    "dns_tcp",
    "curl_public",
    "curl_direct_ip",
    "wget_public",
    "bash_tcp",
    "node_http",
    "pip_download",
    "git_https",
)


def _reports():
    control = {
        "netns": "host",
        "uid": 1013,
        "interfaces": [[1, "lo"], [2, "eth0"]],
        "routes_v4": "header\nroute\n",
        "results": {name: {"status": "reachable"} for name in _NAMES},
    }
    isolated = {
        "netns": "private",
        "uid": 1013,
        "interfaces": [[1, "lo"]],
        "routes_v4": "header\n",
        "results": {name: {"status": "blocked"} for name in _NAMES},
    }
    for name in ("raw_ipv4", "raw_ipv6", "route_add", "unix_path_host", "nsenter_pid1"):
        isolated["results"][name] = {"status": "blocked"}
    isolated["security"] = [
        f"{name}: 0000000000000000"
        for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    ] + ["NoNewPrivs: 1"]
    isolated["results"]["own_loopback"] = {"status": "reachable"}
    return control, isolated


def test_working_controls_and_private_namespace_pass():
    """Working endpoints and a private namespace establish the tested boundary."""
    control, isolated = _reports()
    result = assess(control, isolated, "host")
    assert not result["failures"]
    assert not result["inconclusive"]
    assert set(result["passed"]) == set(_NAMES)


@pytest.mark.parametrize(
    "name", _NAMES + ("raw_ipv4", "raw_ipv6", "route_add", "unix_path_host")
)
def test_any_successful_escape_fails(name):
    """Any reachable forbidden endpoint invalidates isolation."""
    control, isolated = _reports()
    isolated["results"][name]["status"] = "reachable"
    assert name in assess(control, isolated, "host")["failures"]


@pytest.mark.parametrize("status", ["missing", "error", "timeout", "blocked", "failed"])
def test_unreachable_public_control_is_inconclusive(status):
    """A failed positive control cannot prove a negative."""
    control, isolated = _reports()
    control["results"]["curl_public"]["status"] = status
    result = assess(control, isolated, "host")
    assert "curl_public" in result["inconclusive"]
    assert "curl_public" not in result["passed"]


def test_missing_pip_is_not_a_network_block():
    """Absent package tooling must remain inconclusive."""
    control, isolated = _reports()
    isolated["results"]["pip_download"] = {"status": "missing"}
    result = assess(control, isolated, "host")
    assert "pip_download" in result["inconclusive"]


def test_dead_local_canary_invalidates_audit():
    """A broken owned endpoint invalidates the test setup."""
    control, isolated = _reports()
    control["results"]["tcp_host_ipv4"] = {"status": "blocked"}
    assert (
        "invalid_control_tcp_host_ipv4" in assess(control, isolated, "host")["failures"]
    )


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("netns", "host", "isolated_still_shares_host_network"),
        ("interfaces", [[1, "lo"], [2, "eth0"]], "isolated_has_external_interfaces"),
        ("routes_v4", "header\nroute\n", "isolated_has_ipv4_routes"),
        ("uid", 0, "agent_is_root"),
    ],
)
def test_structural_boundary_is_required(field, value, expected):
    """An apparent connectivity block cannot replace namespace invariants."""
    control, isolated = _reports()
    isolated[field] = value
    assert expected in assess(control, isolated, "host")["failures"]


def test_nsenter_must_not_recover_host_namespace():
    """Rejoining the same namespace is harmless; reaching the host is a breach."""
    control, isolated = _reports()
    isolated["results"]["nsenter_pid1"] = {"status": "reachable", "detail": "host\n"}
    assert "nsenter_escaped" in assess(control, isolated, "host")["failures"]
    isolated["results"]["nsenter_pid1"]["detail"] = "private\n"
    assert "nsenter_escaped" not in assess(control, isolated, "host")["failures"]


def test_broken_local_socket_environment_cannot_pass():
    """Outer socket restrictions must not masquerade as Apptainer isolation."""
    control, isolated = _reports()
    isolated["results"]["own_loopback"] = {"status": "blocked"}
    assert (
        "invalid_isolated_loopback_control"
        in assess(control, isolated, "host")["failures"]
    )


def test_capabilities_invalidate_isolation_assessment():
    """An agent with retained capabilities fails the boundary check."""
    control, isolated = _reports()
    isolated["security"][0] = "CapInh: 0000000000001000"
    assert "agent_retains_capabilities" in assess(control, isolated, "host")["failures"]


@pytest.mark.parametrize("no_agent", [False, True])
def test_apptainer_defaults_to_controlled_audit(tmp_path, monkeypatch, no_agent):
    """The original CLI forwards image, backend, and budget choices to the audit."""
    calls = []
    monkeypatch.setattr(
        probe, "run_network_audit", lambda *a, **kw: calls.append((a, kw))
    )
    args = argparse.Namespace(
        probe=None,
        container="apptainer",
        no_agent=no_agent,
        results_dir=tmp_path / "results",
        image_dir=tmp_path / "images",
        backend="claude",
        model="test-model",
        max_budget_usd=0.75,
        strict_sif_path=tmp_path / "strict.sif",
    )
    assert asyncio.run(probe._run(args)) == 0  # pylint: disable=protected-access
    assert calls == [
        (
            (args.results_dir, args.image_dir, None if no_agent else "claude"),
            {
                "model": "test-model",
                "max_budget_usd": 0.75,
                "strict_sif_path": args.strict_sif_path,
            },
        )
    ]


def test_docker_keeps_webpage_probe(monkeypatch):
    """Existing Docker commands retain their live webpage behavior."""
    calls = []

    async def webpage(args):
        calls.append(args)
        return 2

    monkeypatch.setattr(probe, "_run_webpage_probe", webpage)
    args = argparse.Namespace(probe=None, container="docker", no_agent=False)
    assert asyncio.run(probe._run(args)) == 2  # pylint: disable=protected-access
    assert calls == [args]
