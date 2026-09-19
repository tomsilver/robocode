"""Prevent false isolation passes from missing probes or broken controls."""

import pytest

from integration_tests.apptainer_network_audit import assess

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
