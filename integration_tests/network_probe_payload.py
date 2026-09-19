"""Dependency-free, deterministic probes executed INSIDE the test container."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from http import client as http_client
from pathlib import Path
from typing import Any


def broker_probes() -> dict[str, int]:
    """Attack both local TCP and direct Unix access to the host broker."""
    responses = {}
    requests = [
        ("CONNECT", "example.com:443", None),
        ("GET", "/v1/responses", None),
        ("POST", "https://example.com/v1/responses", {"model": "x"}),
        (
            "POST",
            "/v1/responses",
            {"model": "x", "input": "search", "tools": [{"type": "web_search"}]},
        ),
        (
            "POST",
            "/v1/messages",
            {
                "model": "x",
                "messages": [],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            },
        ),
        (
            "POST",
            "/v1/responses",
            {
                "model": "x",
                "input": [
                    {"type": "input_image", "image_url": "https://example.com/p.png"}
                ],
            },
        ),
        (
            "POST",
            "/v1/messages",
            {"model": "x", "mcp_servers": [{"url": "https://example.com"}]},
        ),
    ]
    for transport in ("tcp", "unix"):
        for index, (method, path, body) in enumerate(requests):
            conn = http_client.HTTPConnection("127.0.0.1", 18080, timeout=10)
            if transport == "unix":
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect("/run/robocode-broker/model.sock")
                conn.sock = sock
            conn.request(
                method, path, body=json.dumps(body) if body is not None else None
            )
            response = conn.getresponse()
            responses[f"{transport}_{index}"] = response.status
            response.read()
            conn.close()
    return responses


def probe(config: dict) -> dict:
    """Collect actual operations, never treating absent tools as blocked access."""
    results: dict[str, dict[str, Any]] = {}

    def attempt(name: str, fn: Callable[[], object]) -> None:
        try:
            detail = fn()
            results[name] = {"status": "reachable", "detail": str(detail)[:300]}
        except OSError as exc:
            results[name] = {
                "status": "blocked",
                "detail": str(exc),
                "errno": exc.errno,
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            results[name] = {"status": "error", "detail": repr(exc)}

    def tcp(host: str, port: int) -> object:
        with socket.create_connection((host, port), timeout=3) as sock:
            return sock.getpeername()

    def udp(host: str, port: int) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(3)
            sock.sendto(config["nonce"].encode(), (host, port))
            return sock.recv(100).decode()

    def unix(address: str) -> str:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            sock.connect(address)
            return sock.recv(100).decode()

    def http(url: str) -> object:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=5) as response:
            return response.status, response.read(100)

    def command(name: str, args: list[str], timeout: int = 20) -> None:
        if not shutil.which(args[0]):
            results[name] = {"status": "missing", "detail": args[0]}
            return
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout, check=False
            )
            results[name] = {
                "status": "reachable" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "detail": (proc.stdout + proc.stderr)[-1500:],
            }
        except subprocess.TimeoutExpired:
            results[name] = {"status": "timeout"}

    def own_loopback() -> object:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            return tcp("127.0.0.1", listener.getsockname()[1])

    attempt("own_loopback", own_loopback)
    command(
        "explicit_host_proxy",
        [
            "curl",
            "--noproxy",
            "",
            "--proxy",
            config["http_url"],
            "-fsS",
            "--max-time",
            "6",
            "http://network-audit.invalid/",
            "-o",
            "/dev/null",
        ],
    )
    attempt("tcp_host_ipv4", lambda: tcp("127.0.0.1", config["tcp_port"]))
    if config["tcp6_port"]:
        attempt("tcp_host_ipv6", lambda: tcp("::1", config["tcp6_port"]))
    attempt("udp_host", lambda: udp("127.0.0.1", config["udp_port"]))
    attempt("unix_abstract_host", lambda: unix("\0" + config["abstract_name"]))
    attempt("unix_path_host", lambda: unix(config["unix_path"]))
    attempt("http_host", lambda: http(config["http_url"]))
    attempt("https_public", lambda: http("https://pypi.org/simple/pip/"))
    attempt("tcp_public_ipv4", lambda: tcp(config["public_ipv4"], 443))
    attempt("tcp_public_ipv6", lambda: tcp("2606:4700:4700::1111", 443))

    def dns() -> str:
        # A real DNS query (example.com A), sent without the system resolver.
        packet = bytes.fromhex(
            "123401000001000000000000076578616d706c6503636f6d0000010001"
        )
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(3)
            sock.sendto(packet, (config["dns_server"], 53))
            return sock.recv(512).hex()

    attempt("dns_udp", dns)
    attempt("dns_tcp", lambda: tcp(config["dns_server"], 53))
    command(
        "curl_public",
        [
            "curl",
            "--noproxy",
            "*",
            "-fsS",
            "--max-time",
            "6",
            "https://pypi.org/simple/pip/",
            "-o",
            "/dev/null",
        ],
    )
    command(
        "curl_direct_ip",
        [
            "curl",
            "--noproxy",
            "*",
            "--resolve",
            f"pypi.org:443:{config['public_ipv4']}",
            "-fsS",
            "--max-time",
            "6",
            "https://pypi.org/simple/pip/",
            "-o",
            "/dev/null",
        ],
    )
    command(
        "wget_public",
        [
            "wget",
            "--no-proxy",
            "-q",
            "-T",
            "6",
            "-t",
            "1",
            "-O",
            "/dev/null",
            "https://pypi.org/simple/pip/",
        ],
    )
    command(
        "bash_tcp", ["bash", "-c", f"exec 3<>/dev/tcp/127.0.0.1/{config['tcp_port']}"]
    )
    command(
        "node_http",
        [
            "node",
            "-e",
            f"fetch({json.dumps(config['http_url'])})"
            ".then(r=>{if(!r.ok)process.exit(2)})"
            ".catch(e=>{console.error(e);process.exit(1)})",
        ],
    )
    # The regular uv venv intentionally lacks pip. Find an installed pip rather
    # than counting "No module named pip" as a network block.
    pip_python = next(
        (
            p
            for p in (
                sys.executable,
                "/opt/robocode-strict/bin/python",
                "/usr/bin/python3",
                "/usr/local/bin/python3",
            )
            if Path(p).exists()
            and subprocess.run(
                [p, "-m", "pip", "--version"], capture_output=True, check=False
            ).returncode
            == 0
        ),
        None,
    )
    if pip_python is None:
        results["pip_download"] = {"status": "missing", "detail": "pip"}
    else:
        command(
            "pip_download",
            [
                pip_python,
                "-m",
                "pip",
                "--isolated",
                "download",
                "--no-cache-dir",
                "--no-deps",
                "--disable-pip-version-check",
                "--retries",
                "0",
                "--timeout",
                "5",
                "--index-url",
                "https://pypi.org/simple",
                "--dest",
                "/tmp/network-probe-download",
                "six==1.17.0",
            ],
        )
    command(
        "git_https",
        [
            "git",
            "-c",
            "http.proxy=",
            "ls-remote",
            "https://github.com/pypa/sampleproject.git",
            "HEAD",
        ],
        timeout=12,
    )

    def raw(family: int) -> str:
        with socket.socket(family, socket.SOCK_RAW, socket.IPPROTO_RAW):
            return "raw socket created"

    attempt("raw_ipv4", lambda: raw(socket.AF_INET))
    attempt("raw_ipv6", lambda: raw(socket.AF_INET6))
    command("route_add", ["ip", "route", "add", "default", "dev", "lo"])
    command(
        "nsenter_pid1",
        ["nsenter", "--net=/proc/1/ns/net", "--", "readlink", "/proc/self/ns/net"],
    )
    # A nested user/net namespace cannot restore an ancestor's network access.
    command(
        "nested_namespace",
        [
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "sh",
            "-c",
            "cat /proc/net/route",
        ],
    )
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    return {
        "uid": os.getuid(),
        "netns": os.readlink("/proc/self/ns/net"),
        "interfaces": socket.if_nameindex(),
        "routes_v4": Path("/proc/net/route").read_text(encoding="utf-8"),
        "routes_v6": Path("/proc/net/ipv6_route").read_text(encoding="utf-8"),
        "security": [
            line
            for line in status.splitlines()
            if line.startswith(("Cap", "NoNewPrivs"))
        ],
        "results": results,
        "broker_rejections": broker_probes() if config.get("test_broker") else {},
    }


if __name__ == "__main__":
    print(
        json.dumps(
            probe(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))), indent=2
        )
    )
