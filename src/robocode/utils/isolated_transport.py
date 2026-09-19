"""Small fixed-destination stream relays; runnable with the container stdlib.

The container listeners connect ONLY to named Unix sockets. There is no network
bridge, DNS forwarding, SOCKS negotiation, CONNECT support, or destination field.
Host-side environment relays have one destination selected by trusted code.
"""

from __future__ import annotations

import json
import os
import pkgutil
import select
import signal
import socket
import socketserver
import subprocess
import sys
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Any


def copy_streams(left: socket.socket, right: socket.socket) -> None:
    """Copy duplex streams while preserving half-close semantics."""
    readable = [left, right]
    while readable:
        ready, _, _ = select.select(readable, [], [], 120)
        if not ready:
            return
        for source in ready:
            target = right if source is left else left
            data = source.recv(65536)
            if data:
                target.sendall(data)
            else:
                readable.remove(source)
                target.shutdown(socket.SHUT_WR)


class RelayHandler(socketserver.BaseRequestHandler):
    """Relay bytes to the one address configured by the trusted parent."""

    def handle(self) -> None:
        try:
            target = self.server.target  # type: ignore[attr-defined]
            if isinstance(target, str):
                remote = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                remote.settimeout(120)
                remote.connect(target)
            else:
                remote = socket.create_connection(target, timeout=120)
            with remote:
                copy_streams(self.request, remote)
        except OSError:
            pass


class UnixRelay(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Host endpoint pinned to a single loopback environment server."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, path: str, target: tuple[str, int]):
        self.target = target
        super().__init__(path, RelayHandler)


class TCPRelay(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Container loopback endpoint pinned to a mounted Unix socket."""

    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, port: int, target: str):
        self.target = target
        super().__init__(("127.0.0.1", port), RelayHandler)


def verify_namespace() -> None:
    """Refuse to execute any agent unless the kernel boundary is established."""
    if os.getuid() == 0 or {name for _, name in socket.if_nameindex()} != {"lo"}:
        raise RuntimeError(
            "Apptainer isolation requires a non-root, loopback-only namespace"
        )
    if len(Path("/proc/net/route").read_text(encoding="utf-8").splitlines()) != 1:
        raise RuntimeError("Unexpected route in isolated namespace")
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    )
    for key in ("CapEff", "CapPrm", "CapBnd", "CapInh", "CapAmb"):
        if int(status[key].strip(), 16):
            raise RuntimeError("Agent retains capabilities")
    if status["NoNewPrivs"].strip() != "1":
        raise RuntimeError("NoNewPrivs is required")


def verify_strict_runtime() -> None:
    """Reject old images and readable third-party Python package environments.

    This startup guard supplements the image audit; a virtualenv alone does not
    stop an agent from reading another interpreter's packages.
    """
    if (
        Path("/opt/robocode-mcp").exists()
        or not Path("/opt/robocode-render/strict_server.py").is_file()
    ):
        raise RuntimeError("Rebuild the strict image: legacy MCP environment is unsafe")
    roots = [Path("/opt"), Path("/usr/lib"), Path("/usr/local/lib")]
    package_dirs = [
        directory
        for root in roots
        for pattern in ("**/site-packages", "**/dist-packages")
        for directory in root.glob(pattern)
        if directory.is_dir()
    ]
    unexpected = {
        module.name
        for module in pkgutil.iter_modules([str(path) for path in package_dirs])
        if module.name not in {"numpy", "scipy"}
    }
    if unexpected:
        raise RuntimeError(f"Unexpected strict-image packages: {sorted(unexpected)}")


def main() -> None:
    """Start local relays only after verifying isolation, then supervise the CLI."""
    verify_namespace()
    config: dict[str, Any] = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if config.get("strict_blackbox"):
        verify_strict_runtime()
    with ExitStack() as stack:
        for listener in config["listeners"]:
            server = stack.enter_context(TCPRelay(listener["port"], listener["socket"]))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, 5)
            stack.callback(server.shutdown)
        child = subprocess.Popen(sys.argv[2:])  # pylint: disable=consider-using-with
        signal.signal(signal.SIGTERM, lambda *_: child.terminate())
        try:
            code = child.wait()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        raise SystemExit(code)


if __name__ == "__main__":
    main()
