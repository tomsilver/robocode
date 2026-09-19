"""Standalone strict render tools using only stdlib and the numerical env client.

Copied into the strict image as a plain script, never as a robocode package.
The small stateless MCP HTTP surface deliberately has no framework environment
that an agent or rendered policy could import. Policies execute in this same
isolated container. The host receives only the existing environment protocol.

Transport: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

VERSIONS = ("2025-03-26", "2025-06-18", "2025-11-25")
PROPERTIES: dict[str, dict[str, Any]] = {
    "seed": {"type": "integer", "default": 42},
    "object_count": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
    "state": {
        "anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}],
        "default": None,
    },
    "label": {"type": "string", "default": ""},
    "approach_dir": {"type": "string", "default": "."},
    "max_steps": {"type": "integer", "default": 1000},
    "max_frames": {"type": "integer", "default": 100},
}
TOOL_ARGUMENTS = {
    "render_state": ("seed", "state", "label", "object_count"),
    "render_policy": (
        "approach_dir",
        "seed",
        "max_steps",
        "max_frames",
        "object_count",
    ),
}
TOOL_DESCRIPTIONS = {
    "render_state": (
        "Render a reset state (seed) or observation vector (state) "
        "as a PNG. Returns the saved image path."
    ),
    "render_policy": (
        "Run approach_dir/approach.py inside the isolated container "
        "and save episode frames as PNGs. Returns saved image paths."
    ),
}


class RenderTools:
    """The strict environment client is the only dependency beyond stdlib."""

    def __init__(self, metadata: Path, tools: list[str]):
        if not set(tools) <= TOOL_ARGUMENTS.keys():
            raise ValueError("Unknown strict render tool")
        self.metadata = metadata.resolve()
        self.tools = tools

    def list_tools(self) -> list[dict[str, Any]]:
        """Describe the two fixed tools without a schema-generation dependency."""
        return [
            {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        key: PROPERTIES[key] for key in TOOL_ARGUMENTS[name]
                    },
                    "additionalProperties": False,
                },
            }
            for name in self.tools
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> str | list[str]:
        """Use a fresh connection per call; policies never share a host process."""
        # env_client is installed beside this standalone script, not in a project
        # package. It contains only generic protocol/observation handling.
        # pylint: disable=import-outside-toplevel,import-error
        from env_client import BlackboxEnv  # type: ignore[import-not-found]

        # pylint: enable=import-outside-toplevel,import-error

        if name not in self.tools or not set(arguments) <= set(TOOL_ARGUMENTS[name]):
            raise ValueError("Unknown tool or arguments")
        root = self.metadata.parent
        meta = json.loads(self.metadata.read_text(encoding="utf-8"))
        if meta.get("strict") is not True:
            raise ValueError("Strict rendering requires strict environment metadata")
        with BlackboxEnv(meta, sandbox_root=root) as client:
            if name == "render_state":
                return str(root / client.render_state(**arguments))
            kwargs = dict(arguments)
            approach_dir = kwargs.pop("approach_dir", ".")
            paths = client.render_policy(
                approach_path=root / approach_dir / "approach.py", **kwargs
            )
            return [str(root / path) for path in paths]

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle MCP lifecycle and tool calls; no resources, prompts, or proxies."""
        if "id" not in request:
            return None
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": request["id"]}
        method, params = request.get("method"), request.get("params", {})
        if method == "initialize":
            version = params.get("protocolVersion")
            response["result"] = {
                "protocolVersion": version if version in VERSIONS else VERSIONS[-1],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "robocode-tools", "version": "1.0"},
            }
        elif method == "ping":
            response["result"] = {}
        elif method == "tools/list":
            response["result"] = {"tools": self.list_tools()}
        elif method == "tools/call":
            try:
                value = self.call(params["name"], params.get("arguments", {}))
                response["result"] = {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                value if isinstance(value, str) else json.dumps(value)
                            ),
                        }
                    ],
                    "isError": False,
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logging.exception("Render tool failed")
                response["result"] = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
        else:
            response["error"] = {"code": -32601, "message": "Method not found"}
        return response


def serve(tools: RenderTools, host: str, port: int) -> None:
    """Serve JSON responses on the MCP HTTP endpoint; optional SSE is unsupported."""
    if host != "127.0.0.1":
        raise ValueError("Strict MCP must bind only to loopback")

    class Handler(BaseHTTPRequestHandler):
        """No files, uploads, URL fetching, or arbitrary RPC dispatch."""

        def reply(self, status: int, value: Any = None) -> None:
            """Write one JSON response with an explicit length."""
            payload = b"" if value is None else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def valid(self) -> bool:
            """Reject unrelated origins, paths, and protocol versions."""
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{port}":
                self.reply(403)
                return False
            if urlsplit(self.path).path != "/mcp":
                self.reply(404)
                return False
            version = self.headers.get("MCP-Protocol-Version")
            if version and version not in VERSIONS:
                self.reply(400)
                return False
            return True

        def do_POST(self) -> None:  # pylint: disable=invalid-name
            """Handle one bounded JSON-RPC message."""
            if not self.valid():
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                    raise ValueError("Expected JSON-RPC object")
                response = tools.dispatch(request)
            except (ValueError, TypeError, KeyError):
                self.reply(400)
                return
            self.reply(202 if response is None else 200, response)

        def do_GET(self) -> None:  # pylint: disable=invalid-name
            """This stateless server has no optional server-to-client stream."""
            if self.valid():
                self.reply(405)

        do_DELETE = do_GET

    with ThreadingHTTPServer((host, port), Handler) as server:
        server.serve_forever()


def main() -> None:
    """Start the only strict render server, under the strict Python interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-spaces", type=Path, required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--transport", choices=["http"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    serve(RenderTools(args.env_spaces, args.tools.split(",")), args.host, args.port)


if __name__ == "__main__":
    main()
