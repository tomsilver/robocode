"""Strict rendering must work without importing project or MCP dependencies."""

import asyncio
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from robocode.mcp.strict_server import RenderTools


def test_only_render_capabilities(tmp_path):
    """Unknown RPCs cannot reach arbitrary client/server attributes."""
    tools = RenderTools(tmp_path / "env_spaces.json", ["render_state", "render_policy"])
    assert {t["name"] for t in tools.list_tools()} == {"render_state", "render_policy"}
    assert tools.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"})[
        "result"
    ]["capabilities"] == {"tools": {"listChanged": False}}
    assert tools.dispatch({"id": 2, "method": "getattr"})["error"]["code"] == -32601
    assert tools.dispatch({"method": "notifications/initialized"}) is None
    with pytest.raises(ValueError):
        RenderTools(tmp_path / "meta", ["execute_python"])


def test_official_mcp_client_interoperability(tmp_path):
    """Exercise initialize, tools/list, success/error calls through the SDK client."""
    source = Path(__file__).resolve().parents[2] / "src/robocode/mcp/strict_server.py"
    (tmp_path / "strict_server.py").write_bytes(source.read_bytes())
    (tmp_path / "env_spaces.json").write_text(json.dumps({"strict": True}))
    (tmp_path / "env_client.py").write_text("""class BlackboxEnv:
 def __init__(self, *args, **kwargs): pass
 def __enter__(self): return self
 def __exit__(self, *args): pass
 def render_state(self, **kwargs): return 'state.png'
 def render_policy(self, **kwargs): return ['frame.png']
""")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with subprocess.Popen(
        [
            sys.executable,
            str(tmp_path / "strict_server.py"),
            "--env-spaces",
            str(tmp_path / "env_spaces.json"),
            "--tools",
            "render_state,render_policy",
            "--transport",
            "http",
            "--port",
            str(port),
            "--log-file",
            str(tmp_path / "server.log"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as process:

        async def check():
            for _ in range(100):
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    del reader
                    writer.close()
                    await writer.wait_closed()
                    break
                except OSError:
                    await asyncio.sleep(0.05)
            async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    assert len((await session.list_tools()).tools) == 2
                    state = await session.call_tool("render_state", {"seed": 3})
                    assert not state.isError and "state.png" in state.content[0].text
                    policy = await session.call_tool("render_policy", {"max_steps": 2})
                    assert not policy.isError and "frame.png" in policy.content[0].text
                    bad = await session.call_tool("arbitrary_command", {})
                    assert bad.isError

        try:
            asyncio.run(check())
        finally:
            process.terminate()
            process.wait(timeout=5)
