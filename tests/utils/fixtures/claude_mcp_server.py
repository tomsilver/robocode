"""Harmless MCP tool for the Docker Claude isolation tests."""

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("probe")


@mcp.tool()
def probe_token() -> str:
    """Return the secret test token."""
    with Path("/sandbox/calls.txt").open("a", encoding="utf-8") as log:
        log.write("called\n")
    return sys.argv[1]


if __name__ == "__main__":
    mcp.run(transport="stdio")
