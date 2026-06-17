"""Rendering helpers for visual debugging (agent tooling, not policy primitives).

Split out of the ``robocode.primitives`` package so the normal-mode MCP render
server can render without importing that package (which is stripped from the
agentic sandbox). These modules depend only on the environments (present in
normal mode) and the episode-running helpers, never on the primitive registry.
"""

from __future__ import annotations

from robocode.rendering.render_policy import render_policy
from robocode.rendering.render_state import render_state

__all__ = ["render_policy", "render_state"]
