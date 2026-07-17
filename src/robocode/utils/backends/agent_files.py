"""Sandbox agent-memory files (CLAUDE.md, AGENTS.md).

The project-memory files the coding agent reads inside its sandbox working
directory: CLAUDE.md for the Claude Code backend, AGENTS.md for OpenCode. Kept
here, dependency-free, so this sandbox-facing text lives in one evident place
instead of inline in the backend setup code.

The container variants add a context-management block (the sandbox agent has a
tight context window); the local variant is just the relative-paths rule.
OpenCode's block omits the source-reading-delegation items because its subagents
do not read source the way Claude Code's do.
"""

from __future__ import annotations

_RELATIVE_PATHS_CONTAINER = (
    "All files you create MUST use relative paths so they stay in the current "
    "working directory (/sandbox). Never write files using absolute paths."
)
_RELATIVE_PATHS_LOCAL = (
    "All files you create MUST use relative paths so they stay in the current "
    "working directory. Never write files using absolute paths."
)
_PRIMITIVES_NOTE = "Primitive source files (for reference) are in ./primitives/"
_RUN_INSTRUCTIONS = (
    "The Python interpreter is at {python}\n"
    "Run test scripts with:\n"
    "    {python} test_approach.py"
)

_CLAUDE_CONTEXT_MANAGEMENT = (
    "CONTEXT MANAGEMENT: your context window is limited and you MUST protect it "
    "aggressively: "
    "(1) Delegate ALL source code reading, exploration, and deep reasoning to "
    "subagents: have them return only concise summaries and actionable "
    "suggestions. "
    "(2) Never read large files directly; spawn a subagent to read and summarize "
    "them. "
    "(3) When running Bash commands, pipe output through `head` or `tail` to "
    "limit verbosity if possible. "
    "(4) Keep your thinking brief, do not write long reasoning traces. If you "
    "need to reason deeply about a design decision, delegate that to a subagent "
    "and have it return the conclusion. "
    "(5) Keep your main conversation focused on writing and testing code, not on "
    "reading source or reasoning at length."
)

_OPENCODE_CONTEXT_MANAGEMENT = (
    "CONTEXT MANAGEMENT: your context window is limited and you MUST protect it "
    "aggressively: "
    "(1) When running Bash commands, pipe output through `head` or `tail` to "
    "limit verbosity if possible. "
    "(2) Keep your thinking brief, do not write long reasoning traces. "
    "(3) Keep your main conversation focused on writing and testing code, not on "
    "reading source or reasoning at length."
)


def _container_body(
    context_management: str, docker_python: str, primitive_names: tuple[str, ...]
) -> str:
    body = "\n\n".join(
        (
            _RELATIVE_PATHS_CONTAINER,
            context_management,
            _RUN_INSTRUCTIONS.format(python=docker_python),
        )
    )
    body += "\n"
    if primitive_names:
        body += "\n" + _PRIMITIVES_NOTE + "\n"
    return body


def build_claude_md(docker_python: str, primitive_names: tuple[str, ...] = ()) -> str:
    """The CLAUDE.md body; container variant when docker_python is set, else local."""
    if not docker_python:
        return _RELATIVE_PATHS_LOCAL + "\n"
    return _container_body(_CLAUDE_CONTEXT_MANAGEMENT, docker_python, primitive_names)


def build_agents_md(docker_python: str, primitive_names: tuple[str, ...] = ()) -> str:
    """The AGENTS.md body (OpenCode); the caller prepends the system prompt."""
    if not docker_python:
        return _RELATIVE_PATHS_LOCAL + "\n"
    return _container_body(_OPENCODE_CONTEXT_MANAGEMENT, docker_python, primitive_names)
