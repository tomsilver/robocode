"""Static import allowlist for programs written under the strict blackbox.

In strict blackbox mode the agent develops in a generic image that holds only the
Python standard library and a few generic numerical packages. The frozen program is
scored on the host, whose environment has every RoboCode dependency installed, so a
program could in principle import at scoring time something it never had while it
was written (a speculative ``try: import shapely`` fallback, or an environment
class). This check closes that gap statically: before the program is loaded, every
import reachable from ``approach.py`` through its sibling modules must resolve to
the standard library, an allowed package, or another sibling file.

This is a cooperative guardrail with a loud failure, not an adversarial sandbox. Its
purpose is catching dependencies picked up by accident, so the direct ways to reach
a module by name at run time are rejected too: ``importlib``, ``runpy``,
``__import__``, and ``sys.modules``. Indirections such as ``getattr(sys,
"modules")``, ``os.sys``, or a symlinked sibling file are not detected.
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

# What the strict image installs beyond the standard library. The prompt, the
# Dockerfile, and the rejection message all describe exactly this set.
STRICT_ALLOWED_PACKAGES: frozenset[str] = frozenset({"networkx", "numpy", "scipy"})

# Standard-library modules that import by name at run time.
_DYNAMIC_IMPORT_MODULES: frozenset[str] = frozenset({"importlib", "runpy"})
# Those, plus the sandbox client, which has no server to talk to once scoring starts.
_FORBIDDEN_MODULES: frozenset[str] = _DYNAMIC_IMPORT_MODULES | {"env_client"}

STRICT_BLACKBOX_IMAGE = "robocode-strict-blackbox"
STRICT_BLACKBOX_PYTHON = "/opt/robocode-strict/bin/python"
STRICT_BLACKBOX_MCP_PYTHON = "/opt/robocode-mcp/bin/python"


class StrictImportError(ValueError):
    """A strict-blackbox program imports something outside its allowlist."""


def strict_allowlist_description() -> str:
    """The allowlist in prose, shared by the agent prompt and the rejection message."""
    packages = ", ".join(f"`{name}`" for name in sorted(STRICT_ALLOWED_PACKAGES))
    dynamic = ", ".join(f"`{name}`" for name in sorted(_DYNAMIC_IMPORT_MODULES))
    return (
        f"the Python standard library, {packages}, and sibling files "
        f"({dynamic}, `__import__`, and `sys.modules` are rejected)"
    )


def module_locations(module: ModuleType) -> list[Path]:
    """Resolved filesystem locations a loaded module runs from: file and package dirs.

    Empty for built-in and other location-less modules.
    """
    locations: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        locations.append(Path(module_file))
    locations.extend(Path(str(entry)) for entry in getattr(module, "__path__", ()))
    return [location.resolve() for location in locations]


def _module_files(candidate: Path) -> list[Path]:
    """Files a resolved sibling import can execute: a module, or a whole package.

    Importing any part of a sibling package makes every module in it reachable
    through its ``__init__`` or later imports, so the whole package is scanned.
    """
    if candidate.is_file():
        return [candidate]
    if (candidate / "__init__.py").is_file():
        return sorted(candidate.rglob("*.py"))
    return []


def _sibling_files(root: Path, head: str) -> list[Path]:
    """Files a top-level import of *head* loads from *root*; empty if not a sibling."""
    return _module_files(root / f"{head}.py") + _module_files(root / head)


def _relative_sibling_files(
    root: Path, path: Path, node: ast.ImportFrom
) -> list[Path] | None:
    """Files a relative import loads, or ``None`` when it climbs above *root*."""
    base = path.parent
    for _ in range(node.level - 1):
        base = base.parent
    if not base.is_relative_to(root):
        return None
    if node.module:
        base = base.joinpath(*node.module.split("."))
    files = _module_files(base.with_suffix(".py")) + _module_files(base)
    for alias in node.names:
        files += _module_files(base / f"{alias.name}.py") + _module_files(
            base / alias.name
        )
    return files


def _import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return [node.module or ""]


def _sys_aliases(tree: ast.AST) -> set[str]:
    """Names the file binds to the ``sys`` module."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "sys"
    }


def _cached_outside_root(name: str, root: Path) -> bool:
    """Whether *name* is cached as a module that did not come from *root*."""
    if name not in sys.modules:
        return False
    locations = module_locations(sys.modules[name])
    return not locations or any(not loc.is_relative_to(root) for loc in locations)


def check_strict_imports(
    entry: Path, *, reject_cached_siblings: bool = False
) -> set[str]:
    """Verify every import reachable from *entry* is stdlib, allowed, or a sibling.

    Returns the external top-level packages the program uses, for logging. Raises
    :class:`StrictImportError` naming each offending import and where it occurs.
    When *reject_cached_siblings* is true, also reject sibling names already bound
    to host modules, because ``sys.modules`` takes precedence over ``sys.path``.
    """
    root = entry.resolve().parent
    pending = [entry.resolve()]
    seen: set[Path] = set()
    external: set[str] = set()
    violations: list[str] = []
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        where = path.relative_to(root)
        sys_aliases = _sys_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "__import__":
                violations.append(f"{where}:{node.lineno}: __import__")
            if isinstance(node, ast.Attribute):
                if node.attr == "__import__":
                    violations.append(f"{where}:{node.lineno}: __import__")
                if (
                    node.attr == "modules"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in sys_aliases
                ):
                    violations.append(f"{where}:{node.lineno}: sys.modules")
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.ImportFrom) and node.level:
                relative = _relative_sibling_files(root, path, node)
                if relative is None:
                    violations.append(
                        f"{where}:{node.lineno}: relative import above the program"
                    )
                else:
                    pending.extend(relative)
                continue
            # ``import a, b`` is one node but two independent imports; validate each.
            for name in _import_names(node):
                head = name.split(".", 1)[0]
                if head in _FORBIDDEN_MODULES:
                    violations.append(f"{where}:{node.lineno}: import {head}")
                    continue
                if (
                    head == "sys"
                    and isinstance(node, ast.ImportFrom)
                    and any(alias.name == "modules" for alias in node.names)
                ):
                    violations.append(f"{where}:{node.lineno}: sys.modules")
                    continue
                siblings = _sibling_files(root, head)
                if siblings:
                    if reject_cached_siblings and _cached_outside_root(head, root):
                        violations.append(
                            f"{where}:{node.lineno}: cached host module {head} "
                            "overrides sibling import"
                        )
                    else:
                        pending.extend(siblings)
                    continue
                if head in sys.stdlib_module_names:
                    continue
                if head in STRICT_ALLOWED_PACKAGES:
                    external.add(head)
                    continue
                violations.append(f"{where}:{node.lineno}: import {head}")
    if violations:
        raise StrictImportError(
            "Strict blackbox program imports outside its allowlist of "
            f"{strict_allowlist_description()}:\n  " + "\n  ".join(violations)
        )
    logger.info(
        "Strict import check passed for %s: external packages %s",
        entry,
        sorted(external) or "none",
    )
    return external
