"""Static import allowlist for programs written under the strict blackbox.

In strict blackbox mode the agent develops in a generic image that holds only the
Python standard library, NumPy, and SciPy. The frozen program is scored on the host,
whose environment has every RoboCode dependency installed, so a program could in
principle import at scoring time something it never had while it was written (a
speculative ``try: import shapely`` fallback, or an environment class). This check
closes that gap statically: before the program is loaded, every import reachable
from ``approach.py`` through its sibling modules must resolve to the standard
library, an allowed package, or another sibling file.

This is a cooperative guardrail with a loud failure, not an adversarial sandbox:
``importlib``, ``runpy``, and ``__import__`` are rejected because they are the
direct ways around a static check, and nothing more elaborate is attempted.
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# What the strict image installs beyond the standard library. The prompt and the
# Dockerfile describe exactly this set.
STRICT_ALLOWED_PACKAGES: frozenset[str] = frozenset({"numpy", "scipy"})

# Modules that would let a program import by name at run time, and the sandbox
# client, which has no server to talk to once scoring starts.
_FORBIDDEN_MODULES: frozenset[str] = frozenset({"importlib", "runpy", "env_client"})

STRICT_BLACKBOX_IMAGE = "robocode-strict-blackbox"
STRICT_BLACKBOX_PYTHON = "/opt/robocode-strict/bin/python"
STRICT_BLACKBOX_MCP_PYTHON = "/opt/robocode-mcp/bin/python"


class StrictImportError(ValueError):
    """A strict-blackbox program imports something outside its allowlist."""


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


def _resolve_sibling(
    root: Path,
    path: Path,
    node: ast.Import | ast.ImportFrom,
    names: list[str] | None = None,
) -> list[Path] | None:
    """Sibling files an import node loads, or ``None`` if it is not a sibling."""
    files: list[Path] = []
    if isinstance(node, ast.ImportFrom) and node.level:
        # Relative import: anchor at the importing file's package directory.
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        if not base.is_relative_to(root):
            return None
        if node.module:
            base = base.joinpath(*node.module.split("."))
        files += _module_files(base.with_suffix(".py")) + _module_files(base)
        for alias in node.names:
            files += _module_files(base / f"{alias.name}.py")
            files += _module_files(base / alias.name)
        return files or None
    for name in names if names is not None else _import_names(node):
        head = name.split(".", 1)[0]
        files += _module_files(root / f"{head}.py") + _module_files(root / head)
    return files or None


def _import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return [node.module or ""]


def _cached_outside_root(name: str, root: Path) -> bool:
    """Whether *name* is cached as a module that did not come from *root*."""
    if name not in sys.modules:
        return False
    module = sys.modules[name]
    locations: list[str] = []
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        locations.append(module_file)
    module_path = getattr(module, "__path__", ()) or ()
    locations.extend(str(location) for location in module_path)
    if not locations:
        return True
    return any(
        not Path(location).resolve().is_relative_to(root) for location in locations
    )


def check_strict_imports(
    entry: Path, *, reject_cached_siblings: bool = False
) -> set[str]:
    """Verify every import reachable from *entry* is stdlib, allowed, or a sibling.

    Returns the external top-level modules the program uses, for logging. Raises
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
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "__import__":
                violations.append(f"{where}:{node.lineno}: __import__")
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            # ``import a, b`` is one AST node but performs two independent imports.
            # Resolve and validate each alias separately so one local sibling cannot
            # make another, host-only alias look local too.  ImportFrom has one module
            # target regardless of how many names it imports.
            imported_names = _import_names(node)
            name_groups = (
                [[name] for name in imported_names]
                if isinstance(node, ast.Import)
                else [imported_names]
            )
            for names in name_groups:
                heads = {name.split(".", 1)[0] for name in names}
                forbidden = sorted(heads & _FORBIDDEN_MODULES)
                if forbidden:
                    violations.append(f"{where}:{node.lineno}: import {forbidden[0]}")
                    continue
                siblings = _resolve_sibling(root, path, node, names)
                if siblings is not None:
                    if reject_cached_siblings:
                        conflicts = sorted(
                            head
                            for head in heads
                            if head and _cached_outside_root(head, root)
                        )
                        if conflicts:
                            violations.append(
                                f"{where}:{node.lineno}: cached host module "
                                f"{conflicts[0]} overrides sibling import"
                            )
                            continue
                    pending.extend(siblings)
                    continue
                for head in sorted(heads):
                    if head in sys.stdlib_module_names:
                        continue
                    if head in STRICT_ALLOWED_PACKAGES:
                        external.add(head)
                        continue
                    violations.append(f"{where}:{node.lineno}: import {head}")
    if violations:
        allowed = ", ".join(sorted(STRICT_ALLOWED_PACKAGES))
        raise StrictImportError(
            "Strict blackbox program imports outside its allowlist (standard "
            f"library, {allowed}, and sibling files):\n  " + "\n  ".join(violations)
        )
    logger.info(
        "Strict import check passed for %s: external packages %s",
        entry,
        sorted(external) or "none",
    )
    return external
