"""Resolve transitive local source dependencies for a Python file."""

from pathlib import Path


def collect_local_deps(
    source: Path, pkg_root: Path, package: str = "robocode"
) -> list[Path]:
    """Collect source and its transitive imports from a single package.

    Parses ``from <package>.x.y import ...`` and ``import <package>.x.y``
    statements, resolves them relative to *pkg_root*, and follows them
    recursively. The parser is line-based and only matches simple import
    forms, so the result is a best-effort local source bundle, not a
    guaranteed-complete transitive closure.

    Args:
        source: The starting source file (e.g. maze_env.py).
        pkg_root: The directory that contains the *package* directory
            (i.e. the parent of the ``<package>/`` directory).
        package: The top-level package whose imports are followed (e.g.
            ``"robocode"`` or ``"kinder"``).

    Returns:
        Sorted list of resolved absolute paths for *source* and all of
        its discovered local dependencies.
    """
    from_prefix = f"from {package}."
    import_prefix = f"import {package}."
    collected: set[Path] = set()
    stack = [source.resolve()]
    while stack:
        current = stack.pop()
        if current in collected:
            continue
        collected.add(current)
        for line in current.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(from_prefix):
                # e.g. "from robocode.environments.base_env import BaseEnv"
                parts = stripped.split()[1].split(".")
                rel = Path(*parts[:-1]) / (parts[-1] + ".py")
                dep = (pkg_root / rel).resolve()
                if dep.exists():
                    stack.append(dep)
            elif stripped.startswith(import_prefix):
                # e.g. "import robocode.environments.base_env"
                parts = stripped.split()[1].rstrip(",").split(".")
                rel = Path(*parts[:-1]) / (parts[-1] + ".py")
                dep = (pkg_root / rel).resolve()
                if dep.exists():
                    stack.append(dep)
    return sorted(collected)
