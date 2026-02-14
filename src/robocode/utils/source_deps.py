"""Resolve transitive local source dependencies for a Python file."""

from pathlib import Path


def collect_local_deps(source: Path, pkg_root: Path) -> list[Path]:
    """Collect source and its transitive ``robocode`` imports.

    Parses ``from robocode.x.y import ...`` and ``import robocode.x.y``
    statements, resolves them relative to *pkg_root*, and follows them
    recursively.

    Args:
        source: The starting source file (e.g. maze_env.py).
        pkg_root: The directory that contains the ``robocode`` package
            (i.e. the parent of the ``robocode/`` directory).

    Returns:
        Sorted list of resolved absolute paths for *source* and all of
        its transitive local dependencies.
    """
    collected: set[Path] = set()
    stack = [source.resolve()]
    while stack:
        current = stack.pop()
        if current in collected:
            continue
        collected.add(current)
        for line in current.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("from robocode."):
                # e.g. "from robocode.environments.base_env import BaseEnv"
                parts = stripped.split()[1].split(".")
                rel = Path(*parts[:-1]) / (parts[-1] + ".py")
                dep = (pkg_root / rel).resolve()
                if dep.exists():
                    stack.append(dep)
            elif stripped.startswith("import robocode."):
                # e.g. "import robocode.environments.base_env"
                parts = stripped.split()[1].rstrip(",").split(".")
                rel = Path(*parts[:-1]) / (parts[-1] + ".py")
                dep = (pkg_root / rel).resolve()
                if dep.exists():
                    stack.append(dep)
    return sorted(collected)
