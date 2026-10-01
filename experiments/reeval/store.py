"""Checkpoint store layout shared by prepare, the workers and collect.

A store holds one directory per run, ``runs/<run_key>/``, with ``repo.bundle`` (the
sandbox git history as one file) and ``run.json`` (eval settings, commit list and the
original results). Outputs go to ``<out>/<run_key>/<tree>.json``, one per distinct
sandbox tree, so a checkpoint is evaluated once however many commits share it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_FILE = "run.json"
BUNDLE_FILE = "repo.bundle"


@dataclass(frozen=True)
class Task:
    """One sandbox tree of one run to evaluate on the held-out suite."""

    run_key: str
    tree: str
    est_seconds: float


def load_run(store: Path, run_key: str) -> dict[str, Any]:
    """Read one run's metadata."""
    return json.loads((store / "runs" / run_key / RUN_FILE).read_text())


def iter_runs(store: Path) -> list[dict[str, Any]]:
    """Return every run in the store, sorted by key."""
    paths = sorted((store / "runs").glob(f"*/{RUN_FILE}"))
    return [json.loads(p.read_text()) for p in paths]


def checkpoint_trees(run: dict[str, Any]) -> list[str]:
    """Distinct trees that contain approach.py, in commit order."""
    trees: list[str] = []
    for commit in run["commits"]:
        if commit["has_approach"] and commit["tree"] not in trees:
            trees.append(commit["tree"])
    return trees


def final_tree(run: dict[str, Any]) -> str | None:
    """Tree of the last commit that contains approach.py."""
    trees = [c["tree"] for c in run["commits"] if c["has_approach"]]
    return trees[-1] if trees else None


def list_tasks(store: Path, select: str) -> list[Task]:
    """All tasks in the store; ``select`` is ``all`` or ``final``."""
    tasks: list[Task] = []
    for run in iter_runs(store):
        if select == "final":
            final = final_tree(run)
            trees = [final] if final else []
        else:
            trees = checkpoint_trees(run)
        tasks.extend(Task(run["run_key"], t, run["est_eval_seconds"]) for t in trees)
    return tasks


def output_path(out: Path, task: Task) -> Path:
    """Where the record for ``task`` lives."""
    return out / task.run_key / f"{task.tree}.json"


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
