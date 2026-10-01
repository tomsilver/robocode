"""Merge checkpoint records into tables and check them against the original runs.

Writes ``checkpoints.csv`` (one row per commit of every run, with the score of its
tree) and ``validation.csv`` (final checkpoint against the run's own results.json).

Example:
    python -m experiments.reeval.collect --store <store> --out <out> --dest <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.reeval.store import Task, final_tree, iter_runs, output_path


def _record(out: Path, run_key: str, tree: str) -> dict[str, Any] | None:
    path = output_path(out, Task(run_key, tree, 0.0))
    return json.loads(path.read_text()) if path.is_file() else None


def checkpoint_rows(run: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    """One row per commit; commits sharing a tree share its score."""
    rows = []
    for index, commit in enumerate(run["commits"]):
        record = None
        if commit["has_approach"]:
            record = _record(out, run["run_key"], commit["tree"])
        results = (record or {}).get("results") or {}
        if not commit["has_approach"]:
            status = "no_approach"
        else:
            status = record["status"] if record else "pending"
        rows.append(
            {
                "run_key": run["run_key"],
                "experiment_dir": run["experiment_dir"],
                "experiment_id": run["experiment_id"],
                "replicate_seed": run["replicate_seed"],
                "commit_index": index,
                "num_commits": len(run["commits"]),
                "sha": commit["sha"],
                "tree": commit["tree"],
                "time": commit["time"],
                "subject": commit["subject"],
                "status": status,
                "solve_rate": results.get("solve_rate"),
                "num_crashed_episodes": results.get("num_crashed_episodes"),
                "eval_complete": results.get("eval_complete"),
                "wall_seconds": (record or {}).get("wall_seconds"),
                "cpu_model": (record or {}).get("cpu_model"),
            }
        )
    return rows


def validation_row(run: dict[str, Any], out: Path) -> dict[str, Any]:
    """Final checkpoint's re-evaluation against the original results.json."""
    tree = final_tree(run)
    record = _record(out, run["run_key"], tree) if tree else None
    results = (record or {}).get("results") or {}
    original = run.get("original") or {}
    old = original.get("solved") or []
    new = [e.get("solved") for e in results.get("per_episode", [])]
    same = sum(a == b for a, b in zip(old, new)) if old and new else None
    return {
        "run_key": run["run_key"],
        "status": record["status"] if record else "pending",
        "original_solve_rate": original.get("solve_rate"),
        "reeval_solve_rate": results.get("solve_rate"),
        "episodes_compared": min(len(old), len(new)) if same is not None else 0,
        "episodes_agreeing": same,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    runs = iter_runs(args.store)
    checkpoints = pd.DataFrame(
        [row for run in runs for row in checkpoint_rows(run, args.out)]
    )
    validation = pd.DataFrame([validation_row(run, args.out) for run in runs])
    args.dest.mkdir(parents=True, exist_ok=True)
    checkpoints.to_csv(args.dest / "checkpoints.csv", index=False)
    validation.to_csv(args.dest / "validation.csv", index=False)

    print(checkpoints["status"].value_counts().to_string())
    done = validation[validation["reeval_solve_rate"].notna()]
    if not done.empty:
        delta = done["reeval_solve_rate"] - done["original_solve_rate"]
        exact = (done["episodes_agreeing"] == done["episodes_compared"]).sum()
        print(
            f"final checkpoints re-evaluated: {len(done)}; identical per-episode "
            f"outcomes: {exact}; |delta solve rate| > 0.05: {(delta.abs() > 0.05).sum()}"
        )


if __name__ == "__main__":
    main()
