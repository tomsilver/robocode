"""Pack finished agentic runs into a compact checkpoint store.

Each replicate directory with a ``sandbox/.git`` history becomes
``<store>/runs/<run_key>/`` holding a single-file ``git bundle`` and ``run.json``.
The store has a few files per run, so it copies quickly to cluster filesystems that
penalize many small files.

Example:
    python -m experiments.reeval.prepare --root <results dir> --store <store dir> \
        [--include paths.txt]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from experiments.reeval.store import BUNDLE_FILE, RUN_FILE, write_json_atomic

logger = logging.getLogger(__name__)

_LOG_TIME_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]")
_EVAL_START = "held-out evaluation episodes"
_SEP = "\x1f"
# Fallback eval cost (seconds per 100 episodes) when a run left no timing behind.
_DEFAULT_EVAL_SECONDS = 600.0
_DEFAULT_EVAL_SECONDS_DYNAMIC3D = 5000.0


def _git(sandbox: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=sandbox,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout


def run_key_for(replicate_dir: Path) -> str:
    """``<experiment dir>__<timestamp dir>__<replicate dir>``."""
    parts = replicate_dir.parts[-3:]
    return "__".join(parts)


def find_replicates(root: Path, include: list[str] | None) -> list[Path]:
    """Replicate dirs below ``root`` (or the listed relative paths) with git history."""
    if include is not None:
        candidates = [root / rel for rel in include]
    else:
        candidates = sorted(p.parent.parent for p in root.rglob("sandbox/.git"))
    return candidates


def read_commits(sandbox: Path) -> list[dict[str, Any]]:
    """Commits on HEAD, oldest first, with their tree and approach.py presence."""
    log = _git(sandbox, "log", "--reverse", f"--format=%H{_SEP}%T{_SEP}%aI{_SEP}%s")
    commits = []
    for line in log.splitlines():
        sha, tree, time, subject = line.split(_SEP, 3)
        has_approach = (
            subprocess.run(
                ["git", "cat-file", "-e", f"{sha}:approach.py"],
                cwd=sandbox,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )
        commits.append(
            {
                "sha": sha,
                "tree": tree,
                "time": time,
                "subject": subject,
                "has_approach": has_approach,
            }
        )
    return commits


def eval_seconds_estimate(
    replicate: Path, results: dict[str, Any] | None
) -> float | None:
    """Wall time of the original eval, from the log or the per-episode timing."""
    log = replicate / "run_experiment.log"
    if log.is_file():
        start = end = None
        for line in log.read_text(errors="replace").splitlines():
            match = _LOG_TIME_RE.match(line)
            if not match:
                continue
            stamp = datetime.fromisoformat(match.group(1))
            if _EVAL_START in line:
                start = stamp
            end = stamp
        if start is not None and end is not None and end > start:
            return (end - start).total_seconds()
    episodes = (results or {}).get("per_episode", [])
    timed = [e.get("policy_time_s", 0) + e.get("env_time_s", 0) for e in episodes]
    if timed and any(timed):
        return float(sum(timed))
    return None


def build_run(replicate: Path) -> dict[str, Any]:
    """Assemble ``run.json`` for one replicate directory."""
    cfg = yaml.safe_load((replicate / ".hydra" / "config.yaml").read_text())
    approach = cfg["approach"]
    results_path = replicate / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else None
    commits = read_commits(replicate / "sandbox")
    est = eval_seconds_estimate(replicate, results)
    if est is None:
        dynamic3d = "dynamic3d" in json.dumps(cfg["environment"])
        est = _DEFAULT_EVAL_SECONDS_DYNAMIC3D if dynamic3d else _DEFAULT_EVAL_SECONDS
    original = None
    if results is not None:
        original = {
            "solve_rate": results.get("solve_rate"),
            "solved": [e.get("solved") for e in results.get("per_episode", [])],
        }
    return {
        "run_key": run_key_for(replicate),
        "source": str(replicate),
        "experiment_dir": replicate.parts[-3],
        "experiment_id": cfg.get("experiment_id"),
        "replicate_seed": cfg["replicate_seed"],
        "environment": cfg["environment"],
        "blackbox": bool(approach.get("blackbox", False)),
        "blackbox_strict": bool(approach.get("blackbox_strict", False)),
        "max_steps": cfg["max_steps"],
        "num_eval_tasks": cfg["num_eval_tasks"],
        "eval_timeout": cfg["eval_timeout"],
        "head": commits[-1]["sha"] if commits else None,
        "commits": commits,
        "original": original,
        "est_eval_seconds": est,
    }


def add_run(replicate: Path, store: Path) -> str:
    """Write one run into the store; returns ``added``, ``present`` or a skip reason."""
    if not (replicate / "sandbox" / ".git").is_dir():
        return "skipped: no sandbox git history"
    if not (replicate / ".hydra" / "config.yaml").is_file():
        return "skipped: no .hydra/config.yaml"
    run = build_run(replicate)
    if not run["commits"]:
        return "skipped: empty history"
    run_dir = store / "runs" / run["run_key"]
    existing = run_dir / RUN_FILE
    if existing.is_file():
        old = json.loads(existing.read_text())
        if old["head"] == run["head"]:
            return "present"
        return f"skipped: key clash with different HEAD ({old['source']})"
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = (run_dir / BUNDLE_FILE).resolve()
    _git(replicate / "sandbox", "bundle", "create", "--quiet", str(bundle), "HEAD")
    write_json_atomic(existing, run)
    return "added"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--include",
        type=Path,
        help="file of replicate paths relative to --root, one per line",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    include = None
    if args.include:
        include = [ln.strip() for ln in args.include.read_text().splitlines()]
        include = [ln for ln in include if ln]
    replicates = find_replicates(args.root, include)
    counts: dict[str, int] = {}
    for replicate in replicates:
        try:
            status = add_run(replicate, args.store)
        except (subprocess.CalledProcessError, KeyError, ValueError) as exc:
            status = f"skipped: {type(exc).__name__}"
            logger.warning("%s: %s", replicate, exc)
        if status.startswith("skipped"):
            logger.info("%s: %s", replicate, status)
        counts[status] = counts.get(status, 0) + 1
    for status, n in sorted(counts.items()):
        logger.info("%6d %s", n, status)


if __name__ == "__main__":
    main()
