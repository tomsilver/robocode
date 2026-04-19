"""Extract and replay approach.py versions from sandbox git history.

After an agentic run, the sandbox contains a git history with auto-snapshot commits.
This module walks that history, exports each version to an isolated copy, runs an
episode, and saves GIFs + metrics.  The original sandbox is never modified.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robocode.approaches.agentic_approach import AgenticApproach
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.episode import run_episode, save_video

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """A commit in the sandbox history."""

    version: int
    commit_hash: str
    timestamp: str  # ISO-8601
    message: str


_LOG_SEP = "\x1f"  # ASCII unit separator — won't appear in commit messages


def get_snapshots(sandbox_dir: Path) -> list[Snapshot]:
    """Return every commit in *sandbox_dir*, oldest first."""
    result = subprocess.run(
        ["git", "log", "--all", "--reverse", f"--format=%H{_LOG_SEP}%aI{_LOG_SEP}%s"],
        cwd=str(sandbox_dir),
        capture_output=True,
        text=True,
        check=True,
    )

    snapshots: list[Snapshot] = []
    for line in result.stdout.strip().splitlines():
        commit_hash, timestamp, message = line.split(_LOG_SEP, 2)

        # Skip commits that don't contain approach.py.
        has_approach = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_hash}:approach.py"],
            cwd=str(sandbox_dir),
            capture_output=True,
            check=False,
        )
        if has_approach.returncode != 0:
            continue

        snapshots.append(
            Snapshot(
                version=len(snapshots),
                commit_hash=commit_hash,
                timestamp=timestamp,
                message=message,
            )
        )

    logger.info("Found %d snapshots in %s", len(snapshots), sandbox_dir)
    return snapshots


def _export_snapshot(sandbox_dir: Path, commit_hash: str, dest: Path) -> None:
    """Export the sandbox tree at *commit_hash* into *dest* without modifying the
    repo."""
    dest.mkdir(parents=True, exist_ok=True)
    # git archive exports the tree without .git metadata
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit_hash],
        cwd=str(sandbox_dir),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "xf", "-"],
        cwd=str(dest),
        input=archive.stdout,
        capture_output=True,
        check=True,
    )


def record_episodes(
    snapshots: list[Snapshot],
    sandbox_dir: Path,
    env: Any,
    primitives: dict[str, Any],
    seed: int,
    output_dir: Path,
    max_steps: int = 100,
) -> list[dict[str, Any]]:
    """Run one episode per snapshot, saving GIFs and metrics.

    For each commit, the sandbox tree is exported to an isolated copy under
    *output_dir*/approach_history/vNNN/sandbox_<hash>/.  The original sandbox
    is never modified.  Results go to *output_dir*/approach_history/vNNN/.
    """
    history_dir = output_dir / "approach_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    env.reset(seed=seed)
    caller_state = env.get_state()

    records: list[dict[str, Any]] = []

    for snap in snapshots:
        short_hash = snap.commit_hash[:8]
        version_dir = history_dir / f"v{snap.version:03d}"
        # AgenticApproach expects load_dir/sandbox/approach.py, so we
        # export into a directory literally named "sandbox".
        snapshot_sandbox = version_dir / "sandbox"

        # Export this commit's files into an isolated directory.
        if snapshot_sandbox.exists():
            shutil.rmtree(snapshot_sandbox)
        _export_snapshot(sandbox_dir, snap.commit_hash, snapshot_sandbox)

        # Purge any cached sandbox modules (obs_helpers, act_helpers, etc.)
        # so the next load picks up files from the new exported copy.
        sandbox_modules = [
            name
            for name, mod in sys.modules.items()
            if hasattr(mod, "__file__")
            and mod.__file__ is not None
            and "sandbox" in mod.__file__
        ]
        for name in sandbox_modules:
            del sys.modules[name]

        try:
            approach = AgenticApproach(
                action_space=env.action_space,
                observation_space=env.observation_space,
                seed=seed,
                primitives=primitives,
                backend=DEFAULT_BACKEND_CFG,
                load_dir=str(version_dir),
            )
            approach.train()

            saved_state = env.get_state()
            metrics, frames = run_episode(env, approach, seed, max_steps, render=True)
            env.set_state(saved_state)

            if frames:
                save_video(frames, version_dir / "episode.gif")

            record = {
                "version": snap.version,
                "commit_hash": snap.commit_hash,
                "timestamp": snap.timestamp,
                "message": snap.message,
                **metrics,
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("v%03d (%s) failed: %s", snap.version, short_hash, e)
            record = {
                "version": snap.version,
                "commit_hash": snap.commit_hash,
                "timestamp": snap.timestamp,
                "message": snap.message,
                "total_reward": float("nan"),
                "num_steps": 0,
                "solved": False,
                "error": f"{type(e).__name__}: {e}",
            }

        records.append(record)

        with open(version_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=_json_default)

        logger.info(
            "v%03d: reward=%.2f, steps=%d, solved=%s",
            snap.version,
            record.get("total_reward", float("nan")),
            record.get("num_steps", 0),
            record["solved"],
        )

    with open(history_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=_json_default)

    if caller_state is not None:
        env.set_state(caller_state)

    return records


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
