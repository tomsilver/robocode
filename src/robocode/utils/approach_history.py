"""Extract and replay approach.py versions from sandbox git history.

After an agentic run, the sandbox contains a git history with auto-snapshot commits.
This module walks that history, checks out each version, runs an episode, and saves GIFs
+ metrics.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robocode.approaches.agentic_approach import AgenticApproach
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


def _checkout(sandbox_dir: Path, ref: str) -> None:
    """Checkout *ref* in the sandbox repo."""
    subprocess.run(
        ["git", "checkout", ref],
        cwd=str(sandbox_dir),
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

    Checks out each commit in the sandbox, runs an episode, then restores the sandbox to
    its original HEAD.  Results go to *output_dir*/approach_history/vNNN/.
    """
    history_dir = output_dir / "approach_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    env.reset(seed=seed)
    caller_state = env.get_state()

    # Remember current ref so we can restore it (branch name if on one,
    # otherwise the raw hash for detached HEAD).
    ref_result = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        cwd=str(sandbox_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if ref_result.returncode == 0:
        head = ref_result.stdout.strip()
    else:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(sandbox_dir),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    records: list[dict[str, Any]] = []

    for snap in snapshots:
        version_dir = history_dir / f"v{snap.version:03d}"
        version_dir.mkdir(parents=True, exist_ok=True)

        _checkout(sandbox_dir, snap.commit_hash)

        try:
            approach = AgenticApproach(
                action_space=env.action_space,
                observation_space=env.observation_space,
                seed=seed,
                primitives=primitives,
                load_dir=str(sandbox_dir / ".."),
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
            logger.warning(
                "v%03d (%s) failed: %s", snap.version, snap.commit_hash[:8], e
            )
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

    # Restore sandbox to original state.
    _checkout(sandbox_dir, head)

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
