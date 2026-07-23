"""Local web viewer for robocode experiment runs.

A stdlib-only HTTP server (no framework) that discovers run directories under a
root, shows metrics, per-episode success/failure GIFs, the sandbox git history
of the generated approach.py (with diffs and the agent's reasoning per commit),
and can re-render a GIF for any episode on the current or a past approach
version. Modeled on the TensorBoard-style predicators log_viewer.

The scan walks --root recursively and finds every run regardless of where it
lives (outputs/, multirun/, or a curated collection like viewer_runs/). Point
--root at a subtree to view only that subtree, or pass --exclude to prune
directory names from a broader scan.

Launch:  python experiments/results_viewer.py --root . --port 8000
         python experiments/results_viewer.py --root viewer_runs
         python experiments/results_viewer.py --exclude outputs multirun
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

# Repo root (this file lives in <repo>/experiments/).
REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass
class RunInfo:
    """Identity and location of one discovered experiment run."""

    run_id: str  # path relative to the scan root, the stable URL key
    path: Path
    approach: str
    environment: str
    seed: Optional[int]
    budget: Optional[float]
    num_eval_tasks: int
    per_instance: bool


@dataclass
class _Scan:
    """Where to look for runs; populated from argv at startup."""

    root: Path = REPO_ROOT
    exclude_dirs: set[str] = field(default_factory=set)  # extra dirs to prune


SCAN = _Scan()
RUNS: dict[str, RunInfo] = {}
_RUNS_LOCK = threading.Lock()


def _parse_overrides(hydra_dir: Path) -> dict[str, str]:
    """Parse .hydra/overrides.yaml (a flat list of `- key=value` lines)."""
    out: dict[str, str] = {}
    f = hydra_dir / "overrides.yaml"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line.startswith("- ") or "=" not in line:
            continue
        key, val = line[2:].split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _choice(hydra_dir: Path, key: str) -> Optional[str]:
    """Read hydra.runtime.choices.<key> from .hydra/hydra.yaml (cheap text scan)."""
    f = hydra_dir / "hydra.yaml"
    if not f.exists():
        return None
    m = re.search(rf"^\s+{re.escape(key)}:\s*(\S+)\s*$", f.read_text(), re.MULTILINE)
    return m.group(1) if m else None


def _primitives(hydra_dir: Path) -> str:
    """The run's primitives as a hydra override value, e.g. ``[BiRRT,csp]``.

    A generated approach.py indexes the exact primitives it was written against, so a
    replay must rebuild the same set or it raises KeyError. Prefer the recorded
    override; fall back to the resolved list in config.yaml.
    """
    ov = _parse_overrides(hydra_dir)
    if "primitives" in ov:
        return ov["primitives"]
    cfg = hydra_dir / "config.yaml"
    if not cfg.exists():
        return "[]"
    lines = cfg.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("primitives:"):
            continue
        inline = line.split(":", 1)[1].strip()
        if inline:
            return inline
        names = []
        for entry in lines[i + 1 :]:
            if not entry.startswith("- "):
                break
            names.append(entry[2:].strip())
        return "[" + ",".join(names) + "]"
    return "[]"


# Trees that never hold experiment runs; pruned so the recursive scan stays fast.
_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "third-party"}


def _find_results(root: Path) -> list[Path]:
    """Every results.json under root, at any depth, skipping vendored/dep trees."""
    skip = _SKIP_DIRS | SCAN.exclude_dirs
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        if "results.json" in filenames:
            out.append(Path(dirpath) / "results.json")
    return out


def _discover_runs(root: Path) -> dict[str, RunInfo]:
    runs: dict[str, RunInfo] = {}
    for results in _find_results(root):
        run_dir = results.parent
        hydra_dir = run_dir / ".hydra"
        if not hydra_dir.is_dir():
            continue  # re-eval render dirs have results.json but no .hydra/
        ov = _parse_overrides(hydra_dir)
        if "approach.load_dir" in ov:
            continue  # a re-eval / render pass, not a real generation run
        run_id = str(run_dir.relative_to(root))
        budget = ov.get("approach.max_budget_usd")
        if budget is None:
            m = re.search(r"budget_(\d+)", run_id)
            budget = m.group(1) if m else None
        seed = ov.get("seed")
        m2 = re.search(r"[/_]s(\d+)", run_id)
        approach = ov.get("approach") or _choice(hydra_dir, "approach") or "?"
        num_eval = 100
        cfg = hydra_dir / "config.yaml"
        if cfg.exists():
            mm = re.search(r"^num_eval_tasks:\s*(\d+)", cfg.read_text(), re.MULTILINE)
            if mm:
                num_eval = int(mm.group(1))
        runs[run_id] = RunInfo(
            run_id=run_id,
            path=run_dir,
            approach=approach,
            environment=ov.get("environment")
            or _choice(hydra_dir, "environment")
            or "?",
            seed=(
                int(seed)
                if seed and seed.isdigit()
                else (int(m2.group(1)) if m2 else None)
            ),
            budget=(
                float(budget)
                if budget and str(budget).replace(".", "").isdigit()
                else None
            ),
            num_eval_tasks=num_eval,
            per_instance=("per_instance" in approach or "best_of_k" in approach),
        )
    return runs


def refresh_runs() -> None:
    """Rebuild the run index from disk."""
    with _RUNS_LOCK:
        RUNS.clear()
        RUNS.update(_discover_runs(SCAN.root))


def _run(run_id: str) -> Optional[RunInfo]:
    with _RUNS_LOCK:
        return RUNS.get(run_id)


# --------------------------------------------------------------------------- #
# Live status
# --------------------------------------------------------------------------- #

_PS_CACHE: dict[str, Any] = {"t": 0.0, "dirs": set()}


def _running_dirs() -> set[str]:
    """Set of hydra.run.dir paths for live run_experiment.py processes (cached 5s)."""
    now = time.monotonic()
    if now - _PS_CACHE["t"] < 5.0:
        return _PS_CACHE["dirs"]
    dirs: set[str] = set()
    res = subprocess.run(
        ["pgrep", "-af", "run_experiment.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in res.stdout.splitlines():
        m = re.search(r"hydra\.run\.dir=(\S+)", line)
        if m:
            dirs.add(m.group(1).rstrip("/"))
    _PS_CACHE.update(t=now, dirs=dirs)
    return dirs


def _status(run: RunInfo) -> str:
    if (run.path / "results.json").exists():
        return "done"
    rel = str(run.path)
    if rel in _running_dirs() or str(run.path.relative_to(SCAN.root)) in {
        d.replace(str(SCAN.root) + "/", "") for d in _running_dirs()
    }:
        return "running"
    return "interrupted"


# --------------------------------------------------------------------------- #
# Metrics / summaries
# --------------------------------------------------------------------------- #


def _results(run: RunInfo) -> dict[str, Any]:
    f = run.path / "results.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _summary(run: RunInfo) -> dict[str, Any]:
    r = _results(run)
    return {
        "run_id": run.run_id,
        "approach": run.approach,
        "environment": run.environment,
        "seed": run.seed,
        "budget": run.budget,
        "status": _status(run),
        "solve_rate": r.get("solve_rate"),
        "mean_eval_reward": r.get("mean_eval_reward"),
        "agent_cost_usd": r.get("agent_cost_usd"),
        "gen_total_tokens": r.get("gen_total_tokens"),
        "gen_wall_time_s": r.get("gen_wall_time_s"),
        "gen_cli_duration_ms": r.get("gen_cli_duration_ms"),
        "num_crashed_episodes": r.get("num_crashed_episodes"),
        "largest_count_all_solved": r.get("largest_count_all_solved"),
        "num_eval_tasks": run.num_eval_tasks,
        "has_history": (run.path / "sandbox" / ".git").is_dir()
        and not run.per_instance,
    }


def _episode_outcome(e: dict[str, Any]) -> str:
    if e.get("crashed"):
        return "crashed"
    if e.get("timed_out"):
        return "timed_out"
    return "success" if e.get("solved") else "failure"


def _generation_time_breakdown(results: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Estimate Claude API wait versus local experiment/tool time.

    Claude Code reports ``duration_api_ms`` directly. The complement of that
    duration within our end-to-end generation timer is time outside the model:
    environment rollouts, shell/file tools, CLI orchestration, and sandbox
    startup. Successful tool-result boundaries were not timestamped, so the
    experiment component cannot be separated further for historical runs.
    """
    wall_s = results.get("gen_wall_time_s")
    cli_ms = results.get("gen_cli_duration_ms")
    if isinstance(wall_s, (int, float)) and wall_s > 0:
        total_s = float(wall_s)
        basis = "generation wall time"
    elif isinstance(cli_ms, (int, float)) and cli_ms > 0:
        total_s = float(cli_ms) / 1000
        basis = "Claude CLI duration"
    else:
        return None
    model_raw = results.get("gen_model_wait_time_s")
    experiment_raw = results.get("gen_experiment_time_s")
    other_raw = results.get("gen_other_tool_time_s")
    if (
        isinstance(model_raw, (int, float))
        and model_raw >= 0
        and isinstance(experiment_raw, (int, float))
        and experiment_raw >= 0
        and isinstance(other_raw, (int, float))
        and other_raw >= 0
    ):
        model_s = float(model_raw)
        experiment_s = float(experiment_raw)
        other_s = float(other_raw)
        measured_s = model_s + experiment_s + other_s
        total_s = max(total_s, measured_s)
        other_s += max(0.0, total_s - measured_s)
        return {
            "total_s": total_s,
            "claude_s": model_s,
            "experiments_tools_s": experiment_s,
            "other_s": other_s,
            "claude_fraction": model_s / total_s,
            "experiments_tools_fraction": experiment_s / total_s,
            "other_fraction": other_s / total_s,
            "basis": "instrumented stream wall time",
            "instrumented": True,
            "clamped": False,
        }

    api_ms = results.get("gen_cli_duration_api_ms")
    if not isinstance(api_ms, (int, float)) or api_ms <= 0:
        return None
    claude_s = min(float(api_ms) / 1000, total_s)
    local_s = max(0.0, total_s - claude_s)
    return {
        "total_s": total_s,
        "claude_s": claude_s,
        "experiments_tools_s": local_s,
        "other_s": 0.0,
        "claude_fraction": claude_s / total_s,
        "experiments_tools_fraction": local_s / total_s,
        "other_fraction": 0.0,
        "basis": basis,
        "instrumented": False,
        "clamped": float(api_ms) / 1000 > total_s,
    }


def _run_detail(run: RunInfo) -> dict[str, Any]:
    r = _results(run)
    per = r.get("per_episode", [])
    eval_seeds = _eval_seeds(run)
    episodes = []
    for i, e in enumerate(per):
        episodes.append(
            {
                "i": i,
                "seed": eval_seeds[i] if i < len(eval_seeds) else None,
                "object_count": e.get("object_count"),
                "num_steps": e.get("num_steps"),
                "total_reward": e.get("total_reward"),
                "outcome": _episode_outcome(e),
                "has_gif": _eval_gif_path(run, i) is not None,
            }
        )
    env_desc = run.path / "env_description.md"
    logs = [
        n for n in ("run_experiment.log", "env_config.json") if (run.path / n).exists()
    ] + [
        f"sandbox/{n}"
        for n in ("agent_log.txt", "CLAUDE.md", "approach.py")
        if (run.path / "sandbox" / n).exists()
    ]
    metrics = {k: v for k, v in r.items() if k != "per_episode"}
    return {
        "summary": _summary(run),
        "metrics": metrics,
        "generation_time_breakdown": _generation_time_breakdown(r),
        "episodes": episodes,
        "env_description": env_desc.read_text() if env_desc.exists() else None,
        "logs": logs,
        "has_history": _summary(run)["has_history"],
    }


# --------------------------------------------------------------------------- #
# GIF source resolution
# --------------------------------------------------------------------------- #


def _campaign(run: RunInfo) -> Path:
    # <campaign>/budget_B/sS  ->  campaign is run_dir.parent.parent
    return run.path.parent.parent


def _eval_gif_path(run: RunInfo, i: int) -> Optional[Path]:
    """Locate an already-rendered eval GIF for episode i, else None."""
    cand = [run.path / "videos" / f"episode_{i}.gif"]
    if run.budget is not None and run.seed is not None:
        b = int(run.budget)
        cand.append(
            _campaign(run)
            / "renders"
            / f"budget_{b}_s{run.seed}"
            / "videos"
            / f"episode_{i}.gif"
        )
        # named_gifs: needs count + outcome from per_episode
        r = _results(run)
        per = r.get("per_episode", [])
        if i < len(per):
            e = per[i]
            c = e.get("object_count")
            outcome = "success" if e.get("solved") else "failure"
            if c is not None:
                cand.append(
                    _campaign(run)
                    / "named_gifs"
                    / f"b{b:02d}_s{run.seed}_c{c:02d}_{outcome}_ep{i:03d}.gif"
                )
    for p in cand:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _history_gif_path(run: RunInfo, version: int, i: int) -> Optional[Path]:
    version_dir = run.path / "approach_history" / f"v{version:03d}"
    candidates = [version_dir / f"episode_{i}.gif"]
    # record_episodes() historically wrote one unnumbered replay per version.
    if i == 0:
        candidates.append(version_dir / "episode.gif")
    return next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)


def _eval_seeds(run: RunInfo) -> list[int]:
    """Recreate the held-out episode seeds used by run_experiment.py."""
    if run.seed is None:
        return []
    # Keep numpy out of the read-only viewer path. PCG64 and Generator.integers
    # are used by the experiment, so use numpy when installed and degrade to no
    # seed labels on a lightweight viewer-only machine.
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        rng = np.random.default_rng(run.seed)
        return [int(rng.integers(0, 2**63)) for _ in range(run.num_eval_tasks)]
    except ImportError:
        return []


# --------------------------------------------------------------------------- #
# Git history
# --------------------------------------------------------------------------- #

_SEP = "\x1f"


def _git(sandbox: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(sandbox), capture_output=True, text=True, check=False
    )


def _snapshots(run: RunInfo) -> list[dict[str, Any]]:
    """Version list from the sandbox git history (approach.py-bearing commits)."""
    sandbox = run.path / "sandbox"
    if not (sandbox / ".git").is_dir():
        return []
    res = _git(sandbox, "log", "--all", "--reverse", f"--format=%H{_SEP}%aI{_SEP}%s")
    stream = _stream_index(run)
    snaps: list[dict[str, Any]] = []
    previous_ts: Optional[dt.datetime] = None
    for line in res.stdout.strip().splitlines():
        h, ts, msg = line.split(_SEP, 2)
        try:
            commit_ts = dt.datetime.fromisoformat(ts)
        except ValueError:
            commit_ts = None
        elapsed_s = (
            max(0.0, (commit_ts - previous_ts).total_seconds())
            if commit_ts is not None and previous_ts is not None
            else None
        )
        if commit_ts is not None:
            previous_ts = commit_ts
        if _git(sandbox, "cat-file", "-e", f"{h}:approach.py").returncode != 0:
            continue
        v = len(snaps)
        effort = stream["effort_by_msg"].get(msg, {})
        if not effort:
            effort = next(
                (
                    value
                    for key, value in stream["effort_by_msg"].items()
                    if key[:30] == msg[:30]
                ),
                {},
            )
        numstat = _git(
            sandbox, "show", "--format=", "--numstat", h, "--", "approach.py"
        ).stdout.strip()
        additions = deletions = 0
        for stat_line in numstat.splitlines():
            parts = stat_line.split("\t")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                additions += int(parts[0])
                deletions += int(parts[1])
        snaps.append(
            {
                "version": v,
                "commit_hash": h,
                "short": h[:8],
                "timestamp": ts,
                "message": msg,
                "effort": {
                    "elapsed_s": elapsed_s,
                    "tokens": effort.get("tokens"),
                    "input_tokens": effort.get("input_tokens"),
                    "output_tokens": effort.get("output_tokens"),
                    "turns": effort.get("turns"),
                    "tool_calls": effort.get("tool_calls"),
                    "additions": additions,
                    "deletions": deletions,
                },
                "evaluation": _history_evaluation(run, v),
                "has_gif": (
                    _history_gif_path(run, v, 0) is not None
                    or any(
                        (run.path / "approach_history" / f"v{v:03d}").glob(
                            "episode_*.gif"
                        )
                    )
                    if (run.path / "approach_history" / f"v{v:03d}").is_dir()
                    else False
                ),
            }
        )
    # The run's results.json was produced by the final committed approach, so it
    # is a complete, already-paid-for evaluation of the last snapshot. Older
    # viewer versions discarded metrics for ad-hoc historical GIF replays, but
    # the final commit should never appear unevaluated.
    final_evaluation = _final_evaluation(run)
    if snaps and final_evaluation is not None:
        replay_episodes = snaps[-1]["evaluation"]["episodes"]
        for episode_index, replay in replay_episodes.items():
            if replay.get("has_gif") and episode_index in final_evaluation["episodes"]:
                final_evaluation["episodes"][episode_index].update(
                    has_gif=True, gif_kind="history"
                )
        snaps[-1]["evaluation"] = final_evaluation
    return snaps


def _commit_hash(run: RunInfo, version: int) -> Optional[str]:
    snaps = _snapshots(run)
    return snaps[version]["commit_hash"] if 0 <= version < len(snaps) else None


def _approach_source(run: RunInfo, version: int) -> str:
    h = _commit_hash(run, version)
    if h is None:
        return ""
    return _git(run.path / "sandbox", "show", f"{h}:approach.py").stdout


def _approach_diff(run: RunInfo, a: int, b: int) -> str:
    ha, hb = _commit_hash(run, a), _commit_hash(run, b)
    if ha is None or hb is None:
        return ""
    return _git(run.path / "sandbox", "diff", ha, hb, "--", "approach.py").stdout


def _history_episode_records(run: RunInfo, version: int) -> dict[int, dict[str, Any]]:
    """Return recorded replay outcomes for a version, including legacy data."""
    version_dir = run.path / "approach_history" / f"v{version:03d}"
    records: dict[int, dict[str, Any]] = {}
    episodes_file = version_dir / "episodes.json"
    if episodes_file.exists():
        try:
            raw = json.loads(episodes_file.read_text())
            values = raw.values() if isinstance(raw, dict) else raw
            for record in values:
                i = int(record.get("episode_index", len(records)))
                records[i] = record
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Older record_episodes runs stored a single metrics.json replay. It was not
    # tied to a held-out episode, so expose its score but don't claim a seed.
    legacy = version_dir / "metrics.json"
    if legacy.exists() and not records:
        try:
            record = json.loads(legacy.read_text())
            # The legacy replay used the training seed, not eval episode zero.
            records[int(record.get("episode_index", -1))] = record
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return records


def _history_evaluation(run: RunInfo, version: int) -> dict[str, Any]:
    records = _history_episode_records(run, version)
    eval_records = {i: record for i, record in records.items() if i >= 0}
    solved = sum(bool(r.get("solved")) for r in eval_records.values())
    failures = [
        {
            "episode_index": i,
            "seed": r.get("seed"),
            "outcome": _episode_outcome(r),
            "num_steps": r.get("num_steps"),
            "total_reward": r.get("total_reward"),
            "error": r.get("error"),
            "has_gif": _history_gif_path(run, version, i) is not None,
        }
        for i, r in sorted(eval_records.items())
        if not r.get("solved")
    ]
    return {
        "evaluated": len(eval_records),
        "solved": solved,
        "solve_rate": solved / len(eval_records) if eval_records else None,
        "episodes": {
            str(i): {
                "solved": bool(r.get("solved")),
                "outcome": _episode_outcome(r),
                "error": r.get("error"),
                "has_gif": _history_gif_path(run, version, i) is not None,
                "gif_kind": "history",
            }
            for i, r in sorted(eval_records.items())
        },
        "failures": failures,
        "source": "replays",
    }


def _final_evaluation(run: RunInfo) -> Optional[dict[str, Any]]:
    """Evaluation summary for HEAD from the original experiment results."""
    per = _results(run).get("per_episode", [])
    if not per:
        return None
    seeds = _eval_seeds(run)
    solved = sum(bool(record.get("solved")) for record in per)
    failures = [
        {
            "episode_index": i,
            "seed": seeds[i] if i < len(seeds) else None,
            "outcome": _episode_outcome(record),
            "num_steps": record.get("num_steps"),
            "total_reward": record.get("total_reward"),
            "error": record.get("error"),
            "has_gif": _eval_gif_path(run, i) is not None,
        }
        for i, record in enumerate(per)
        if not record.get("solved")
    ]
    return {
        "evaluated": len(per),
        "solved": solved,
        "solve_rate": solved / len(per),
        "episodes": {
            str(i): {
                "solved": bool(record.get("solved")),
                "outcome": _episode_outcome(record),
                "error": record.get("error"),
                "has_gif": _eval_gif_path(run, i) is not None,
                "gif_kind": "eval",
            }
            for i, record in enumerate(per)
        },
        "failures": failures,
        "source": "final run",
    }


# --------------------------------------------------------------------------- #
# stream.jsonl -> "why" reasoning per commit
# --------------------------------------------------------------------------- #

_STREAM_CACHE: dict[str, dict[str, Any]] = {}


def _commit_subject_from_command(command: str) -> Optional[str]:
    match = re.search(r"git commit[^\"']*-m\s+[\"'](.+?)[\"']", command, re.DOTALL)
    return match.group(1).splitlines()[0].strip() if match else None


def _seed_mentions(command: str) -> dict[int, set[str]]:
    """Extract literal environment seeds from an agent's Bash command.

    The coding agent usually embeds small Python test programs in heredocs. We
    intentionally recognize only explicit seed assignments/lists/ranges, not
    arbitrary numbers, to avoid confusing step counts and geometry constants
    with seeds.
    """
    mentions: dict[int, set[str]] = {}

    def add(seed: int, kind: str) -> None:
        if 0 <= seed < 1_000_000:
            mentions.setdefault(seed, set()).add(kind)

    for match in re.finditer(r"(?:reset\s*\(\s*seed|\bseed)\s*=\s*(\d+)", command):
        add(int(match.group(1)), "literal")
    for match in re.finditer(r"for\s+seed\s+in\s*\[([^\]]+)\]", command):
        for value in re.findall(r"\b\d+\b", match.group(1)):
            add(int(value), "list")
    for match in re.finditer(
        r"for\s+seed\s+in\s+range\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)", command
    ):
        first, second = match.groups()
        start, stop = (0, int(first)) if second is None else (int(first), int(second))
        for seed in range(start, min(stop, start + 500)):
            add(seed, "range")
    # Occasionally a diagnostic table is written as ``seeds = [(0, ...), ...]``.
    for match in re.finditer(r"\bseeds\s*=\s*\[([^\]]+)\]", command, re.DOTALL):
        for value in re.findall(r"\(\s*(\d+)\s*,", match.group(1)):
            add(int(value), "list")
    return mentions


def _stream_index(run: RunInfo) -> dict[str, Any]:
    """Map commit messages to reasoning and best-effort per-commit effort.

    Claude stream events contain per-turn token usage but no stable wall-clock
    timestamps. We group turns and tool calls up to each ``git commit``; elapsed
    time comes independently from adjacent git commit timestamps.
    """
    f = run.path / "stream.jsonl"
    if not f.exists():
        return {"by_msg": {}, "effort_by_msg": {}, "seed_events": []}
    key = run.run_id
    mtime = f.stat().st_mtime
    cached = _STREAM_CACHE.get(key)
    if cached and cached["mtime"] == mtime:
        return cached["idx"]

    by_msg: dict[str, dict[str, Any]] = {}
    effort_by_msg: dict[str, dict[str, Any]] = {}
    seed_events: list[dict[str, Any]] = []
    current_commit: Optional[str] = None
    prev_text = ""
    effort = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "turns": 0,
        "tool_calls": 0,
    }
    usage_seen = False
    for line in f.open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "assistant":
            continue
        message = d.get("message") or {}
        blocks = message.get("content", [])
        commands = [
            (block.get("input") or {}).get("command", "")
            for block in blocks
            if block.get("type") == "tool_use" and block.get("name") == "Bash"
        ]
        message_commit = next(
            (
                subject
                for command in commands
                if (subject := _commit_subject_from_command(command)) is not None
            ),
            None,
        )
        event_commit = message_commit or current_commit
        for command in commands:
            mentions = _seed_mentions(command)
            if mentions:
                relevant_lines = [
                    line.strip()
                    for line in command.splitlines()
                    if re.search(r"\bseed|range\(", line, re.IGNORECASE)
                ]
                seed_events.append(
                    {
                        "commit": event_commit,
                        "seeds": sorted(mentions),
                        "kinds": {
                            str(seed): sorted(kinds) for seed, kinds in mentions.items()
                        },
                        "evidence": " | ".join(relevant_lines)[:500],
                    }
                )
        if message_commit is not None:
            current_commit = message_commit
        effort["turns"] += 1
        usage = message.get("usage") or d.get("usage") or {}
        token_keys = {
            "input_tokens": "input_tokens",
            "output_tokens": "output_tokens",
            "cache_read_input_tokens": "cache_read_tokens",
            "cache_creation_input_tokens": "cache_creation_tokens",
        }
        for source, dest in token_keys.items():
            value = usage.get(source)
            if isinstance(value, (int, float)):
                effort[dest] += int(value)
                usage_seen = True
        thinking = " ".join(
            b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
        )
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        for b in blocks:
            if b.get("type") == "tool_use":
                effort["tool_calls"] += 1
            if b.get("type") == "tool_use" and b.get("name") == "Bash":
                cmd = (b.get("input") or {}).get("command", "")
                subject = _commit_subject_from_command(cmd)
                if subject:
                    by_msg[subject] = {
                        "thinking": (thinking or prev_text)[:6000],
                        "text": text[:4000],
                    }
                    effort_by_msg[subject] = {
                        **effort,
                        "tokens": (
                            effort["input_tokens"]
                            + effort["output_tokens"]
                            + effort["cache_read_tokens"]
                            + effort["cache_creation_tokens"]
                            if usage_seen
                            else None
                        ),
                    }
                    effort = {key: 0 for key in effort}
                    usage_seen = False
        if thinking or text:
            prev_text = thinking or text
    idx = {
        "by_msg": by_msg,
        "effort_by_msg": effort_by_msg,
        "seed_events": seed_events,
    }
    _STREAM_CACHE[key] = {"mtime": mtime, "idx": idx}
    return idx


def _why(run: RunInfo, version: int) -> dict[str, Any]:
    snaps = _snapshots(run)
    if not 0 <= version < len(snaps):
        return {}
    msg = snaps[version]["message"]
    idx = _stream_index(run)
    hit = idx["by_msg"].get(msg)
    if hit:
        return {"matched_by": "message", "message": msg, **hit}
    # loose fallback: prefix match on the subject
    for k, v in idx["by_msg"].items():
        if k[:30] == msg[:30]:
            return {"matched_by": "message~", "message": msg, **v}
    return {"matched_by": "none", "message": msg, "thinking": "", "text": ""}


def _training_seed_info(run: RunInfo) -> dict[str, Any]:
    """Summarize literal seeds the coding agent exercised during generation."""
    snapshots = _snapshots(run)
    events = _stream_index(run)["seed_events"]
    versions_by_message = {
        snapshot["message"]: snapshot["version"] for snapshot in snapshots
    }

    def version_for(message: Optional[str]) -> int:
        if message is None:
            return 0
        if message in versions_by_message:
            return versions_by_message[message]
        return next(
            (
                version
                for subject, version in versions_by_message.items()
                if subject[:30] == message[:30]
            ),
            0,
        )

    by_seed: dict[int, dict[str, Any]] = {}
    by_version: dict[int, set[int]] = {}
    normalized_events = []
    for event in events:
        version = version_for(event["commit"])
        seeds = event["seeds"]
        by_version.setdefault(version, set()).update(seeds)
        normalized_events.append({**event, "version": version})
        for seed in seeds:
            kinds = event["kinds"].get(str(seed), [])
            item = by_seed.setdefault(
                seed,
                {
                    "seed": seed,
                    "mentions": 0,
                    "explicit_mentions": 0,
                    "versions": set(),
                    "evidence": [],
                },
            )
            item["mentions"] += 1
            item["explicit_mentions"] += int(any(kind != "range" for kind in kinds))
            item["versions"].add(version)
            if event["evidence"] and event["evidence"] not in item["evidence"]:
                item["evidence"].append(event["evidence"])

    seeds_out = []
    for item in by_seed.values():
        versions = sorted(item["versions"])
        seeds_out.append(
            {
                **item,
                "versions": versions,
                "first_version": versions[0],
                "last_version": versions[-1],
                "evidence": item["evidence"][:4],
                # Individually targeted seeds are more diagnostic than a seed
                # that merely appeared once inside range(200).
                "interest": item["explicit_mentions"] * 100 + item["mentions"],
            }
        )
    seeds_out.sort(key=lambda item: (-item["interest"], item["seed"]))
    return {
        "seeds": seeds_out,
        "events": normalized_events,
        "by_version": {
            str(version): sorted(seeds) for version, seeds in sorted(by_version.items())
        },
    }


# --------------------------------------------------------------------------- #
# Render jobs (single worker; heavy imports are lazy)
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    """State of one background GIF-render job, polled by the client."""

    job_id: str
    state: str = "queued"  # queued | running | done | error
    message: str = ""
    gif_url: Optional[str] = None
    current: int = 0
    total: int = 0


JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_RENDER_Q: "queue.Queue[tuple[str, str, str, int]]" = queue.Queue()
_JOB_SEQ = [0]


def _enqueue_render(run_id: str, target: str, episode_index: int) -> str:
    with _JOBS_LOCK:
        _JOB_SEQ[0] += 1
        job_id = f"job{_JOB_SEQ[0]}"
        JOBS[job_id] = Job(job_id=job_id, message=f"queued: {target} ep{episode_index}")
    _RENDER_Q.put((job_id, run_id, target, episode_index))
    return job_id


def _enqueue_history_evaluation(run_id: str) -> str:
    """Queue one metrics-only evaluation across every incomplete commit."""
    with _JOBS_LOCK:
        _JOB_SEQ[0] += 1
        job_id = f"job{_JOB_SEQ[0]}"
        JOBS[job_id] = Job(job_id=job_id, message="queued full history evaluation")
    _RENDER_Q.put((job_id, run_id, "EVAL_HISTORY", -1))
    return job_id


def _purge_sandbox_modules() -> None:
    """Drop cached ``*sandbox*`` modules so the next version loads fresh files."""
    for name in [
        n
        for n, m in list(sys.modules.items())
        if getattr(m, "__file__", None) and "sandbox" in (m.__file__ or "")
    ]:
        del sys.modules[name]


def _jsonable(value: Any) -> Any:
    """Convert numpy-ish metric values without importing numpy in the server."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _record_history_episode(
    run: RunInfo, version: int, episode_index: int, record: dict[str, Any]
) -> None:
    """Persist one replay so progress survives viewer restarts."""
    version_dir = run.path / "approach_history" / f"v{version:03d}"
    version_dir.mkdir(parents=True, exist_ok=True)
    path = version_dir / "episodes.json"
    try:
        records = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        records = {}
    records[str(episode_index)] = record
    tmp = version_dir / "episodes.json.tmp"
    tmp.write_text(json.dumps(records, indent=2, default=_jsonable) + "\n")
    tmp.replace(path)


def _record_history_episodes(
    run: RunInfo, version: int, records: dict[int, dict[str, Any]]
) -> None:
    """Atomically replace the complete metrics-only evaluation for a version."""
    version_dir = run.path / "approach_history" / f"v{version:03d}"
    version_dir.mkdir(parents=True, exist_ok=True)
    path = version_dir / "episodes.json"
    tmp = version_dir / "episodes.json.tmp"
    tmp.write_text(
        json.dumps(
            {str(index): record for index, record in records.items()},
            indent=2,
            default=_jsonable,
        )
        + "\n"
    )
    tmp.replace(path)


def _evaluate_history(run: RunInfo, job: Job) -> None:
    """Evaluate every historical commit on the final run's full eval suite.

    This intentionally records metrics only. Rendering hundreds of GIFs would
    dominate runtime and disk; individual failures can still be rendered from
    the matrix after the comparable solve-rate curve is available.
    """
    # pylint: disable=import-outside-toplevel
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from robocode.primitives import build_primitives
    from robocode.utils.approach_history import _export_snapshot
    from robocode.utils.episode import run_episode

    snapshots = _snapshots(run)
    eval_seeds = _eval_seeds(run)
    final_per_episode = _results(run).get("per_episode", [])
    if not eval_seeds or not final_per_episode:
        raise RuntimeError("the run has no reproducible final evaluation suite")
    if len(eval_seeds) < len(final_per_episode):
        raise RuntimeError("fewer reconstructed seeds than final evaluation episodes")

    # results.json already fully evaluates the final snapshot. Re-run only
    # historical versions whose persisted suite is incomplete.
    versions = [
        snapshot
        for snapshot in snapshots[:-1]
        if snapshot["evaluation"]["evaluated"] < len(final_per_episode)
    ]
    with _JOBS_LOCK:
        job.total = len(versions) * len(final_per_episode)
        job.current = 0
    conf_dir = str(REPO_ROOT / "experiments" / "conf")

    for snapshot in versions:
        version = snapshot["version"]
        records: dict[int, dict[str, Any]] = {}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _export_snapshot(
                    run.path / "sandbox",
                    snapshot["commit_hash"],
                    Path(tmp) / "sandbox",
                )
                with initialize_config_dir(version_base=None, config_dir=conf_dir):
                    cfg = compose(
                        config_name="config",
                        overrides=[
                            f"approach={run.approach}",
                            f"approach.load_dir={tmp}",
                            f"approach.output_dir={tmp}/out",
                            f"primitives={_primitives(run.path / '.hydra')}",
                            "mcp_tools=[]",
                            f"environment={run.environment}",
                            f"seed={run.seed}",
                        ],
                    )
                env = instantiate(cfg.environment)
                primitives = build_primitives(env, cfg.primitives)
                approach_ctor = instantiate(
                    cfg.approach,
                    action_space=env.action_space,
                    observation_space=env.observation_space,
                    seed=cfg.seed,
                    env_description_path=None,
                    mcp_tools=(),
                    env_name=run.environment,
                    env=env,
                    env_cfg=json.dumps({}),
                    max_steps=cfg.max_steps,
                    eval_timeout=cfg.eval_timeout,
                    _partial_=True,
                )
                _purge_sandbox_modules()
                approach = approach_ctor(primitives=primitives)
                approach.train()

                for i, final_record in enumerate(final_per_episode):
                    count = final_record.get("object_count")
                    max_steps = (
                        env.max_steps_for_count(count)
                        if count is not None and hasattr(env, "max_steps_for_count")
                        else int(cfg.max_steps)
                    )
                    try:
                        metrics, _frames, _ = run_episode(
                            env,
                            approach,
                            eval_seeds[i],
                            max_steps,
                            render=False,
                            count=count,
                        )
                        records[i] = {
                            "episode_index": i,
                            "seed": eval_seeds[i],
                            "object_count": count,
                            **metrics,
                        }
                    except Exception as error:  # pylint: disable=broad-exception-caught
                        records[i] = {
                            "episode_index": i,
                            "seed": eval_seeds[i],
                            "object_count": count,
                            "solved": False,
                            "crashed": True,
                            "num_steps": 0,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    with _JOBS_LOCK:
                        job.current += 1
                        job.message = (
                            f"v{version} episode {i + 1}/{len(final_per_episode)}"
                        )
        except Exception as error:  # pylint: disable=broad-exception-caught
            # A version that cannot even load is a crash on every seed, not a
            # reason to lose the rest of the history comparison.
            records = {
                i: {
                    "episode_index": i,
                    "seed": eval_seeds[i],
                    "object_count": final_record.get("object_count"),
                    "solved": False,
                    "crashed": True,
                    "num_steps": 0,
                    "error": f"{type(error).__name__}: {error}",
                }
                for i, final_record in enumerate(final_per_episode)
            }
            with _JOBS_LOCK:
                job.current += len(final_per_episode)
                job.message = f"v{version} could not load"
        _record_history_episodes(run, version, records)


def _render_worker() -> None:
    # pylint: disable=import-outside-toplevel
    conf_dir = str(REPO_ROOT / "experiments" / "conf")

    while True:
        job_id, run_id, target, i = _RENDER_Q.get()
        run = _run(run_id)
        history_version: Optional[int] = None
        replay_seed: Optional[int] = None
        replay_count: Optional[int] = None
        with _JOBS_LOCK:
            job = JOBS[job_id]
            job.state, job.message = "running", f"rendering {target} ep{i}"
        try:
            # Heavy deps are imported on the first render so the server and all
            # read-only browsing need nothing but the Python stdlib (useful for
            # inspecting a downloaded outputs/ folder on a laptop).
            from hydra import compose, initialize_config_dir
            from hydra.utils import instantiate

            from robocode.primitives import build_primitives
            from robocode.utils.approach_history import _export_snapshot
            from robocode.utils.episode import run_episode, save_video

            if run is None:
                raise RuntimeError(f"unknown run {run_id}")
            if target == "EVAL_HISTORY":
                with _JOBS_LOCK:
                    job.message = "preparing full history evaluation"
                _evaluate_history(run, job)
                with _JOBS_LOCK:
                    job.state = "done"
                    job.message = "full history evaluation complete"
                continue
            r = _results(run)
            per = r.get("per_episode", [])
            count = per[i].get("object_count") if i < len(per) else None
            eval_seeds = _eval_seeds(run)
            if not 0 <= i < len(eval_seeds):
                raise IndexError(
                    f"episode {i} is outside the {len(eval_seeds)} eval seeds"
                )
            replay_seed, replay_count = eval_seeds[i], count

            with tempfile.TemporaryDirectory() as tmp:
                if target == "HEAD":
                    load_dir = str(run.path)
                    out = run.path / "videos" / f"episode_{i}.gif"
                else:
                    version = int(target)
                    history_version = version
                    h = _commit_hash(run, version)
                    assert h is not None, f"no commit for version {version}"
                    _export_snapshot(run.path / "sandbox", h, Path(tmp) / "sandbox")
                    load_dir = tmp
                    out = (
                        run.path
                        / "approach_history"
                        / f"v{version:03d}"
                        / f"episode_{i}.gif"
                    )
                out.parent.mkdir(parents=True, exist_ok=True)

                with initialize_config_dir(version_base=None, config_dir=conf_dir):
                    cfg = compose(
                        config_name="config",
                        overrides=[
                            f"approach={run.approach}",
                            f"approach.load_dir={load_dir}",
                            f"approach.output_dir={tmp}/out",
                            f"primitives={_primitives(run.path / '.hydra')}",
                            "mcp_tools=[]",
                            f"environment={run.environment}",
                            f"seed={run.seed}",
                        ],
                    )
                env = instantiate(cfg.environment)
                primitives = build_primitives(env, cfg.primitives)
                approach_ctor = instantiate(
                    cfg.approach,
                    action_space=env.action_space,
                    observation_space=env.observation_space,
                    seed=cfg.seed,
                    env_description_path=None,
                    mcp_tools=(),
                    env_name=run.environment,
                    env=env,
                    env_cfg=json.dumps({}),
                    max_steps=cfg.max_steps,
                    eval_timeout=cfg.eval_timeout,
                    _partial_=True,
                )
                _purge_sandbox_modules()
                approach = approach_ctor(primitives=primitives)
                approach.train()

                if count is not None and hasattr(env, "max_steps_for_count"):
                    max_steps = env.max_steps_for_count(count)
                else:
                    max_steps = int(cfg.max_steps)
                metrics, frames, _ = run_episode(
                    env, approach, eval_seeds[i], max_steps, render=True, count=count
                )
                if not frames:
                    raise RuntimeError("no frames rendered")
                save_video(frames, out)
                if history_version is not None:
                    _record_history_episode(
                        run,
                        history_version,
                        i,
                        {
                            "episode_index": i,
                            "seed": replay_seed,
                            "object_count": replay_count,
                            **metrics,
                        },
                    )

            with _JOBS_LOCK:
                job.state = "done"
                job.message = "rendered"
                kind = "eval" if target == "HEAD" else "history"
                key = i if target == "HEAD" else target
                job.gif_url = (
                    f"/api/gif?run={run_id}&kind={kind}"
                    f"&key={key}&i={i}&t={int(time.time())}"
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            # A render can fail many ways (bad approach, env error); surface it to
            # the UI job instead of killing the single worker thread.
            with _JOBS_LOCK:
                job.state = "error"
                job.message = f"{type(e).__name__}: {e}"
            if run is not None and history_version is not None:
                _record_history_episode(
                    run,
                    history_version,
                    i,
                    {
                        "episode_index": i,
                        "seed": replay_seed,
                        "object_count": replay_count,
                        "solved": False,
                        "crashed": True,
                        "num_steps": 0,
                        "error": f"{type(e).__name__}: {e}",
                    },
                )
        finally:
            _RENDER_Q.task_done()


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #

_LOG_ALLOW = {
    "run_experiment.log",
    "env_config.json",
    "env_description.md",
    "sandbox/agent_log.txt",
    "sandbox/CLAUDE.md",
    "sandbox/approach.py",
}


class Handler(BaseHTTPRequestHandler):
    """Routes API + static requests; all filesystem access is run-scoped."""

    def log_message(self, *_: Any) -> None:  # quiet
        pass

    # -- helpers --
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _err(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _safe(self, run: RunInfo, rel: str) -> Optional[Path]:
        """Resolve rel within the run dir; None if it escapes."""
        p = (run.path / rel).resolve()
        return p if p.is_relative_to(run.path.resolve()) else None

    # -- GET --
    def do_GET(self) -> None:  # noqa: N802
        """Route GET requests (static assets, run/metric/gif/history APIs)."""
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        path = u.path

        if path == "/":
            return self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
        if path == "/static/app.js":
            return self._send(200, APP_JS.encode(), "application/javascript")
        if path == "/static/app.css":
            return self._send(200, APP_CSS.encode(), "text/css")
        if path == "/api/runs":
            with _RUNS_LOCK:
                runs = list(RUNS.values())
            return self._json(
                [_summary(r) for r in sorted(runs, key=lambda r: r.run_id)]
            )
        if path == "/api/stamp":
            with _RUNS_LOCK:
                stamp = sum(
                    1 for r in RUNS.values() if (r.path / "results.json").exists()
                )
            return self._json({"stamp": f"{len(RUNS)}:{stamp}"})
        if path == "/api/job":
            with _JOBS_LOCK:
                j = JOBS.get(q.get("job", ""))
                return self._json(j.__dict__ if j else {"error": "unknown job"})

        # Every route past here is run-scoped (run-less routes returned above).
        run = _run(q.get("run", ""))
        if run is None:
            return self._err(404, "unknown run")

        if path == "/api/run":
            return self._json(_run_detail(run))
        if path == "/api/log":
            name = q.get("name", "")
            if name not in _LOG_ALLOW:
                return self._err(403, "not allowed")
            p = self._safe(run, name)
            if p is None or not p.exists():
                return self._err(404, "missing")
            return self._send(200, p.read_bytes(), "text/plain; charset=utf-8")
        if path == "/api/gif":
            kind, key = q.get("kind", "eval"), q.get("key", "0")
            if kind == "eval":
                p = _eval_gif_path(run, int(key))
            else:
                p = _history_gif_path(run, int(key), int(q.get("i", "0")))
            if p is None:
                return self._err(404, "not rendered")
            return self._send(200, p.read_bytes(), "image/gif")
        if path == "/api/history":
            return self._json(_snapshots(run))
        if path == "/api/history/training-seeds":
            return self._json(_training_seed_info(run))
        if path == "/api/history/source":
            return self._send(
                200,
                _approach_source(run, int(q["v"])).encode(),
                "text/plain; charset=utf-8",
            )
        if path == "/api/history/diff":
            return self._send(
                200,
                _approach_diff(run, int(q["a"]), int(q["b"])).encode(),
                "text/plain; charset=utf-8",
            )
        if path == "/api/history/why":
            return self._json(_why(run, int(q["v"])))
        return self._err(404, "no route")

    # -- POST --
    def do_POST(self) -> None:  # noqa: N802
        """Route POST requests (refresh, enqueue render job)."""
        u = urlparse(self.path)
        if u.path == "/api/refresh":
            refresh_runs()
            return self._json({"ok": True, "n": len(RUNS)})
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or "{}")
        if u.path == "/api/history/evaluate":
            run = _run(body.get("run", ""))
            if run is None:
                return self._err(404, "unknown run")
            return self._json({"job_id": _enqueue_history_evaluation(run.run_id)})
        if u.path == "/api/render":
            run = _run(body.get("run", ""))
            if run is None:
                return self._err(404, "unknown run")
            job_id = _enqueue_render(
                run.run_id, str(body.get("target", "HEAD")), int(body["episode_index"])
            )
            return self._json({"job_id": job_id})
        return self._err(404, "no route")


# --------------------------------------------------------------------------- #
# Frontend (HTML / CSS / JS as constants)
# --------------------------------------------------------------------------- #

INDEX_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>robocode results</title>
<link rel="stylesheet" href="/static/app.css">
</head><body>
<header><a href="#/index" class="brand">robocode results</a>
  <button id="refresh">refresh</button></header>
<main id="app"></main>
<script src="/static/app.js"></script>
</body></html>
"""

APP_CSS = """
:root{--bg:#fbfbfa;--fg:#20211f;--muted:#6b6f6a;--line:#e3e4e0;--card:#f3f4ef;
  --ok:#2f7d4f;--no:#c0492f;--warn:#b8860b;--accent:#3a6b5c;--code:#f6f7f2;}
@media(prefers-color-scheme:dark){:root{--bg:#12140f;--fg:#e9ebe4;--muted:#9aa094;
  --line:#2b2e26;--card:#1b1e17;--ok:#5fb07f;--no:#e08267;--warn:#d8b45a;
  --accent:#7fb3a1;--code:#171a13;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:12px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
.brand{font-weight:700;text-decoration:none;color:var(--fg)}
button{font:inherit;padding:5px 11px;border:1px solid var(--line);border-radius:7px;
  background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--accent)}button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
main{padding:18px 22px;max-width:1240px;margin:0 auto}
a{color:var(--accent)}
.facets{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:16px}
.facet{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.facet .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-right:2px}
.chip{padding:3px 9px;font-size:12.5px;border-radius:99px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:11px;padding:13px 14px;background:var(--card);text-decoration:none;color:var(--fg);display:block}
.card:hover{border-color:var(--accent)}
.card .id{font:12px ui-monospace,Menlo,monospace;color:var(--muted);word-break:break-all}
.sr{font-size:30px;font-weight:700;line-height:1.1;margin:4px 0}
.row{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.pill{font-size:11px;padding:2px 7px;border-radius:99px;background:var(--bg);border:1px solid var(--line);color:var(--muted)}
.pill.running{color:#fff;background:var(--warn);border-color:var(--warn)}
.pill.done{color:var(--ok);border-color:var(--ok)}
.pill.interrupted{color:var(--no);border-color:var(--no)}
h1{font-size:20px;margin:2px 0 4px}h2{font-size:15px;margin:22px 0 9px;border-bottom:1px solid var(--line);padding-bottom:5px}
.mono{font:12px/1.5 ui-monospace,Menlo,monospace}
.mgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:9px}
.metric{border:1px solid var(--line);border-radius:9px;padding:8px 10px;background:var(--card)}
.metric .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.metric .v{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}
.time-split{border:1px solid var(--line);border-radius:9px;padding:10px 12px;background:var(--card);margin-top:9px}
.time-track{height:18px;display:flex;border-radius:5px;overflow:hidden;background:var(--bg);border:1px solid var(--line)}
.time-claude{background:var(--accent)}.time-local{background:var(--warn)}.time-other{background:var(--muted)}
.time-legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:7px;font-variant-numeric:tabular-nums}
.time-key{display:inline-flex;align-items:center;gap:6px}.time-swatch{width:9px;height:9px;border-radius:2px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:800px){.cols{grid-template-columns:1fr}}
.eph{display:flex;flex-wrap:wrap;gap:8px}
.ep{width:132px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--card)}
.ep img{width:100%;display:block;aspect-ratio:1;object-fit:contain;background:#fff}
.ep .cap{font:11px ui-monospace,Menlo,monospace;color:var(--muted);padding:5px 6px}
.ep .ph{aspect-ratio:1;display:flex;align-items:center;justify-content:center}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}
.bars{display:flex;flex-direction:column;gap:5px}
.bar{display:flex;align-items:center;gap:8px;font-size:12px}
.bar .track{flex:1;height:14px;background:var(--bg);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.bar .fill{height:100%;background:var(--accent)}
.steps{display:flex;gap:6px;overflow-x:auto;padding-bottom:6px}
.step{flex:none;border:1px solid var(--line);border-radius:8px;padding:7px 10px;background:var(--card);cursor:pointer;min-width:150px}
.step.sel{border-color:var(--accent);background:var(--bg)}
.step .h{font:11px ui-monospace,Menlo,monospace;color:var(--muted)}
.step .m{font-size:12px;margin-top:3px}
.history-overview{margin-top:22px;border-top:1px solid var(--line);padding-top:12px}
.history-head{display:grid;grid-template-columns:minmax(280px,1fr) minmax(340px,430px);gap:18px;align-items:center}
.solve-lineplot{border:1px solid var(--line);border-radius:9px;padding:8px 10px;background:var(--card)}
.solve-lineplot svg{display:block;width:100%;height:auto}.solve-lineplot .title{font-size:11px;color:var(--muted);margin-bottom:3px}
@media(max-width:800px){.history-head{grid-template-columns:1fr}}
.chart-scroll,.matrix-scroll{overflow-x:auto;padding-bottom:6px}
.progress-chart{display:grid;gap:7px;min-width:max-content;align-items:end;margin-top:9px}
.progress-col{width:105px;border:1px solid var(--line);border-radius:8px;padding:7px;background:var(--card);cursor:pointer}
.progress-col:hover{border-color:var(--accent)}
.mini-metric{display:grid;grid-template-columns:27px 1fr 30px;gap:4px;align-items:center;font-size:9px;color:var(--muted);margin:3px 0}
.mini-track{height:8px;border-radius:3px;background:var(--bg);border:1px solid var(--line);overflow:hidden}
.mini-fill{height:100%;background:var(--accent)}.mini-fill.tokens{background:var(--warn)}.mini-fill.time{background:var(--muted)}
.chart-legend{display:flex;gap:13px;flex-wrap:wrap;font-size:11px;color:var(--muted)}
.seed-matrix{border-collapse:separate;border-spacing:4px;font-size:11px;min-width:max-content}
.seed-matrix th{font-weight:500;color:var(--muted);text-align:center}.seed-matrix th:first-child{text-align:left}
.seed-cell{width:54px;padding:5px 3px;font:11px ui-monospace,Menlo,monospace}
.seed-cell.ok{color:var(--ok);border-color:var(--ok)}.seed-cell.fail{color:var(--no);border-color:var(--no)}
.seed-cell.unknown{color:var(--muted);border-style:dashed}.seed-cell.busy{color:var(--warn)}
.seed-cell.tested{color:#fff;background:var(--accent);border-color:var(--accent)}
.seed-label{white-space:nowrap;text-align:left}.seed-label .mono{font-size:10px}
.seed-badge{font-size:9px;padding:1px 5px;border-radius:99px;border:1px solid var(--line);margin-left:4px;font-weight:500}
.seed-badge.fixed{color:var(--ok);border-color:var(--ok)}.seed-badge.regression{color:var(--warn);border-color:var(--warn)}
.seed-badge.persistent{color:var(--no);border-color:var(--no)}
pre{background:var(--code);border:1px solid var(--line);border-radius:9px;padding:12px;overflow:auto;font:12px/1.5 ui-monospace,Menlo,monospace;max-height:520px}
.diff .add{color:var(--ok)}.diff .del{color:var(--no)}
.why{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:11px 13px;white-space:pre-wrap;font-size:13px;max-height:340px;overflow:auto}
.muted{color:var(--muted)}details summary{cursor:pointer;color:var(--accent)}
.lb{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.82);display:flex;
  align-items:center;justify-content:center;cursor:zoom-out;padding:24px}
.lb img{max-width:92vw;max-height:88vh;image-rendering:auto;border-radius:10px;
  box-shadow:0 12px 48px rgba(0,0,0,.5);cursor:default}
.lbx{position:fixed;top:16px;right:18px}
"""

APP_JS = r"""
const $=(s,r=document)=>r.querySelector(s), h=(t,a={},...k)=>{const e=document.createElement(t);
 for(const[x,y]of Object.entries(a)){if(x==="class")e.className=y;else if(x==="html")e.innerHTML=y;
 else if(x.startsWith("on"))e.addEventListener(x.slice(2),y);else e.setAttribute(x,y);}
 for(const c of k.flat())e.append(c?.nodeType?c:document.createTextNode(c??""));return e;};
const j=(u,o)=>fetch(u,o).then(r=>r.json());
const num=(x,d=2)=>x==null?"-":(typeof x==="number"?(Math.abs(x)>=1000?x.toFixed(0):x.toFixed(d)):x);
const enc=encodeURIComponent;
const duration=x=>x==null?"-":x<60?`${Math.round(x)}s`:x<3600?`${(x/60).toFixed(1)}m`:`${(x/3600).toFixed(1)}h`;
const compact=x=>x==null?"-":x>=1e6?`${(x/1e6).toFixed(1)}m`:x>=1e3?`${(x/1e3).toFixed(1)}k`:String(x);

// A gif thumbnail that opens a large, replaying view on click.
function gifImg(src){return h("img",{loading:"lazy",src,style:"cursor:zoom-in",title:"click to enlarge",
 onclick:()=>openLightbox(src)});}
function openLightbox(src){
 // new <img> element => the gif restarts from frame 0 ("see it play")
 const ov=h("div",{class:"lb",onclick:()=>ov.remove()},
  h("img",{src,onclick:e=>e.stopPropagation()}),
  h("button",{class:"lbx",onclick:()=>ov.remove()},"close (esc)"));
 const esc=e=>{if(e.key==="Escape"){ov.remove();document.removeEventListener("keydown",esc);}};
 document.addEventListener("keydown",esc);
 document.body.append(ov);
}

let ALL=[], FILTER={};
async function loadRuns(){ALL=await j("/api/runs");}

function facetBar(){
 const keys=["approach","environment","seed","budget"];
 const bar=h("div",{class:"facets"});
 for(const k of keys){
  const vals=[...new Set(ALL.map(r=>r[k]).filter(v=>v!=null))].sort();
  if(vals.length<2)continue;
  const f=h("div",{class:"facet"},h("span",{class:"lbl"},k));
  for(const v of vals){
   const on=(FILTER[k]||[]).includes(String(v));
   f.append(h("button",{class:"chip"+(on?" active":""),onclick:()=>{
    FILTER[k]=FILTER[k]||[];const i=FILTER[k].indexOf(String(v));
    if(i<0)FILTER[k].push(String(v));else FILTER[k].splice(i,1);
    if(!FILTER[k].length)delete FILTER[k];location.hash="#/index";renderIndex();}},String(v)));
  }
  bar.append(f);
 }
 return bar;
}
function match(r){for(const[k,vs]of Object.entries(FILTER))if(vs.length&&!vs.includes(String(r[k])))return false;return true;}

function srColor(v){if(v==null)return"var(--muted)";const g=Math.round(160*v);return`rgb(${160-g},${90+g*0.6},${70+g*0.3})`;}

function renderIndex(){
 const app=$("#app");app.innerHTML="";
 app.append(facetBar());
 const runs=ALL.filter(match);
 app.append(h("div",{class:"muted",style:"margin-bottom:8px"},`${runs.length} run(s)`));
 const g=h("div",{class:"grid"});
 for(const r of runs){
  g.append(h("a",{class:"card",href:"#/run/"+enc(r.run_id)},
   h("div",{class:"sr",style:`color:${srColor(r.solve_rate)}`},r.solve_rate==null?"-":r.solve_rate.toFixed(2)),
   h("div",{class:"id"},r.run_id),
   h("div",{class:"row"},
    h("span",{class:"pill "+r.status},r.status),
    r.budget!=null?h("span",{class:"pill"},"$"+r.budget):"",
    r.seed!=null?h("span",{class:"pill"},"s"+r.seed):"",
    r.agent_cost_usd!=null?h("span",{class:"pill"},"$"+num(r.agent_cost_usd)):"",
    r.num_crashed_episodes?h("span",{class:"pill"},r.num_crashed_episodes+" crash"):"",
    r.has_history?h("span",{class:"pill"},"history"):"" )));
 }
 app.append(g);
}

const GENK=["gen_num_turns","gen_total_tokens","gen_output_tokens","gen_num_tool_calls","gen_num_autocompactions","gen_stop_reason","gen_wall_time_s","gen_cli_duration_ms"];
const TOPK=["solve_rate","mean_eval_reward","mean_eval_steps","num_evaluated_episodes","num_crashed_episodes","agent_cost_usd","largest_count_all_solved","largest_count_any_solved"];

function metricCard(k,v){return h("div",{class:"metric"},h("div",{class:"k"},k.replace(/_/g," ")),h("div",{class:"v"},num(v)));}

function generationTimeSplit(b){
 const claudePct=b.claude_fraction*100,localPct=b.experiments_tools_fraction*100,otherPct=b.other_fraction*100;
 const key=(color,label,value)=>h("span",{class:"time-key"},
  h("span",{class:"time-swatch",style:`background:${color}`}),
  h("span",{},`${label}: ${value}`));
 const box=h("div",{class:"time-split"},
  h("div",{class:"muted",style:"font-size:11px;margin-bottom:6px"},"generation time estimate"),
  h("div",{class:"time-track",title:`Measured from ${b.basis}`},
   h("div",{class:"time-claude",style:`width:${claudePct}%`}),
   h("div",{class:"time-local",style:`width:${localPct}%`}),
   h("div",{class:"time-other",style:`width:${otherPct}%`})),
  h("div",{class:"time-legend"},
   key("var(--accent)","Claude API / thinking",`${claudePct.toFixed(1)}% · ${duration(b.claude_s)}`),
   key("var(--warn)",b.instrumented?"environment experiments":"experiments & tools",`${localPct.toFixed(1)}% · ${duration(b.experiments_tools_s)}`),
   b.instrumented?key("var(--muted)","other tools / overhead",`${otherPct.toFixed(1)}% · ${duration(b.other_s)}`):""));
 box.append(h("div",{class:"muted",style:"font-size:10.5px;margin-top:6px"},
  b.instrumented?
   "Basis: live tool-call/result wall timing. Parallel calls are counted once; sandbox startup and unclassified time are included in overhead.":
   `Basis: ${b.basis}. The non-Claude remainder includes environment runs, shell/file tools, CLI overhead, and sandbox startup; historical logs cannot isolate experiments exactly.`,
  b.clamped?" Claude API time exceeded the available timer and was capped.":""));
 return box;
}

function byCountBars(bc){
 const wrap=h("div",{class:"bars"});
 for(const c of Object.keys(bc).sort((a,b)=>+a-+b)){
  const o=bc[c],sr=o.solve_rate||0;
  wrap.append(h("div",{class:"bar"},h("span",{style:"width:60px"},c+" obj"),
   h("div",{class:"track"},h("div",{class:"fill",style:`width:${sr*100}%`})),
   h("span",{class:"mono",style:"width:120px"},`${(sr*100).toFixed(0)}%  ${num(o.mean_num_steps,0)} steps`)));
 }
 return wrap;
}

function epCard(run,e){
 const c=h("div",{class:"ep"});
 const color=e.outcome==="success"?"var(--ok)":e.outcome==="failure"?"var(--no)":"var(--warn)";
 const cap=h("div",{class:"cap"},h("span",{class:"dot",style:`background:${color}`}),
  `ep${e.i} c${e.object_count} ${num(e.num_steps,0)}st`);
 if(e.has_gif){c.append(gifImg(`/api/gif?run=${enc(run)}&kind=eval&key=${e.i}`));}
 else{const ph=h("div",{class:"ph"},h("button",{onclick:ev=>recordGif(ev,run,"HEAD",e.i,c)},"record gif"));c.append(ph);}
 c.append(cap);return c;
}

async function recordGif(ev,run,target,i,cardEl){
 const btn=ev.target;btn.disabled=true;btn.textContent="rendering...";
 const {job_id}=await j("/api/render",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({run,target,episode_index:i})});
 const poll=setInterval(async()=>{
  const s=await j("/api/job?job="+job_id);
  if(s.state==="done"){clearInterval(poll);
   const img=gifImg(s.gif_url);
   const ph=cardEl.querySelector(".ph");if(ph)ph.replaceWith(img);else cardEl.prepend(img);}
  else if(s.state==="error"){clearInterval(poll);btn.disabled=false;btn.textContent="retry";
   btn.after(h("div",{class:"mono",style:"color:var(--no);font-size:10px;padding:4px"},s.message));}
  else{btn.textContent=s.state;}
 },1500);
}

async function renderRun(id){
 const app=$("#app");app.innerHTML="loading...";
 const d=await j("/api/run?run="+enc(id));
 const m=d.metrics,s=d.summary;app.innerHTML="";
 app.append(h("h1",{},id),
  h("div",{class:"row"},h("span",{class:"pill "+s.status},s.status),
   s.budget!=null?h("span",{class:"pill"},"$"+s.budget):"",s.seed!=null?h("span",{class:"pill"},"s"+s.seed):"",
   h("span",{class:"pill"},s.approach),h("span",{class:"pill"},s.environment)));
 // logs
 const logs=h("div",{class:"row",style:"margin-top:8px"});
 for(const l of d.logs)logs.append(h("a",{class:"pill",href:`/api/log?run=${enc(id)}&name=${enc(l)}`,target:"_blank"},l));
 app.append(logs);
 if(d.env_description)app.append(h("details",{style:"margin-top:8px"},h("summary",{},"env description"),
  h("pre",{},d.env_description)));

 // metrics
 app.append(h("h2",{},"metrics"));
 const mg=h("div",{class:"mgrid"});
 for(const k of TOPK)if(k in m)mg.append(metricCard(k,m[k]));
 app.append(mg);
 app.append(h("h2",{},"generation"));
 const gg=h("div",{class:"mgrid"});
 for(const k of GENK)if(k in m)gg.append(metricCard(k,m[k]));
 app.append(gg);
 if(d.generation_time_breakdown)app.append(generationTimeSplit(d.generation_time_breakdown));
 if(m.by_count){app.append(h("h2",{},"scaling by object count"));app.append(byCountBars(m.by_count));}

 // episodes
 const succ=d.episodes.filter(e=>e.outcome==="success"),fail=d.episodes.filter(e=>e.outcome!=="success");
 app.append(h("h2",{},`episodes (${succ.length} solved / ${fail.length} failed)`));
 const cols=h("div",{class:"cols"});
 const cS=h("div",{},h("div",{class:"muted",style:"margin-bottom:6px"},"Solved"),h("div",{class:"eph"}));
 const cF=h("div",{},h("div",{class:"muted",style:"margin-bottom:6px"},"Failed"),h("div",{class:"eph"}));
 for(const e of succ.slice(0,60))cS.querySelector(".eph").append(epCard(id,e));
 for(const e of fail.slice(0,60))cF.querySelector(".eph").append(epCard(id,e));
 cols.append(cS,cF);app.append(cols);

 if(d.has_history)await renderHistory(app,id,d.episodes);
}

async function renderHistory(app,id,episodes){
 app.append(h("h2",{},"approach history"));
 const [snaps,training]=await Promise.all([
  j("/api/history?run="+enc(id)),j("/api/history/training-seeds?run="+enc(id))]);
 if(!snaps.length){app.append(h("div",{class:"muted"},"no versioned approach.py commits"));return;}
 const steps=h("div",{class:"steps"});
 const panel=h("div",{});
 let sel=snaps.length-1;
 const pick=async v=>{sel=v;[...steps.children].forEach((c,i)=>c.classList.toggle("sel",i===v));
  await showVersion(panel,id,snaps,v);};
 snaps.forEach((sn,i)=>steps.append(h("div",{class:"step",onclick:()=>pick(i)},
  h("div",{class:"h"},`v${sn.version} ${sn.short}`),h("div",{class:"m"},sn.message))));
 app.append(steps,panel);await pick(sel);
 app.append(historyOverview(id,snaps,episodes,pick,training));
}

function solveLinePlot(snaps,totalEpisodes){
 const W=420,H=145,L=34,R=12,T=10,B=25,iw=W-L-R,ih=H-T-B;
 const x=i=>L+(snaps.length===1?iw/2:i*iw/(snaps.length-1));
 const y=rate=>T+(1-rate)*ih;
 const complete=snaps.map((sn,i)=>sn.evaluation.evaluated===totalEpisodes?{i,rate:sn.evaluation.solve_rate}:null);
 const segments=[];let segment=[];
 complete.forEach(point=>{if(point)segment.push(point);else if(segment.length){segments.push(segment);segment=[];}});
 if(segment.length)segments.push(segment);
 const grid=[0,.5,1].map(rate=>`<line x1="${L}" y1="${y(rate)}" x2="${W-R}" y2="${y(rate)}" stroke="var(--line)"/><text x="${L-5}" y="${y(rate)+3}" text-anchor="end" fill="var(--muted)" font-size="9">${Math.round(rate*100)}%</text>`).join("");
 const paths=segments.map(points=>`<polyline points="${points.map(p=>`${x(p.i)},${y(p.rate)}`).join(" ")}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round"/>`).join("");
 const dots=complete.filter(Boolean).map(p=>`<circle cx="${x(p.i)}" cy="${y(p.rate)}" r="4" fill="var(--accent)"><title>v${p.i}: ${Math.round(p.rate*100)}% (${totalEpisodes} seeds)</title></circle>`).join("");
 const labels=snaps.map((sn,i)=>`<text x="${x(i)}" y="${H-6}" text-anchor="middle" fill="var(--muted)" font-size="9">v${i}</text>`).join("");
 return h("div",{class:"solve-lineplot"},h("div",{class:"title"},`full-suite solve rate · ${totalEpisodes} identical seeds`),
  h("div",{html:`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="solve rate by commit">${grid}${paths}${dots}${labels}</svg>`}));
}

function trainingSeedEvolution(snaps,training){
 if(!training.seeds.length)return null;
 const section=h("div",{},h("h2",{},"training-seed exploration"),
  h("div",{class:"muted"},
   `Recovered ${training.seeds.length} literal environment seeds from the agent's test commands. Filled cells show when a seed was exercised around that commit; they are training probes, not held-out eval episodes.`));
 const matrix=items=>{
  const table=h("table",{class:"seed-matrix"});
  const head=h("tr",{},h("th",{},"training seed"));
  snaps.forEach((sn,i)=>head.append(h("th",{},`v${i}`)));head.append(h("th",{},"mentions"));table.append(head);
  items.forEach(item=>{
   const evidence=item.evidence.join("\n");
   const row=h("tr",{},h("th",{class:"seed-label"},`seed ${item.seed}`));
   snaps.forEach((sn,v)=>row.append(h("td",{},item.versions.includes(v)?
    h("button",{class:"seed-cell tested",title:evidence},"tested"):
    h("span",{class:"seed-cell muted",style:"display:inline-block;text-align:center"},"·"))));
   row.append(h("td",{class:"mono muted"},String(item.mentions)));table.append(row);
  });
  return h("div",{class:"matrix-scroll"},table);
 };
 const focused=training.seeds.slice(0,20);
 section.append(matrix(focused));
 if(training.seeds.length>focused.length){
  const more=h("details",{},h("summary",{},`show all ${training.seeds.length} recovered seeds`));
  more.append(matrix(training.seeds));section.append(more);
 }
 return section;
}

function seedStory(snaps,index){
 const values=snaps.map(sn=>sn.evaluation.episodes[String(index)]?.solved);
 let firstFail=values.findIndex(v=>v===false),firstFix=-1,firstRegression=-1,recovery=-1;
 if(firstFail>=0)for(let i=firstFail+1;i<values.length;i++)if(values[i]===true){firstFix=i;break;}
 const firstSolve=values.findIndex(v=>v===true);
 if(firstSolve>=0)for(let i=firstSolve+1;i<values.length;i++)if(values[i]===false){firstRegression=i;break;}
 if(firstRegression>=0)for(let i=firstRegression+1;i<values.length;i++)if(values[i]===true){recovery=i;break;}
 const lastKnown=[...values].reverse().find(v=>v!=null);
 if(firstRegression>=0&&recovery>=0)return{priority:0,cls:"regression",label:`regressed v${firstRegression} · recovered v${recovery}`};
 if(firstFail>=0&&firstFix>=0)return{priority:1,cls:"fixed",label:`fixed in v${firstFix}`};
 if(firstRegression>=0)return{priority:2,cls:"regression",label:`regressed in v${firstRegression}`};
 if(lastKnown===false)return{priority:3,cls:"persistent",label:"still failing"};
 return{priority:4,cls:"",label:"observed failure"};
}

function historyOverview(id,snaps,episodes,pick,training){
 const wrap=h("section",{class:"history-overview"},h("h2",{},"progress by commit"));
 const legend=h("div",{class:"chart-legend"},
  h("span",{},"solve = recorded replay solve rate"),h("span",{},"tokens = agent tokens before commit"),
  h("span",{},"time = time since previous commit"));
 const incomplete=snaps.slice(0,-1).filter(sn=>sn.evaluation.evaluated<episodes.length);
 if(incomplete.length){
  const rollouts=incomplete.length*episodes.length;
  const evaluate=h("button",{class:"chip",onclick:async()=>{
   evaluate.disabled=true;
   evaluate.textContent="queueing full evaluation...";
   const {job_id}=await j("/api/history/evaluate",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({run:id})});
   const poll=setInterval(async()=>{const state=await j("/api/job?job="+job_id);
    const progress=state.total?` ${state.current}/${state.total}`:"";
    evaluate.textContent=state.message+progress;
    if(state.state==="done"){clearInterval(poll);renderRun(id);}
    if(state.state==="error"){clearInterval(poll);evaluate.disabled=false;evaluate.classList.add("fail");}
   },1500);
  }},`evaluate all ${episodes.length} seeds × ${incomplete.length} commits (${rollouts} rollouts)`);
  legend.append(evaluate);
 }
 wrap.append(h("div",{class:"history-head"},legend,solveLinePlot(snaps,episodes.length)));
 const maxTok=Math.max(1,...snaps.map(s=>s.effort.tokens||0));
 const maxTime=Math.max(1,...snaps.map(s=>s.effort.elapsed_s||0));
 const chart=h("div",{class:"progress-chart",style:`grid-template-columns:repeat(${snaps.length},105px)`});
 const metric=(label,value,pct,cls="")=>h("div",{class:"mini-metric"},h("span",{},label),
  h("div",{class:"mini-track"},h("div",{class:"mini-fill "+cls,style:`width:${Math.max(0,Math.min(100,pct))}%`})),
  h("span",{},value));
 snaps.forEach((sn,i)=>{
  const ev=sn.evaluation,e=sn.effort;
  const fullRate=ev.evaluated===episodes.length?ev.solve_rate:null;
  chart.append(h("div",{class:"progress-col",onclick:()=>pick(i),title:sn.message},
   h("div",{class:"mono",style:"margin-bottom:5px"},`v${i} ${sn.short}`),
   metric("solve",fullRate==null?"-":`${Math.round(fullRate*100)}%`,fullRate==null?0:fullRate*100),
   metric("tok",compact(e.tokens),e.tokens==null?0:e.tokens/maxTok*100,"tokens"),
   metric("time",duration(e.elapsed_s),e.elapsed_s==null?0:e.elapsed_s/maxTime*100,"time"),
   h("div",{class:"muted",style:"font-size:10px;margin-top:5px"},
    `${ev.evaluated}/${episodes.length} seeds · +${e.additions}/-${e.deletions}`)));
 });
 wrap.append(h("div",{class:"chart-scroll"},chart));
 const trainingDiagram=trainingSeedEvolution(snaps,training);
 if(trainingDiagram)wrap.append(trainingDiagram);

 const episodeById=Object.fromEntries(episodes.map(e=>[e.i,e]));
 const relevant=new Set(episodes.filter(e=>e.outcome!=="success").map(e=>e.i));
 snaps.forEach(sn=>Object.entries(sn.evaluation.episodes).forEach(([i,e])=>{if(!e.solved)relevant.add(+i);}));
 if(!relevant.size){wrap.append(h("div",{class:"muted",style:"margin-top:12px"},"No failed seeds recorded."));return wrap;}
 wrap.append(h("h2",{},"interesting held-out seed evolution"),h("div",{class:"muted"},
  "Prioritizes failures that were fixed, regressions, and persistent failures. Each row is the same held-out seed across code versions; click a cell to record or view its replay."));
 const table=h("table",{class:"seed-matrix"});
 const head=h("tr",{},h("th",{},"episode / seed"));
 snaps.forEach((sn,i)=>head.append(h("th",{},`v${i}`)));head.append(h("th",{},"compare"));table.append(head);
 [...relevant].map(i=>({i,story:seedStory(snaps,i)})).sort((a,b)=>a.story.priority-b.story.priority||a.i-b.i).forEach(({i,story})=>{
  const ep=episodeById[i]||{};
  const row=h("tr",{},h("th",{class:"seed-label"},`ep${i} `,
   h("span",{class:"mono muted",title:String(ep.seed??"")},ep.seed==null?"":String(ep.seed).slice(-8)),
   h("span",{class:`seed-badge ${story.cls}`},story.label)));
  const cells=[];
  snaps.forEach((sn,v)=>{
   const rec=sn.evaluation.episodes[String(i)];
   const cls=rec?(rec.solved?"ok":"fail"):"unknown";
   const label=rec?(rec.solved?"solved":rec.outcome==="crashed"?"crash":"failed"):"record";
   const cell=h("button",{class:`seed-cell ${cls}`,title:rec?.error||`${sn.message} · episode ${i}`},label);
   cell.addEventListener("click",async()=>{
    if(rec?.has_gif){const kind=rec.gif_kind||"history",key=kind==="eval"?i:v;
     return openLightbox(`/api/gif?run=${enc(id)}&kind=${kind}&key=${key}&i=${i}`);}
    cell.className="seed-cell busy";cell.textContent="queued";
    await replayOne(id,v,i,cell);renderRun(id);
   });
   cells.push({cell,rec,v});row.append(h("td",{},cell));
  });
  const all=h("button",{class:"chip",onclick:async()=>{
   all.disabled=true;all.textContent="replaying...";
   for(const x of cells.filter(x=>!x.rec)){x.cell.className="seed-cell busy";x.cell.textContent="queued";}
   await Promise.all(cells.filter(x=>!x.rec).map(x=>replayOne(id,x.v,i,x.cell)));
   renderRun(id);
  }},"replay row");
  row.append(h("td",{},all));table.append(row);
 });
 wrap.append(h("div",{class:"matrix-scroll"},table));
 return wrap;
}

async function replayOne(id,v,i,cell){
 const {job_id}=await j("/api/render",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({run:id,target:String(v),episode_index:i})});
 return new Promise(resolve=>{const poll=setInterval(async()=>{const s=await j("/api/job?job="+job_id);
  if(s.state==="running"){cell.textContent="running";}
  if(s.state==="done"||s.state==="error"){clearInterval(poll);cell.textContent=s.state;resolve(s);}},1500);});
}

async function showVersion(panel,id,snaps,v){
 panel.innerHTML="loading...";
 const sn=snaps[v];
 const [src,why]=await Promise.all([
  fetch(`/api/history/source?run=${enc(id)}&v=${v}`).then(r=>r.text()),
  j(`/api/history/why?run=${enc(id)}&v=${v}`)]);
 panel.innerHTML="";
 const ev=sn.evaluation,e=sn.effort;
 panel.append(h("div",{class:"mono muted",style:"margin:6px 0"},`${sn.short} · ${sn.timestamp}`),
  h("div",{class:"row"},
   h("span",{class:"pill"},`${ev.evaluated} replays · ${ev.solve_rate==null?"-":Math.round(ev.solve_rate*100)+"%"} solved`),
   h("span",{class:"pill"},`${compact(e.tokens)} tokens`),h("span",{class:"pill"},`${duration(e.elapsed_s)} effort`),
   h("span",{class:"pill"},`+${e.additions} / -${e.deletions} lines`)));
 // why
 panel.append(h("div",{class:"muted",style:"margin:8px 0 4px"},`why (matched: ${why.matched_by})`));
 panel.append(h("div",{class:"why"},(why.thinking||why.text||"(no reasoning found)")));
 // controls: full-source / diff toggle + record
 const ctl=h("div",{class:"row",style:"margin:10px 0"});
 const view=h("div",{});
 const bSrc=h("button",{},`full approach.py (${src.split("\n").length} lines)`);
 const bDiff=h("button",{},"diff vs prev");
 const setActive=b=>[bSrc,bDiff].forEach(x=>x.classList.toggle("active",x===b));
 const showSrc=()=>{setActive(bSrc);view.innerHTML="";view.append(h("pre",{},src));};
 const showDiff=async()=>{setActive(bDiff);
  if(v===0){view.innerHTML="";view.append(h("div",{class:"muted",style:"margin:6px 0"},"v0 has no previous version"),h("pre",{},src));return;}
  const t=await fetch(`/api/history/diff?run=${enc(id)}&a=${v-1}&b=${v}`).then(r=>r.text());
  view.innerHTML="";const pre=h("pre",{class:"diff"});
  t.split("\n").forEach(l=>{const cls=l.startsWith("+")?"add":l.startsWith("-")?"del":"";
   pre.append(h("span",{class:cls},l+"\n"));});view.append(pre);};
 bSrc.addEventListener("click",showSrc);bDiff.addEventListener("click",showDiff);
 ctl.append(bSrc,bDiff);
 // record for this version on a chosen episode
 const epIn=h("input",{type:"number",value:"0",min:"0",style:"width:70px;padding:5px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)"});
 const recBtn=h("button",{onclick:ev=>recordGif(ev,id,String(v),+epIn.value,gifBox)},"record gif");
 ctl.append(h("span",{class:"muted",style:"margin-left:10px"},"episode"),epIn,recBtn);
 const gifBox=h("div",{class:"ep",style:"width:180px;margin-top:8px"},h("div",{class:"ph muted"},"no gif yet"));
 panel.append(ctl,gifBox,view);
 showDiff();
}

async function route(){
 const hash=location.hash||"#/index";
 if(!ALL.length)await loadRuns();
 if(hash.startsWith("#/run/"))renderRun(decodeURIComponent(hash.slice(6)));
 else renderIndex();
}
window.addEventListener("hashchange",route);
$("#refresh").addEventListener("click",async()=>{await fetch("/api/refresh",{method:"POST"});await loadRuns();route();});
// auto-refresh on the index when runs finish
let lastStamp="";
setInterval(async()=>{if(!location.hash.startsWith("#/run/")){const{stamp}=await j("/api/stamp");
 if(lastStamp&&stamp!==lastStamp){await loadRuns();if(!location.hash.startsWith("#/run/"))renderIndex();}lastStamp=stamp;}},10000);
route();
"""


def main() -> None:
    """Discover runs, start the render worker, serve forever."""
    ap = argparse.ArgumentParser(description="robocode results viewer")
    ap.add_argument("--root", default=str(REPO_ROOT), help="root to scan for runs")
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="DIR",
        help="directory names to prune from the scan (e.g. --exclude outputs multirun)",
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; pass 0.0.0.0 to reach the viewer from another machine",
    )
    args = ap.parse_args()

    SCAN.root = Path(args.root).resolve()
    SCAN.exclude_dirs = set(args.exclude)
    refresh_runs()
    print(f"discovered {len(RUNS)} runs under {SCAN.root}")

    threading.Thread(target=_render_worker, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
