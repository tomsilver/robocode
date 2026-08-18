"""Local web viewer for robocode experiment runs.

A stdlib-only HTTP server (no framework) that discovers run directories under a
root, shows metrics, per-episode success/failure GIFs, the sandbox git history
of the generated approach.py (with diffs and the agent's reasoning per commit),
and can re-render a GIF for any episode on the current or a past approach
version. Modeled on the TensorBoard-style predicators log_viewer.

The scan walks --root recursively and finds every run regardless of where it
lives (outputs/, multirun/, or a curated collection like viewer_runs/). Point
--root at a subtree to view only that subtree, or pass --exclude to prune
directory names from a broader scan. Alternatively, --drive-folder downloads
ZIP archives from a Google Drive folder into a private local cache before
scanning. Drive access is read-only; rendered GIFs remain in that local cache.

Launch:  python -m experiments.results_viewer --root . --port 8000
         python -m experiments.results_viewer --root viewer_runs
         python -m experiments.results_viewer --exclude outputs multirun
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Hashable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import imageio.v3 as iio
import imageio_ffmpeg

from experiments import drive_results

# Repo root (this file lives in <repo>/experiments/).
REPO_ROOT = Path(__file__).resolve().parent.parent
# The viewer is often launched with a virtualenv from the experiment checkout it
# is inspecting. Prefer this checkout's implementation while reusing that env's
# installed third-party dependencies.
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
    primitives: str  # coarse bucket: "none", "low level", or "bilevel"
    seed: Optional[int]
    budget: Optional[float]
    num_eval_tasks: int
    per_instance: bool
    # Cells whose suite or policy is not the run's own synthesis record these in an
    # env_variant.json marker next to results.json (see the reset-distribution
    # campaigns, where band and standard cells record byte-identical env configs).
    policy_source: Optional[str] = None  # scan-root-relative dir holding sandbox/
    eval_seed: Optional[int] = None  # pinned eval-suite seed; None follows `seed`
    explicit_seeds: bool = False  # per_episode records carry their own seeds
    # The environment group whose sampler the policy saw during synthesis.
    # `environment` names the evaluated suite; the two differ in cross-eval cells.
    trained_environment: str = ""
    # Conditioned instance slice this cell was scored on (e.g. "high-button
    # mid-stick"); empty for the plain uniform suite. Display-only label.
    slice: str = ""
    # Short label for the synthesis model (e.g. "opus-5"); None when the run has
    # no LLM backend.
    model: Optional[str] = None
    # "whitebox" when the agent could read the environment source, "blackbox" when it
    # could not; None for approaches with no such setting.
    env_access: Optional[str] = None


@dataclass
class _Scan:
    """Where to look for runs; populated from argv at startup."""

    root: Path = REPO_ROOT
    exclude_dirs: set[str] = field(default_factory=set)  # extra dirs to prune
    tag: str = ""  # "<root name>:<port>", identifies this instance in copied refs


SCAN = _Scan()
RUNS: dict[str, RunInfo] = {}
_RUNS_LOCK = threading.Lock()
DRIVE_SYNC: Any = None


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


def _primitives_category(value: str) -> str:
    """Group a primitives override value into the viewer's coarse filter buckets.

    ``none`` for an empty set, ``bilevel`` when the bilevel planning primitive is
    granted, and ``low level`` for any other geometric/controller primitive set.
    """
    names = [n for n in re.split(r"[\[\],\s]+", value) if n]
    if not names:
        return "none"
    if any("bilevel" in n for n in names):
        return "bilevel"
    return "low level"


def _variant_marker(run_dir: Path) -> dict[str, Any]:
    """Optional env_variant.json next to results.json, labelling special cells.

    Records the environment group whose sampler produced the recorded suite, the
    run dir holding the evaluated policy for re-eval cells, and, for cells without
    their own .hydra, enough (approach/primitives/eval_seed) to rebuild a replay.
    """
    f = run_dir / "env_variant.json"
    if not f.exists():
        return {}
    try:
        marker = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return marker if isinstance(marker, dict) else {}


def _policy_dir(run: RunInfo) -> Path:
    """The run dir whose sandbox/ holds the policy this run evaluated."""
    return SCAN.root / run.policy_source if run.policy_source else run.path


def _environment_of_dir(run_dir: Path) -> Optional[str]:
    """The environment group a run dir's own records name, marker first."""
    marker = _variant_marker(run_dir)
    if marker.get("environment"):
        return str(marker["environment"])
    hydra_dir = run_dir / ".hydra"
    ov = _parse_overrides(hydra_dir)
    return ov.get("environment") or _choice(hydra_dir, "environment")


def _replay_primitives(run: RunInfo) -> str:
    """The primitives override a replay must rebuild for this run's policy."""
    marker = _variant_marker(run.path)
    if isinstance(marker.get("primitives"), str):
        return marker["primitives"]
    if (run.path / ".hydra").is_dir():
        return _primitives(run.path / ".hydra")
    if run.policy_source:
        return _primitives(_policy_dir(run) / ".hydra")
    return "[]"


# Short labels for approach.backend.model values. Bare names are CLI aliases;
# the pinned sandbox image resolves them to these versions.
_MODEL_LABELS = {
    "claude-opus-5": "opus-5",
    "claude-sonnet-5": "sonnet-5",
    "opus": "opus-4.8",
    "sonnet": "sonnet-4.6",
}


def _backend_model(hydra_dir: Path) -> Optional[str]:
    """Short label for approach.backend.model from the resolved config.

    ``model:`` only occurs under ``approach.backend`` in the resolved config, so a
    flat scan is safe; runs without an LLM backend (e.g. bilevel planning) have no
    such line and return None.
    """
    cfg = hydra_dir / "config.yaml"
    if not cfg.exists():
        return None
    m = re.search(r"^\s+model:\s*(\S+)\s*$", cfg.read_text(), re.MULTILINE)
    if m is None:
        return None
    raw = m.group(1).strip("'\"")
    return _MODEL_LABELS.get(raw, raw.removeprefix("claude-"))


_ACCESS_LABELS = {"true": "blackbox", "false": "whitebox"}


def _env_access(hydra_dir: Path) -> Optional[str]:
    """Whether the agent could read the environment source, from the resolved config.

    ``blackbox:`` only occurs under ``approach`` in the resolved config, so a flat scan
    is safe; approaches without the flag (e.g. bilevel planning) return None.
    """
    cfg = hydra_dir / "config.yaml"
    if cfg.exists():
        m = re.search(r"^\s+blackbox:\s*(\S+)\s*$", cfg.read_text(), re.MULTILINE)
        if m is not None:
            return _ACCESS_LABELS.get(m.group(1).strip("'\"").lower())
    raw = _parse_overrides(hydra_dir).get("approach.blackbox")
    return _ACCESS_LABELS.get(raw.strip("'\"").lower()) if raw else None


# Environment pairs that sample the same task family from different spawn ranges;
# an episode of one can be replayed under the other for a matched comparison.
_ENV_COUNTERPART = {
    "stickbutton2d_generalized": "stickbutton2d_generalized_a",
    "stickbutton2d_generalized_a": "stickbutton2d_generalized",
}


def _configured_object_counts(run: RunInfo) -> dict[str, list[int]]:
    """Read variable-object-count train/eval domains from resolved Hydra YAML.

    Keep this deliberately small and stdlib-only: the resolved config emitted by
    Hydra uses a simple top-level ``environment`` mapping and integer lists.
    """
    config = run.path / ".hydra" / "config.yaml"
    if not config.exists():
        return {"design": [], "eval": []}
    text = config.read_text()
    environment = re.search(
        r"^environment:\s*\n(?P<body>(?:^[ \t]+.*\n?)*)", text, re.MULTILINE
    )
    if environment is None:
        return {"design": [], "eval": []}
    body = environment.group("body")

    def values(key: str) -> list[int]:
        match = re.search(
            rf"^[ \t]+{key}:[ \t]*(?P<inline>\[[^\]]*\])?[ \t]*\n"
            rf"(?P<items>(?:^[ \t]*-[ \t]*-?\d+[ \t]*\n?)*)",
            body,
            re.MULTILINE,
        )
        if match is None:
            return []
        return [
            int(value)
            for value in re.findall(
                r"-?\d+", (match.group("inline") or "") + match.group("items")
            )
        ]

    return {"design": values("design_counts"), "eval": values("eval_counts")}


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
        marker = _variant_marker(run_dir)
        if not hydra_dir.is_dir() and not marker:
            continue  # re-eval render dirs have results.json but no .hydra/
        ov = _parse_overrides(hydra_dir)
        if "approach.load_dir" in ov and not marker:
            continue  # an unlabelled re-eval / render pass, not a generation run
        run_id = str(run_dir.relative_to(root))
        try:
            r = json.loads(results.read_text())
        except (OSError, json.JSONDecodeError):
            r = {}
        per = r.get("per_episode") or []
        budget = ov.get("approach.max_budget_usd")
        if budget is None:
            m = re.search(r"budget_(\d+)", run_id)
            budget = m.group(1) if m else None
        # replicate_seed labels the run; older runs called the same knob seed.
        seed = ov.get("replicate_seed") or ov.get("seed")
        m2 = re.search(r"[/_]s(\d+)", run_id)
        approach = (
            marker.get("approach")
            or ov.get("approach")
            or _choice(hydra_dir, "approach")
            or "?"
        )
        num_eval = 100
        cfg = hydra_dir / "config.yaml"
        if cfg.exists():
            mm = re.search(r"^num_eval_tasks:\s*(\d+)", cfg.read_text(), re.MULTILINE)
            if mm:
                num_eval = int(mm.group(1))
        elif per:
            num_eval = len(per)
        eval_seed = marker.get("eval_seed", r.get("eval_seed"))
        primitives_value = (
            marker["primitives"]
            if isinstance(marker.get("primitives"), str)
            else _primitives(hydra_dir)
        )
        environment = (
            marker.get("environment")
            or ov.get("environment")
            or _choice(hydra_dir, "environment")
            or "?"
        )
        trained = marker.get("trained_environment")
        if not trained and marker.get("policy_source"):
            trained = _environment_of_dir(root / marker["policy_source"])
        model = _backend_model(hydra_dir)
        env_access = _env_access(hydra_dir)
        if marker.get("policy_source"):
            # A cross-eval cell replays a policy synthesized elsewhere.
            policy_hydra = root / marker["policy_source"] / ".hydra"
            if model is None:
                model = _backend_model(policy_hydra)
            if env_access is None:
                env_access = _env_access(policy_hydra)
        runs[run_id] = RunInfo(
            run_id=run_id,
            path=run_dir,
            approach=approach,
            environment=environment,
            primitives=_primitives_category(primitives_value),
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
            # The bilevel planner is also evaluated independently per held-out
            # seed (it has no train-once policy to replay).
            per_instance=(
                "per_instance" in approach
                or "best_of_k" in approach
                or "bilevel_planning" in approach
            ),
            policy_source=marker.get("policy_source"),
            eval_seed=int(eval_seed) if eval_seed is not None else None,
            explicit_seeds=bool(per and isinstance(per[0], dict) and "seed" in per[0]),
            trained_environment=trained or environment,
            slice=str(marker.get("slice", "")),
            model=model,
            env_access=env_access,
        )
    return runs


def refresh_runs() -> None:
    """Rebuild the run index from disk."""
    with _RUNS_LOCK:
        RUNS.clear()
        RUNS.update(_discover_runs(SCAN.root))


def sync_drive_and_refresh() -> dict[str, int]:
    """Synchronize the optional Drive source, then rebuild the local run index."""
    report: dict[str, int] = {}
    if DRIVE_SYNC is not None:
        sync_report = DRIVE_SYNC.sync()
        report = {
            key: int(getattr(sync_report, key))
            for key in ("downloaded", "unchanged", "removed", "ignored")
        }
    refresh_runs()
    return report


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
        "primitives": run.primitives,
        "seed": run.seed,
        "budget": run.budget,
        "model": run.model,
        "env_access": run.env_access,
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
        "policy_source": run.policy_source,
        "trained_environment": run.trained_environment,
        "slice": run.slice or None,
        "environment_counterpart": _ENV_COUNTERPART.get(run.environment),
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
    counterpart = _ENV_COUNTERPART.get(run.environment)
    episodes = []
    for i, e in enumerate(per):
        gif_path = _eval_gif_path(run, i)
        episodes.append(
            {
                "i": i,
                # Sent as a string: eval seeds run to 2**63, and JSON numbers
                # lose exactness past 2**53 once the browser parses them.
                "seed": str(eval_seeds[i]) if i < len(eval_seeds) else None,
                "object_count": e.get("object_count"),
                "num_steps": e.get("num_steps"),
                "total_reward": e.get("total_reward"),
                "outcome": _episode_outcome(e),
                "has_gif": gif_path is not None,
                "gif_path": str(gif_path) if gif_path is not None else None,
                # Planner runs record these; an episode with no plan executed no
                # action, so its GIF is a single frame and reads as a stuck render.
                "plan_found": e.get("plan_found"),
                "planning_time": e.get("planning_time"),
                "gif_preview": (
                    gif_path is not None
                    and gif_path.with_suffix(gif_path.suffix + ".preview").exists()
                ),
                "xenv_has_gif": bool(
                    counterpart
                    and _eval_gif_path(run, i, environment=counterpart) is not None
                ),
            }
        )
    env_desc = run.path / "env_description.md"
    logs = [
        n
        for n in ("run_experiment.log", "env_config.json", "env_variant.json")
        if (run.path / n).exists()
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
        "viewer": SCAN.tag,
        "run_path": str(run.path),
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


def _eval_gif_path(
    run: RunInfo, i: int, environment: Optional[str] = None
) -> Optional[Path]:
    """Locate an already-rendered eval GIF for episode i, else None.

    With ``environment`` set to a group other than the run's own, look up the
    cross-environment replay of the same episode seed instead.
    """
    if environment and environment != run.environment:
        p = run.path / f"videos_env_{environment}" / f"episode_{i}.gif"
        return p if p.exists() and p.stat().st_size > 0 else None
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


def _gif_from_query(run: RunInfo, q: dict[str, str]) -> Optional[Path]:
    """Resolve the episode gif addressed by /api/gif- or /api/video-style params."""
    kind, key = q.get("kind", "eval"), q.get("key", "0")
    if kind == "eval":
        return _eval_gif_path(run, int(key), environment=q.get("env"))
    return _history_gif_path(run, int(key), int(q.get("i", "0")))


def _mp4_for_gif(gif: Path) -> Path:
    """Return a cached MP4 transcode of ``gif`` for seekable in-browser playback.

    The gif stays the artifact of record; the .mp4 next to it is a derived cache,
    refreshed whenever the gif is newer (e.g. after a re-render).
    """
    mp4 = gif.with_suffix(gif.suffix + ".mp4")
    if mp4.exists() and mp4.stat().st_mtime >= gif.stat().st_mtime:
        return mp4
    part = mp4.with_name(mp4.name + ".part")
    # ffmpeg's gif demuxer appends a duplicate of the final frame; cap the
    # output so mp4 frame k stays episode step k.
    n_frames = iio.improps(str(gif)).n_images
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            *("-y", "-v", "error"),
            *("-i", str(gif)),
            *("-frames:v", str(n_frames)),
            # yuv420p needs even dimensions; scale at most one pixel down.
            *("-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"),
            *("-pix_fmt", "yuv420p"),
            *("-movflags", "+faststart"),
            *("-f", "mp4"),
            str(part),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg transcode failed: {result.stderr.decode(errors='replace').strip()}"
        )
    part.replace(mp4)
    return mp4


def _history_gif_path(run: RunInfo, version: int, i: int) -> Optional[Path]:
    version_dir = run.path / "approach_history" / f"v{version:03d}"
    candidates = [version_dir / f"episode_{i}.gif"]
    # record_episodes() historically wrote one unnumbered replay per version.
    if i == 0:
        candidates.append(version_dir / "episode.gif")
    return next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)


def _eval_seeds(run: RunInfo) -> list[int]:
    """The held-out episode seeds of this run's recorded evaluation suite.

    Prefer per-episode seeds recorded in results.json; otherwise recreate the
    suite the way run_experiment.py draws it, from the pinned eval_seed when the
    run recorded one and from the run seed otherwise.
    """
    if run.explicit_seeds:
        per = _results(run).get("per_episode") or []
        return [int(e["seed"]) for e in per if "seed" in e]
    base = run.eval_seed if run.eval_seed is not None else run.seed
    if base is None:
        return []
    # Keep numpy out of the read-only viewer path. PCG64 and Generator.integers
    # are used by the experiment, so use numpy when installed and degrade to no
    # seed labels on a lightweight viewer-only machine.
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        rng = np.random.default_rng(base)
        return [int(rng.integers(0, 2**63)) for _ in range(run.num_eval_tasks)]
    except ImportError:
        return []


# --------------------------------------------------------------------------- #
# Git history
# --------------------------------------------------------------------------- #

_SEP = "\x1f"
_SNAPSHOT_CACHE: dict[str, tuple[Hashable, list[dict[str, Any]]]] = {}
_SNAPSHOT_LOCKS: dict[str, threading.Lock] = {}
_SNAPSHOT_CACHE_LOCK = threading.Lock()  # guards both dictionaries


def _git(sandbox: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(sandbox), capture_output=True, text=True, check=False
    )


def _path_stamp(path: Path) -> tuple[int, int]:
    """Cheap file-change signature, including replacements with the same mtime."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


def _snapshot_stamp(run: RunInfo) -> Hashable:
    """Inputs that affect the history response.

    ``renderHistory`` requests snapshots and training-seed data concurrently.
    Both need the same expensive Git history. This stamp lets the second request
    reuse it while keeping live runs and newly recorded replays fresh.
    """
    git_dir = run.path / "sandbox" / ".git"
    history = run.path / "approach_history"
    replay_files = []
    if history.is_dir():
        replay_files = sorted(
            (
                str(path.relative_to(history)),
                *_path_stamp(path),
            )
            for pattern in ("v*/episodes.json", "v*/metrics.json")
            for path in history.glob(pattern)
        )
    return (
        _path_stamp(git_dir / "HEAD"),
        _path_stamp(git_dir / "logs" / "HEAD"),
        _path_stamp(git_dir / "packed-refs"),
        _path_stamp(run.path / "stream.jsonl"),
        _path_stamp(run.path / "results.json"),
        _path_stamp(history),
        tuple(replay_files),
    )


def _snapshots_uncached(run: RunInfo) -> list[dict[str, Any]]:
    """Build the version list from approach.py-bearing Git commits."""
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


def _snapshots(run: RunInfo) -> list[dict[str, Any]]:
    """Return history snapshots, sharing work across concurrent page requests."""
    key = str(run.path.resolve())
    with _SNAPSHOT_CACHE_LOCK:
        run_lock = _SNAPSHOT_LOCKS.setdefault(key, threading.Lock())
    # Only duplicate requests for this run wait; other run pages can load in parallel.
    with run_lock:
        stamp = _snapshot_stamp(run)
        with _SNAPSHOT_CACHE_LOCK:
            cached = _SNAPSHOT_CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        snapshots = _snapshots_uncached(run)
        with _SNAPSHOT_CACHE_LOCK:
            _SNAPSHOT_CACHE[key] = (stamp, snapshots)
        return snapshots


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
    intentionally recognize only explicit seed assignments/lists/ranges, not arbitrary
    numbers, to avoid confusing step counts and geometry constants with seeds.
    """
    mentions: dict[int, set[str]] = {}

    def add(seed: int, kind: str) -> None:
        if 0 <= seed < 1_000_000:
            mentions.setdefault(seed, set()).add(kind)

    for match in re.finditer(
        r"(?:reset\s*\(\s*seed|\b[\"']?seed[\"']?)\s*(?:=|:)\s*(\d+)", command
    ):
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


def _variable_integer_values(text: str, variable: str) -> set[int]:
    """Best-effort values for a Python variable used as ``object_count``."""
    values: set[int] = set()
    escaped = re.escape(variable)
    for match in re.finditer(
        rf"(?:for\s+{escaped}\s+in|{escaped}\s*=)\s*\[([^\]]+)\]", text
    ):
        values.update(
            int(value) for value in re.findall(r"(?<![\w.])-?\d+\b", match[1])
        )
    for match in re.finditer(rf"\b{escaped}\s*=\s*(-?\d+)\b", text):
        values.add(int(match[1]))

    # ``cases = [(seed, count), ...]; for seed, count in cases`` is common in
    # diagnostic scripts. Resolve the relevant tuple column.
    for loop in re.finditer(r"for\s+([\w\s,]+)\s+in\s+(\w+)", text):
        variables = [item.strip() for item in loop[1].split(",")]
        if variable not in variables:
            continue
        assignment = re.search(
            rf"\b{re.escape(loop[2])}\s*=\s*\[([^\]]+)\]", text, re.DOTALL
        )
        if assignment is None:
            continue
        column = variables.index(variable)
        for tuple_text in re.findall(r"\(([^()]*)\)", assignment[1]):
            fields = [field.strip() for field in tuple_text.split(",")]
            if column < len(fields) and re.fullmatch(r"-?\d+", fields[column]):
                values.add(int(fields[column]))
    return values


def _object_count_mentions(text: str) -> dict[str, Any]:
    """Recover explicit object counts used in reset/render calls.

    Constructor ``design_counts`` and ``eval_counts`` are intentionally ignored;
    only values passed through an ``object_count`` argument are considered an
    attempted configuration.
    """
    counts: set[int] = set()
    unresolved: set[str] = set()
    for match in re.finditer(
        r"(?:[\"']object_count[\"']|\bobject_count)\s*(?:=|:)\s*"
        r"(?P<value>-?\d+|[A-Za-z_]\w*)",
        text,
    ):
        value = match.group("value")
        if re.fullmatch(r"-?\d+", value):
            counts.add(int(value))
            continue
        resolved = _variable_integer_values(text, value)
        if resolved:
            counts.update(resolved)
        else:
            unresolved.add(value)
    return {"counts": sorted(counts), "unresolved": sorted(unresolved)}


def _stream_index(run: RunInfo) -> dict[str, Any]:
    """Map commit messages to reasoning and best-effort per-commit effort.

    Claude stream events contain per-turn token usage but no stable wall-clock
    timestamps. We group turns and tool calls up to each ``git commit``; elapsed
    time comes independently from adjacent git commit timestamps.
    """
    f = run.path / "stream.jsonl"
    if not f.exists():
        return {
            "by_msg": {},
            "effort_by_msg": {},
            "seed_events": [],
            "count_events": [],
        }
    key = str(run.path.resolve())
    mtime = f.stat().st_mtime
    cached = _STREAM_CACHE.get(key)
    if cached and cached["mtime"] == mtime:
        return cached["idx"]

    by_msg: dict[str, dict[str, Any]] = {}
    effort_by_msg: dict[str, dict[str, Any]] = {}
    seed_events: list[dict[str, Any]] = []
    count_events: list[dict[str, Any]] = []
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
        for block in blocks:
            if block.get("type") != "tool_use":
                continue
            tool_name = block.get("name", "")
            tool_input = block.get("input") or {}
            source = ""
            evidence_source = ""
            if tool_name == "Bash":
                source = tool_input.get("command", "")
                evidence_source = "executed command"
            elif tool_name in {"Write", "Edit"}:
                source = tool_input.get("content") or tool_input.get("new_string") or ""
                evidence_source = "agent-authored test code"
            elif isinstance(tool_input.get("object_count"), int):
                source = json.dumps(tool_input)
                evidence_source = tool_name
            if not source:
                continue
            count_info = _object_count_mentions(source)
            if not count_info["counts"] and not count_info["unresolved"]:
                continue
            mentions = _seed_mentions(source)
            relevant_lines = [
                line.strip()
                for line in source.splitlines()
                if re.search(r"object_count|for\s+\w+\s+in", line)
            ]
            count_events.append(
                {
                    "commit": event_commit,
                    **count_info,
                    "seeds": sorted(mentions),
                    "source": evidence_source,
                    "evidence": " | ".join(relevant_lines)[:800],
                }
            )
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
        "count_events": count_events,
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
    stream = _stream_index(run)
    events = stream["seed_events"]
    count_events = stream["count_events"]
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
                    "object_counts": set(),
                    "object_count_unknown": False,
                },
            )
            item["mentions"] += 1
            item["explicit_mentions"] += int(any(kind != "range" for kind in kinds))
            item["versions"].add(version)
            if event["evidence"] and event["evidence"] not in item["evidence"]:
                item["evidence"].append(event["evidence"])

    normalized_count_events = []
    observed_counts: set[int] = set()
    unresolved: set[str] = set()
    for event in count_events:
        version = version_for(event["commit"])
        normalized = {**event, "version": version}
        normalized_count_events.append(normalized)
        observed_counts.update(event["counts"])
        unresolved.update(event["unresolved"])
        for seed in event["seeds"]:
            if seed not in by_seed:
                by_seed[seed] = {
                    "seed": seed,
                    "mentions": 1,
                    "explicit_mentions": 1,
                    "versions": {version},
                    "evidence": [event["evidence"]] if event["evidence"] else [],
                    "object_counts": set(),
                    "object_count_unknown": False,
                }
                by_version.setdefault(version, set()).add(seed)
            by_seed[seed]["object_counts"].update(event["counts"])
            by_seed[seed]["object_count_unknown"] |= bool(event["unresolved"])

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
                "object_counts": sorted(item["object_counts"]),
                # Individually targeted seeds are more diagnostic than a seed
                # that merely appeared once inside range(200).
                "interest": item["explicit_mentions"] * 100 + item["mentions"],
            }
        )
    seeds_out.sort(key=lambda item: (-item["interest"], item["seed"]))
    configured = _configured_object_counts(run)
    design = set(configured["design"])
    evaluation = set(configured["eval"])
    held_out = evaluation - design
    out_of_domain = observed_counts - design if design else set()
    violations = {
        count
        for count in out_of_domain
        if count in held_out or (design and count > max(design))
    }
    if not design and not evaluation:
        status = "not_applicable"
    elif violations:
        status = "violation"
    elif unresolved or not observed_counts:
        status = "incomplete"
    else:
        status = "no_violation_observed"
    return {
        "seeds": seeds_out,
        "events": normalized_events,
        "count_events": normalized_count_events,
        "by_version": {
            str(version): sorted(seeds) for version, seeds in sorted(by_version.items())
        },
        "object_count_audit": {
            "status": status,
            "design_counts": sorted(design),
            "eval_counts": sorted(evaluation),
            "held_out_counts": sorted(held_out),
            "observed_counts": sorted(observed_counts),
            "violation_counts": sorted(violations),
            "other_out_of_domain_counts": sorted(out_of_domain - violations),
            "unresolved_expressions": sorted(unresolved),
            "evidence_complete": not unresolved and bool(observed_counts),
        },
    }


# --------------------------------------------------------------------------- #
# Render jobs (single worker; heavy imports are lazy)
# --------------------------------------------------------------------------- #


# A failed episode usually runs to its step limit, so a full replay is slow and
# the GIF is long. A preview render stops after this many steps; the full replay
# is rendered on demand.
PREVIEW_STEPS = 100


def _preview_step_limit(value: Any) -> int:
    """Return a validated per-render preview horizon from an API value."""
    if value is None:
        return PREVIEW_STEPS
    if isinstance(value, bool):
        raise ValueError("preview_steps must be a positive integer")
    try:
        steps = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("preview_steps must be a positive integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("preview_steps must be a positive integer")
    if isinstance(value, str) and str(steps) != value.strip():
        raise ValueError("preview_steps must be a positive integer")
    if steps < 1:
        raise ValueError("preview_steps must be a positive integer")
    return steps


@dataclass
class Job:
    """State of one background GIF-render job, polled by the client."""

    job_id: str
    state: str = "queued"  # queued | running | done | error
    phase: str = "queued"
    message: str = ""
    gif_url: Optional[str] = None
    gif_path: Optional[str] = None  # on-disk path, for copyable references
    current: int = 0
    total: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    capped: bool = (
        False  # the requested preview horizon stopped before the episode ended
    )


JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_RENDER_Q: "queue.Queue[tuple[str, str, str, int, bool, Optional[str], int]]" = (
    queue.Queue()
)
_JOB_SEQ = [0]
# Bumped by the cancel endpoint. A running render captures the value at its start
# and aborts at its next progress report once the two disagree.
_CANCEL_EPOCH = [0]


class _Cancelled(Exception):
    """Raised inside a render to unwind it after the user cancels."""


def _raise_if_cancelled(epoch: int) -> None:
    """Abort the running render if a cancel arrived after it started."""
    if _CANCEL_EPOCH[0] != epoch:
        raise _Cancelled


def _cancel_renders() -> dict[str, Any]:
    """Drop every queued render and ask the running one to stop.

    Queued jobs are discarded here; the running job cannot be interrupted mid-step,
    so it unwinds at its next progress report.
    """
    dropped = []
    while True:
        try:
            job_id, *_ = _RENDER_Q.get_nowait()
        except queue.Empty:
            break
        dropped.append(job_id)
        _RENDER_Q.task_done()
    with _JOBS_LOCK:
        _CANCEL_EPOCH[0] += 1
        for job_id in dropped:
            job = JOBS.get(job_id)
            if job is not None:
                job.state, job.phase = "cancelled", "cancelled"
                job.message = "cancelled before starting"
                job.finished_at = time.time()
        running = sum(1 for j in JOBS.values() if j.state == "running")
    return {"dropped": len(dropped), "running": running}


def _enqueue_render(
    run_id: str,
    target: str,
    episode_index: int,
    full: bool = False,
    environment: Optional[str] = None,
    preview_steps: int = PREVIEW_STEPS,
) -> str:
    with _JOBS_LOCK:
        _JOB_SEQ[0] += 1
        job_id = f"job{_JOB_SEQ[0]}"
        mode = "full render" if full else f"{preview_steps}-step preview"
        if environment:
            mode += f" in {environment}"
        ahead = _RENDER_Q.qsize()
        suffix = f" ({ahead} job{'s' if ahead != 1 else ''} ahead)" if ahead else ""
        JOBS[job_id] = Job(
            job_id=job_id,
            message=f"queued {mode}{suffix}",
        )
    _RENDER_Q.put(
        (job_id, run_id, target, episode_index, full, environment, preview_steps)
    )
    return job_id


def _enqueue_history_evaluation(run_id: str) -> str:
    """Queue one metrics-only evaluation across every incomplete commit."""
    with _JOBS_LOCK:
        _JOB_SEQ[0] += 1
        job_id = f"job{_JOB_SEQ[0]}"
        JOBS[job_id] = Job(job_id=job_id, message="queued full history evaluation")
    _RENDER_Q.put((job_id, run_id, "EVAL_HISTORY", -1, True, None, PREVIEW_STEPS))
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


def _safe_json_value(value: Any) -> Any:
    """Recursively replace non-finite metric values with JSON null."""
    if hasattr(value, "item"):
        return _safe_json_value(value.item())
    if hasattr(value, "tolist"):
        return _safe_json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    return value


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


def _replay_overrides(
    run: RunInfo, load_dir: str, output_dir: str, environment: Optional[str] = None
) -> list[str]:
    """Hydra overrides shared by current and historical replay jobs.

    ``++`` supports both generated approaches, which declare these fields, and
    planner configs, which accept them through ``**kwargs`` but do not declare them.
    ``environment`` replays the policy under a different environment group; the
    primitives always follow the policy, since a generated approach.py indexes the
    exact primitives dict it was written against.
    """
    overrides = [
        f"approach={run.approach}",
        f"++approach.load_dir={load_dir}",
        f"++approach.output_dir={output_dir}",
        f"primitives={_replay_primitives(run)}",
        "mcp_tools=[]",
        f"environment={environment or run.environment}",
    ]
    # An unlabelled run keeps the config default rather than replaying under a
    # null seed, which no RNG accepts.
    if run.seed is not None:
        overrides.append(f"replicate_seed={run.seed}")
    return overrides


def _evaluate_history(run: RunInfo, job: Job, epoch: int) -> None:
    """Evaluate every historical commit on the final run's full eval suite.

    This intentionally records metrics only. Rendering hundreds of GIFs would dominate
    runtime and disk; individual failures can still be rendered from the matrix after
    the comparable solve-rate curve is available.
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
                        overrides=_replay_overrides(run, tmp, f"{tmp}/out"),
                    )
                env = instantiate(cfg.environment)
                primitives = build_primitives(env, cfg.primitives)
                approach_ctor = instantiate(
                    cfg.approach,
                    action_space=env.action_space,
                    observation_space=env.observation_space,
                    seed=cfg.replicate_seed,
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
                    _raise_if_cancelled(epoch)
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
        (
            job_id,
            run_id,
            target,
            i,
            full,
            env_override,
            preview_steps,
        ) = _RENDER_Q.get()
        run = _run(run_id)
        epoch = _CANCEL_EPOCH[0]
        history_version: Optional[int] = None
        replay_seed: Optional[int] = None
        replay_count: Optional[int] = None
        completion_message = "rendered"
        with _JOBS_LOCK:
            job = JOBS[job_id]
            job.state = "running"
            job.phase = "preparing"
            job.started_at = time.time()
            job.message = "loading replay configuration"
        try:
            # Heavy deps are imported on the first render so the server and all
            # read-only browsing need nothing but the Python stdlib (useful for
            # inspecting a downloaded outputs/ folder on a laptop).
            from hydra import compose, initialize_config_dir
            from hydra.utils import instantiate

            from robocode.primitives import build_primitives
            from robocode.utils.approach_history import _export_snapshot
            from robocode.utils.episode import (
                open_video_writer,
                run_episode,
                save_video,
            )

            if run is None:
                raise RuntimeError(f"unknown run {run_id}")
            if target == "EVAL_HISTORY":
                with _JOBS_LOCK:
                    job.message = "preparing full history evaluation"
                _evaluate_history(run, job, epoch)
                with _JOBS_LOCK:
                    job.state = "done"
                    job.message = "full history evaluation complete"
                continue
            if env_override and env_override == run.environment:
                env_override = None
            if env_override:
                if target != "HEAD":
                    raise RuntimeError("environment override only replays HEAD")
                if not re.fullmatch(r"[A-Za-z0-9_\-]+", env_override):
                    raise RuntimeError(f"bad environment override {env_override!r}")
            policy_dir = _policy_dir(run)
            if (
                not run.per_instance
                and not (policy_dir / "sandbox" / "approach.py").exists()
            ):
                raise RuntimeError(
                    f"policy sandbox not found locally: {policy_dir / 'sandbox'}"
                )
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
                    load_dir = str(policy_dir)
                    videos = f"videos_env_{env_override}" if env_override else "videos"
                    out = run.path / videos / f"episode_{i}.gif"
                else:
                    version = int(target)
                    history_version = version
                    h = _commit_hash(run, version)
                    assert h is not None, f"no commit for version {version}"
                    _export_snapshot(policy_dir / "sandbox", h, Path(tmp) / "sandbox")
                    load_dir = tmp
                    out = (
                        run.path
                        / "approach_history"
                        / f"v{version:03d}"
                        / f"episode_{i}.gif"
                    )
                out.parent.mkdir(parents=True, exist_ok=True)

                with _JOBS_LOCK:
                    job.message = "building environment and approach"
                with initialize_config_dir(version_base=None, config_dir=conf_dir):
                    cfg = compose(
                        config_name="config",
                        overrides=_replay_overrides(
                            run, load_dir, f"{tmp}/out", environment=env_override
                        ),
                    )
                env = instantiate(cfg.environment)
                primitives = build_primitives(env, cfg.primitives)
                approach_ctor = instantiate(
                    cfg.approach,
                    action_space=env.action_space,
                    observation_space=env.observation_space,
                    seed=cfg.replicate_seed,
                    env_description_path=None,
                    mcp_tools=(),
                    env_name=env_override or run.environment,
                    env=env,
                    env_cfg=json.dumps({}),
                    max_steps=cfg.max_steps,
                    eval_timeout=cfg.eval_timeout,
                    _partial_=True,
                )
                _purge_sandbox_modules()
                if count is not None and hasattr(env, "max_steps_for_count"):
                    max_steps = env.max_steps_for_count(count)
                else:
                    max_steps = int(cfg.max_steps)
                render_steps = max_steps if full else min(max_steps, preview_steps)
                approach = approach_ctor(primitives=primitives)
                capped = False
                # Set by the per-instance branch; the generated-policy branch
                # streams frames straight to disk and leaves this as None.
                frames: Optional[list[Any]] = None

                def report_instance(phase: str, current: int, total: int) -> None:
                    _raise_if_cancelled(epoch)
                    with _JOBS_LOCK:
                        job.phase = "rollout" if total else "preparing"
                        job.message = phase
                        job.current = current
                        job.total = total

                def report_rollout(current: int, total: int) -> None:
                    _raise_if_cancelled(epoch)
                    with _JOBS_LOCK:
                        job.phase = "rollout"
                        job.message = f"rendering step {current}/{total}"
                        job.current = current
                        job.total = total

                if approach.per_instance:
                    result = approach.solve_instance(
                        env=env,
                        seed=eval_seeds[i],
                        budget_usd=float(cfg.approach.max_budget_usd),
                        output_subdir=Path(tmp) / "out" / f"instance_{i}",
                        render=True,
                        count=count,
                        max_steps=render_steps,
                        progress_callback=report_instance,
                    )
                    metrics = {
                        **result.extras,
                        "solved": result.solved,
                        "total_reward": result.total_reward,
                        "num_steps": result.num_steps,
                    }
                    if metrics.get("plan_found") is False:
                        completion_message = (
                            "planner found no plan; rendered initial state"
                        )
                    frames = result.frames or []
                    capped = (
                        not full
                        and render_steps < max_steps
                        and result.num_steps is not None
                        and result.num_steps >= render_steps
                    )
                    # Keep this defensive trim for custom per-instance approaches
                    # that do not yet honor the replay horizon.
                    if len(frames) > render_steps + 1:
                        frames = frames[: render_steps + 1]
                        capped = True
                else:
                    with _JOBS_LOCK:
                        job.phase = "preparing"
                        job.message = "loading generated policy"
                    approach.train()
                    # Stream each frame into the GIF encoder as it is rendered:
                    # holding a long episode's raw RGB frames in memory can OOM
                    # the machine. The partial file replaces ``out`` only once
                    # the rollout finishes.
                    part = out.with_name(out.name + ".part")
                    n_frames = 0
                    try:
                        with open_video_writer(part) as append_frame:

                            def _sink(frame: Any) -> None:
                                nonlocal n_frames
                                append_frame(frame)
                                n_frames += 1

                            metrics, _, _ = run_episode(
                                env,
                                approach,
                                eval_seeds[i],
                                render_steps,
                                render=True,
                                count=count,
                                progress_callback=report_rollout,
                                frame_sink=_sink,
                            )
                        if n_frames == 0:
                            raise RuntimeError("no frames rendered")
                        part.replace(out)
                    finally:
                        part.unlink(missing_ok=True)
                    capped = (
                        render_steps < max_steps
                        and metrics["num_steps"] >= render_steps
                    )
                if frames is not None:
                    if not frames:
                        raise RuntimeError("no frames rendered")
                    n_frames = len(frames)
                    with _JOBS_LOCK:
                        job.phase = "encoding"
                        job.message = f"encoding {n_frames} frames as GIF"
                    save_video(frames, out)
                preview_marker = out.with_suffix(out.suffix + ".preview")
                if capped:
                    preview_marker.write_text(f"{n_frames} frames\n")
                else:
                    preview_marker.unlink(missing_ok=True)
                # A preview stops before the episode ends, so its metrics are not the
                # episode's real outcome; only a full render updates the history record.
                if full and history_version is not None:
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
                job.phase = "done"
                job.capped = capped
                job.finished_at = time.time()
                job.message = (
                    f"rendered first {preview_steps} steps"
                    if capped
                    else completion_message
                )
                kind = "eval" if target == "HEAD" else "history"
                key = i if target == "HEAD" else target
                envq = f"&env={env_override}" if env_override else ""
                job.gif_url = (
                    f"/api/gif?run={run_id}&kind={kind}"
                    f"&key={key}&i={i}{envq}&t={int(time.time())}"
                )
                job.gif_path = str(out)
        except _Cancelled:
            with _JOBS_LOCK:
                job.state = "cancelled"
                job.phase = "cancelled"
                job.finished_at = time.time()
                job.message = "cancelled"
        except Exception as e:  # pylint: disable=broad-exception-caught
            # A render can fail many ways (bad approach, env error); surface it to
            # the UI job instead of killing the single worker thread.
            with _JOBS_LOCK:
                job.state = "error"
                job.phase = "error"
                job.finished_at = time.time()
                job.message = f"{type(e).__name__}: {e}"
            if full and run is not None and history_version is not None:
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
    "env_variant.json",
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

    def _send_file(self, path: Path, ctype: str) -> None:
        """Serve a file honoring single-range requests so <video> can seek."""
        data = path.read_bytes()
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        if m and (m.group(1) or m.group(2)):
            start = int(m.group(1) or 0)
            end = min(int(m.group(2)) if m.group(2) else len(data) - 1, len(data) - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            data = data[start : end + 1]
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(_safe_json_value(obj), allow_nan=False).encode()
        self._send(code, body, "application/json")

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
            p = _gif_from_query(run, q)
            if p is None:
                return self._err(404, "not rendered")
            return self._send(200, p.read_bytes(), "image/gif")
        if path == "/api/video":
            p = _gif_from_query(run, q)
            if p is None:
                return self._err(404, "not rendered")
            try:
                return self._send_file(_mp4_for_gif(p), "video/mp4")
            except RuntimeError as error:
                return self._err(500, str(error))
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
        if u.path == "/api/render/cancel":
            return self._json(_cancel_renders())
        if u.path == "/api/refresh":
            try:
                drive_report = sync_drive_and_refresh()
            except (OSError, RuntimeError) as error:
                return self._err(502, f"refresh failed: {error}")
            return self._json({"ok": True, "n": len(RUNS), "drive": drive_report})
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
            try:
                preview_steps = _preview_step_limit(body.get("preview_steps"))
            except ValueError as error:
                return self._err(400, str(error))
            job_id = _enqueue_render(
                run.run_id,
                str(body.get("target", "HEAD")),
                int(body["episode_index"]),
                bool(body.get("full", False)),
                body.get("environment") or None,
                preview_steps,
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
<header><a id="brand" href="#/index" class="brand">robocode results</a>
  <button id="refresh">refresh</button>
  <button id="cancel-renders" title="Drop every queued render and stop the running one">cancel renders</button></header>
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
.chip .chip-n{margin-left:5px;font-size:11px;opacity:.55;font-variant-numeric:tabular-nums}
.chip.empty{opacity:.4}
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
.ep.pair{width:272px}
.pair2{display:flex;gap:2px}
.pair2>div{flex:1;min-width:0}
.pair2 .pl{font:9px ui-monospace,Menlo,monospace;color:var(--muted);text-align:center;padding:1px 0 0}
.ep img{width:100%;display:block;aspect-ratio:1;object-fit:contain;background:#fff}
.ep .cap{font:11px ui-monospace,Menlo,monospace;color:var(--muted);padding:5px 6px}
.hrow{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
button.cp{font:10px ui-monospace,Menlo,monospace;padding:1px 6px;border-radius:5px;margin-left:6px;
  color:var(--muted);vertical-align:1px}
button.cp:hover{color:var(--fg)}
button.cp.ok{color:var(--ok);border-color:var(--ok)}
.ep .ph{aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ep .ph.noplan{gap:3px;color:var(--muted);font-size:11px;text-align:center;
  background:repeating-linear-gradient(45deg,transparent,transparent 6px,var(--bg) 6px,var(--bg) 12px)}
.render-status{width:100%;min-width:0;box-sizing:border-box;padding:6px;font:10px/1.35 ui-monospace,Menlo,monospace;color:var(--muted);overflow-wrap:anywhere}
.render-meter{height:4px;margin-top:5px;border-radius:3px;overflow:hidden;background:var(--line)}
.render-meter>span{display:block;width:0;height:100%;background:var(--accent);transition:width .2s}
.render-meter.indeterminate>span{width:35%;animation:render-wait 1.1s ease-in-out infinite}
@keyframes render-wait{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
.fullbtn{font-size:10px;margin:4px 6px}
.previewctl{display:grid;grid-template-columns:44px minmax(0,1fr);align-items:center;gap:4px;padding:4px 6px;width:100%;box-sizing:border-box}
.previewctl input{width:100%;min-width:0;box-sizing:border-box;padding:3px 4px;font:10px ui-monospace,Menlo,monospace}
.previewctl button{min-width:0;font-size:10px;line-height:1.15;padding:3px 4px;white-space:normal;overflow-wrap:anywhere}
.previewctl .render-status{grid-column:1/-1;padding:2px 0 0}
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
.history-plots{display:grid;gap:10px}
.solve-lineplot{border:1px solid var(--line);border-radius:9px;padding:8px 10px;background:var(--card)}
.solve-lineplot svg{display:block;width:100%;height:auto}.solve-lineplot .title{font-size:11px;color:var(--muted);margin-bottom:3px}
.chart-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:var(--muted);margin:2px 0 1px}
.chart-key{display:inline-flex;align-items:center;gap:5px}.chart-key::before{content:"";width:12px;height:2px;border-radius:2px;background:var(--key)}
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
.count-audit{border:1px solid var(--line);border-radius:9px;padding:10px 12px;margin:10px 0;background:var(--card)}
.count-audit.violation{border-color:var(--no)}.count-audit.safe{border-color:var(--ok)}
.count-audit.incomplete{border-color:var(--warn)}
.count-audit .audit-title{font-weight:600;margin-bottom:5px}
.count-audit .audit-grid{display:flex;gap:7px 16px;flex-wrap:wrap;font-size:11px}
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
.lb .vwrap{display:flex;flex-direction:column;gap:8px;cursor:default;max-width:92vw}
.lb video{max-width:92vw;max-height:80vh;border-radius:10px;
  box-shadow:0 12px 48px rgba(0,0,0,.5);cursor:pointer}
.vc{display:flex;gap:8px;align-items:center;background:var(--card);
  border:1px solid var(--line);border-radius:9px;padding:8px 12px}
.vc input[type=range]{flex:1;margin:0}
.vc-lbl{font:12px ui-monospace,Menlo,monospace;color:var(--muted);
  min-width:104px;text-align:right;white-space:nowrap}
"""

APP_JS = r"""
const $=(s,r=document)=>r.querySelector(s), h=(t,a={},...k)=>{const e=document.createElement(t);
 for(const[x,y]of Object.entries(a)){if(x==="class")e.className=y;else if(x==="html")e.innerHTML=y;
 else if(x.startsWith("on"))e.addEventListener(x.slice(2),y);else e.setAttribute(x,y);}
 for(const c of k.flat())e.append(c?.nodeType?c:document.createTextNode(c??""));return e;};
const j=(u,o)=>fetch(u,o).then(r=>r.json());
const num=(x,d=2)=>x==null?"-":(typeof x==="number"?(Math.abs(x)>=1000?x.toFixed(0):x.toFixed(d)):x);
const enc=encodeURIComponent;
// Identify what is on screen well enough to paste elsewhere: which viewer
// instance, which run, and the on-disk gif. Set from the run payload.
let VIEWER="";  // "<root name>:<port>"
let RUNTAG="";  // "<env> <primitives> <approach>"

function epRef(run,e){
 return `[${VIEWER}] ${run} | ${RUNTAG} | ep${e.i}`
  +(e.seed!=null?` seed${e.seed}`:"")
  +` c${e.object_count} ${num(e.num_steps,0)}st ${e.outcome}`
  +(e.plan_found===false?` | no plan found after ${num(e.planning_time,1)}s`:"")
  +` | ${e.gif_path||"(gif not rendered)"}`;
}

function copyBtn(getText,label){
 const b=h("button",{class:"cp",title:"copy reference"},label||"copy");
 b.onclick=async ev=>{ev.preventDefault();ev.stopPropagation();
  const t=getText();
  try{await navigator.clipboard.writeText(t);}
  catch(_){const ta=h("textarea",{});ta.value=t;document.body.append(ta);ta.select();
   document.execCommand("copy");ta.remove();}
  const old=b.textContent;b.textContent="copied";b.classList.add("ok");
  setTimeout(()=>{b.textContent=old;b.classList.remove("ok");},1200);};
 return b;
}
const duration=x=>x==null?"-":x<60?`${Math.round(x)}s`:x<3600?`${(x/60).toFixed(1)}m`:`${(x/3600).toFixed(1)}h`;
const compact=x=>x==null?"-":x>=1e6?`${(x/1e6).toFixed(1)}m`:x>=1e3?`${(x/1e3).toFixed(1)}k`:String(x);

// A gif thumbnail that opens a large, replaying view on click.
function gifImg(src){return h("img",{loading:"lazy",src,style:"cursor:zoom-in",title:"click to enlarge",
 onclick:()=>openLightbox(src)});}
const FPS=10; // save_video default; frame 0 = initial state, frame k = after step k
function openLightbox(src){
 const close=()=>{ov.remove();document.removeEventListener("keydown",keys);};
 const ov=h("div",{class:"lb",onclick:close});
 // Interactive player over a server-side mp4 transcode of the gif (seekable,
 // frame-steppable). If the transcode fails, fall back to the plain gif.
 const v=h("video",{src:src.replace("/api/gif?","/api/video?"),autoplay:"",loop:"",
  playsinline:"",onclick:e=>{e.stopPropagation();toggle();},
  title:"click: play/pause | ←/→: step (shift: ×10) | home/end: jump"});
 v.muted=true;
 const nframes=()=>v.duration?Math.max(1,Math.round(v.duration*FPS)):0;
 const seek=f=>{v.pause();
  v.currentTime=Math.min(Math.max((f+0.5)/FPS,0),Math.max(v.duration-1e-3,0));};
 const stepBy=d=>seek(Math.floor(v.currentTime*FPS)+d);
 const toggle=()=>v.paused?v.play():v.pause();
 const bar=h("input",{type:"range",min:0,max:0,step:1,value:0,
  oninput:e=>seek(+e.target.value),onclick:e=>e.stopPropagation()});
 const lbl=h("span",{class:"vc-lbl"},"");
 const playBtn=h("button",{title:"play/pause (click video)",
  onclick:e=>{e.stopPropagation();toggle();}},"⏯");
 const wrap=h("div",{class:"vwrap",onclick:e=>e.stopPropagation()},v,
  h("div",{class:"vc"},playBtn,
   h("button",{title:"back one step (←)",onclick:e=>{e.stopPropagation();stepBy(-1);}},"◀"),
   h("button",{title:"forward one step (→)",onclick:e=>{e.stopPropagation();stepBy(1);}},"▶"),
   bar,lbl));
 v.addEventListener("error",()=>{ // no mp4 => old behavior: replaying gif
  wrap.replaceWith(h("img",{src,onclick:e=>e.stopPropagation()}));});
 const sync=()=>{if(!ov.isConnected)return;
  const n=nframes();
  if(n){bar.max=n-1;const f=Math.min(n-1,Math.floor(v.currentTime*FPS));
   bar.value=f;lbl.textContent=`step ${f} / ${n-1}`;}
  requestAnimationFrame(sync);};
 const keys=e=>{
  if(e.key==="Escape"){close();return;}
  if(!wrap.isConnected)return;
  if(e.key===" "){e.preventDefault();toggle();}
  else if(e.key==="ArrowLeft"){e.preventDefault();stepBy(e.shiftKey?-10:-1);}
  else if(e.key==="ArrowRight"){e.preventDefault();stepBy(e.shiftKey?10:1);}
  else if(e.key==="Home"){e.preventDefault();seek(0);}
  else if(e.key==="End"){e.preventDefault();seek(nframes()-1);}};
 document.addEventListener("keydown",keys);
 ov.append(wrap,h("button",{class:"lbx",onclick:close},"close (esc)"));
 document.body.append(ov);
 requestAnimationFrame(sync);
}

let ALL=[], FILTER={}, INDEX_HASH="#/index";
const FACETS=[["model","model"],["approach","approach"],["env_access","env access"],
 ["environment","eval env"],["trained_environment","trained on"],["slice","slice"],
 ["primitives","primitives"],["seed","seed"],["budget","budget"]];
const FACET_KEYS=new Set(FACETS.map(([key])=>key));
async function loadRuns(){ALL=await j("/api/runs");}

function indexHash(){
 const query=new URLSearchParams();
 for(const [key,values] of Object.entries(FILTER))for(const value of values)query.append(key,value);
 const encoded=query.toString();return "#/index"+(encoded?"?"+encoded:"");
}
function restoreFilters(hash){
 FILTER={};const q=hash.indexOf("?");if(q<0)return;
 for(const [key,value] of new URLSearchParams(hash.slice(q+1))){
  if(!FACET_KEYS.has(key)||!value)continue;
  FILTER[key]=FILTER[key]||[];if(!FILTER[key].includes(value))FILTER[key].push(value);
 }
}

// Stable per-model colors so a run's model is readable at a glance across the
// grid; unknown models get a hue hashed from the name.
const MODEL_COLORS={"opus-5":"#8a63d2","sonnet-5":"#3f8cc9","sonnet-4.6":"#8a8f88","opus-4.8":"#b8860b"};
function modelColor(m){if(!m)return null;if(MODEL_COLORS[m])return MODEL_COLORS[m];
 let x=0;for(const c of m)x=(x*31+c.charCodeAt(0))>>>0;return`hsl(${x%360} 45% 50%)`;}
function modelDot(m){return h("span",{class:"dot",style:`background:${modelColor(m)}`});}
function modelPill(m){return m?h("span",{class:"pill",
 style:`color:${modelColor(m)};border-color:${modelColor(m)}`},m):"";}
// blackbox = the agent could not read the environment source during synthesis.
function accessPill(a){return a?h("span",{class:"pill",
 title:a==="blackbox"?"agent could not read the environment source"
  :"agent could read the environment source"},a):"";}

// Runs surviving every active filter except the named facet, which is the pool a
// facet's own chips are drawn from: picking a model narrows the env chips to the
// envs that model has runs for, while leaving the model chips themselves whole.
function matchExcept(r,skip){
 for(const[k,vs]of Object.entries(FILTER)){
  if(k===skip)continue;
  if(vs.length&&!vs.includes(String(r[k])))return false;
 }
 return true;
}
function facetBar(){
 // environment = the evaluated suite; trained_environment = what the policy saw
 // during synthesis. They differ only in cross-eval cells, so the "trained on"
 // facet mostly mirrors "eval env" but separates e.g. band-trained policies
 // re-scored on the standard suite.
 const bar=h("div",{class:"facets"});
 for(const [k,lbl] of FACETS){
  // A facet appears whenever the whole corpus offers a choice there, so the bar
  // keeps its shape as filters narrow.
  if(new Set(ALL.map(r=>r[k]).filter(v=>v!=null)).size<2)continue;
  const counts=new Map();
  for(const r of ALL){
   if(r[k]==null||!matchExcept(r,k))continue;
   const key=String(r[k]);counts.set(key,(counts.get(key)||0)+1);
  }
  const selected=FILTER[k]||[];
  // Selected values stay on screen at zero so they can be switched back off.
  const vals=[...new Set([...counts.keys(),...selected])].sort();
  if(!vals.length)continue;
  const f=h("div",{class:"facet"},h("span",{class:"lbl"},lbl));
  for(const v of vals){
   const on=selected.includes(v);
   const n=counts.get(v)||0;
   f.append(h("button",{class:"chip"+(on?" active":"")+(n?"":" empty"),
    title:`${n} run(s)`,onclick:()=>{
    FILTER[k]=FILTER[k]||[];const i=FILTER[k].indexOf(v);
    if(i<0)FILTER[k].push(v);else FILTER[k].splice(i,1);
    if(!FILTER[k].length)delete FILTER[k];location.hash=indexHash();}},
    k==="model"?[modelDot(v),v]:v,h("span",{class:"chip-n"},String(n))));
  }
  bar.append(f);
 }
 return bar;
}
function match(r){return matchExcept(r,null);}

function srColor(v){if(v==null)return"var(--muted)";const g=Math.round(160*v);return`rgb(${160-g},${90+g*0.6},${70+g*0.3})`;}

function renderIndex(){
 const app=$("#app");app.innerHTML="";
 app.append(facetBar());
 const runs=ALL.filter(match);
 app.append(h("div",{class:"muted",style:"margin-bottom:8px"},`${runs.length} run(s)`));
 const g=h("div",{class:"grid"});
 for(const r of runs){
  g.append(h("a",{class:"card",href:"#/run/"+enc(r.run_id),
   style:r.model?`border-left:4px solid ${modelColor(r.model)}`:""},
   h("div",{class:"sr",style:`color:${srColor(r.solve_rate)}`},r.solve_rate==null?"-":r.solve_rate.toFixed(2)),
   h("div",{class:"id"},r.run_id),
   h("div",{class:"row"},
    h("span",{class:"pill "+r.status},r.status),
    modelPill(r.model),
    accessPill(r.env_access),
    r.budget!=null?h("span",{class:"pill"},"$"+r.budget):"",
    r.seed!=null?h("span",{class:"pill"},"s"+r.seed):"",
    r.agent_cost_usd!=null?h("span",{class:"pill"},"$"+num(r.agent_cost_usd)):"",
    r.num_crashed_episodes?h("span",{class:"pill"},r.num_crashed_episodes+" crash"):"",
    r.policy_source?h("span",{class:"pill",title:"policy from "+r.policy_source},"xeval"):"",
    r.trained_environment&&r.trained_environment!==r.environment?
     h("span",{class:"pill",style:"color:var(--warn);border-color:var(--warn)",
      title:`evaluated on ${r.environment}`},"trained: "+r.trained_environment):"",
    r.slice?h("span",{class:"pill",style:"color:var(--accent);border-color:var(--accent)",
     title:"conditioned instance slice, not the uniform suite"},r.slice):"",
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

const envShort=n=>n&&n.endsWith("_a")?"pinned-spawn":"standard";
let PAIRVIEW=false;  // run-page toggle: show both env variants' gifs per episode

function previewControls(run,e,cardEl,label="preview gif"){
 const input=h("input",{type:"number",min:"1",step:"1",value:cardEl.dataset.previewSteps||"100",
  title:"Maximum preview steps","aria-label":"Preview steps"});
 const btn=h("button",{title:"Render up to the selected number of steps",onclick:ev=>{
  if(!input.reportValidity())return;
  const steps=Number(input.value);
  if(!Number.isInteger(steps)||steps<1){input.setCustomValidity("Enter a positive whole number");input.reportValidity();return;}
  input.setCustomValidity("");
  recordGif(ev,run,"HEAD",e.i,cardEl,false,null,steps);
 }},label);
 input.addEventListener("input",()=>input.setCustomValidity(""));
 return h("div",{class:"previewctl"},input,btn);
}

function epCard(run,e,s){
 const c=h("div",{class:"ep"});
 if(e.num_steps!=null)c.dataset.steps=e.num_steps;
 c._ep=e;  // recordGif updates e.gif_path here once a render lands
 const color=e.outcome==="success"?"var(--ok)":e.outcome==="failure"?"var(--no)":"var(--warn)";
 const cap=h("div",{class:"cap"},h("span",{class:"dot",style:`background:${color}`}),
  `ep${e.i} `,
  e.seed!=null?h("span",{class:"mono muted",title:`held-out validation seed ${e.seed}`},`seed…${String(e.seed).slice(-8)} `):"",
  `c${e.object_count} ${num(e.num_steps,0)}st`,copyBtn(()=>epRef(run,e)));
 const xenvGrp=s&&s.environment_counterpart;
 const xenvUrl=xenvGrp?`/api/gif?run=${enc(run)}&kind=eval&key=${e.i}&env=${enc(xenvGrp)}`:null;
 // Pair view: the same episode seed under both spawn ranges, side by side. Only
 // episodes with both gifs already recorded widen; the rest keep the single view.
 if(PAIRVIEW&&xenvGrp&&e.has_gif&&e.xenv_has_gif&&e.plan_found!==false){
  c.classList.add("pair");
  c.append(h("div",{class:"pair2"},
   h("div",{},h("div",{class:"pl"},envShort(s.environment)+" (eval)"),
    gifImg(`/api/gif?run=${enc(run)}&kind=eval&key=${e.i}`)),
   h("div",{},h("div",{class:"pl"},envShort(xenvGrp)),gifImg(xenvUrl))));
  c.append(cap);return c;
 }
 if(e.plan_found===false){
  // No plan means no action was executed, so any GIF here is one frame. Say so
  // rather than showing a render that looks frozen.
  c.append(h("div",{class:"ph noplan"},h("div",{},"no plan found"),
   h("div",{class:"mono"},e.planning_time!=null?`${num(e.planning_time,1)}s search`:"")));}
 else if(e.has_gif){c.append(gifImg(`/api/gif?run=${enc(run)}&kind=eval&key=${e.i}`));
  if(e.gif_preview)c.append(previewControls(run,e,c,"re-preview"),
   h("button",{class:"fullbtn",onclick:ev=>recordGif(ev,run,"HEAD",e.i,c,true)},"render full"));}
 else{const ph=h("div",{class:"ph"},previewControls(run,e,c));c.append(ph);}
 // Replay the same policy and episode seed under the paired environment group
 // (matched spawn-range comparison). Opens in the lightbox; the gif persists
 // under videos_env_<group>/ in the run dir.
 if(xenvGrp&&e.plan_found!==false){
  // class "cp", not "fullbtn": recordGif's done handler removes the first
  // .fullbtn to clear a stale "render full" button and must not take this one.
  const xb=h("button",{class:"cp",
   title:`replay this policy on this episode seed under ${xenvGrp}`
    +(e.xenv_has_gif?" (shift-click to re-record)":""),
   onclick:ev=>{if(c._ep.xenv_has_gif&&!ev.shiftKey){openLightbox(xenvUrl+`&t=${Date.now()}`);return;}
    recordGif(ev,run,"HEAD",e.i,c,true,xenvGrp);}},
   (e.xenv_has_gif?"view in ":"↷ ")+envShort(xenvGrp)+" env");
  cap.append(xb);
 }
 c.append(cap);return c;
}

async function recordGif(ev,run,target,i,cardEl,full=false,env=null,previewSteps=100){
 const btn=ev.currentTarget||ev.target;const origLabel=btn.textContent;
 btn.disabled=true;btn.textContent=full?"full render":"preview";
 const oldStatus=cardEl.querySelector(".render-status");if(oldStatus)oldStatus.remove();
 const meter=h("div",{class:"render-meter indeterminate"},h("span",{}));
 const status=h("div",{class:"render-status"},"requesting render…",meter);
 btn.after(status);
 const started=Date.now();
 const body={run,target,episode_index:i,full};if(env)body.environment=env;
 if(!full){body.preview_steps=previewSteps;cardEl.dataset.previewSteps=previewSteps;}
 const {job_id}=await j("/api/render",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify(body)});
 const tick=async()=>{
  const s=await j("/api/job?job="+job_id);
  const elapsed=Math.max(0,Math.round((Date.now()-started)/1000));
  const detail=s.message||s.state;
  status.firstChild.textContent=`${detail} · ${duration(elapsed)}`;
  const determinate=s.phase==="rollout"&&s.total>0;
  meter.classList.toggle("indeterminate",!determinate&&s.state!=="done");
  if(determinate)meter.firstChild.style.width=`${Math.min(100,100*s.current/s.total)}%`;
  if(s.state==="done"){clearInterval(poll);
   status.remove();
   if(env){
    // A cross-environment replay opens in the lightbox and leaves the card's
    // own-suite gif untouched.
    if(cardEl._ep)cardEl._ep.xenv_has_gif=true;
    btn.disabled=false;btn.textContent=origLabel.replace("↷ ","view in ");
    openLightbox(s.gif_url);return;}
   if(cardEl._ep)cardEl._ep.gif_path=s.gif_path;
   const img=gifImg(s.gif_url);
   const ph=cardEl.querySelector(".ph"),old=cardEl.querySelector("img");
   if(ph)ph.replaceWith(img);else if(old)old.replaceWith(img);else cardEl.prepend(img);
   const stale=cardEl.querySelector(".fullbtn");if(stale)stale.remove();
   const oldPreviewCtl=cardEl.querySelector(".previewctl");
   if(!s.capped&&oldPreviewCtl)oldPreviewCtl.remove();
   if(s.capped){const steps=cardEl.dataset.steps;
    if(!oldPreviewCtl)img.after(previewControls(run,cardEl._ep||{i},cardEl,"re-preview"));
    img.after(h("button",{class:"fullbtn",
     onclick:ev2=>recordGif(ev2,run,target,i,cardEl,true)},
     steps?`render full (${steps}st)`:"render full"));}}
  else if(s.state==="error"){clearInterval(poll);btn.disabled=false;btn.textContent="retry";
   status.style.color="var(--no)";}
  else if(s.state==="cancelled"){clearInterval(poll);status.remove();btn.disabled=false;
   btn.textContent=full?"render full":"preview gif";}
  else{btn.textContent=s.phase==="queued"?"queued":full?"rendering full":"rendering preview";}
 };
 const poll=setInterval(tick,750);await tick();
}

async function renderRun(id){
 const app=$("#app");app.innerHTML="loading...";
 const d=await j("/api/run?run="+enc(id));
 const m=d.metrics,s=d.summary;app.innerHTML="";
 VIEWER=d.viewer||VIEWER;
 RUNTAG=[s.model,s.env_access,s.environment,s.primitives,s.approach].filter(Boolean).join(" ");
 const runRef=()=>`[${VIEWER}] ${id} | ${RUNTAG}`
  +` | solve_rate=${num(m.solve_rate)} | ${d.run_path}`;
 app.append(h("div",{class:"hrow"},h("h1",{},id),copyBtn(runRef,"copy run ref")),
  h("div",{class:"row"},h("span",{class:"pill "+s.status},s.status),
   modelPill(s.model),
   accessPill(s.env_access),
   s.budget!=null?h("span",{class:"pill"},"$"+s.budget):"",s.seed!=null?h("span",{class:"pill"},"s"+s.seed):"",
   h("span",{class:"pill"},s.approach),h("span",{class:"pill",title:"evaluated suite"},"eval: "+s.environment),
   s.trained_environment&&s.trained_environment!==s.environment?
    h("span",{class:"pill",style:"color:var(--warn);border-color:var(--warn)"},
     "trained: "+s.trained_environment):"",
   s.slice?h("span",{class:"pill",style:"color:var(--accent);border-color:var(--accent)",
    title:"conditioned instance slice, not the uniform suite"},"slice: "+s.slice):"",
   s.policy_source?h("span",{class:"pill",title:s.policy_source},
    "policy: "+s.policy_source.split("/").slice(-3).join("/")):""));
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
 const epHdr=h("h2",{},`held-out validation episodes (${succ.length} solved / ${fail.length} failed)`);
 if(s.environment_counterpart){
  const paired=d.episodes.filter(e=>e.has_gif&&e.xenv_has_gif).length;
  epHdr.append(h("button",{class:"cp"+(PAIRVIEW?" ok":""),style:"margin-left:10px",
   title:"Episodes with both gifs recorded show this env and "+s.environment_counterpart+" side by side",
   onclick:()=>{PAIRVIEW=!PAIRVIEW;renderRun(id);}},
   PAIRVIEW?"single-env view":`pair both envs (${paired} ready)`));
 }
 app.append(epHdr);
 app.append(h("div",{class:"muted",style:"margin:-6px 0 10px;font-size:12px"},
  "epN is held-out validation episode N; its environment seed (shown per card) is drawn from the run seed and is unrelated to N."));
 const cols=h("div",{class:"cols"});
 const cS=h("div",{},h("div",{class:"muted",style:"margin-bottom:6px"},"Solved"),h("div",{class:"eph"}));
 const cF=h("div",{},h("div",{class:"muted",style:"margin-bottom:6px"},"Failed"),h("div",{class:"eph"}));
 for(const e of succ.slice(0,60))cS.querySelector(".eph").append(epCard(id,e,s));
 for(const e of fail.slice(0,60))cF.querySelector(".eph").append(epCard(id,e,s));
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

function pearson(xs,ys){
 const n=xs.length;if(n<2)return null;
 const mx=xs.reduce((a,b)=>a+b,0)/n,my=ys.reduce((a,b)=>a+b,0)/n;
 let sxy=0,sxx=0,syy=0;
 for(let i=0;i<n;i++){const dx=xs[i]-mx,dy=ys[i]-my;sxy+=dx*dy;sxx+=dx*dx;syy+=dy*dy;}
 return sxx===0||syy===0?null:sxy/Math.sqrt(sxx*syy);
}

// Does each commit take longer as the run goes on, and does that track the tokens
// (accumulated context) fed to the agent? Overlays per-commit wall time and agent
// tokens against commit index, each on its own scale, and reports the Pearson
// correlations that quantify the "more context, slower generation" hypothesis.
function genTimePlot(snaps){
 const W=420,H=168,L=44,R=46,T=12,B=30,iw=W-L-R,ih=H-T-B,n=snaps.length;
 const x=i=>L+(n===1?iw/2:i*iw/(n-1));
 const times=snaps.map(s=>s.effort.elapsed_s),toks=snaps.map(s=>s.effort.tokens);
 const maxTime=Math.max(1,...times.filter(v=>v!=null)),maxTok=Math.max(1,...toks.filter(v=>v!=null));
 const yT=v=>T+(1-v/maxTime)*ih,yK=v=>T+(1-v/maxTok)*ih;
 const line=(vals,yf,color)=>{
  const segs=[];let cur=[];
  vals.forEach((v,i)=>{if(v==null){if(cur.length){segs.push(cur);cur=[];}}else cur.push([i,v]);});
  if(cur.length)segs.push(cur);
  const polys=segs.map(s=>`<polyline points="${s.map(([i,v])=>`${x(i)},${yf(v)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round"/>`).join("");
  const dots=vals.map((v,i)=>v==null?"":`<circle cx="${x(i)}" cy="${yf(v)}" r="3.2" fill="${color}"><title>v${i}: ${duration(times[i])} · ${compact(toks[i])} tok</title></circle>`).join("");
  return polys+dots;
 };
 const grid=[0,.5,1].map(f=>`<line x1="${L}" y1="${T+f*ih}" x2="${W-R}" y2="${T+f*ih}" stroke="var(--line)"/>`).join("");
 const leftAx=[0,.5,1].map(f=>`<text x="${L-5}" y="${T+(1-f)*ih+3}" text-anchor="end" fill="var(--accent)" font-size="9">${duration(maxTime*f)}</text>`).join("");
 const rightAx=[0,.5,1].map(f=>`<text x="${W-R+5}" y="${T+(1-f)*ih+3}" text-anchor="start" fill="var(--warn)" font-size="9">${compact(Math.round(maxTok*f))}</text>`).join("");
 const labelEvery=Math.max(1,Math.ceil(n/8));
 const labels=snaps.map((sn,i)=>(i%labelEvery===0||i===n-1)?`<text x="${x(i)}" y="${H-8}" text-anchor="middle" fill="var(--muted)" font-size="9">v${i}</text>`:"").join("");
 const svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="generation time and tokens by commit">${grid}${leftAx}${rightAx}${line(toks,yK,"var(--warn)")}${line(times,yT,"var(--accent)")}${labels}</svg>`;
 const idx=[],tIdx=[],kBoth=[],tBoth=[];
 snaps.forEach((s,i)=>{const e=s.effort;if(e.elapsed_s!=null){idx.push(i);tIdx.push(e.elapsed_s);
  if(e.tokens!=null){kBoth.push(e.tokens);tBoth.push(e.elapsed_s);}}});
 const fmt=r=>r==null?"n/a":(r>=0?"+":"")+r.toFixed(2);
 return h("div",{class:"solve-lineplot"},
  h("div",{class:"title"},"per-commit generation effort"),
  h("div",{class:"chart-legend"},
   h("span",{class:"chart-key",style:"--key:var(--accent)"},"wall time"),
   h("span",{class:"chart-key",style:"--key:var(--warn)"},"agent tokens")),
  h("div",{html:svg}),
  h("div",{class:"muted",style:"font-size:10px;margin-top:3px"},
   `correlation: time ↔ version ${fmt(pearson(idx,tIdx))} · time ↔ tokens ${fmt(pearson(kBoth,tBoth))}`));
}

function trainingSeedEvolution(snaps,training){
 const audit=training.object_count_audit||{};
 if(!training.seeds.length&&audit.status==="not_applicable")return null;
 const section=h("div",{},h("h2",{},"training-seed exploration"),
  h("div",{class:"muted"},
   `Recovered ${training.seeds.length} literal environment seeds from the agent's test commands. Filled cells show when a seed was exercised around that commit; they are training probes, not held-out eval episodes.`));
 if(audit.status!=="not_applicable"){
  const labels={
   violation:["violation","HELD-OUT COUNT ACCESSED"],
   no_violation_observed:["safe","no held-out count observed"],
   incomplete:["incomplete","audit incomplete"]
  };
  const [cls,title]=labels[audit.status]||["incomplete","audit unavailable"];
  const list=values=>values?.length?values.join(", "):"—";
  const card=h("div",{class:`count-audit ${cls}`},
   h("div",{class:"audit-title"},title),
   h("div",{class:"audit-grid"},
    h("span",{},`allowed during training: ${list(audit.design_counts)}`),
    h("span",{},`held out: ${list(audit.held_out_counts)}`),
    h("span",{},`observed attempts: ${list(audit.observed_counts)}`),
    audit.violation_counts?.length?h("span",{style:"color:var(--no)"},
     `held-out violations: ${list(audit.violation_counts)}`):"",
    audit.other_out_of_domain_counts?.length?h("span",{style:"color:var(--warn)"},
     `other out-of-domain: ${list(audit.other_out_of_domain_counts)}`):""));
  const caveat=audit.status==="no_violation_observed"?
   "Static audit found no out-of-domain object_count arguments. This verifies recorded tool calls and recoverable test code, not activity absent from the stream.":
   audit.status==="incomplete"?
   `Some object-count expressions could not be resolved (${list(audit.unresolved_expressions)}), or no explicit count was recorded.`:
   "At least one explicit attempted object count was outside design_counts, so higher-count evaluation was not fully held out.";
  card.append(h("div",{class:"muted",style:"font-size:10px;margin-top:6px"},caveat));
  const relevant=(training.count_events||[]).filter(event=>
   audit.status!=="violation"||event.counts.some(count=>audit.violation_counts.includes(count)));
  if(relevant.length){
   const details=h("details",{style:"margin-top:7px"},h("summary",{},
    `${relevant.length} object-count evidence event${relevant.length===1?"":"s"}`));
   relevant.forEach(event=>details.append(h("div",{class:"mono",style:"font-size:10px;margin-top:5px"},
    `v${event.version} · counts ${list(event.counts)} · ${event.source}: ${event.evidence||"(structured tool input)"}`)));
   card.append(details);
  }
  section.append(card);
 }
 if(!training.seeds.length)return section;
 const matrix=items=>{
  const table=h("table",{class:"seed-matrix"});
  const head=h("tr",{},h("th",{},"training seed / object count"));
  snaps.forEach((sn,i)=>head.append(h("th",{},`v${i}`)));head.append(h("th",{},"mentions"));table.append(head);
  items.forEach(item=>{
   const evidence=item.evidence.join("\n");
   const countLabel=item.object_counts.length?` · objects ${item.object_counts.join(",")}`:
    (item.object_count_unknown?" · objects ?":" · design-auto");
   const row=h("tr",{},h("th",{class:"seed-label"},`seed ${item.seed}${countLabel}`));
   snaps.forEach((sn,v)=>{
    const versionCounts=(training.count_events||[]).filter(event=>
     event.version===v&&event.seeds.includes(item.seed)).flatMap(event=>event.counts);
    const cellLabel=versionCounts.length?`c${[...new Set(versionCounts)].join(",")}`:"tested";
    row.append(h("td",{},item.versions.includes(v)?
    h("button",{class:"seed-cell tested",title:evidence},cellLabel):
    h("span",{class:"seed-cell muted",style:"display:inline-block;text-align:center"},"·")));
   });
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
    if(state.state==="cancelled"){clearInterval(poll);evaluate.disabled=false;
     evaluate.textContent="cancelled, click to run again";}
   },1500);
  }},`evaluate all ${episodes.length} seeds × ${incomplete.length} commits (${rollouts} rollouts)`);
  legend.append(evaluate);
 }
 wrap.append(h("div",{class:"history-head"},legend,
  h("div",{class:"history-plots"},solveLinePlot(snaps,episodes.length),genTimePlot(snaps))));
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
 // The matrix cell records the definitive solved/failed outcome, so it renders in full.
 const {job_id}=await j("/api/render",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({run:id,target:String(v),episode_index:i,full:true})});
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
 const recBtn=h("button",{onclick:ev=>recordGif(ev,id,String(v),+epIn.value,gifBox,true)},"record gif");
 ctl.append(h("span",{class:"muted",style:"margin-left:10px"},"held-out episode"),epIn,recBtn);
 const gifBox=h("div",{class:"ep",style:"width:180px;margin-top:8px"},h("div",{class:"ph muted"},"no gif yet"));
 panel.append(ctl,gifBox,view);
 showDiff();
}

async function route(){
 const hash=location.hash||"#/index";
 if(!ALL.length)await loadRuns();
 if(hash.startsWith("#/run/")){$("#brand").href=INDEX_HASH;renderRun(decodeURIComponent(hash.slice(6)));}
 else{restoreFilters(hash);INDEX_HASH=indexHash();$("#brand").href=INDEX_HASH;renderIndex();}
}
window.addEventListener("hashchange",route);
$("#refresh").addEventListener("click",async ev=>{
 const b=ev.currentTarget,old=b.textContent;b.disabled=true;b.textContent="syncing…";
 try{
  const r=await fetch("/api/refresh",{method:"POST"}),payload=await r.json();
  if(!r.ok)throw new Error(payload.error||"refresh failed");
  await loadRuns();route();
  const d=payload.drive||{};
  b.textContent=d.downloaded!=null?`synced ${d.downloaded}, kept ${d.unchanged}`:"refreshed";
 }catch(e){b.textContent="sync failed";alert(e.message);}
 finally{setTimeout(()=>{b.textContent=old;b.disabled=false;},1800);}
});
$("#cancel-renders").addEventListener("click",async ev=>{
 const b=ev.currentTarget;b.disabled=true;
 const r=await j("/api/render/cancel",{method:"POST"});
 b.textContent=`dropped ${r.dropped}`;
 setTimeout(()=>{b.textContent="cancel renders";b.disabled=false;},1800);});
// auto-refresh on the index when runs finish
let lastStamp="";
setInterval(async()=>{if(!location.hash.startsWith("#/run/")){const{stamp}=await j("/api/stamp");
 if(lastStamp&&stamp!==lastStamp){await loadRuns();if(!location.hash.startsWith("#/run/"))renderIndex();}lastStamp=stamp;}},10000);
route();
"""


def main() -> None:
    """Discover runs, start the render worker, serve forever."""
    ap = argparse.ArgumentParser(description="robocode results viewer")
    ap.add_argument("--root", help="local root to scan for runs (default: repo root)")
    ap.add_argument(
        "--drive-folder",
        default=os.environ.get("ROBOCODE_RESULTS_DRIVE_FOLDER"),
        metavar="URL_OR_ID",
        help=(
            "read-only Drive folder of ZIP result archives; may also be set with "
            "ROBOCODE_RESULTS_DRIVE_FOLDER"
        ),
    )
    ap.add_argument(
        "--drive-cache",
        help=(
            "private local cache base (default: $ROBOCODE_RESULTS_CACHE or the "
            "platform user cache)"
        ),
    )
    ap.add_argument(
        "--rclone-remote",
        default=os.environ.get("ROBOCODE_RCLONE_REMOTE", "robocode-drive"),
        metavar="NAME",
        help="configured rclone Drive remote name (default: robocode-drive)",
    )
    ap.add_argument(
        "--rclone-config",
        default=os.environ.get("RCLONE_CONFIG"),
        metavar="PATH",
        help="optional rclone config path (default: rclone's platform default)",
    )
    ap.add_argument(
        "--rclone-binary",
        default=os.environ.get("RCLONE_BINARY", "rclone"),
        metavar="PATH",
        help="rclone executable name or path (default: rclone)",
    )
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

    global DRIVE_SYNC  # pylint: disable=global-statement
    if args.drive_folder and args.root:
        ap.error("--root and --drive-folder cannot be used together")
    if args.drive_folder:
        cache_base = (
            Path(args.drive_cache).expanduser()
            if args.drive_cache
            else drive_results.default_cache_base()
        )
        DRIVE_SYNC = drive_results.RcloneResultsSync(
            args.drive_folder,
            cache_base,
            args.rclone_remote,
            rclone_binary=args.rclone_binary,
            config_path=(
                Path(args.rclone_config).expanduser() if args.rclone_config else None
            ),
        )
        SCAN.root = DRIVE_SYNC.runs_dir
    else:
        SCAN.root = Path(args.root or REPO_ROOT).resolve()
    SCAN.exclude_dirs = set(args.exclude)
    SCAN.tag = f"{SCAN.root.name}:{args.port}"
    try:
        drive_report = sync_drive_and_refresh()
    except (OSError, RuntimeError, ValueError) as error:
        ap.error(str(error))
    print(f"discovered {len(RUNS)} runs under {SCAN.root}")
    if drive_report:
        print(
            "Drive sync: "
            f"{drive_report['downloaded']} downloaded, "
            f"{drive_report['unchanged']} unchanged, "
            f"{drive_report['removed']} removed, "
            f"{drive_report['ignored']} non-ZIP files ignored"
        )

    threading.Thread(target=_render_worker, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
