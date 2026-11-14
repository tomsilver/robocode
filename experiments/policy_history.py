"""Analyze every saved version of one agent's policy and plot how it changed.

A coding agent commits ``approach.py`` to its sandbox's git history while it works.
``run`` extracts each version (oldest first), runs the search-explanations analyzer
on it (``python -m robocode.planning_explanations.search --suite policy --source``)
in parallel, and collects one row of numbers per version into ``history.json``.
``plot`` turns ``history.json`` into figures and ``findings.md``, a list of the
notable changes (first search, jumps, verdict changes, solved episodes), optionally
compared with an older ``history.json`` made by an earlier analyzer.

Usage:

    python experiments/policy_history.py run --sandbox <sandbox dir> --out <dir>
    python experiments/policy_history.py plot --out <dir> [--compare <history.json>]

``run`` resumes: versions whose analysis already finished are not analyzed again
(``--force`` re-runs them).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# Program-level numbers reported as a mean over held-out episodes, or withheld
# with a lower bound ("≥") when some loop could not be read.
BOUNDED = (
    "search_score",
    "planning_events",
    "expansion_events",
    "initialization_expansion_events",
    "attributed_expansion_events",
)
# Per action call: what the action relied on (counts in each episode's accounting).
CATEGORIES = (
    ("followed_witnessed_plan", "Followed a plan, witnessed"),
    ("followed_unresolved_plan", "Followed a plan, unresolved"),
    ("followed_unmatched_plan", "Followed a plan not read as a search path"),
    ("followed_disproved_plan", "Followed a plan, disproved"),
    ("ran_search", "Searched in the call"),
    ("unresolved_loop", "Ran an unknown loop"),
    ("neither", "Reactive (no search, no plan)"),
)
FINDINGS = {
    "search_supported": "explained",
    "unreadable_loops": "unknown (≥)",
    "partially_explained": "partly explained (≥)",
    "no_search_parts": "no search",
}


# --------------------------------------------------------------------------- run


def extract(sandbox: Path, out: Path) -> list[dict]:
    """Every committed version of approach.py, oldest first, as checkpoint files."""
    log = (
        subprocess.run(
            [
                "git",
                "-C",
                str(sandbox),
                "log",
                "--reverse",
                "--format=%h%x09%ad%x09%s",
                "--date=format:%Y-%m-%d %H:%M:%S",
                "--",
                "approach.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    versions = []
    for index, line in enumerate(log, 1):
        sha, when, subject = line.split("\t", 2)
        source = subprocess.run(
            ["git", "-C", str(sandbox), "show", f"{sha}:approach.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        name = f"v{index:02d}_{sha}"
        (checkpoints / f"{name}.py").write_text(source, encoding="utf-8")
        versions.append(
            {
                "index": index,
                "sha": sha,
                "time": when,
                "subject": subject,
                "name": name,
                "lines": source.count("\n"),
            }
        )
    (out / "versions.json").write_text(json.dumps(versions, indent=1))
    return versions


def analyze(version: dict, out: Path, minutes: float, force: bool) -> dict:
    """Run the analyzer CLI on one version; returns its status."""
    target = out / "runs" / version["name"]
    status_file = target / "status.json"
    if status_file.exists() and not force:
        return json.loads(status_file.read_text())
    target.mkdir(parents=True, exist_ok=True)
    start = time.time()
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "robocode.planning_explanations.search",
                "--suite",
                "policy",
                "--source",
                str(out / "checkpoints" / f"{version['name']}.py"),
                "--output-dir",
                str(target),
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=minutes * 60,
            check=False,
        )
        error = done.stderr.strip().splitlines()[-1] if done.returncode else ""
        status = {
            "status": "ok" if done.returncode == 0 else "crashed",
            "error": error[:300],
        }
    except subprocess.TimeoutExpired:
        status = {"status": "timeout", "error": f"over {minutes:g} minutes"}
    status["seconds"] = round(time.time() - start)
    status_file.write_text(json.dumps(status))
    print(f"{version['name']}: {status['status']} in {status['seconds']} s", flush=True)
    return status


def _category_shares(episode: dict) -> dict[str, float]:
    program = episode["accounting"]["program"]
    calls = program.get("action_calls") or 0
    return {
        key: (program.get(key, 0) / calls if calls else 0.0) for key, _ in CATEGORIES
    }


def metrics(run_dir: Path) -> dict:
    """One version's numbers from the analyzer's summary.json and report.json."""
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        return {}
    summary = json.loads(summary_file.read_text())[0]
    if summary.get("error") or "complete" not in summary:
        # The policy itself failed (an exception in its own code).
        return {"policy_error": summary.get("error", "no result")}
    row: dict[str, Any] = {
        "finding": summary["finding"]["code"],
        "complete": summary["complete"],
        "solved": int(str(summary["solved"]).split("/", maxsplit=1)[0]),
        "episodes": int(str(summary["solved"]).split("/")[1]),
        "followed_low": summary.get("mean_followed_plan_share"),
        "followed_high": summary.get("mean_followed_plan_share_max"),
        "searched_share": summary.get("mean_searched_share"),
        "max_depth": (
            summary.get("max_expanded_depth")
            if summary.get("max_expanded_depth") is not None
            else summary.get("lower_bound_max_expanded_depth")
        ),
    }
    for key in BOUNDED:
        row[key] = summary.get(f"mean_{key}")
        row[f"{key}_bound"] = summary.get(f"lower_bound_{key}")
    report_file = next(run_dir.glob("*/report.json"), None)
    if report_file is None:
        return row
    report = json.loads(report_file.read_text())
    parts = []
    for site, part in report["frozen"]["parts"].items():
        info = report["sites"][site]
        audit = report["discovery_audit"]["parts"].get(site, {})
        parts.append(
            {
                "site": site,
                "kind": info.get("loop", info["kind"]),
                "status": part["status"],
                "reason": audit.get("unreadable"),
                "function": info["function"],
                "source": info["source"][:90],
            }
        )
    row["parts"] = sorted(parts, key=lambda p: int(p["site"].split(":")[0]))
    late = Counter()
    episodes = []
    for episode in report["validation"]["episodes"]:
        program = episode["accounting"]["program"]
        for entry in episode["runs"]:
            if entry.get("unreadable"):
                late[(entry["site"], entry["unreadable"])] += 1
        score = program.get("search_score")
        episodes.append(
            {
                "case": episode["case"],
                "solved": episode["solved"],
                "steps": episode["steps"],
                "complete": program.get("complete"),
                "search_score": score,
                "search_score_bound": program.get("explained_only_search_score", score),
                "expansion_events": program.get("expansion_events"),
                "planning_events": program.get("planning_events"),
                "followed_low": program.get("followed_plan_share"),
                "followed_high": program.get("followed_plan_share_max"),
                "shares": _category_shares(episode),
            }
        )
    row["episode_rows"] = episodes
    # Loops "not a search" in development but flagged on a held-out run.
    row["validation_unknown"] = [
        {"site": site, "reason": reason, "runs": n}
        for (site, reason), n in late.items()
    ]
    return row


def run(args: argparse.Namespace) -> None:
    """Extract, analyze in parallel, and collect history.json."""
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    versions = extract(args.sandbox, out)
    print(f"{len(versions)} versions extracted", flush=True)
    chosen = (
        [v for v in versions if v["index"] in args.versions]
        if args.versions
        else versions
    )
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda v: analyze(v, out, args.minutes, args.force), chosen))
    # Every version analyzed so far (now or in an earlier, partial run).
    history = []
    for version in versions:
        status_file = out / "runs" / version["name"] / "status.json"
        if status_file.exists():
            status = json.loads(status_file.read_text())
            history.append(
                {**version, **status, **metrics(out / "runs" / version["name"])}
            )
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"Wrote {out / 'history.json'} ({len(history)} of {len(versions)} versions)")


# -------------------------------------------------------------------------- plot

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
HAIRLINE = "#e4e3df"
MUTED = "#a9a8a3"
# Categorical slots, fixed order (validated reference palette, light mode).
SLOTS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": HAIRLINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.color": HAIRLINE,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _at(history: list[dict], index: int) -> dict:
    """The version with this number (histories may be partial)."""
    return next(v for v in history if v["index"] == index)


def _x(history: list[dict]) -> list[int]:
    return [v["index"] for v in history]


def _value(version: dict, key: str) -> tuple[float | None, bool]:
    """A program number and whether it is a lower bound (withheld)."""
    if version.get(key) is not None:
        return version[key], False
    if version.get(f"{key}_bound") is not None:
        return version[f"{key}_bound"], True
    return None, False


def _short(text: str, size: int = 44) -> str:
    return text if len(text) <= size else text[: size - 1] + "…"


def _last(history: list[dict]) -> int:
    return max(v["index"] for v in history)


def _version_axis(ax, history: list[dict]) -> None:
    ax.set_xlim(0.3, _last(history) + 0.7)
    shown = _x(history)
    ax.set_xticks(
        shown if len(shown) <= 16 else [v for v in shown if v == 1 or v % 5 == 0]
    )
    ax.set_xlabel("Version (commit order)")


def _failed_strip(ax, history: list[dict], y: float) -> None:
    xs = [v["index"] for v in history if v.get("status") != "ok"]
    if xs:
        ax.scatter(
            xs,
            [y] * len(xs),
            marker="x",
            color=MUTED,
            s=36,
            linewidths=1.5,
            label="Analysis failed (policy crash or timeout)",
            zorder=3,
        )


def _bounded_series(ax, history, key, color, label) -> None:
    xs, ys, hollow_x, hollow_y = [], [], [], []
    for v in history:
        value, bound = _value(v, key)
        if value is None:
            continue
        xs.append(v["index"])
        ys.append(value)
        if bound:
            hollow_x.append(v["index"])
            hollow_y.append(value)
    ax.plot(xs, ys, color=color, linewidth=1.8, zorder=2, label=label)
    full = [(x, y) for x, y in zip(xs, ys) if x not in hollow_x]
    if full:
        ax.scatter(*zip(*full), color=color, s=22, zorder=3)
    if hollow_x:
        ax.scatter(
            hollow_x,
            hollow_y,
            facecolor=SURFACE,
            edgecolor=color,
            s=40,
            linewidths=1.5,
            zorder=4,
            label="Withheld: lower bound (≥)",
        )


def _annotate(ax, history, picks: list[int], key: str) -> None:
    """Letters at a few chosen versions, and a key under the chart naming each
    commit (labels never collide with the line or each other)."""
    entries = []
    for letter, index in zip("ABCDEFGH", sorted(picks)):
        version = _at(history, index)
        value, _ = _value(version, key)
        if value is None:
            continue
        ax.annotate(
            letter,
            xy=(index, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
        entries.append(f"{letter}  v{index}: {_short(version['subject'], 70)}")
    ax.text(
        0,
        -0.33,
        "\n".join(entries),
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color=INK_2,
        linespacing=1.5,
    )


def _jumps(history: list[dict], key: str, count: int) -> list[tuple[int, float, float]]:
    """Largest changes between consecutive analyzed versions."""
    seen = [
        (v["index"], _value(v, key)[0])
        for v in history
        if _value(v, key)[0] is not None
    ]
    deltas = [(b[0], a[1], b[1]) for a, b in zip(seen, seen[1:]) if a[1] != b[1]]
    return sorted(deltas, key=lambda d: -abs(d[2] - d[1]))[:count]


def _level_changes(
    history: list[dict], key: str, count: int, hold: int = 2
) -> list[tuple[int, float, float]]:
    """Largest changes that last: the new level holds for `hold` more versions
    (within 25%), so a change reverted at once is not picked."""
    seen = [
        (v["index"], _value(v, key)[0])
        for v in history
        if _value(v, key)[0] is not None
    ]
    changes = []
    for i in range(1, len(seen)):
        (index, value), before = seen[i], seen[i - 1][1]
        if value == before:
            continue
        later = [v for _, v in seen[i + 1 : i + 1 + hold]]
        if all(abs(v - value) <= 0.25 * max(abs(value), 0.5) for v in later):
            changes.append((index, before, value))
    return sorted(changes, key=lambda d: -abs(d[2] - d[1]))[:count]


def plot_score(plt, history, out: Path, compare: dict | None) -> Path:
    """Search score per version, with lower bounds and failed versions."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _bounded_series(
        ax, history, "search_score", SLOTS[0], "Search score, current analyzer"
    )
    if compare:
        xs = [v["index"] for v in history if compare.get(v["index"]) is not None]
        ax.scatter(
            xs,
            [compare[x] for x in xs],
            facecolor="none",
            edgecolor=SLOTS[1],
            s=80,
            linewidths=1.3,
            zorder=1,
            label="Earlier analyzer (rings; a ring around a dot means the same value)",
        )
    _failed_strip(ax, history, 0)
    ax.set_ylim(bottom=-0.4, top=ax.get_ylim()[1] * 1.08)
    first = next(
        (v["index"] for v in history if (_value(v, "search_score")[0] or 0) > 0), None
    )
    picks = {first} if first else set()
    picks.update(j[0] for j in _level_changes(history, "search_score", 4))
    _annotate(ax, history, [p for p in picks if p], "search_score")
    ax.set_title("Search score per policy call, across the agent's versions")
    ax.set_ylabel("Cells expanded beyond the start, per call")
    _version_axis(ax, history)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    path = out / "01_search_score.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_effort(plt, history, out: Path) -> Path:
    """Searches, cells expanded and depth per episode, one panel each."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True)
    panels = (
        ("planning_events", "Searches per episode"),
        ("expansion_events", "Cells expanded per episode"),
        ("max_depth", "Deepest expansion (cells)"),
    )
    for ax, (key, label) in zip(axes, panels):
        if key == "max_depth":
            xs = [v["index"] for v in history if v.get("max_depth") is not None]
            ax.plot(
                xs,
                [_at(history, x)["max_depth"] for x in xs],
                color=SLOTS[0],
                linewidth=1.8,
                marker="o",
                markersize=4,
            )
        else:
            _bounded_series(ax, history, key, SLOTS[0], label)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[0].set_title("How much the policy searches, per held-out episode (mean of 3)")
    handles, labels = axes[0].get_legend_handles_labels()
    keep = [(h, lbl) for h, lbl in zip(handles, labels) if "Withheld" in lbl]
    if keep:
        axes[0].legend(*zip(*keep), loc="upper left")
    _version_axis(axes[-1], history)
    path = out / "02_search_effort.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_share(plt, history, out: Path) -> Path:
    """Followed-plan share as a range per version, and the searched share."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for v in history:
        low, high = v.get("followed_low"), v.get("followed_high")
        if low is None:
            continue
        ax.plot(
            [v["index"]] * 2,
            [low, high],
            color=SLOTS[0],
            linewidth=1.6,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter([v["index"]], [high], marker="_", color=SLOTS[0], s=60, zorder=3)
        ax.scatter([v["index"]], [low], color=SLOTS[0], s=22, zorder=4)
    searched = [
        (v["index"], v["searched_share"])
        for v in history
        if v.get("searched_share") is not None
    ]
    if searched:
        ax.plot(
            *zip(*searched),
            color=SLOTS[1],
            linewidth=1.4,
            label="Share of calls that search themselves",
        )
    ax.scatter(
        [],
        [],
        color=SLOTS[0],
        s=22,
        label="Followed a plan: witnessed (dot) to max (bar top)",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("What the actions relied on: share of action calls")
    ax.set_ylabel("Share of action calls")
    _version_axis(ax, history)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    path = out / "03_plan_use.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_categories(plt, history, out: Path) -> Path:
    """What each action call did, stacked to 1."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    rows = []
    for v in history:
        episodes = v.get("episode_rows") or []
        if not episodes:
            continue
        shares = {
            k: sum(e["shares"][k] for e in episodes) / len(episodes)
            for k, _ in CATEGORIES
        }
        rows.append((v["index"], shares))
    bottom = {x: 0.0 for x, _ in rows}
    for slot, (key, label) in enumerate(CATEGORIES):
        heights = [s[key] for _, s in rows]
        if not any(heights):
            continue
        color = MUTED if key == "neither" else SLOTS[slot]
        ax.bar(
            [x for x, _ in rows],
            heights,
            bottom=[bottom[x] for x, _ in rows],
            width=0.82,
            color=color,
            edgecolor=SURFACE,
            linewidth=0.8,
            label=label,
        )
        for x, s in rows:
            bottom[x] += s[key]
    ax.set_ylim(0, 1.0)
    ax.set_title("What each action call did, per version (mean of 3 held-out episodes)")
    ax.set_ylabel("Share of action calls")
    _version_axis(ax, history)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
    path = out / "04_call_kinds.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _part_group(part: dict) -> str:
    if part["status"] == "search":
        return "Explained search"
    if part["status"] in ("unreadable_loop", "no_explanation"):
        return "Unknown (maybe a search)"
    if part["status"] == "not_executed_in_development":
        return "Not run in practice episodes"
    return "Not a search"


def plot_parts(plt, history, out: Path) -> Path:
    """Parts by verdict per version, and lines of code."""
    groups = (
        "Explained search",
        "Unknown (maybe a search)",
        "Not a search",
        "Not run in practice episodes",
    )
    colors = (SLOTS[0], SLOTS[1], MUTED, "#d9d8d3")
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]}
    )
    rows = [
        (v["index"], Counter(_part_group(p) for p in v.get("parts", [])))
        for v in history
        if v.get("parts") is not None
    ]
    bottom = {x: 0 for x, _ in rows}
    for group, color in zip(groups, colors):
        heights = [c[group] for _, c in rows]
        axes[0].bar(
            [x for x, _ in rows],
            heights,
            bottom=[bottom[x] for x, _ in rows],
            width=0.82,
            color=color,
            edgecolor=SURFACE,
            linewidth=0.8,
            label=group,
        )
        for x, c in rows:
            bottom[x] += c[group]
    axes[0].set_title("Loops and parts the analyzer watched, by verdict")
    axes[0].set_ylabel("Parts (loops, recursion, library calls)")
    axes[0].legend(loc="upper left", ncol=2)
    axes[1].plot(
        _x(history), [v["lines"] for v in history], color=SLOTS[0], linewidth=1.8
    )
    axes[1].set_ylabel("Lines of code")
    axes[1].set_ylim(bottom=0)
    _version_axis(axes[1], history)
    path = out / "05_parts_and_size.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_episodes(plt, history, out: Path) -> Path:
    """Search score per held-out episode (heat map), and episodes solved."""
    from matplotlib.colors import (  # pylint: disable=import-outside-toplevel
        LinearSegmentedColormap,
    )

    cases = []
    for v in history:
        for e in v.get("episode_rows") or []:
            if tuple(e["case"]) not in cases:
                cases.append(tuple(e["case"]))
    grid = [[None] * _last(history) for _ in cases]
    hatch = []
    for v in history:
        col = v["index"] - 1
        for e in v.get("episode_rows") or []:
            row = cases.index(tuple(e["case"]))
            value = (
                e["search_score"]
                if e["search_score"] is not None
                else e["search_score_bound"]
            )
            grid[row][col] = value
            if e["search_score"] is None:
                hatch.append((col, row))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad("#efeeea")
    import numpy as np  # pylint: disable=import-outside-toplevel

    data = np.array(
        [[np.nan if c is None else c for c in r] for r in grid], dtype=float
    )
    image = axes[0].imshow(
        data,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=(0.5, _last(history) + 0.5, len(cases) - 0.5, -0.5),
    )
    for col, row in hatch:
        axes[0].add_patch(
            plt.Rectangle(
                (col + 0.5, row - 0.5),
                1,
                1,
                fill=False,
                hatch="////",
                edgecolor=INK_2,
                linewidth=0,
            )
        )
    axes[0].set_yticks(range(len(cases)))
    axes[0].set_yticklabels([f"{c[0]} obstacles, seed {c[1]}" for c in cases])
    axes[0].grid(False)
    axes[0].set_title("Search score in each held-out episode (hatched: lower bound)")
    # A colorbar beside the heat map that does not narrow it (panels stay aligned).
    scale = fig.colorbar(image, cax=axes[0].inset_axes([1.01, 0, 0.012, 1]))
    scale.outline.set_visible(False)
    scale.set_label("Search score")
    for slot, case in enumerate(cases):
        points = [
            (v["index"], e["steps"], e["solved"])
            for v in history
            for e in v.get("episode_rows") or []
            if tuple(e["case"]) == case
        ]
        axes[1].plot(
            [x for x, _, _ in points],
            [y for _, y, _ in points],
            color=SLOTS[slot],
            linewidth=1.6,
            label=f"{case[0]} obstacles, seed {case[1]}",
        )
        won = [(x, y) for x, y, ok in points if ok]
        if won:
            axes[1].scatter(*zip(*won), color=SLOTS[slot], s=26, zorder=3)
    rows = [e for v in history for e in v.get("episode_rows") or []]
    solved_any = any(e["solved"] for e in rows)
    constant = all(
        len({e["steps"] for e in rows if tuple(e["case"]) == case}) == 1
        for case in cases
    )
    if solved_any:
        title = "Episode length (a dot marks a solved episode)"
    elif constant:
        title = "Episode length: every version ran to the step limit (none solved)"
    else:
        title = "Episode length (no version solved any held-out episode)"
    axes[1].set_title(title)
    axes[1].set_ylabel("Steps")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.3), ncol=3)
    _version_axis(axes[1], history)
    path = out / "06_episodes.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_runtime(plt, history, out: Path) -> Path:
    """Minutes the analysis took per version."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    # A version copied from an earlier analysis has no time of its own.
    ok = [
        (v["index"], v["seconds"] / 60)
        for v in history
        if v.get("status") == "ok" and v.get("seconds") is not None
    ]
    ax.bar([x for x, _ in ok], [m for _, m in ok], width=0.82, color=SLOTS[0])
    bad = [
        (v["index"], v["seconds"] / 60)
        for v in history
        if v.get("status") != "ok" and v.get("seconds") is not None
    ]
    if bad:
        ax.bar(
            [x for x, _ in bad],
            [m for _, m in bad],
            width=0.82,
            color=MUTED,
            label="Analysis failed",
        )
        ax.legend(loc="upper left")
    ax.set_title("Analysis time per version")
    ax.set_ylabel("Minutes")
    _version_axis(ax, history)
    path = out / "07_analysis_time.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- findings


def _fmt(version: dict, key: str, digits: int = 2) -> str:
    value, bound = _value(version, key)
    if value is None:
        return "—"
    text = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return f"≥ {text}" if bound else text


def _old_scores(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    old = json.loads(path.read_text())
    scores = {}
    for v in old:
        summary = v.get("summary") or v
        value = summary.get("mean_search_score")
        if value is None:
            value = summary.get("lower_bound_search_score")
        if value is not None:
            scores[v["index"]] = value
    return scores


def _old_findings(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    old = json.loads(path.read_text())
    out = {}
    for v in old:
        summary = v.get("summary") or v
        finding = summary.get("finding")
        if isinstance(finding, dict):
            out[v["index"]] = finding.get("code")
    return out


def findings(history: list[dict], compare: Path | None) -> str:
    """Notable changes, found by rules (no judgment): read them, then check."""
    lines = ["# Notable changes across versions", ""]
    ok = [v for v in history if v.get("status") == "ok"]
    failed = [v for v in history if v.get("status") != "ok"]
    verdicts = Counter(FINDINGS.get(v.get("finding"), v.get("finding")) for v in ok)
    failures = ", ".join(f"v{v['index']} {v.get('status')}" for v in failed)
    lines += [
        f"- {len(history)} versions; {len(ok)} analyzed, {len(failed)} failed "
        f"({failures or 'none'}).",
        "- Verdicts: " + ", ".join(f"{n} {k}" for k, n in verdicts.most_common()) + ".",
    ]
    first = next(
        (
            v
            for v in ok
            if v.get("finding") == "search_supported"
            and (_value(v, "search_score")[0] or 0) > 0
        ),
        None,
    )
    if first:
        lines.append(
            f"- First explained search: v{first['index']} ({first['time']}, "
            f"\"{first['subject']}\"), search score {_fmt(first, 'search_score')}."
        )
    unknown = [
        v for v in ok if v.get("finding") in ("unreadable_loops", "partially_explained")
    ]
    if unknown:
        lines.append(
            f"- Versions with an unknown loop (totals shown as ≥): "
            f"{', '.join('v' + str(v['index']) for v in unknown)}."
        )
    for v in failed:
        lines.append(
            f"- v{v['index']} failed: {v.get('policy_error') or v.get('error')}"
        )
    for key, title in (
        ("search_score", "search score"),
        ("planning_events", "searches per episode"),
    ):
        lines += [
            "",
            f"## Largest lasting changes in {title}",
            "",
            "Changes whose new level held for the next two analyzed versions.",
            "",
            "| Version | Commit | From | To |",
            "| --- | --- | --- | --- |",
        ]
        for index, before, after in _level_changes(history, key, 5):
            v = _at(history, index)
            lines.append(
                f"| v{index} | {_short(v['subject'], 64)} | {before:.2f} | "
                f"{after:.2f} |"
            )
    lines += ["", "## Changes the agent reverted", ""]
    reverts = []
    for v in ok:
        if not v["subject"].startswith('Revert "'):
            continue
        undone = v["subject"][len('Revert "') :].rstrip('"')
        source = next(
            (
                u
                for u in reversed(history)
                if u["index"] < v["index"] and u["subject"] == undone
            ),
            None,
        )
        if source is None:
            continue
        earlier = [u for u in history if u["index"] < source["index"]]
        if not earlier:
            continue
        before = earlier[-1]
        reverts.append(
            f"- v{source['index']} \"{_short(undone, 60)}\" changed the search score "
            f"{_fmt(before, 'search_score')} → {_fmt(source, 'search_score')}, "
            f"searches per episode {_fmt(before, 'planning_events')} → "
            f"{_fmt(source, 'planning_events')}, plan share "
            f"{_fmt(before, 'followed_low')} → {_fmt(source, 'followed_low')}; "
            f"v{v['index']} reverted it (score {_fmt(v, 'search_score')})."
        )
    lines += reverts or ["- None."]
    lines += ["", "## Verdict changes", ""]
    changes = []
    previous = None
    for v in ok:
        if previous is not None and v.get("finding") != previous.get("finding"):
            why = [
                f"{p['source']} ({p['reason']})"
                for p in v.get("parts", [])
                if p["status"] in ("unreadable_loop", "no_explanation")
            ]
            changes.append(
                f"- v{v['index']} \"{_short(v['subject'], 60)}\": "
                f"{FINDINGS.get(previous.get('finding'))} → "
                f"{FINDINGS.get(v.get('finding'))}"
                + (f". Unknown: {'; '.join(why[:3])}" if why else "")
            )
        previous = v
    lines += changes or ["- None."]
    lines += ["", "## Held-out episodes solved", ""]
    changes = []
    previous = None
    for v in ok:
        if previous is not None and v.get("solved") != previous.get("solved"):
            changes.append(
                f"- v{v['index']} \"{_short(v['subject'], 60)}\": "
                f"{previous.get('solved')}/3 → {v.get('solved')}/3"
            )
        previous = v
    solved = sorted({v.get("solved") for v in ok if v.get("solved") is not None})
    lines += changes or [
        f"- No change: every analyzed version solved {solved[0] if solved else '—'} "
        "of the 3 held-out episodes."
    ]
    lines += ["", "## Plan use (followed-plan share, witnessed – max)", ""]
    changes = []
    previous = None
    for v in ok:
        if previous is not None and None not in (
            v.get("followed_low"),
            previous.get("followed_low"),
        ):
            change = max(
                abs(v["followed_low"] - previous["followed_low"]),
                abs(v["followed_high"] - previous["followed_high"]),
            )
            if change >= 0.1:
                changes.append(
                    f"- v{v['index']} \"{_short(v['subject'], 60)}\": "
                    f"{previous['followed_low']:.2f}–{previous['followed_high']:.2f} → "
                    f"{v['followed_low']:.2f}–{v['followed_high']:.2f}"
                )
        previous = v
    lines += changes or ["- No change of 0.1 or more between versions."]
    held_out = [(v["index"], u) for v in ok for u in v.get("validation_unknown", [])]
    if held_out:
        lines += ["", "## Loops flagged only on held-out episodes", ""]
        lines += [
            f"- v{i}: {u['site']} {u['reason']} ({u['runs']} runs)" for i, u in held_out
        ]
    old_scores, old_findings = _old_scores(compare), _old_findings(compare)
    if old_scores or old_findings:
        rows = []
        for v in ok:
            before = old_scores.get(v["index"])
            now = _value(v, "search_score")[0]
            changed_verdict = old_findings.get(v["index"]) not in (
                None,
                v.get("finding"),
            )
            changed_score = (before is None) != (now is None) or (
                before is not None
                and now is not None
                and abs(before - now) > 0.01 * max(1, abs(before))
            )
            if not (changed_verdict or changed_score):
                continue
            why = [
                f"{p['source'][:40]} ({p['reason']})"
                for p in v.get("parts", [])
                if p["status"] in ("unreadable_loop", "no_explanation")
            ]
            rows.append(
                f"| v{v['index']} | {FINDINGS.get(old_findings.get(v['index']), '—')} | "
                f"{FINDINGS.get(v.get('finding'))} | "
                f"{'—' if before is None else f'{before:.2f}'} | "
                f"{_fmt(v, 'search_score')} | "
                f"{'; '.join(why[:2]) or '—'} |"
            )
        lines += ["", "## Compared with the earlier analyzer", ""]
        if rows:
            lines += [
                "| Version | Earlier verdict | Now | Earlier score | Now "
                "| Why (unknown parts now) |",
                "| --- | --- | --- | --- | --- | --- |",
                *rows,
            ]
        else:
            compared = sum(1 for v in ok if v["index"] in old_scores)
            lines.append(
                f"- No change: all {compared} versions both analyzers scored have the "
                "same verdict and the same search score (within 1%)."
            )
    lines += [
        "",
        "## Every version",
        "",
        "| v | Time | Commit | Lines | Verdict | Score | Searches | Cells "
        "| Plan share | Solved |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for v in history:
        share = (
            "—"
            if v.get("followed_low") is None
            else f"{v['followed_low']:.2f}–{v['followed_high']:.2f}"
        )
        verdict = FINDINGS.get(v.get("finding"), v.get("status"))
        lines.append(
            f"| {v['index']} | {v['time'][-8:]} | {_short(v['subject'], 48)} | "
            f"{v['lines']} | "
            f"{verdict} | {_fmt(v, 'search_score')} | {_fmt(v, 'planning_events')} | "
            f"{_fmt(v, 'expansion_events', 0)} | {share} | "
            f"{'—' if v.get('solved') is None else str(v['solved']) + '/3'} |"
        )
    return "\n".join(lines) + "\n"


def plot(args: argparse.Namespace) -> None:
    """Figures and findings.md from history.json."""
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    _style(plt)
    history = json.loads((args.out / "history.json").read_text())
    figures = args.out / "plots"
    figures.mkdir(exist_ok=True)
    compare = _old_scores(args.compare) or None
    paths = [
        plot_score(plt, history, figures, compare),
        plot_effort(plt, history, figures),
        plot_share(plt, history, figures),
        plot_categories(plt, history, figures),
        plot_parts(plt, history, figures),
        plot_episodes(plt, history, figures),
        plot_runtime(plt, history, figures),
    ]
    (args.out / "findings.md").write_text(
        findings(history, args.compare), encoding="utf-8"
    )
    for path in paths:
        print(path)
    print(args.out / "findings.md")


def _minutes(version: dict, start) -> float:
    from datetime import datetime  # pylint: disable=import-outside-toplevel

    when = datetime.strptime(version["time"], "%Y-%m-%d %H:%M:%S")
    return (when - start).total_seconds() / 60


def compare_histories(args: argparse.Namespace) -> None:
    """Several agents' histories on one time axis: minutes since the first commit.

    Each `--history` is `Label=dir` (a `run` output directory). Versions that were
    not analyzed are marked as ticks along the bottom.
    """
    from datetime import datetime  # pylint: disable=import-outside-toplevel

    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    _style(plt)
    series = []
    for item in args.history:
        label, folder = item.split("=", 1)
        folder = Path(folder)
        history = json.loads((folder / "history.json").read_text())
        versions = json.loads((folder / "versions.json").read_text())
        start = min(datetime.strptime(v["time"], "%Y-%m-%d %H:%M:%S") for v in versions)
        series.append((label, history, versions, start))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    panels = (
        ("search_score", "Search score per call"),
        ("planning_events", "Searches per episode"),
    )
    for ax, (key, ylabel) in zip(axes, panels):
        for slot, (label, history, versions, start) in enumerate(series):
            color = SLOTS[slot]
            points = []
            for v in history:
                value, bound = _value(v, key)
                if v.get("status") == "ok" and value is not None:
                    points.append((_minutes(v, start), value, bound, v["index"]))
            xs = [x for x, _, _, _ in points]
            ys = [y for _, y, _, _ in points]
            # Solid between consecutive versions; dashed across versions that
            # were not analyzed (the path between them was not measured).
            for a, b in zip(points, points[1:]):
                gap = b[3] != a[3] + 1
                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color=color,
                    linewidth=1.2 if gap else 1.8,
                    linestyle=(0, (4, 3)) if gap else "-",
                    zorder=2,
                )
            ax.scatter(
                xs,
                ys,
                color=color,
                s=18 if len(points) > 20 else 34,
                zorder=3,
                label=label,
            )
            if len(points) <= 20:
                for x, y, _, index in points:
                    ax.annotate(
                        f"v{index}",
                        xy=(x, y),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=INK_2,
                    )
            analyzed = {v["index"] for v in history}
            missing = [
                _minutes(v, start) for v in versions if v["index"] not in analyzed
            ]
            if missing:
                ax.scatter(
                    missing,
                    [-0.07 - 0.05 * slot] * len(missing),
                    transform=ax.get_xaxis_transform(),
                    marker="|",
                    color=color,
                    s=60,
                    linewidths=1.2,
                    zorder=1,
                    clip_on=False,
                    label=f"{label}: commit not analyzed (tick under the axis)",
                )
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel)
    axes[0].set_title(
        "Planning across each agent's commits, by time since its first commit"
    )
    from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK_2, linewidth=1.2, linestyle=(0, (4, 3))))
    labels.append("Dashed: versions in between not analyzed")
    axes[1].legend(handles, labels, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2)
    axes[1].set_xlabel("Minutes since the agent's first commit", labelpad=16)
    fig.subplots_adjust(hspace=0.3)
    fig.text(
        0.01,
        -0.1,
        "Log scale above 1. Every analyzed version is a dot; held-out episodes "
        "(obstacles, seed): (3, 100), (15, 101), (20, 102).",
        fontsize=8.5,
        color=INK_2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(args.out)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="analyze every saved version")
    run_parser.add_argument(
        "--sandbox",
        type=Path,
        required=True,
        help="the agent's sandbox (a git repo holding approach.py)",
    )
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, default=5)
    run_parser.add_argument(
        "--minutes", type=float, default=40, help="time limit per version"
    )
    run_parser.add_argument(
        "--force", action="store_true", help="re-analyze versions that already finished"
    )
    run_parser.add_argument(
        "--versions",
        type=lambda text: {int(v) for v in text.split(",")},
        help="only these version numbers, e.g. 1,10 (default: all)",
    )
    plot_parser = commands.add_parser("plot", help="figures and findings.md")
    plot_parser.add_argument("--out", type=Path, required=True)
    plot_parser.add_argument(
        "--compare", type=Path, help="an older history.json to compare against"
    )
    compare_parser = commands.add_parser(
        "compare", help="several histories on one time axis"
    )
    compare_parser.add_argument(
        "--history",
        action="append",
        required=True,
        help="Label=run output directory (repeat for each agent)",
    )
    compare_parser.add_argument("--out", type=Path, required=True, help="PNG file")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "compare":
        compare_histories(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
