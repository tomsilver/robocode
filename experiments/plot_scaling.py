"""Plot solve-rate-vs-object-count scaling curves from experiment results.

Reads ``results.json`` files (with ``.hydra`` sidecars) from one or more run
directories and renders, per environment family:

* solve rate vs object count, one line per run (two runs of the same approach are
  kept separate), with **honest denominators** -- every scheduled
  episode at a count counts, crashes and unattempted included; and
* the planner's mean planning time and plan-found rate vs object count, which is the
  degradation the generalized program is meant to amortize away.

Usage:

    python experiments/plot_scaling.py <run dirs...> [--out scaling.png]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Render to files without a display.
plt.switch_backend("Agg")


def _run_label(job_dir: Path) -> tuple[str, str] | None:
    """Return ``(environment, approach)`` for a run, or None if unavailable."""
    config_path = job_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return None
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    environment = approach = None
    overrides_path = job_dir / ".hydra" / "overrides.yaml"
    if overrides_path.exists():
        for override in OmegaConf.load(overrides_path):
            key, _, val = str(override).partition("=")
            if key == "environment":
                environment = val
            elif key == "approach":
                approach = val
    if environment is None:
        environment = cfg["environment"]["_target_"].rsplit(".", 1)[-1]
    if approach is None:
        approach = cfg["approach"]["_target_"].rsplit(".", 1)[-1]
    return environment, approach


def collect(
    dirs: list[Path],
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Map environment -> approach -> source-dir name -> per-episode dicts.

    Keeping the source (the CLI dir a run was found under) lets two runs of the
    same approach that differ only in config -- e.g. one with primitives and one
    without -- stay separate instead of pooling into a single blended line.
    """
    data: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for search_dir in dirs:
        for results_path in sorted(search_dir.rglob("results.json")):
            job_dir = results_path.parent
            label = _run_label(job_dir)
            if label is None:
                continue
            environment, approach = label
            results = json.loads(results_path.read_text())
            episodes = [
                e
                for e in results.get("per_episode", [])
                if e.get("object_count") is not None
            ]
            if episodes:
                data[environment][approach][search_dir.name].extend(episodes)
    return data


def series(
    by_approach: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    """Flatten approach -> {source: episodes} into the label -> episodes to plot.

    A single source keeps the plain approach name; several sources (seeds of one run
    pool under one source, so this means genuinely distinct runs) are split and suffixed
    with the source dir so each stays its own line.
    """
    labelled: dict[str, list[dict[str, Any]]] = {}
    for approach, by_source in by_approach.items():
        if len(by_source) == 1:
            (episodes,) = by_source.values()
            labelled[approach] = episodes
        else:
            for source, episodes in by_source.items():
                labelled[f"{approach} ({source})"] = episodes
    return labelled


def by_count(
    episodes: list[dict[str, Any]], key: str, reducer: str = "mean"
) -> tuple[list[int], list[float]]:
    """Aggregate ``key`` per object count.

    reducer: 'mean' or 'rate' (of truthy).
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for episode in episodes:
        count = episode["object_count"]
        value = episode.get(key)
        if value is None:
            continue
        buckets[count].append(float(bool(value)) if reducer == "rate" else float(value))
    counts = sorted(buckets)
    values = [float(np.mean(buckets[c])) for c in counts]
    return counts, values


def plot_environment(
    environment: str, approaches: dict[str, list[dict[str, Any]]], out: Path
) -> None:
    """Render the two-panel solve-rate and planner-cost figure for one environment."""
    fig, (ax_solve, ax_plan) = plt.subplots(1, 2, figsize=(12, 4.5))

    for approach, episodes in sorted(approaches.items()):
        counts, solve = by_count(episodes, "solved", reducer="rate")
        if counts:
            ax_solve.plot(counts, solve, marker="o", label=approach)

    ax_solve.set_title(f"{environment}: solve rate vs object count")
    ax_solve.set_xlabel("object count")
    ax_solve.set_ylabel("solve rate (all scheduled episodes)")
    ax_solve.set_ylim(-0.02, 1.02)
    ax_solve.grid(True, alpha=0.3)
    ax_solve.legend()

    # Planner degradation: only approaches that record planning_time contribute.
    plotted = False
    for approach, episodes in sorted(approaches.items()):
        counts, plan_time = by_count(episodes, "planning_time")
        if not counts:
            continue
        plotted = True
        ax_plan.plot(
            counts, plan_time, marker="s", label=f"{approach} planning time (s)"
        )
        pf_counts, plan_found = by_count(episodes, "plan_found", reducer="rate")
        ax_twin = ax_plan.twinx()
        ax_twin.plot(
            pf_counts,
            plan_found,
            marker="^",
            linestyle="--",
            color="tab:red",
            label=f"{approach} plan-found rate",
        )
        ax_twin.set_ylabel("plan-found rate")
        ax_twin.set_ylim(-0.02, 1.02)

    ax_plan.set_title(f"{environment}: planner cost vs object count")
    ax_plan.set_xlabel("object count")
    ax_plan.set_ylabel("mean planning time (s)")
    ax_plan.grid(True, alpha=0.3)
    if plotted:
        ax_plan.legend(loc="upper left")
    else:
        ax_plan.text(
            0.5,
            0.5,
            "no planner runs",
            ha="center",
            va="center",
            transform=ax_plan.transAxes,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path, help="run dirs to scan")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scaling.png"),
        help="output PNG (one per environment: <stem>_<env><suffix>)",
    )
    args = parser.parse_args()
    data = collect(args.dirs)
    if not data:
        print("No variable-count results found (per_episode needs object_count).")
        return
    for environment, by_approach in sorted(data.items()):
        out = args.out.with_name(f"{args.out.stem}_{environment}{args.out.suffix}")
        plot_environment(environment, series(by_approach), out)


if __name__ == "__main__":
    _main()
