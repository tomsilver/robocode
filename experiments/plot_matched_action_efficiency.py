"""Plot action efficiency on held-out episodes solved by every main method.

The input archives are the three top-level ZIP files exported for the paper.
They contain nested ZIP files. No extraction is required: result JSON files are
read directly from the nested archives.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


METHODS = ("Claude Code", "Codex", "GenPlan", "Planner")
COLORS = {
    "Claude Code": "#D55E00",
    "Codex": "#0072B2",
    "GenPlan": "#009E73",
    "Planner": "#555555",
}
DISPLAY_NAMES = {
    "balancebeam3d": "BalanceBeam",
    "clutteredretrieval2d_generalized": "ClutteredRetrieval 2D",
    "clutteredstorage2d_generalized": "ClutteredStorage 2D",
    "constrainedcupboard3d_generalized": "ConstrainedCupboard",
    "dynamicshelf3d_generalized": "DynamicShelf",
    "dynamo3d_generalized": "Dynamo",
    "dynobstruction2d_generalized": "Dynamic Obstruction 2D",
    "dynpushpullhook2d_generalized": "Dynamic PushPullHook 2D",
    "dynpusht2d": "Dynamic PushT 2D",
    "dynscooppour2d_generalized": "Dynamic ScoopPour 2D",
    "motion2d_generalized": "Motion 2D",
    "obstruction2d_generalized": "Obstruction 2D",
    "obstruction3d_generalized": "Obstruction 3D",
    "packing3d_generalized": "Packing 3D",
    "pr2blocked_generalized": "PDDLStream Blocked",
    "pr2packed_generalized": "PDDLStream Packing",
    "pushpullhook2d": "PushPullHook 2D",
    "rearrange3d": "Rearrange",
    "rovers_generalized": "PDDLStream Rovers",
    "scooppour3d_generalized": "ScoopPour 3D",
    "shelf3d_generalized": "Shelf 3D",
    "sortclutteredblocks3d_generalized": "SortClutteredBlocks",
    "stickbutton2d_generalized": "StickButton 2D",
    "sweepintodrawer3d": "SweepIntoDrawer",
    "sweepsimple3d_generalized": "SweepSimple",
    "table3d_generalized": "Table 3D",
    "tossing3d_generalized": "Tossing",
    "transport3d_generalized": "Transport 3D",
}


@dataclass(frozen=True)
class Run:
    """One method/environment/synthesis-seed evaluation result."""

    method: str
    environment: str
    replicate_seed: int
    per_episode: tuple[dict[str, Any], ...]
    source: str
    rank: tuple[int, str]


def _method_for_member(archive_label: str, member: str) -> str | None:
    normalized = member.replace("\\", "/")
    if "/__MACOSX/" in f"/{normalized}" or Path(normalized).name.startswith("._"):
        return None
    if archive_label == "baselines":
        if normalized.startswith("Baselines/Planner/"):
            return "Planner"
        if normalized.startswith("Baselines/GenPlan/"):
            return "GenPlan"
    if archive_label == "blackbox":
        if normalized.startswith("Blackbox/Codex GPT5.6 Sol/"):
            return "Codex"
        if normalized.startswith("Blackbox/Claude Code Opus 5/"):
            return "Claude Code"
    return None


def _environment(result: dict[str, Any], member: str) -> str | None:
    experiment_id = str(result.get("experiment_id", ""))
    if "__" in experiment_id:
        return experiment_id.split("__", maxsplit=1)[0]
    match = re.search(r"([^/]+)__(?:agentic|llm_genplan|bilevel_planning)", member)
    return match.group(1) if match else None


def _timestamp(member: str) -> str:
    matches = re.findall(r"20\d\d-\d\d-\d\d[_T-]\d\d[-:]\d\d[-:]\d\d", member)
    return matches[-1] if matches else ""


def _iter_results(inner: zipfile.ZipFile) -> Iterable[tuple[str, dict[str, Any]]]:
    for name in inner.namelist():
        if name.endswith("/results.json") and "__MACOSX" not in name:
            try:
                yield name, json.loads(inner.read(name))
            except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
                continue


def load_runs(archive: Path, archive_label: str, temp_dir: Path) -> list[Run]:
    """Read all supported results from nested ZIP members."""
    runs: list[Run] = []
    with zipfile.ZipFile(archive) as outer:
        for member in outer.namelist():
            method = _method_for_member(archive_label, member)
            if method is None or not member.endswith(".zip"):
                continue
            with tempfile.SpooledTemporaryFile(max_size=32 << 20, dir=temp_dir) as stream:
                with outer.open(member) as source:
                    shutil.copyfileobj(source, stream, length=8 << 20)
                stream.seek(0)
                try:
                    with zipfile.ZipFile(stream) as inner:
                        for result_member, result in _iter_results(inner):
                            episodes = result.get("per_episode")
                            environment = _environment(result, result_member)
                            seed = result.get("replicate_seed")
                            if (
                                environment is None
                                or not isinstance(seed, int)
                                or not isinstance(episodes, list)
                                or not episodes
                                or result.get("eval_complete") is False
                            ):
                                continue
                            outdated = int("outdated" not in member.lower())
                            runs.append(
                                Run(
                                    method,
                                    environment,
                                    seed,
                                    tuple(episodes),
                                    f"{archive.name}!/{member}!/{result_member}",
                                    (outdated, _timestamp(result_member)),
                                )
                            )
                except zipfile.BadZipFile:
                    continue
    return runs


def select_latest(runs: Iterable[Run]) -> dict[tuple[str, str, int], Run]:
    """Select the newest non-outdated result for each method/environment/seed."""
    selected: dict[tuple[str, str, int], Run] = {}
    for run in runs:
        key = (run.method, run.environment, run.replicate_seed)
        if key not in selected or run.rank > selected[key].rank:
            selected[key] = run
    return selected


def match_episodes(
    runs: dict[tuple[str, str, int], Run],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, set[int]], dict[str, Any]]:
    """Return planner-normalized steps for episodes solved by all methods."""
    environments = sorted({key[1] for key in runs})
    matched: dict[str, dict[str, list[float]]] = {}
    shared_seeds: dict[str, set[int]] = {}
    coverage: dict[str, Any] = {}
    for environment in environments:
        seeds_by_method = {
            method: {
                seed
                for present_method, present_environment, seed in runs
                if present_method == method and present_environment == environment
            }
            for method in METHODS
        }
        common_seeds = set.intersection(*(seeds_by_method[method] for method in METHODS))
        values = {method: [] for method in METHODS}
        for seed in sorted(common_seeds):
            method_runs = {method: runs[(method, environment, seed)] for method in METHODS}
            episode_count = min(len(run.per_episode) for run in method_runs.values())
            for episode_index in range(episode_count):
                episodes = {
                    method: method_runs[method].per_episode[episode_index]
                    for method in METHODS
                }
                if not all(episode.get("solved") is True for episode in episodes.values()):
                    continue
                steps = {method: episodes[method].get("num_steps") for method in METHODS}
                if not all(isinstance(value, (int, float)) and value > 0 for value in steps.values()):
                    continue
                planner_steps = float(steps["Planner"])
                for method in METHODS:
                    values[method].append(float(steps[method]) / planner_steps)
        coverage[environment] = {
            "available_seeds": {method: sorted(seeds_by_method[method]) for method in METHODS},
            "shared_seeds": sorted(common_seeds),
            "matched_episodes": len(values["Planner"]),
        }
        if values["Planner"]:
            matched[environment] = values
            shared_seeds[environment] = common_seeds
    return matched, shared_seeds, coverage


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _ecdf(values: Iterable[float], grid: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(tuple(values), dtype=float))
    return np.searchsorted(sorted_values, grid, side="right") / len(sorted_values)


def _draw_page(
    values_by_method: dict[str, list[float]],
    title: str,
    subtitle: str,
    environment_values: dict[str, dict[str, list[float]]] | None = None,
) -> plt.Figure:
    all_values = [value for values in values_by_method.values() for value in values]
    lower = max(0.05, min(all_values) * 0.8)
    upper = max(1.25, max(all_values) * 1.2)
    grid = np.geomspace(lower, upper, 600)
    fig, axis = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    if environment_values is None:
        curves = {method: _ecdf(values_by_method[method], grid) for method in METHODS}
    else:
        curves = {
            method: np.mean(
                [_ecdf(values[method], grid) for values in environment_values.values()],
                axis=0,
            )
            for method in METHODS
        }
    for method in METHODS[:-1]:
        axis.plot(grid, curves[method], color=COLORS[method], label=method)
    axis.axvline(1.0, color=COLORS["Planner"], linestyle="--", linewidth=1.3, label="Planner (1×)")
    axis.set_xscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("Actions relative to planner (lower is better)")
    axis.set_ylabel("Fraction of matched\nsolved executions")
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    axis.grid(True, which="minor", axis="x", color="#EEEEEE", linewidth=0.35, alpha=0.65)
    axis.set_title(title, fontweight="bold", pad=12)
    axis.text(0.5, 1.015, subtitle, transform=axis.transAxes, ha="center", va="bottom", fontsize=7)
    axis.legend(loc="lower right", frameon=False)
    return fig


def _draw_no_data_page(environment: str, details: dict[str, Any]) -> plt.Figure:
    """Draw an explicit page for an environment with no four-way matched episode."""
    fig, axis = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    axis.axis("off")
    axis.set_title(DISPLAY_NAMES.get(environment, environment), fontweight="bold", pad=8)
    missing = [method for method in METHODS if not details["available_seeds"][method]]
    if missing:
        message = "No four-method comparison available"
        explanation = "Missing results: " + ", ".join(missing)
    elif not details["shared_seeds"]:
        message = "No shared synthesis seeds across all four methods"
        explanation = "Available seed coverage differs between methods."
    else:
        message = "No held-out episode was solved by all four methods"
        explanation = (
            f"All four methods share {len(details['shared_seeds'])} synthesis seeds, "
            "but their solved-instance intersection is empty."
        )
    axis.text(0.5, 0.56, message, ha="center", va="center", fontsize=9, fontweight="bold", wrap=True)
    axis.text(0.5, 0.40, explanation, ha="center", va="center", fontsize=7, color="#555555", wrap=True)
    return fig


def write_outputs(
    matched: dict[str, dict[str, list[float]]],
    shared_seeds: dict[str, set[int]],
    coverage: dict[str, Any],
    output_pdf: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    _configure_style()
    pooled = {
        method: [
            value
            for environment_values in matched.values()
            for value in environment_values[method]
        ]
        for method in METHODS
    }
    total_matched = len(pooled["Planner"])
    with PdfPages(output_pdf) as pdf:
        figure = _draw_page(
            pooled,
            "Matched-instance action efficiency",
            f"Equal-weight mean across {len(matched)} environments · {total_matched} matched executions",
            environment_values=matched,
        )
        pdf.savefig(figure)
        plt.close(figure)
        for environment in sorted(coverage, key=lambda name: DISPLAY_NAMES.get(name, name)):
            if environment in matched:
                values = matched[environment]
                seed_count = len(shared_seeds[environment])
                figure = _draw_page(
                    values,
                    DISPLAY_NAMES.get(environment, environment),
                    f"{len(values['Planner'])} matched executions · {seed_count} shared synthesis seeds",
                )
            else:
                figure = _draw_no_data_page(environment, coverage[environment])
            pdf.savefig(figure)
            plt.close(figure)

    summary_csv = output_pdf.with_suffix(".csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["environment", "matched_executions", "shared_seeds"]
            + [f"{method}_median_relative_actions" for method in METHODS]
        )
        for environment, values in sorted(matched.items()):
            writer.writerow(
                [environment, len(values["Planner"]), ";".join(map(str, sorted(shared_seeds[environment])))]
                + [float(np.median(values[method])) for method in METHODS]
            )
    output_pdf.with_name(f"{output_pdf.stem}-coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--blackbox", type=Path, required=True)
    parser.add_argument("--whitebox", type=Path, help="accepted for provenance; not used in the main-method match")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    temp_dir = args.temp_dir or args.output.parent / ".nested-zip-cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.baselines, "baselines", temp_dir)
    runs.extend(load_runs(args.blackbox, "blackbox", temp_dir))
    selected = select_latest(runs)
    matched, shared_seeds, coverage = match_episodes(selected)
    if not matched:
        raise RuntimeError("No episodes were solved by all four methods")
    write_outputs(matched, shared_seeds, coverage, args.output)
    print(f"Wrote {args.output} with {1 + len(coverage)} pages")
    print(f"Matched {sum(len(values['Planner']) for values in matched.values())} executions")


if __name__ == "__main__":
    main()
