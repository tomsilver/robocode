"""Plot absolute actions for method runs that solve all held-out episodes.

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
    "Planner": "#CC79A7",
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


def fully_solved_run_actions(
    runs: dict[tuple[str, str, int], Run],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
    """Return per-run mean actions for final policies that solve 100/100."""
    environments = sorted({key[1] for key in runs})
    successful: dict[str, dict[str, list[float]]] = {}
    coverage: dict[str, Any] = {}
    for environment in environments:
        values = {method: [] for method in METHODS}
        successful_seeds = {method: [] for method in METHODS}
        for method in METHODS:
            method_runs = sorted(
                (
                    run for (present_method, present_environment, _), run in runs.items()
                    if present_method == method and present_environment == environment
                ),
                key=lambda run: run.replicate_seed,
            )
            for run in method_runs:
                if len(run.per_episode) != 100 or not all(
                    episode.get("solved") is True for episode in run.per_episode
                ):
                    continue
                steps = [episode.get("num_steps") for episode in run.per_episode]
                if not all(isinstance(value, (int, float)) and value > 0 for value in steps):
                    continue
                values[method].append(float(np.mean(steps)))
                successful_seeds[method].append(run.replicate_seed)
        coverage[environment] = {
            "successful_seeds": successful_seeds,
        }
        if any(values.values()):
            successful[environment] = values
    return successful, coverage


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


def _mean_ci(values: list[float]) -> tuple[float, float]:
    """Return the mean and normal-approximation 95% CI half-width."""
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))


def _draw_absolute_actions(successful: dict[str, dict[str, list[float]]]) -> plt.Figure:
    environments = sorted(successful, key=lambda name: DISPLAY_NAMES.get(name, name), reverse=True)
    positions = np.arange(len(environments), dtype=float)
    offsets = dict(zip(METHODS, (-0.24, -0.08, 0.08, 0.24), strict=True))
    fig_height = max(3.2, 0.24 * len(environments) + 1.15)
    fig, axis = plt.subplots(figsize=(3.45, fig_height), constrained_layout=True)
    for method in METHODS:
        present = [index for index, environment in enumerate(environments) if successful[environment][method]]
        summaries = [_mean_ci(successful[environments[index]][method]) for index in present]
        means = [summary[0] for summary in summaries]
        intervals = [summary[1] for summary in summaries]
        axis.barh(
            np.asarray(present) + offsets[method], means, height=0.16, xerr=intervals,
            color=COLORS[method], edgecolor="white", linewidth=0.35,
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
            label=method,
        )
    axis.set_yticks(positions)
    axis.set_yticklabels([DISPLAY_NAMES.get(environment, environment) for environment in environments])
    axis.set_xlabel("Mean actions in 100%-successful runs\n(lower is better)")
    axis.set_ylabel("Environment")
    axis.set_title("Actions for 100%-successful policies", fontweight="bold", pad=6)
    axis.grid(True, axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="outside lower center")
    axis.margins(y=0.015)
    return fig


def write_outputs(
    successful: dict[str, dict[str, list[float]]],
    coverage: dict[str, Any],
    output_pdf: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    _configure_style()
    with PdfPages(output_pdf) as pdf:
        figure = _draw_absolute_actions(successful)
        pdf.savefig(figure)
        plt.close(figure)

    summary_csv = output_pdf.with_suffix(".csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["environment"]
            + [f"{method}_mean_actions" for method in METHODS]
            + [f"{method}_successful_seeds" for method in METHODS]
        )
        for environment, values in sorted(successful.items()):
            writer.writerow(
                [environment]
                + [float(np.mean(values[method])) if values[method] else "" for method in METHODS]
                + [";".join(map(str, coverage[environment]["successful_seeds"][method])) for method in METHODS]
            )
    output_pdf.with_name(f"{output_pdf.stem}-coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--blackbox", type=Path, required=True)
    parser.add_argument("--whitebox", type=Path, help="accepted for provenance; not used in this plot")
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
    successful, coverage = fully_solved_run_actions(selected)
    if not successful:
        raise RuntimeError("No 100%-successful runs found")
    write_outputs(successful, coverage, args.output)
    print(f"Wrote {args.output} with 1 page")
    print(f"Included {len(successful)} environments")


if __name__ == "__main__":
    main()
