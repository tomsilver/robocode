"""Plot policy computation, environment, and computation-per-action timings."""

# pylint: disable=line-too-long,missing-function-docstring

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


METHODS = ("claude", "claude_whitebox", "codex", "genplan", "planner")
LABELS = {
    "claude": "Claude Code", "claude_whitebox": "CC + source",
    "codex": "Codex", "genplan": "GenPlan", "planner": "Planner",
}
COLORS = {
    "claude": "#D55E00", "claude_whitebox": "#E69F00", "codex": "#0072B2",
    "genplan": "#009E73", "planner": "#CC79A7",
}


def _display_name(environment: str) -> str:
    if environment == "__overall__":
        return "Overall average"
    special = {
        "pr2blocked_generalized": "PDDLStream Blocked",
        "pr2packed_generalized": "PDDLStream Packing",
        "rovers_generalized": "PDDLStream Rovers",
        "sortclutteredblocks3d_generalized": "SortClutteredBlocks",
        "sweepintodrawer3d": "SweepIntoDrawer",
        "sweepsimple3d_generalized": "SweepSimple",
    }
    if environment in special:
        return special[environment]
    name = environment.removesuffix("_generalized").replace("_", " ")
    name = re.sub(r"(2d|3d)$", lambda match: " " + match.group(1).upper(), name)
    return name.title().replace(" 2D", " 2D").replace(" 3D", " 3D")


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Times"],
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 15,
        "legend.fontsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _episode_values(method: str, episode: dict) -> tuple[float, float, float] | None:
    steps = episode.get("num_steps")
    if not isinstance(steps, (int, float)) or steps <= 0:
        return None
    if method == "planner":
        computation: object = float(episode.get("planning_time") or 0) + float(
            episode.get("execution_time") or 0
        )
        # PDDLStream planner logs contain planning time but no separate
        # execution/environment timing; that is sufficient for computation/action.
        environment: object = float(episode.get("env_step_time") or 0)
    else:
        computation = episode.get("policy_time_s")
        environment = episode.get("env_time_s")
    if not isinstance(computation, (int, float)) or not isinstance(environment, (int, float)):
        return None
    return float(computation), float(environment), 1000.0 * float(computation) / float(steps)


GROUPS = (
    ("all", "All runs", lambda rate: True),
    ("exactly_100", "Runs with 100% success", lambda rate: rate == 1.0),
    ("below_100", "Runs below 100% success", lambda rate: rate < 1.0),
)


def load(root: Path) -> tuple[
    dict[str, dict[str, dict[str, list[float]]]],
    dict[str, dict[str, dict[str, list[float]]]],
]:
    manifest = json.loads((root / "manifest.json").read_text())
    solve_rates = {
        (row["method"], row["environment"], int(row["seed"])): row.get("solve_rate")
        for row in manifest
    }
    per_run: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for result_path in root.glob("*/*/seed_*/results.json"):
        parts = result_path.relative_to(root).parts
        method, environment, seed = parts[0], parts[1], int(parts[2].removeprefix("seed_"))
        if method not in METHODS:
            continue
        result = json.loads(result_path.read_text())
        observations = []
        for episode in result.get("per_episode", []):
            measured = _episode_values(method, episode)
            if measured is not None:
                observations.append(measured[2])
        if observations:
            per_run[environment][method][seed] = float(np.median(observations))

    grouped: dict[str, dict[str, dict[str, list[float]]]] = {
        key: defaultdict(lambda: defaultdict(list)) for key, _, _ in GROUPS
    }
    scores: dict[str, dict[str, dict[str, list[float]]]] = {
        key: defaultdict(lambda: defaultdict(list)) for key, _, _ in GROUPS
    }
    for group_key, _, predicate in GROUPS:
        for environment, methods in per_run.items():
            claude = methods.get("claude", {})
            for method, seed_values in methods.items():
                for seed, value in seed_values.items():
                    rate = solve_rates.get((method, environment, seed))
                    reference = claude.get(seed)
                    if rate is None or reference is None or reference <= 0 or not predicate(rate):
                        continue
                    grouped[group_key][environment][method].append(value / reference)
                    scores[group_key][environment][method].append(float(rate))
    return grouped, scores


def _summary(observations: list[float]) -> tuple[float, float]:
    array = np.asarray(observations)
    mean = float(np.mean(array))
    ci = 0.0 if len(array) < 2 else float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))
    return mean, min(ci, mean * 0.99)


def _add_overall_row(
    data: dict[str, dict[str, list[float]]],
    scores: dict[str, dict[str, list[float]]],
) -> None:
    """Add an equal-environment-weighted summary row in place."""
    for method in METHODS:
        environment_means = [
            float(np.mean(methods[method]))
            for environment, methods in data.items()
            if environment != "__overall__" and methods[method]
        ]
        score_means = [
            float(np.mean(methods[method]))
            for environment, methods in scores.items()
            if environment != "__overall__" and methods[method]
        ]
        data["__overall__"][method].extend(environment_means)
        scores["__overall__"][method].extend(score_means)


def draw(
    data: dict[str, dict[str, list[float]]],
    title: str,
    scores: dict[str, dict[str, list[float]]] | None = None,
) -> plt.Figure:
    environments = sorted(data, reverse=True)
    positions = np.arange(len(environments))
    offsets = dict(zip(METHODS, (-0.28, -0.14, 0.0, 0.14, 0.28), strict=True))
    display_limit = 12.5
    figure, axis = plt.subplots(
        figsize=(14.0, max(9.0, 0.48 * len(environments) + 2.6)),
        constrained_layout=True,
    )
    for method in METHODS:
        present = [i for i, environment in enumerate(environments) if data[environment][method]]
        summaries = [_summary(data[environments[i]][method]) for i in present]
        axis.barh(
            np.asarray(present) + offsets[method], [mean for mean, _ in summaries],
            height=0.135, xerr=[ci for _, ci in summaries],
            color=COLORS[method], edgecolor="white", linewidth=0.35,
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
            label=LABELS[method],
        )
        for position, (mean, ci) in zip(present, summaries, strict=True):
            environment = environments[position]
            if mean > display_limit:
                axis.text(
                    display_limit * 0.985, position + offsets[method], f"{mean:.0f}×→",
                    ha="right", va="center", fontsize=9, color=COLORS[method],
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85},
                )
    if scores is not None:
        score_x = {
            "claude": -3.9, "claude_whitebox": -3.0, "codex": -2.1,
            "genplan": -1.2, "planner": -0.35,
        }
        score_headers = {
            "claude": "CC", "claude_whitebox": "CC+src", "codex": "Codex",
            "genplan": "GP", "planner": "Plan",
        }
        for position, environment in enumerate(environments):
            for method in METHODS:
                if scores[environment][method]:
                    score = 100.0 * float(np.mean(scores[environment][method]))
                    axis.text(
                        score_x[method], position, f"{score:.0f}%",
                        ha="center", va="center", fontsize=9, color=COLORS[method],
                    )
        for method in METHODS:
            axis.text(
                score_x[method], len(environments) - 0.48, score_headers[method],
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=COLORS[method],
            )
    axis.set_yticks(positions)
    axis.set_yticklabels([_display_name(environment) for environment in environments])
    axis.axvline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlim(-4.45 if scores is not None else 0, display_limit)
    axis.set_xlabel("Computation / action relative to Claude Code (×)")
    axis.set_ylabel("Environment")
    axis.set_title(title, fontweight="bold")
    if "__overall__" in environments:
        overall_position = environments.index("__overall__")
        axis.axhline(overall_position - 0.5, color="#777777", linewidth=0.8)
    axis.grid(True, axis="x", which="both", color="#DDDDDD", linewidth=0.4)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=5, loc="outside lower center")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values, scores = load(args.input)
    _style()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        for group_key, title, _ in GROUPS:
            page_title = title
            page_scores = None
            if group_key == "all":
                page_title = "All runs"
                page_scores = scores[group_key]
                _add_overall_row(values[group_key], page_scores)
            figure = draw(values[group_key], page_title, page_scores)
            pdf.savefig(figure)
            plt.close(figure)
    print(f"Wrote {args.output} with {len(GROUPS)} pages")


if __name__ == "__main__":
    main()
