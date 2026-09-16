"""Export matched perfect seeds and render the compact timing summary table."""

# pylint: disable=line-too-long,missing-function-docstring

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.plot_final_timing_sample import _display_name, _episode_values


METHODS = ("claude", "claude_whitebox")
LABELS = {
    "claude": ("Claude Code", "blackbox"),
    "claude_whitebox": ("Claude Code + source", "whitebox"),
}


def _load(
    root: Path,
) -> tuple[
    dict[str, list[int]],
    dict[tuple[str, str, int], float],
]:
    manifest = json.loads((root / "manifest.json").read_text())
    rates = {
        (str(row["method"]), str(row["environment"]), int(row["seed"])): row.get(
            "solve_rate"
        )
        for row in manifest
        if row.get("method") in METHODS
    }
    timings: dict[tuple[str, str, int], float] = {}
    for result_path in root.glob("*/*/seed_*/results.json"):
        method, environment, seed_dir = result_path.relative_to(root).parts[:3]
        if method not in METHODS:
            continue
        seed = int(seed_dir.removeprefix("seed_"))
        result = json.loads(result_path.read_text())
        observations = [
            measured[2]
            for episode in result.get("per_episode", [])
            if (measured := _episode_values(method, episode)) is not None
        ]
        if observations:
            timings[(method, environment, seed)] = float(np.median(observations))

    matched: dict[str, list[int]] = defaultdict(list)
    pairs = {
        (environment, seed)
        for method, environment, seed in rates
        if method == "claude"
    } & {
        (environment, seed)
        for method, environment, seed in rates
        if method == "claude_whitebox"
    }
    for environment, seed in sorted(pairs):
        if not all(rates[(method, environment, seed)] == 1.0 for method in METHODS):
            continue
        if not all((method, environment, seed) in timings for method in METHODS):
            continue
        matched[environment].append(seed)
    return dict(matched), timings


def _summaries(
    matched: dict[str, list[int]],
    timings: dict[tuple[str, str, int], float],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    environment_values: dict[str, dict[str, list[float]]] = {
        method: {"percent": [], "milliseconds": []} for method in METHODS
    }
    for environment, seeds in matched.items():
        baseline = [timings[("claude", environment, seed)] for seed in seeds]
        for method in METHODS:
            method_times = [timings[(method, environment, seed)] for seed in seeds]
            ratios = [
                method_time / baseline_time * 100.0
                for method_time, baseline_time in zip(
                    method_times, baseline, strict=True
                )
            ]
            environment_values[method]["percent"].append(float(np.mean(ratios)))
            environment_values[method]["milliseconds"].append(
                float(np.mean(method_times))
            )

    return {
        method: {
            metric: (float(np.mean(values)), min(values), max(values))
            for metric, values in metrics.items()
        }
        for method, metrics in environment_values.items()
    }


def _write_seed_tables(
    csv_path: Path,
    markdown_path: Path,
    matched: dict[str, list[int]],
    timings: dict[tuple[str, str, int], float],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["environment", "seed", "claude_ms_per_action", "cc_source_ms_per_action"]
        )
        for environment, seeds in sorted(matched.items()):
            for seed in seeds:
                writer.writerow(
                    [
                        environment,
                        seed,
                        f"{timings[('claude', environment, seed)]:.6f}",
                        f"{timings[('claude_whitebox', environment, seed)]:.6f}",
                    ]
                )

    lines = [
        "| Environment | Seeds perfect for both | Count |",
        "|---|---:|---:|",
    ]
    for environment, seeds in sorted(matched.items()):
        lines.append(
            f"| {_display_name(environment)} | {', '.join(map(str, seeds))} | {len(seeds)} |"
        )
    lines.append(f"| **Total** |  | **{sum(map(len, matched.values()))}** |")
    markdown_path.write_text("\n".join(lines) + "\n")


def _render(
    output: Path,
    summaries: dict[str, dict[str, tuple[float, float, float]]],
    matched_count: int,
    environment_count: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times"],
            "mathtext.fontset": "stix",
        }
    )
    figure, axis = plt.subplots(figsize=(12.5, 4.35))
    axis.set_xlim(0, 12.5)
    axis.set_ylim(0, 4.35)
    axis.axis("off")

    axis.add_line(Line2D([0.25, 12.25], [4.08, 4.08], color="black", lw=1.8))
    axis.text(0.4, 3.56, "Method", fontsize=23, weight="bold", va="center")
    axis.text(7.05, 3.56, "Time/action (%)", fontsize=21, weight="bold", ha="center", va="center")
    axis.text(10.55, 3.56, "Time/action (ms)", fontsize=21, weight="bold", ha="center", va="center")
    axis.add_line(Line2D([0.25, 12.25], [3.20, 3.20], color="black", lw=1.2))

    for method, y in (("claude", 2.52), ("claude_whitebox", 1.28)):
        label, qualifier = LABELS[method]
        axis.text(0.4, y + 0.18, label, fontsize=22, va="center")
        axis.text(0.4, y - 0.31, qualifier, fontsize=14, va="center")
        for x, metric, decimals in (
            (7.05, "percent", 0),
            (10.55, "milliseconds", 3),
        ):
            mean, low, high = summaries[method][metric]
            axis.text(x, y + 0.18, f"{mean:,.{decimals}f}", fontsize=22, ha="center", va="center")
            axis.text(
                x,
                y - 0.31,
                f"[{low:,.{decimals}f}–{high:,.{decimals}f}]",
                fontsize=14,
                ha="center",
                va="center",
            )

    axis.add_line(Line2D([0.25, 12.25], [0.55, 0.55], color="black", lw=1.8))
    axis.text(
        0.25,
        0.12,
        f"Equal-environment mean [min–max] across {environment_count} environments; "
        f"{matched_count} matched perfect seeds.",
        fontsize=11.5,
        va="center",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-csv", type=Path, required=True)
    parser.add_argument("--seed-markdown", type=Path, required=True)
    args = parser.parse_args()

    matched, timings = _load(args.input)
    summaries = _summaries(matched, timings)
    _write_seed_tables(args.seed_csv, args.seed_markdown, matched, timings)
    _render(
        args.output,
        summaries,
        sum(map(len, matched.values())),
        len(matched),
    )
    print(f"Matched seeds: {sum(map(len, matched.values()))}")
    print(json.dumps(summaries, indent=2))
    print(f"Wrote {args.seed_csv}")
    print(f"Wrote {args.seed_markdown}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
