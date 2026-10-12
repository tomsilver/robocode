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
from matplotlib.patches import Rectangle

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


def _environment_summaries(
    matched: dict[str, list[int]],
    timings: dict[tuple[str, str, int], float],
) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    summaries: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for environment, seeds in matched.items():
        baseline = [timings[("claude", environment, seed)] for seed in seeds]
        for method in METHODS:
            milliseconds = [timings[(method, environment, seed)] for seed in seeds]
            percentages = [
                method_time / baseline_time * 100.0
                for method_time, baseline_time in zip(
                    milliseconds, baseline, strict=True
                )
            ]
            for metric, values in (
                ("percent", percentages),
                ("milliseconds", milliseconds),
            ):
                summaries[(environment, method, metric)] = (
                    float(np.mean(values)),
                    min(values),
                    max(values),
                )
    return summaries


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


def _write_tex_table(
    output: Path,
    summaries: dict[str, dict[str, tuple[float, float, float]]],
    matched_count: int,
    environment_count: int,
) -> None:
    """Write the publication table with absolute milliseconds only."""
    blackbox = summaries["claude"]["milliseconds"]
    source = summaries["claude_whitebox"]["milliseconds"]
    text = rf"""% Generated by experiments/generate_cc_source_efficiency.py.
\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lc}}
\toprule
\textbf{{Method}} &
\textbf{{Time/action (ms)}} \\
\midrule
AgenticGenPlan & {blackbox[0]:.3f} \\[-2.5pt]
{{\tiny Opus 5}} & {{\tiny [{blackbox[1]:.3f}--{blackbox[2]:.3f}]}} \\
AgenticGenPlan + source & {source[0]:.3f} \\[-2.5pt]
{{\tiny Opus 5}} & {{\tiny [{source[1]:.3f}--{source[2]:.3f}]}} \\
\bottomrule
\end{{tabular}}
}}
\caption{{\textbf{{Policy-computation time per action}} on {matched_count} matched seeds
across {environment_count} environments for which both settings achieved 100\% held-out
success. Times are equal-environment means over the same matched seeds, with
[min--max] across environment-level means below. AgenticGenPlan + source is
much slower, as its programs invoke the environment source code as part of
planning at decision time.}}
\label{{tab:efficiency}}
\end{{table}}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


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


def _render_dashboard(
    output: Path,
    matched: dict[str, list[int]],
    timings: dict[tuple[str, str, int], float],
) -> None:
    environment_stats = _environment_summaries(matched, timings)
    overall = _summaries(matched, timings)
    environments = sorted(matched)
    rows = ["__overall__", *environments]
    method_width = 3.0
    sub_width = 1.45
    table_width = method_width + 6 * sub_width
    figure, axis = plt.subplots(figsize=(21, max(9.5, 0.62 * len(rows) + 3.6)))
    axis.set_xlim(0, table_width)
    axis.set_ylim(0, len(rows) + 4.4)
    axis.axis("off")
    edge = "#cbd2da"
    header = "#e7ebf0"

    def cell(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        face: str = "white",
        color: str = "#111111",
        weight: str = "normal",
        size: float = 10.0,
        align: str = "center",
    ) -> None:
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.7,
            )
        )
        axis.text(
            x + (0.14 if align == "left" else width / 2),
            y + height / 2,
            text,
            ha=align,
            va="center",
            fontsize=size,
            color=color,
            fontweight=weight,
            linespacing=1.25,
        )

    top = len(rows) + 2.15
    cell(0, top - 1, method_width, 2, "Environment", header, weight="bold", align="left")
    for method_index, method in enumerate(METHODS):
        x = method_width + method_index * 3 * sub_width
        title = "Claude Code" if method == "claude" else "CC + source"
        cell(x, top, 3 * sub_width, 1, title, header, weight="bold", size=11)
        for sub_index, title in enumerate(
            ("Matched seeds", "Compute/action (%)", "Time/action (ms)")
        ):
            cell(
                x + sub_index * sub_width,
                top - 1,
                sub_width,
                1,
                title,
                header,
                weight="bold",
                size=8.5,
            )

    for row_index, environment in enumerate(rows):
        y = top - 2 - row_index
        is_overall = environment == "__overall__"
        name = "Overall average" if is_overall else _display_name(environment)
        cell(
            0,
            y,
            method_width,
            1,
            name,
            "#eef1f5" if is_overall else "white",
            weight="bold" if is_overall else "normal",
            align="left",
        )
        for method_index, method in enumerate(METHODS):
            x = method_width + method_index * 3 * sub_width
            if is_overall:
                matched_count = sum(map(len, matched.values()))
                denominator = 5 * len(matched)
                percent = overall[method]["percent"]
                milliseconds = overall[method]["milliseconds"]
            else:
                matched_count = len(matched[environment])
                denominator = 5
                percent = environment_stats[(environment, method, "percent")]
                milliseconds = environment_stats[
                    (environment, method, "milliseconds")
                ]
            seed_face = (
                "white"
                if method == "claude"
                else plt.get_cmap("RdYlGn")(matched_count / denominator, 0.22)
            )
            cell(
                x,
                y,
                sub_width,
                1,
                f"{matched_count}/{denominator}",
                seed_face,
                weight="bold" if is_overall else "normal",
            )
            cell(
                x + sub_width,
                y,
                sub_width,
                1,
                f"{percent[0]:,.0f}\n[{percent[1]:,.0f}–{percent[2]:,.0f}]",
                weight="bold" if is_overall else "normal",
                size=8.8,
            )
            cell(
                x + 2 * sub_width,
                y,
                sub_width,
                1,
                f"{milliseconds[0]:.3f}\n"
                f"[{milliseconds[1]:.3f}–{milliseconds[2]:.3f}]",
                weight="bold" if is_overall else "normal",
                size=8.8,
            )

    figure.suptitle(
        "Policy computation — matched seeds with 100% held-out success",
        fontsize=21,
        weight="bold",
        y=0.985,
    )
    axis.text(
        table_width / 2,
        top + 1.35,
        "Mean [min–max]; percentages are relative to Claude Code blackbox "
        "within each environment",
        ha="center",
        va="center",
        fontsize=12,
        color="#52606d",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dashboard-output", type=Path)
    parser.add_argument("--seed-csv", type=Path, required=True)
    parser.add_argument("--seed-markdown", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    args = parser.parse_args()

    matched, timings = _load(args.input)
    summaries = _summaries(matched, timings)
    _write_seed_tables(args.seed_csv, args.seed_markdown, matched, timings)
    if args.tex_output:
        _write_tex_table(
            args.tex_output,
            summaries,
            sum(map(len, matched.values())),
            len(matched),
        )
    _render(
        args.output,
        summaries,
        sum(map(len, matched.values())),
        len(matched),
    )
    if args.dashboard_output:
        _render_dashboard(args.dashboard_output, matched, timings)
    print(f"Matched seeds: {sum(map(len, matched.values()))}")
    print(json.dumps(summaries, indent=2))
    print(f"Wrote {args.seed_csv}")
    print(f"Wrote {args.seed_markdown}")
    print(f"Wrote {args.output}")
    if args.dashboard_output:
        print(f"Wrote {args.dashboard_output}")
    if args.tex_output:
        print(f"Wrote {args.tex_output}")


if __name__ == "__main__":
    main()
