"""Plot mean and standard deviation from program-complexity summaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/robocode-matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANELS = [
    ("cyclomatic_total", "Total cyclomatic complexity"),
    ("cyclomatic_max_function", "Maximum function complexity"),
    ("max_nesting_depth", "Maximum nesting depth"),
    ("persistent_state_fields", "Persistent state fields"),
    ("loop_count", "Loops"),
    ("source_loc", "Lines of code"),
]


def plot_summary(summary_path: Path, output_path: Path) -> None:
    """Create a six-panel horizontal mean ± sample-SD plot."""
    frame = pd.read_csv(summary_path)
    labels = frame["environment"].str.replace("_generalized", "", regex=False)
    labels = labels + " (" + frame["approach"].astype(str) + ")"
    labels = labels + frame["access"].map(
        lambda access: " (BB)" if access == "blackbox" else " (WB)"
    )
    y = np.arange(len(frame))
    colors = frame["access"].map({"whitebox": "#3B82C4", "blackbox": "#E07A3F"})

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for axis, (metric, title) in zip(axes.flat, PANELS, strict=True):
        means = frame[f"{metric}_mean"]
        deviations = frame[f"{metric}_std"].fillna(0)
        axis.barh(
            y,
            means,
            xerr=deviations,
            color=colors,
            alpha=0.88,
            error_kw={"ecolor": "#333333", "capsize": 3, "elinewidth": 1},
        )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=9)
        axis.set_yticks(y, labels, fontsize=9)
        axis.invert_yaxis()

    figure.suptitle(
        "Static complexity of final code policies\nMean ± 1 sample SD across seeds",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "BB = blackbox; WB = whitebox. Five programs per condition.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary", nargs="?", type=Path, default=Path("program_complexity_summary.csv")
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("program_complexity_summary.png"),
    )
    args = parser.parse_args()
    plot_summary(args.summary, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    _main()
