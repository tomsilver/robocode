"""Build per-environment policy, trajectory, and environment-complexity figures."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import yaml

METRICS = [
    ("source_loc", "Lines of code"),
    ("logical_loc", "Logical LOC (comments excluded)"),
    ("ast_nodes", "AST nodes"),
    ("cyclomatic_total", "Cyclomatic complexity"),
    ("cyclomatic_max_function", "Maximum function complexity"),
    ("max_nesting_depth", "Maximum nesting depth"),
    ("persistent_state_fields", "Persistent state fields"),
]

METHOD_COLORS = {
    "agentic/blackbox/claude-opus-5": "#17658c",
    "agentic/blackbox/gpt-5.6-sol": "#b45f24",
    "agentic/whitebox/claude-opus-5": "#3a8f6b",
    "llm_genplan": "#c23b43",
}

MEASUREMENTS_MD = """# Static-complexity measurements

The analysis parses Python source into an abstract syntax tree (AST); it does not
import or execute the generated programs.

## Program and environment metrics

### Lines of code

Number of physical source lines, including blank and comment-only lines. This
measures overall source-file size and is intentionally not logical LOC.

### Logical lines of code (comments excluded)

Number of logical Python statements, counted from tokenizer `NEWLINE` tokens.
Blank lines, comment-only lines, and physical continuation lines are excluded.
Docstrings remain executable Python statements and are therefore counted.

### AST nodes

Total nodes in Python's parsed abstract syntax tree. This measures syntactic
structure independently of formatting.

### Cyclomatic complexity

McCabe complexity of the entire module: one plus decision points from branches,
loops, exception handlers, conditional expressions, Boolean terms,
comprehensions, and `match` cases. This estimates the number of independent
control-flow paths.

### Maximum function complexity

The largest cyclomatic-complexity value among the functions in the file. This
captures complexity concentrated in the single most complicated function.

### Maximum nesting depth

Deepest nesting of `if`, loops, `with`, `try`, and `match` constructs. Function
and class definitions do not themselves increase this depth.

### Persistent state fields

Number of distinct attributes assigned through `self.<name>`. This measures the
size of the policy's explicit persistent object state.

All seven measurements are counts. Larger values indicate more source or structural
complexity, but none is by itself a measure of correctness or software quality.

### Complexity evolution

Each saved synthesis history is sampled at the nearest available revision to 0%,
10%, ..., 100% progress. Replicate seeds are averaged within each environment
first; the plotted line then averages environments equally. Shaded bands are 95%
normal confidence intervals across environments (mean plus or minus 1.96 standard
errors). For every averaged figure, the environment set is restricted after
filtering to the intersection represented by all four evolving methods. The exact
included and excluded environments are printed above each figure. The plotted
endpoint is the final saved implementation; solve rate is not included in the
trajectory panels because it is reported in the paper's table.

Exclusions are reported separately for environments that were not run on all four
methods and environments that have full method coverage but do not have a
qualifying result for every method in the selected performance subset.

Per-environment figures instead average the available replicate seeds and show 95%
normal confidence intervals across those seeds. Each page reports the actual seed
count per method; conditional pages can contain fewer than five qualifying seeds.

Three versions use the same calculation: **all runs**, runs whose final held-out
solve rate is exactly 1.0 (**100% successful**), and runs whose final held-out solve
rate is below 1.0 (**not 100% successful**). Runs without completed evaluation
results appear only in the all-runs figure; they are excluded from both conditional
figures. The downloadable trajectory-subset counts report the number of represented
runs and environments for each method.
"""


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def _save_figure(figure: Any, path: Path, dpi: int = 300) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _load_analyzer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("complexity_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_shots(repo: Path, analyzer: Any) -> pd.DataFrame:
    rows = []
    for source in repo.glob(
        "first_attempt_eval/*/replicate_*/loaded_policy/sandbox/approach.py"
    ):
        run = source.parents[3]
        result_path = source.parents[2] / "results.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        rows.append(
            {
                "source": str(source),
                "approach": "llm_genplan_one_shot",
                "environment": run.name.split("__", 1)[0],
                "access": "whitebox",
                "backend": "claude-opus-5",
                "replicate_seed": int(
                    source.parents[2].name.removeprefix("replicate_")
                ),
                "has_results": result_path.exists(),
                "result_solve_rate": result.get("solve_rate"),
                **analyzer._source_metrics(source.read_text()),
            }
        )
    return pd.DataFrame(rows)


def _genplan_trajectories(repo: Path, analyzer: Any) -> pd.DataFrame:
    rows = []
    for sandbox in repo.glob("multirun/*__llm_genplan__*/*/replicate_*/sandbox"):
        candidates = sorted(
            sandbox.glob("impl*_candidate.py"),
            key=lambda p: int(re.search(r"impl(\d+)", p.name).group(1)),
        )
        if not candidates:
            continue
        for index, source in enumerate(candidates):
            score_path = source.with_name(source.name.replace("_candidate.py", "_score.json"))
            score_data = json.loads(score_path.read_text()) if score_path.exists() else {}
            score = score_data.get("score")
            rows.append(
                {
                    "environment": sandbox.parents[2].name.split("__", 1)[0],
                    "method": "llm_genplan",
                    "replicate_seed": sandbox.parent.name.removeprefix("replicate_"),
                    "revision_progress": index / max(1, len(candidates) - 1),
                    "training_solve_rate": (
                        score["num_solved"] / score["num_total"] if score else 0.0
                    ),
                    **analyzer._source_metrics(source.read_text()),
                }
            )
    return pd.DataFrame(rows)


def _agentic_trajectories(repo: Path) -> pd.DataFrame:
    path = repo / "program_complexity_history.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["method"] = "agentic/" + frame["access"] + "/" + frame["backend"]
    return frame


def _module_path(module: str, roots: list[Path]) -> Path | None:
    relative = Path(*module.split("."))
    for root in roots:
        for candidate in (
            root / relative.with_suffix(".py"),
            root / relative / "__init__.py",
        ):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _local_closure(starts: set[str], roots: list[Path]) -> list[Path]:
    pending = list(starts)
    seen_modules: set[str] = set()
    files: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in seen_modules:
            continue
        seen_modules.add(module)
        path = _module_path(module, roots)
        if path is None:
            continue
        files.add(path)
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        package = module.rsplit(".", 1)[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 0:
                    pending.append(node.module)
                else:
                    prefix = package.split(".")[: -node.level + 1]
                    pending.append(".".join([*prefix, node.module]))
    return sorted(files)


def _config_modules(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "_target_" and isinstance(child, str):
                found.add(child.rsplit(".", 1)[0])
            found |= _config_modules(child)
    elif isinstance(value, list):
        for child in value:
            found |= _config_modules(child)
    return found


def _aggregate_sources(paths: list[Path], analyzer: Any) -> dict[str, float]:
    rows = [
        analyzer._source_metrics(path.read_text(errors="replace")) for path in paths
    ]
    output: dict[str, float] = {"source_file_count": float(len(paths))}
    for metric, _ in METRICS:
        values = [float(row.get(metric, 0) or 0) for row in rows]
        output[metric] = (
            max(values, default=0) if metric == "max_nesting_depth" else sum(values)
        )
    return output


def _environment_complexity(
    repo: Path, environments: set[str], analyzer: Any
) -> pd.DataFrame:
    roots = [
        repo / "src",
        repo / "third-party/kindergarden/src",
        repo / "third-party/ss-pybullet",
    ]
    rows = []
    for environment in sorted(environments):
        config_path = repo / "experiments/conf/environment" / f"{environment}.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text()) or {}
        starts = _config_modules(config)
        backend_path = config.get("constant_object_env_path")
        if isinstance(backend_path, str):
            starts.add(backend_path.split(":", 1)[0])
        env_id = config.get("env_id")
        if isinstance(env_id, str):
            try:
                import gymnasium as gym
                import kinder

                kinder.register_all_environments()
                entry_point = gym.spec(env_id).entry_point
                if isinstance(entry_point, str):
                    starts.add(entry_point.split(":", 1)[0])
            except Exception:  # best-effort static source resolution
                pass
        own_paths = [path for module in starts if (path := _module_path(module, roots))]
        closure = _local_closure(starts, roots)
        row: dict[str, Any] = {"environment": environment}
        row.update(
            {f"own_{k}": v for k, v in _aggregate_sources(own_paths, analyzer).items()}
        )
        row.update(
            {
                f"closure_{k}": v
                for k, v in _aggregate_sources(closure, analyzer).items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_final_by_environment(frame: pd.DataFrame, output: Path) -> list[Path]:
    directory = output / "final_by_environment"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for environment, group in frame.groupby("environment"):
        methods = sorted(group["display_method"].unique())
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for axis, (metric, title) in zip(axes.flat, METRICS, strict=True):
            for x, method in enumerate(methods):
                values = (
                    group.loc[group.display_method == method, metric]
                    .dropna()
                    .to_numpy()
                )
                jitter = (
                    np.linspace(-0.12, 0.12, len(values))
                    if len(values) > 1
                    else np.zeros(len(values))
                )
                axis.scatter(x + jitter, values, alpha=0.72, s=28)
                if len(values):
                    axis.hlines(
                        np.mean(values), x - 0.22, x + 0.22, color="black", linewidth=2
                    )
            axis.set_title(title)
            axis.set_xticks(
                range(len(methods)), methods, rotation=25, ha="right", fontsize=8
            )
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Final and one-shot policy complexity — {environment}")
        fig.tight_layout()
        path = directory / f"{environment}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_evolution(frame: pd.DataFrame, output: Path) -> list[Path]:
    directory = output / "evolution"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    stages = np.linspace(0, 1, 11)
    for (environment, method), group in frame.groupby(["environment", "method"]):
        sampled = []
        for (_, seed), trajectory in group.groupby(["method", "replicate_seed"]):
            for stage in stages:
                sampled.append(
                    trajectory.loc[
                        (trajectory.revision_progress - stage).abs().idxmin()
                    ].to_dict()
                    | {"stage": stage}
                )
        data = pd.DataFrame(sampled)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for axis, (metric, title) in zip(axes.flat, METRICS, strict=True):
            stats = data.groupby("stage")[metric].agg(["mean", "std"])
            axis.plot(stats.index, stats["mean"], marker="o")
            axis.fill_between(
                stats.index,
                stats["mean"] - stats["std"].fillna(0),
                stats["mean"] + stats["std"].fillna(0),
                alpha=0.2,
            )
            axis.set_title(title)
            axis.set_xlabel("Normalized revision / attempt progress")
            axis.grid(alpha=0.25)
        fig.suptitle(f"Policy complexity evolution — {environment} — {method}")
        fig.tight_layout()
        path = (
            directory / f"{environment}__{re.sub('[^a-zA-Z0-9_-]+', '_', method)}.png"
        )
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_correlations(
    programs: pd.DataFrame, envs: pd.DataFrame, output: Path
) -> pd.DataFrame:
    means = (
        programs.groupby(["environment", "display_method"])[[m for m, _ in METRICS]]
        .mean()
        .reset_index()
    )
    merged = means.merge(envs, on="environment")
    rows = []
    for metric, _ in METRICS:
        xcol = f"closure_{metric}"
        for method, group in merged.groupby("display_method"):
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_environments": len(group),
                    "pearson_r": group[xcol].corr(group[metric]),
                    "spearman_rho": group[xcol].rank().corr(group[metric].rank()),
                }
            )
    return pd.DataFrame(rows)


def _short_method(method: str) -> str:
    return {
        "agentic/blackbox/claude-opus-5": "Agentic blackbox Claude",
        "agentic/blackbox/gpt-5.6-sol": "Agentic blackbox GPT",
        "agentic/whitebox/claude-opus-5": "Agentic whitebox Claude",
        "llm_genplan": "GenPlan",
        "llm_genplan/whitebox/claude-opus-5": "GenPlan final",
        "llm_genplan_one_shot/whitebox/claude-opus-5": "GenPlan one-shot",
    }.get(method, method)


def _sample_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stages = np.linspace(0, 1, 11)
    for keys, trajectory in frame.groupby(
        ["method", "environment", "replicate_seed"], dropna=False
    ):
        trajectory = trajectory.sort_values("revision_progress")
        samples = []
        for stage in stages:
            sample = trajectory.loc[
                (trajectory.revision_progress - stage).abs().idxmin()
            ].to_dict()
            sample["stage"] = stage
            samples.append(sample)
        sampled = pd.DataFrame(samples)
        for metric, _ in METRICS:
            values = sampled[metric].fillna(0).astype(float).to_numpy()
            progress = sampled.stage.to_numpy()
            rho = (
                pd.Series(progress).rank().corr(pd.Series(values).rank())
                if len(np.unique(values)) > 1
                else 0.0
            )
            rows.append(
                {
                    "method": keys[0],
                    "environment": keys[1],
                    "replicate_seed": keys[2],
                    "metric": metric,
                    "spearman_rho": rho,
                    "endpoint_change_pct": 100
                    * (values[-1] - values[0])
                    / max(abs(values[0]), 1),
                    "nondecreasing_step_fraction": np.mean(np.diff(values) >= 0),
                }
            )
    return pd.DataFrame(rows)


def _trajectory_checkpoints(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method", "environment", "replicate_seed", "stage", "metric",
        "value", "baseline", "change_pct",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (method, environment, seed), trajectory in frame.groupby(
        ["method", "environment", "replicate_seed"], dropna=False
    ):
        trajectory = trajectory.sort_values("revision_progress")
        for stage in np.linspace(0, 1, 11):
            sample = trajectory.loc[
                (trajectory.revision_progress - stage).abs().idxmin()
            ]
            for metric, _ in METRICS:
                rows.append(
                    {
                        "method": method,
                        "environment": environment,
                        "replicate_seed": seed,
                        "stage": stage,
                        "metric": metric,
                        "value": float(sample[metric] or 0),
                    }
                )
    result = pd.DataFrame(rows)
    baseline = result.loc[result.stage == 0, [
        "method", "environment", "replicate_seed", "metric", "value"
    ]].rename(columns={"value": "baseline"})
    result = result.merge(
        baseline,
        on=["method", "environment", "replicate_seed", "metric"],
        validate="many_to_one",
    )
    result["change_pct"] = (
        100 * (result.value - result.baseline) / result.baseline.abs().clip(lower=1)
    )
    return result


def _plot_cross_method_evolution(
    programs: pd.DataFrame, trajectories: pd.DataFrame, output: Path
) -> list[Path]:
    trajectories = trajectories.copy()
    trajectories["outcome_solve_rate"] = trajectories["final_solve_rate"]
    # GenPlan trajectory snapshots do not carry evaluation results, so attach the
    # result from the corresponding final program. Agentic trajectories already
    # identify the exact archived run and retain their own result.
    genplan_outcomes = (
        programs[
            programs.display_method.eq("llm_genplan/whitebox/claude-opus-5")
        ]
        .groupby(["environment", "replicate_seed"])
        .result_solve_rate.mean()
    )
    genplan = trajectories.method.eq("llm_genplan")
    genplan_keys = pd.MultiIndex.from_frame(
        trajectories.loc[genplan, ["environment", "replicate_seed"]]
    )
    trajectories.loc[genplan, "outcome_solve_rate"] = genplan_keys.map(
        genplan_outcomes
    )

    all_environments = set(trajectories.environment)
    variants = (
        (
            "all",
            "All seeds (no performance filter)",
            "cross_method_evolution_absolute.png",
            pd.Series(True, index=trajectories.index),
        ),
        (
            "fully_successful",
            "Seeds with 100% final evaluation success",
            "cross_method_evolution_successful.png",
            trajectories.outcome_solve_rate.eq(1.0),
        ),
        (
            "not_fully_successful",
            "Seeds below 100% final evaluation success",
            "cross_method_evolution_not_fully_successful.png",
            trajectories.outcome_solve_rate.lt(1.0),
        ),
    )
    methods = sorted(trajectories.method.unique())
    all_method_coverage = {
        method: set(group.environment)
        for method, group in trajectories.groupby("method")
    }
    covered_by_all_methods = set.intersection(
        *(all_method_coverage.get(method, set()) for method in methods)
    )
    missing_method_coverage = all_environments - covered_by_all_methods
    colors = {method: METHOD_COLORS[method] for method in methods}
    paths = []
    summaries = []
    counts = []
    report_path = output.parent / "complexity_evolution_report.pdf"

    def draw(
        environment_means: pd.DataFrame,
        figure_title: str,
        detail_lines: list[str],
        empty_message: str = (
            "No environment has a qualifying run for every approach;\n"
            "a matched cross-method average cannot be computed."
        ),
    ) -> Any:
        fig, axes_grid = plt.subplots(3, 3, figsize=(7.2, 5.8))
        axes = list(axes_grid.flat)
        if environment_means.empty:
            for axis in axes:
                axis.set_visible(False)
            fig.text(
                0.5,
                0.49,
                empty_message,
                ha="center",
                va="center",
                fontsize=10,
            )
        for axis in axes[len(METRICS) :]:
            axis.set_visible(False)
        for axis, (metric, metric_title) in zip(
            axes[: len(METRICS)], METRICS, strict=True
        ):
            subset = environment_means[environment_means.metric == metric]
            for method, group in subset.groupby("method"):
                stats = group.groupby("stage")["value"].agg(["mean", "sem"])
                ci = 1.96 * stats["sem"].fillna(0)
                axis.plot(
                    100 * stats.index,
                    stats["mean"],
                    marker="o",
                    markersize=1.8,
                    linewidth=1.25,
                    color=colors[method],
                    label=_short_method(method),
                )
                axis.fill_between(
                    100 * stats.index,
                    stats["mean"] - ci,
                    stats["mean"] + ci,
                    color=colors[method],
                    alpha=0.10,
                )
            axis.set_title(metric_title)
            axis.set_xlabel("Normalized revision progress (%)")
            axis.set_ylabel("Mean complexity")
            axis.grid(axis="y", alpha=0.25, linewidth=0.5)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle(figure_title, fontsize=9, fontweight="bold", y=0.995)
        y = 0.965
        for line in detail_lines:
            wrapped = textwrap.fill(line, width=128)
            fig.text(0.5, y, wrapped, ha="center", va="top", fontsize=5.5)
            y -= 0.028 * (wrapped.count("\n") + 1)
        if not environment_means.empty:
            fig.tight_layout(rect=(0, 0.065, 1, y - 0.005), w_pad=0.9, h_pad=1.0)
        return fig

    variant_data = []
    for variant, figure_title, filename, mask in variants:
        selected = trajectories[mask].copy()
        available = {
            method: set(group.environment)
            for method, group in selected.groupby("method")
        }
        included = set.intersection(
            *(available.get(method, set()) for method in methods)
        )
        excluded_by_criterion = covered_by_all_methods - included
        averaged = selected[selected.environment.isin(included)]
        criterion_label = {
            "all": "Excluded — no qualifying run for every method",
            "fully_successful": "Excluded — no 100%-successful run for every method",
            "not_fully_successful": "Excluded — no below-100% run for every method",
        }[variant]
        detail_lines = [
            "Aggregation: filter individual seeds by final evaluation, average "
            "qualifying seeds within each environment, retain only environments "
            "represented by every approach, then average environments equally.",
            f"Included ({len(included)}): {', '.join(sorted(included)) or 'none'}",
            f"Excluded — not run on all methods ({len(missing_method_coverage)}): "
            f"{', '.join(sorted(missing_method_coverage)) or 'none'}",
            f"{criterion_label} ({len(excluded_by_criterion)}): "
            f"{', '.join(sorted(excluded_by_criterion)) or 'none'}",
        ]
        variant_data.append(
            (variant, figure_title, filename, selected, included, detail_lines)
        )
        selected_runs = averaged[
            ["method", "environment", "replicate_seed"]
        ].drop_duplicates()
        run_counts = selected_runs.groupby("method").agg(
            trajectories=("replicate_seed", "size"),
            environments=("environment", "nunique"),
        )
        for method in methods:
            row = run_counts.loc[method].to_dict() if method in run_counts.index else {
                "trajectories": 0,
                "environments": 0,
            }
            counts.append({"subset": variant, "method": method, **row})
        checkpoints = _trajectory_checkpoints(averaged)
        # Average seeds first, then average the identical environment set for each
        # method. The ribbon represents a 95% CI across those environments.
        # The ribbon then represents a 95% CI across environments.
        environment_means = (
            checkpoints.groupby(["method", "environment", "stage", "metric"])[
                ["value", "change_pct"]
            ]
            .mean()
            .reset_index()
        )
        environment_means["subset"] = variant
        summaries.append(environment_means)
        fig = draw(
            environment_means,
            f"Aggregate across matched environments: {figure_title}",
            detail_lines,
        )
        path = output / filename
        _save_figure(fig, path)
        plt.close(fig)
        paths.append(path)

    with PdfPages(report_path) as report:
        methods_fig = plt.figure(figsize=(7.2, 5.8))
        methods_fig.suptitle(
            "Methods: performance filtering and aggregation",
            fontsize=13,
            fontweight="bold",
            y=0.94,
        )
        methods_text = (
            "Filtering unit: individual seed/run\n\n"
            "A seed is in the 100% subset when its final held-out evaluation solve "
            "rate is exactly 1.0 (normally 100 of 100 tasks solved). A seed is in "
            "the below-100% subset when that rate is less than 1.0. An environment "
            "can appear in both subsets when different seeds have different outcomes.\n\n"
            "Per-environment pages\n\n"
            "Each page shows every approach with at least one qualifying seed for "
            "that environment. Curves average qualifying seeds; ribbons show 95% "
            "confidence intervals across seeds. A retained empty page explicitly "
            "states when no seed from any approach qualifies.\n\n"
            "Aggregate pages\n\n"
            "Seeds are filtered first and averaged within each environment. An "
            "environment enters the aggregate only when every approach has at least "
            "one qualifying seed, ensuring that all approach curves use the same "
            "environment set. The retained environments are then averaged equally; "
            "ribbons show 95% confidence intervals across environments.\n\n"
            "The all-seeds aggregate applies no performance filter, but it retains "
            "the same complete cross-approach environment requirement."
        )
        methods_fig.text(
            0.10,
            0.84,
            textwrap.fill(methods_text, width=92, replace_whitespace=False),
            ha="left",
            va="top",
            fontsize=9,
            linespacing=1.35,
        )
        report.savefig(methods_fig)
        plt.close(methods_fig)
        for variant, figure_title, _, selected, included, detail_lines in variant_data:
            averaged = selected[selected.environment.isin(included)]
            checkpoints = _trajectory_checkpoints(averaged)
            means = (
                checkpoints.groupby(["method", "environment", "stage", "metric"])[
                    ["value", "change_pct"]
                ]
                .mean()
                .reset_index()
            )
            fig = draw(
                means,
                f"Aggregate across matched environments: {figure_title}",
                detail_lines,
            )
            report.savefig(fig)
            plt.close(fig)
        for environment in sorted(all_environments):
            for variant, figure_title, _, selected, _, _ in variant_data:
                environment_rows = selected[selected.environment.eq(environment)]
                checkpoints = _trajectory_checkpoints(environment_rows)
                seed_counts = (
                    environment_rows[
                        ["method", "environment", "replicate_seed"]
                    ]
                    .drop_duplicates()
                    .groupby("method")
                    .size()
                )
                present = set(checkpoints.method)
                availability_labels = {
                    "all": ("Approaches with trajectory data", "Approaches without trajectory data"),
                    "fully_successful": ("Approaches with 100% seeds", "Approaches with no 100% seeds"),
                    "not_fully_successful": (
                        "Approaches with below-100% seeds",
                        "Approaches with no below-100% seeds",
                    ),
                }[variant]
                empty_message = {
                    "all": "No trajectory seeds are available for this environment.",
                    "fully_successful": (
                        "No seeds got 100% eval in this environment."
                    ),
                    "not_fully_successful": (
                        "No seeds got less than 100% eval in this environment."
                    ),
                }[variant]
                fig = draw(
                    checkpoints,
                    f"{environment}: {figure_title}",
                    [
                        f"{availability_labels[0]}: {', '.join(_short_method(m) for m in methods if m in present) or 'none'}",
                        f"{availability_labels[1]}: {', '.join(_short_method(m) for m in methods if m not in present) or 'none'}",
                        "Seed trajectories: "
                        + ", ".join(
                            f"{_short_method(method)} n={int(seed_counts[method])}"
                            for method in methods
                            if method in seed_counts
                        ),
                    ],
                    empty_message=empty_message,
                )
                report.savefig(fig)
                plt.close(fig)
    pd.concat(summaries, ignore_index=True).to_csv(
        output.parent / "cross_method_evolution.csv", index=False
    )
    pd.DataFrame(counts).to_csv(
        output.parent / "cross_method_evolution_counts.csv", index=False
    )
    return paths


def _heatmap(
    axis: Any,
    values: pd.DataFrame,
    title: str,
    fmt: str = ".2f",
    cmap: str = "RdBu_r",
    limits: tuple[float, float] | None = (-1, 1),
) -> None:
    kwargs = {"cmap": cmap, "aspect": "auto"}
    if limits:
        kwargs |= {"vmin": limits[0], "vmax": limits[1]}
    image = axis.imshow(values.to_numpy(), **kwargs)
    axis.set_xticks(range(len(values.columns)), values.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(values.index)), values.index)
    for row in range(len(values.index)):
        for col in range(len(values.columns)):
            value = values.iloc[row, col]
            axis.text(col, row, format(value, fmt), ha="center", va="center", fontsize=8)
    axis.set_title(title)
    plt.colorbar(image, ax=axis, shrink=0.75)


def _plot_pattern_summaries(
    programs: pd.DataFrame,
    trajectories: pd.DataFrame,
    environments: pd.DataFrame,
    correlations: pd.DataFrame,
    output: Path,
) -> tuple[list[Path], pd.DataFrame]:
    directory = output / "patterns"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    sampled = _sample_trajectories(trajectories)
    sampled.to_csv(output / "trajectory_pattern_summary.csv", index=False)

    paths.extend(_plot_cross_method_evolution(programs, trajectories, directory))

    ordered = environments.sort_values("own_source_loc", ascending=False)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(7.2, 8.5), sharex=True)
    positions = np.arange(len(ordered))
    for axis, (metric, title) in zip(axes, METRICS, strict=True):
        axis.bar(
            positions,
            ordered[f"closure_{metric}"],
            width=0.78,
            label="Local dependency closure",
            color="#aab4bd",
        )
        axis.bar(
            positions,
            ordered[f"own_{metric}"],
            width=0.48,
            label="Environment-owned source",
            color="#17658c",
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    labels = [
        value.removesuffix("_generalized")
        .replace("cluttered", "clut.")
        .replace("obstruction", "obstr.")
        .replace("constrainedcupboard", "cupboard")
        .replace("sortclutteredblocks", "sort-blocks")
        .replace("sweepintodrawer", "sweep-drawer")
        for value in ordered.environment
    ]
    axes[-1].set_xticks(positions, labels, rotation=58, ha="right", fontsize=5)
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    fig.tight_layout(h_pad=0.8)
    path = directory / "environment_static_complexity.png"
    _save_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    valid = programs.dropna(subset=["result_solve_rate"]).copy()
    perf_rows = []
    for method, group in valid.groupby("display_method"):
        for metric, _ in METRICS:
            for environment, environment_group in group.groupby("environment"):
                complete = environment_group[[metric, "result_solve_rate"]].dropna()
                if (
                    len(complete) >= 3
                    and complete[metric].nunique() > 1
                    and complete.result_solve_rate.nunique() > 1
                ):
                    perf_rows.append(
                        {
                            "method": _short_method(method),
                            "metric": metric,
                            "environment": environment,
                            "spearman_rho": complete[metric]
                            .rank()
                            .corr(complete.result_solve_rate.rank()),
                            "n_seeds": len(complete),
                        }
                    )
    performance = pd.DataFrame(perf_rows)
    performance.to_csv(output / "performance_correlations.csv", index=False)
    loc = performance[performance.metric == "source_loc"]
    methods = sorted(loc.method.unique())
    rng = np.random.default_rng(0)
    fig, axis = plt.subplots(figsize=(7.2, 2.45))
    for position, method in enumerate(methods):
        values = loc.loc[loc.method == method, "spearman_rho"].to_numpy()
        jitter = rng.uniform(-0.10, 0.10, len(values))
        axis.scatter(values, position + jitter, color=f"C{position}", alpha=0.45, s=12)
        bootstrap_means = np.mean(
            rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
        )
        mean = np.mean(values)
        low, high = np.quantile(bootstrap_means, [0.025, 0.975])
        axis.errorbar(
            mean,
            position,
            xerr=[[mean - low], [high - mean]],
            fmt="D",
            color="black",
            markersize=4,
            capsize=3,
            linewidth=1.2,
        )
        axis.text(1.03, position, f"n={len(values)}", va="center", fontsize=6)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(-1.05, 1.18)
    axis.set_yticks(range(len(methods)), methods)
    axis.set_xlabel("Spearman ρ: final lines of code vs. final solve rate")
    axis.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    path = directory / "performance_correlations.png"
    _save_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    env_corr = correlations.pivot(index="method", columns="metric", values="spearman_rho")
    env_corr.index = [_short_method(value) for value in env_corr.index]
    env_corr = env_corr[[metric for metric, _ in METRICS]].rename(columns=dict(METRICS))
    fig, axis = plt.subplots(figsize=(7.2, 2.6))
    _heatmap(axis, env_corr, "Environment complexity vs. policy complexity (Spearman ρ)")
    fig.tight_layout()
    path = directory / "environment_policy_correlation_heatmap.png"
    _save_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths, performance


def _gallery(
    output: Path,
    pattern_paths: list[Path],
) -> None:
    markdown_path = output / "measurements.md"
    markdown_path.write_text(MEASUREMENTS_MD)
    ms = subprocess.run(
        ["pandoc", "--from=markdown", "--to=man", str(markdown_path)],
        check=True,
        capture_output=True,
    )
    postscript = subprocess.run(
        ["groff", "-t", "-Tps", "-man"],
        check=True,
        input=b'.TH "STATIC COMPLEXITY MEASUREMENTS" 1\n' + ms.stdout,
        capture_output=True,
    )
    subprocess.run(
        ["ps2pdf", "-", str(output / "measurements.pdf")],
        check=True,
        env={**os.environ, "TMPDIR": "/tmp"},
        input=postscript.stdout,
    )
    sections = [("Complexity results", pattern_paths)]
    figure_titles = {
        "cross_method_evolution_absolute": "Complexity evolution — all runs",
        "cross_method_evolution_successful": "Complexity evolution — 100% successful runs",
        "cross_method_evolution_not_fully_successful": "Complexity evolution — runs below 100% success",
    }
    cards = []
    for title, paths in sections:
        cards.append(f"<h2>{title}</h2>")
        cards.extend(
            f'<article><h3>{figure_titles.get(path.stem, path.stem)}</h3>'
            f'<p><a download href="{path.with_suffix(".pdf").relative_to(output)}">Download PDF</a></p>'
            f'<img loading="lazy" src="{path.relative_to(output)}"></article>'
            for path in paths
        )
    (output / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>RoboCode complexity</title><style>body{max-width:1500px;margin:auto;padding:24px;font-family:system-ui;background:#f5f7fa}article{background:white;padding:16px;margin:20px 0;border-radius:10px}img{width:100%;height:auto}a{margin-right:16px}</style><h1>RoboCode static complexity</h1><p><a href=complexity_evolution_report.pdf download>Complete evolution report (PDF)</a><a href=measurements.pdf download>Measurement definitions (PDF)</a><a href=measurements.md download>Markdown</a><a href=cross_method_evolution_counts.csv>Trajectory subset counts</a><a href=program_complexity_all_methods.csv>Programs CSV</a><a href=environment_complexity.csv>Environments CSV</a><a href=correlations.csv>Correlations CSV</a></p>"
        + "".join(cards)
    )


def main() -> None:
    _set_paper_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analyzer = _load_analyzer(Path(__file__).with_name("analyze_program_complexity.py"))
    final = pd.read_csv(output / "program_complexity.csv")
    one_shot = _one_shots(repo, analyzer)
    programs = pd.concat([final, one_shot], ignore_index=True, sort=False)
    programs["display_method"] = (
        programs["approach"] + "/" + programs["access"] + "/" + programs["backend"]
    )
    programs.to_csv(output / "program_complexity_all_methods.csv", index=False)
    agentic_history = _agentic_trajectories(repo)
    genplan_history = _genplan_trajectories(repo, analyzer)
    trajectories = pd.concat(
        [agentic_history, genplan_history], ignore_index=True, sort=False
    )
    trajectories.to_csv(output / "complexity_trajectories.csv", index=False)
    envs = _environment_complexity(repo, set(programs.environment), analyzer)
    envs.to_csv(output / "environment_complexity.csv", index=False)
    correlations = _plot_correlations(programs, envs, output)
    correlations.to_csv(output / "correlations.csv", index=False)
    pattern_paths, _ = _plot_pattern_summaries(
        programs, trajectories, envs, correlations, output
    )
    _gallery(output, pattern_paths)
    print(
        f"Wrote {len(programs)} programs, {len(trajectories)} trajectory revisions, "
        f"{len(envs)} environments, and {len(pattern_paths)} result figures"
    )


if __name__ == "__main__":
    main()
