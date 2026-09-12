"""Build per-environment policy, trajectory, and environment-complexity figures."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

METRICS = [
    ("source_loc", "Lines of code"),
    ("ast_nodes", "AST nodes"),
    ("cyclomatic_total", "Cyclomatic complexity"),
    ("cyclomatic_max_function", "Maximum function complexity"),
    ("max_nesting_depth", "Maximum nesting depth"),
    ("persistent_state_fields", "Persistent state fields"),
]


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
            rows.append(
                {
                    "environment": sandbox.parents[2].name.split("__", 1)[0],
                    "method": "llm_genplan",
                    "replicate_seed": sandbox.parent.name.removeprefix("replicate_"),
                    "revision_progress": index / max(1, len(candidates) - 1),
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
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (metric, title) in zip(axes.flat, METRICS, strict=True):
        xcol = f"closure_{metric}"
        for method, group in merged.groupby("display_method"):
            axis.scatter(group[xcol], group[metric], label=method, alpha=0.75)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_environments": len(group),
                    "pearson_r": group[xcol].corr(group[metric]),
                    "spearman_rho": group[xcol].rank().corr(group[metric].rank()),
                }
            )
        axis.set_title(title)
        axis.set_xlabel("Environment dependency-closure complexity")
        axis.set_ylabel("Mean policy complexity")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Environment source complexity vs. generated-policy complexity")
    fig.tight_layout()
    fig.savefig(
        output / "environment_policy_correlations.png", dpi=170, bbox_inches="tight"
    )
    plt.close(fig)
    return pd.DataFrame(rows)


def _gallery(
    output: Path, final_paths: list[Path], evolution_paths: list[Path]
) -> None:
    sections = [
        (
            "Environment vs. policy correlation",
            [output / "environment_policy_correlations.png"],
        ),
        ("Final policies by environment", final_paths),
        ("Complexity evolution", evolution_paths),
    ]
    cards = []
    for title, paths in sections:
        cards.append(f"<h2>{title}</h2>")
        cards.extend(
            f'<article><h3>{path.stem}</h3><img loading="lazy" src="{path.relative_to(output)}"></article>'
            for path in paths
        )
    (output / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>RoboCode complexity</title><style>body{max-width:1500px;margin:auto;padding:24px;font-family:system-ui;background:#f5f7fa}article{background:white;padding:16px;margin:20px 0;border-radius:10px}img{width:100%;height:auto}a{margin-right:16px}</style><h1>RoboCode static complexity</h1><p><a href=program_complexity_all_methods.csv>Programs CSV</a><a href=environment_complexity.csv>Environments CSV</a><a href=correlations.csv>Correlations CSV</a></p>"
        + "".join(cards)
    )


def main() -> None:
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
    finals = _plot_final_by_environment(programs, output)
    evolution = _plot_evolution(trajectories, output)
    correlations = _plot_correlations(programs, envs, output)
    correlations.to_csv(output / "correlations.csv", index=False)
    _gallery(output, finals, evolution)
    print(
        f"Wrote {len(programs)} programs, {len(trajectories)} trajectory revisions, {len(envs)} environments, {len(finals)} final figures, and {len(evolution)} evolution figures"
    )


if __name__ == "__main__":
    main()
