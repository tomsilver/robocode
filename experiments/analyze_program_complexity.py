"""Measure static complexity of final generated Agentic and GenPlan programs.

Inputs may be ordinary result directories, ZIP archives, or directories containing
ZIP archives.  One CSV row is emitted for each final ``sandbox/approach.py``.  The
program is parsed but never imported or executed.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import tokenize
import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

# Keep batch analysis friendly to shared workstations. NumPy-backed libraries can
# otherwise start a thread per core merely while pandas is importing or aggregating.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import pandas as pd
import yaml

_DECISIONS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler)
_NESTING = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
)

# These are the measurements summarized across replicate seeds. Metadata and result
# columns are deliberately excluded.
COMPLEXITY_METRICS = [
    "source_loc",
    "ast_nodes",
    "function_count",
    "branch_count",
    "loop_count",
    "max_nesting_depth",
    "cyclomatic_total",
    "cyclomatic_mean_function",
    "cyclomatic_max_function",
    "get_action_cyclomatic",
    "persistent_state_fields",
    "unique_numeric_literals",
    "halstead_effort",
]


class _Metrics(ast.NodeVisitor):
    """Collect syntax-tree metrics without evaluating generated code."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.max_nesting = 0
        self._nesting = 0
        self.functions: list[tuple[str, int]] = []
        self.self_fields: set[str] = set()
        self.numeric_literals: set[float | int | complex] = set()

    def generic_visit(self, node: ast.AST) -> None:
        self.counts[type(node).__name__] += 1
        nested = isinstance(node, _NESTING)
        if nested:
            self._nesting += 1
            self.max_nesting = max(self.max_nesting, self._nesting)
        super().generic_visit(node)
        if nested:
            self._nesting -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append((node.name, _cyclomatic_complexity(node)))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append((node.name, _cyclomatic_complexity(node)))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Store)
        ):
            self.self_fields.add(node.attr)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (int, float, complex)) and not isinstance(
            node.value, bool
        ):
            self.numeric_literals.add(node.value)
        self.generic_visit(node)


def _cyclomatic_complexity(node: ast.AST) -> int:
    """Return McCabe complexity (one plus independent decision points)."""
    score = 1
    for child in ast.walk(node):
        if isinstance(child, _DECISIONS):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += max(0, len(child.values) - 1)
        elif isinstance(child, ast.comprehension):
            score += 1 + len(child.ifs)
        elif isinstance(child, ast.Match):
            score += max(0, len(child.cases) - 1)
    return score


def _source_metrics(source: str) -> dict[str, Any]:
    lines = source.splitlines()
    nonblank = sum(bool(line.strip()) for line in lines)
    operators: Counter[str] = Counter()
    operands: Counter[str] = Counter()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        for token in tokens:
            if token.type == tokenize.OP or (
                token.type == tokenize.NAME
                and token.string
                in {
                    "and",
                    "or",
                    "not",
                    "in",
                    "is",
                    "if",
                    "else",
                    "for",
                    "while",
                    "return",
                    "yield",
                    "raise",
                    "await",
                }
            ):
                operators[token.string] += 1
            elif token.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
                operands[token.string] += 1
    except (IndentationError, tokenize.TokenError):
        pass

    base: dict[str, Any] = {
        "source_loc": len(lines),
        "nonblank_loc": nonblank,
    }
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        return {**base, "syntax_valid": False, "syntax_error": str(err)}

    visitor = _Metrics()
    visitor.visit(tree)
    function_cc = [cc for _, cc in visitor.functions]
    named_cc = dict(visitor.functions)
    module_cc = _cyclomatic_complexity(tree)

    n1, n2 = len(operators), len(operands)
    total_tokens = sum(operators.values()) + sum(operands.values())
    vocabulary = n1 + n2
    volume = total_tokens * math.log2(vocabulary) if vocabulary else 0.0
    difficulty = (n1 / 2) * (sum(operands.values()) / n2) if n2 else 0.0
    effort = volume * difficulty
    return {
        **base,
        "syntax_valid": True,
        "syntax_error": "",
        "ast_nodes": sum(visitor.counts.values()),
        "class_count": visitor.counts["ClassDef"],
        "function_count": len(visitor.functions),
        "helper_function_count": sum(
            name not in {"__init__", "reset", "get_action", "update"}
            for name, _ in visitor.functions
        ),
        "branch_count": visitor.counts["If"] + visitor.counts["IfExp"],
        "loop_count": (
            visitor.counts["For"] + visitor.counts["AsyncFor"] + visitor.counts["While"]
        ),
        "exception_handler_count": visitor.counts["ExceptHandler"],
        "max_nesting_depth": visitor.max_nesting,
        "cyclomatic_total": module_cc,
        "cyclomatic_mean_function": (
            sum(function_cc) / len(function_cc) if function_cc else 0.0
        ),
        "cyclomatic_max_function": max(function_cc, default=0),
        "get_action_cyclomatic": named_cc.get("get_action"),
        "reset_cyclomatic": named_cc.get("reset"),
        "update_cyclomatic": named_cc.get("update"),
        "persistent_state_fields": len(visitor.self_fields),
        "unique_numeric_literals": len(visitor.numeric_literals),
        "halstead_vocabulary": vocabulary,
        "halstead_length": total_tokens,
        "halstead_volume": volume,
        "halstead_difficulty": difficulty,
        "halstead_effort": effort,
    }


def _flatten_results(results: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"result_{key}"] = value
        elif key == "by_count" and isinstance(value, dict):
            for count, metrics in value.items():
                if isinstance(metrics, dict) and "solve_rate" in metrics:
                    row[f"result_solve_rate@{count}"] = metrics["solve_rate"]
    return row


def _metadata(
    config_text: str | None, overrides_text: str | None = None
) -> dict[str, Any]:
    if not config_text:
        return {}
    config = yaml.safe_load(config_text) or {}
    approach = config.get("approach", {})
    environment = config.get("environment", {})
    row = {
        "approach": str(approach.get("_target_", "")).rsplit(".", 1)[-1],
        "environment": str(environment.get("_target_", "")).rsplit(".", 1)[-1],
        "replicate_seed": config.get("replicate_seed", config.get("seed")),
        "experiment_id": config.get("experiment_id"),
        "access": "blackbox" if approach.get("blackbox", False) else "whitebox",
        "backend": str(
            approach.get("backend", {}).get("model", approach.get("model", "unknown"))
        ),
    }
    if overrides_text:
        for override in yaml.safe_load(overrides_text) or []:
            key, separator, value = str(override).partition("=")
            if separator and key in {"environment", "approach", "replicate_seed"}:
                row[key] = yaml.safe_load(value)
    return row


def _generated_approach(config_text: str | None, container_name: str) -> str | None:
    if config_text:
        target = str(
            (yaml.safe_load(config_text) or {}).get("approach", {}).get("_target_", "")
        )
        if target.endswith("AgenticApproach"):
            return "agentic"
        if target.endswith("LLMGenPlanApproach"):
            return "llm_genplan"
        return None
    if "__agentic__" in container_name:
        return "agentic"
    if "__llm_genplan__" in container_name:
        return "llm_genplan"
    return None


def _row(
    source: str,
    source_id: str,
    config: str | None,
    overrides: str | None,
    results: str | None,
) -> dict[str, Any]:
    row = {
        "source": source_id,
        **_metadata(config, overrides),
        **_source_metrics(source),
    }
    row["has_results"] = results is not None
    if results:
        row.update(_flatten_results(json.loads(results)))
    return row


def _rows_from_zip(path: Path) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for name in sorted(names):
            if not name.endswith("/sandbox/approach.py"):
                continue
            run = PurePosixPath(name).parent.parent
            config_name = str(run / ".hydra" / "config.yaml")
            overrides_name = str(run / ".hydra" / "overrides.yaml")
            results_name = str(run / "results.json")
            config = (
                archive.read(config_name).decode() if config_name in names else None
            )
            approach = _generated_approach(config, path.name)
            if approach is None:
                continue
            results = (
                archive.read(results_name).decode() if results_name in names else None
            )
            overrides = (
                archive.read(overrides_name).decode()
                if overrides_name in names
                else None
            )
            source = archive.read(name).decode("utf-8")
            row = _row(source, f"{path}!/{name}", config, overrides, results)
            row["approach"] = approach
            yield row


def _rows_from_directory(path: Path) -> Iterable[dict[str, Any]]:
    for approach_path in sorted(path.rglob("sandbox/approach.py")):
        run = approach_path.parent.parent
        config_path = run / ".hydra" / "config.yaml"
        overrides_path = run / ".hydra" / "overrides.yaml"
        results_path = run / "results.json"
        config = config_path.read_text() if config_path.exists() else None
        approach = _generated_approach(config, str(path))
        if approach is None:
            continue
        results = results_path.read_text() if results_path.exists() else None
        overrides = overrides_path.read_text() if overrides_path.exists() else None
        row = _row(
            approach_path.read_text(), str(approach_path), config, overrides, results
        )
        row["approach"] = approach
        yield row


def collect_complexity(inputs: list[Path]) -> pd.DataFrame:
    """Collect one row per final Agentic or GenPlan generated program."""
    rows: list[dict[str, Any]] = []
    seen_zips: set[Path] = set()
    for path in inputs:
        if path.is_file() and path.suffix.lower() == ".zip":
            rows.extend(_rows_from_zip(path))
            seen_zips.add(path.resolve())
        elif path.is_dir():
            rows.extend(_rows_from_directory(path))
            for zip_path in sorted(path.rglob("*.zip")):
                if zip_path.resolve() not in seen_zips and (
                    "__agentic__" in zip_path.name or "__llm_genplan__" in zip_path.name
                ):
                    rows.extend(_rows_from_zip(zip_path))
                    seen_zips.add(zip_path.resolve())
        else:
            raise FileNotFoundError(path)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop_duplicates(subset="source")
    # The same completed run may exist both unpacked and inside an upload archive.
    # Prefer the first copy and analyze each experiment/seed exactly once.
    identified = frame["experiment_id"].notna() & frame["replicate_seed"].notna()
    known = frame[identified].drop_duplicates(
        subset=["experiment_id", "replicate_seed"], keep="first"
    )
    return pd.concat([known, frame[~identified]], ignore_index=True)


def summarize_complexity(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and sample standard deviation across replicate seeds."""
    if frame.empty:
        return pd.DataFrame()
    metrics = [metric for metric in COMPLEXITY_METRICS if metric in frame.columns]
    grouped = frame.groupby(
        ["approach", "environment", "access", "backend"], sort=True, dropna=False
    )
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_programs", grouped.size())
    summary.insert(1, "n_with_results", grouped["has_results"].sum().astype(int))
    return summary.reset_index()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[Path(".")])
    parser.add_argument(
        "--output", "-o", type=Path, help="Write per-program CSV to this path"
    )
    parser.add_argument(
        "--summary-output",
        "-s",
        type=Path,
        help="Write environment-level mean/std CSV to this path",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON, not CSV")
    args = parser.parse_args()
    frame = collect_complexity(args.inputs)
    summary = summarize_complexity(frame)
    if args.output:
        frame.to_csv(args.output, index=False)
        print(f"Wrote {len(frame)} programs to {args.output}")
    if args.summary_output:
        summary.to_csv(args.summary_output, index=False)
        print(f"Wrote {len(summary)} environment summaries to {args.summary_output}")
    elif args.json and not args.output:
        print(frame.to_json(orient="records", indent=2))
    elif not args.output:
        print(frame.to_csv(index=False), end="")


if __name__ == "__main__":
    _main()
