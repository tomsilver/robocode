#!/usr/bin/env python3
"""Evaluate saved GenPlan first attempts on their original held-out suites.

The GenPlan refinement loop saves its initial generated policy as
``sandbox/impl0_candidate.py``. This script discovers those policies, selects the
newest run for each (experiment, replicate) pair, and invokes the normal
experiment runner with ``approach.load_dir`` so no generation or feedback occurs.

Each evaluation is isolated in its own subprocess and output directory. Completed
``results.json`` files are skipped by default, making the command resumable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from tqdm import tqdm  # type: ignore[import-untyped]
from tqdm.contrib.logging import logging_redirect_tqdm  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "experiments" / "run_experiment.py"


@dataclass(frozen=True)
class Evaluation:
    """One first-attempt policy and the original run that defines its protocol."""

    experiment: str
    replicate_seed: int
    run_dir: Path
    candidate: Path
    overrides: tuple[str, ...]

    @property
    def key(self) -> tuple[str, int]:
        return self.experiment, self.replicate_seed


@dataclass(frozen=True)
class Outcome:
    """Result of launching or reusing one evaluation."""

    evaluation: Evaluation
    status: str
    elapsed_s: float
    result_dir: Path
    message: str = ""


def _parse_override(overrides: Iterable[str], key: str) -> str | None:
    prefix = f"{key}="
    matches = [value[len(prefix) :] for value in overrides if value.startswith(prefix)]
    return matches[-1] if matches else None


def _load_overrides(run_dir: Path) -> tuple[str, ...]:
    path = run_dir / ".hydra" / "overrides.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"expected a string list in {path}")
    return tuple(raw)


def _configured_num_eval_tasks(run_dir: Path) -> int:
    """Read the suite size from the historical composed config."""
    path = run_dir / ".hydra" / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("num_eval_tasks"), int):
        raise ValueError(f"num_eval_tasks is absent or invalid in {path}")
    return raw["num_eval_tasks"]


def _source_eval_complete(run_dir: Path) -> bool:
    """Whether the source run recorded every scheduled held-out episode."""
    try:
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    per_episode = results.get("per_episode")
    num_tasks = results.get("num_eval_tasks")
    return (
        isinstance(per_episode, list)
        and isinstance(num_tasks, int)
        and len(per_episode) == num_tasks
    )


def _selection_key(evaluation: Evaluation) -> tuple[bool, str, str]:
    """Prefer completed sources, then the campaign timestamp and full path."""
    timestamp = evaluation.run_dir.parent.name
    return _source_eval_complete(evaluation.run_dir), timestamp, str(evaluation.run_dir)


def discover_evaluations(search_roots: Iterable[Path]) -> list[Evaluation]:
    """Return the newest candidate for every experiment and replicate."""
    selected: dict[tuple[str, int], Evaluation] = {}
    for root in search_roots:
        for candidate in sorted(root.rglob("sandbox/impl0_candidate.py")):
            run_dir = candidate.parent.parent
            try:
                overrides = _load_overrides(run_dir)
                seed_text = _parse_override(overrides, "replicate_seed")
                if seed_text is None:
                    raise ValueError("replicate_seed is absent")
                replicate_seed = int(seed_text)
            except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
                LOGGER.warning("Skipping %s: %s", run_dir, error)
                continue
            relative_parts = candidate.relative_to(root).parts
            experiment = (
                root.name if "__llm_genplan__" in root.name else relative_parts[0]
            )
            evaluation = Evaluation(
                experiment, replicate_seed, run_dir, candidate, overrides
            )
            current = selected.get(evaluation.key)
            # Prefer a reproducible final evaluation over a newer interrupted rerun.
            # Within the same completion class, campaign directories begin with an
            # ISO timestamp; the full path is only a deterministic tie breaker.
            if current is None or _selection_key(evaluation) > _selection_key(current):
                selected[evaluation.key] = evaluation
    return sorted(selected.values(), key=lambda item: item.key)


def build_command(
    evaluation: Evaluation,
    result_dir: Path,
    *,
    num_eval_tasks: int | None = None,
) -> list[str]:
    """Build a runner command that loads impl0 without invoking the model."""
    load_dir = result_dir / "loaded_policy"
    # Loading the saved, fully composed Hydra config protects historical protocol
    # values from later changes to environment and approach defaults.
    config_dir = result_dir / "source_config"
    overrides: list[str] = []
    if num_eval_tasks is not None:
        overrides.append(f"num_eval_tasks={num_eval_tasks}")
    overrides.extend(
        [
            f"approach.load_dir={load_dir.resolve()}",
            "mcp_tools=[]",
            "record_approach_history=false",
            "render_videos=false",
            f"hydra.run.dir={result_dir.resolve()}",
        ]
    )
    return [
        sys.executable,
        str(RUNNER),
        f"--config-dir={config_dir.resolve()}",
        "--config-name=first_attempt_config",
        *overrides,
    ]


def _is_complete(results_path: Path, expected_tasks: int) -> bool:
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    per_episode = results.get("per_episode")
    num_tasks = results.get("num_eval_tasks")
    # Policy crashes are valid zero-score outcomes, but the shared summary marks
    # eval_complete=false when any occurred. Treat the suite as finished when every
    # scheduled episode has a record so resumptions do not rerun crashing policies.
    return (
        isinstance(per_episode, list)
        and isinstance(num_tasks, int)
        and num_tasks == expected_tasks
        and len(per_episode) == num_tasks
    )


def evaluate_one(
    evaluation: Evaluation,
    output_root: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    num_eval_tasks: int | None = None,
) -> Outcome:
    """Stage and evaluate one candidate, returning a serializable summary."""
    result_root = (
        output_root
        if num_eval_tasks is None
        else output_root / f"benchmark_{num_eval_tasks}_tasks"
    )
    result_dir = (
        result_root / evaluation.experiment / f"replicate_{evaluation.replicate_seed}"
    )
    results_path = result_dir / "results.json"
    expected_tasks = (
        num_eval_tasks
        if num_eval_tasks is not None
        else _configured_num_eval_tasks(evaluation.run_dir)
    )
    if not force and _is_complete(results_path, expected_tasks):
        return Outcome(evaluation, "skipped", 0.0, result_dir, "already complete")

    command = build_command(evaluation, result_dir, num_eval_tasks=num_eval_tasks)
    if dry_run:
        return Outcome(evaluation, "dry-run", 0.0, result_dir, " ".join(command))

    policy_dir = result_dir / "loaded_policy" / "sandbox"
    policy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(evaluation.candidate, policy_dir / "approach.py")
    config_dir = result_dir / "source_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        evaluation.run_dir / ".hydra" / "config.yaml",
        config_dir / "first_attempt_config.yaml",
    )
    metadata = {
        "source_run": str(evaluation.run_dir.resolve()),
        "source_candidate": str(evaluation.candidate.resolve()),
        "experiment": evaluation.experiment,
        "replicate_seed": evaluation.replicate_seed,
        "command": command,
    }
    (result_dir / "first_attempt_eval.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    log_path = result_dir / "evaluation.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        return Outcome(
            evaluation,
            "failed",
            elapsed,
            result_dir,
            f"exit {completed.returncode}; see {log_path}",
        )
    if not results_path.is_file():
        return Outcome(
            evaluation, "failed", elapsed, result_dir, "missing results.json"
        )
    return Outcome(evaluation, "complete", elapsed, result_dir)


def _result_summary(outcome: Outcome) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": outcome.evaluation.experiment,
        "replicate_seed": outcome.evaluation.replicate_seed,
        "status": outcome.status,
        "elapsed_s": round(outcome.elapsed_s, 3),
        "result_dir": str(outcome.result_dir),
        "message": outcome.message,
    }
    results_path = outcome.result_dir / "results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return row
    for key in (
        "solve_rate",
        "mean_eval_reward",
        "mean_eval_steps",
        "num_eval_tasks",
        "num_evaluated_episodes",
        "num_crashed_episodes",
        "eval_complete",
    ):
        row[key] = results.get(key)
    return row


def _write_summary(output_root: Path, outcomes: list[Outcome]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [_result_summary(outcome) for outcome in outcomes]
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


def _default_jobs() -> int:
    # Rollouts are often CPU-heavy and may themselves spawn a timeout worker. A
    # conservative default avoids oversubscribing shared experiment machines.
    return max(1, min(4, (os.cpu_count() or 1) // 2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "search_roots",
        nargs="*",
        type=Path,
        default=[REPO_ROOT / "multirun"],
        help="experiment directories or parents to scan (default: multirun)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "first_attempt_eval",
    )
    parser.add_argument("--jobs", type=int, default=_default_jobs())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-progress", action="store_true", help="disable the tqdm progress bar"
    )
    parser.add_argument(
        "--num-eval-tasks",
        type=int,
        help="override suite size (intended only for benchmarking)",
    )
    parser.add_argument(
        "--limit", type=int, help="evaluate only the first N discovered policies"
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.num_eval_tasks is not None and args.num_eval_tasks <= 0:
        parser.error("--num-eval-tasks must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> int:
    """CLI entry point."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluations = discover_evaluations(path.resolve() for path in args.search_roots)
    if args.limit is not None:
        evaluations = evaluations[: args.limit]
    if not evaluations:
        LOGGER.error("No completed GenPlan runs with impl0_candidate.py were found")
        return 2
    LOGGER.info(
        "Evaluating %d first attempts with %d workers", len(evaluations), args.jobs
    )

    outcomes: list[Outcome] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_eval = {
            executor.submit(
                evaluate_one,
                evaluation,
                args.output_dir.resolve(),
                force=args.force,
                dry_run=args.dry_run,
                num_eval_tasks=args.num_eval_tasks,
            ): evaluation
            for evaluation in evaluations
        }
        completed_futures = concurrent.futures.as_completed(future_to_eval)
        with (
            logging_redirect_tqdm(),
            tqdm(
                completed_futures,
                total=len(future_to_eval),
                desc="First attempts",
                unit="policy",
                disable=args.no_progress,
            ) as progress,
        ):
            for future in progress:
                evaluation = future_to_eval[future]
                try:
                    outcome = future.result()
                except Exception as error:  # pylint: disable=broad-exception-caught
                    result_root = (
                        args.output_dir.resolve()
                        if args.num_eval_tasks is None
                        else args.output_dir.resolve()
                        / f"benchmark_{args.num_eval_tasks}_tasks"
                    )
                    outcome = Outcome(
                        evaluation,
                        "failed",
                        0.0,
                        result_root
                        / evaluation.experiment
                        / f"replicate_{evaluation.replicate_seed}",
                        f"{type(error).__name__}: {error}",
                    )
                outcomes.append(outcome)
                LOGGER.info(
                    "%s seed=%s: %s (%.1fs)%s",
                    outcome.evaluation.experiment,
                    outcome.evaluation.replicate_seed,
                    outcome.status,
                    outcome.elapsed_s,
                    f" — {outcome.message}" if outcome.message else "",
                )
                _write_summary(
                    args.output_dir.resolve(),
                    sorted(outcomes, key=lambda o: o.evaluation.key),
                )
    failed = sum(outcome.status == "failed" for outcome in outcomes)
    LOGGER.info("Finished %d evaluations (%d failed)", len(outcomes), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
