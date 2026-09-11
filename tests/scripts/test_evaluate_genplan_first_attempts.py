"""Tests for the GenPlan first-attempt bulk evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.evaluate_genplan_first_attempts import (
    Evaluation,
    build_command,
    discover_evaluations,
    evaluate_one,
)


def _make_run(root: Path, stamp: str, seed: int, *, complete: bool = True) -> Path:
    run = root / stamp / f"replicate_{seed}"
    (run / "sandbox").mkdir(parents=True)
    (run / ".hydra").mkdir()
    (run / "sandbox" / "impl0_candidate.py").write_text(
        "class GeneratedApproach:\n    pass\n", encoding="utf-8"
    )
    (run / ".hydra" / "overrides.yaml").write_text(
        yaml.safe_dump(
            [
                "environment=motion2d_generalized",
                "approach=llm_genplan",
                "primitive_level=none",
                f"replicate_seed={seed}",
                "eval_seed=1234",
                "num_eval_tasks=100",
                "mcp_tools=[render_state]",
            ]
        ),
        encoding="utf-8",
    )
    if complete:
        (run / "results.json").write_text("{}", encoding="utf-8")
    return run


def test_discover_selects_newest_run_per_replicate(tmp_path: Path) -> None:
    experiment = tmp_path / "example__llm_genplan__none"
    old = _make_run(experiment, "2026-01-01", 42)
    new = _make_run(experiment, "2026-01-02", 42)
    interrupted = _make_run(experiment, "2026-01-03", 24, complete=False)
    old.touch()
    new.touch()

    evaluations = discover_evaluations([experiment])

    assert len(evaluations) == 2
    by_seed = {evaluation.replicate_seed: evaluation for evaluation in evaluations}
    assert by_seed[42].run_dir == new
    assert by_seed[24].run_dir == interrupted


def test_build_command_preserves_protocol_and_replaces_runtime_options(
    tmp_path: Path,
) -> None:
    evaluation = Evaluation(
        "example",
        42,
        tmp_path / "run",
        tmp_path / "impl0_candidate.py",
        (
            "environment=motion2d_generalized",
            "approach=llm_genplan",
            "replicate_seed=42",
            "eval_seed=9876",
            "num_eval_tasks=100",
            "mcp_tools=[render_state]",
            "hydra.run.dir=old",
        ),
    )

    command = build_command(evaluation, tmp_path / "output", num_eval_tasks=3)
    joined = " ".join(command)

    assert "eval_seed=9876" in command
    assert "num_eval_tasks=3" in command
    assert "num_eval_tasks=100" not in command
    assert "mcp_tools=[]" in command
    assert "hydra.run.dir=old" not in command
    assert "approach.load_dir=" in joined


def test_evaluate_one_skips_complete_result(tmp_path: Path) -> None:
    evaluation = Evaluation(
        "example",
        42,
        tmp_path / "run",
        tmp_path / "candidate.py",
        ("num_eval_tasks=1",),
    )
    result_dir = tmp_path / "out" / "example" / "replicate_42"
    result_dir.mkdir(parents=True)
    (result_dir / "results.json").write_text(
        json.dumps(
            {
                "eval_complete": False,
                "num_eval_tasks": 1,
                "per_episode": [{"crashed": True, "solved": False}],
            }
        ),
        encoding="utf-8",
    )

    outcome = evaluate_one(evaluation, tmp_path / "out")

    assert outcome.status == "skipped"

    benchmark_outcome = evaluate_one(
        evaluation, tmp_path / "out", dry_run=True, num_eval_tasks=2
    )
    assert benchmark_outcome.status == "dry-run"
