"""Tests for the GenPlan first-attempt bulk evaluator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

import scripts.evaluate_genplan_first_attempts as evaluator
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
    (run / ".hydra" / "config.yaml").write_text(
        "max_steps: 321\nnum_eval_tasks: 100\n", encoding="utf-8"
    )
    if complete:
        (run / "results.json").write_text(
            json.dumps({"num_eval_tasks": 1, "per_episode": [{}]}),
            encoding="utf-8",
        )
    return run


def test_discover_selects_newest_run_per_replicate(tmp_path: Path) -> None:
    experiment = tmp_path / "example__llm_genplan__none"
    old = _make_run(experiment, "2026-01-01", 42)
    new = _make_run(experiment, "2026-01-02", 42)
    _make_run(experiment, "2026-01-04", 42, complete=False)
    interrupted = _make_run(experiment, "2026-01-03", 24, complete=False)
    old.touch()
    new.touch()

    evaluations = discover_evaluations([experiment])

    assert len(evaluations) == 2
    by_seed = {evaluation.replicate_seed: evaluation for evaluation in evaluations}
    assert by_seed[42].run_dir == new
    assert by_seed[24].run_dir == interrupted


def test_discover_compares_timestamps_across_roots(tmp_path: Path) -> None:
    """A search-root prefix must not outweigh the campaign timestamp."""
    name = "example__llm_genplan__none"
    older = _make_run(tmp_path / "z_archive" / name, "2026-01-01", 42)
    newer = _make_run(tmp_path / "a_archive" / name, "2026-01-02", 42)

    evaluations = discover_evaluations(
        [tmp_path / "z_archive" / name, tmp_path / "a_archive" / name]
    )

    assert len(evaluations) == 1
    assert evaluations[0].run_dir == newer
    assert evaluations[0].run_dir != older


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

    assert f"--config-dir={(tmp_path / 'output/source_config').resolve()}" in command
    assert "--config-name=first_attempt_config" in command
    assert "eval_seed=9876" not in command
    assert "num_eval_tasks=3" in command
    assert "num_eval_tasks=100" not in command
    assert "mcp_tools=[]" in command
    assert "hydra.run.dir=old" not in command
    assert "approach.load_dir=" in joined


def test_evaluate_one_skips_complete_result(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "source" / "example__llm_genplan__none", "stamp", 42)
    evaluation = Evaluation(
        "example",
        42,
        run,
        run / "sandbox" / "impl0_candidate.py",
        ("num_eval_tasks=1",),
    )
    result_dir = tmp_path / "out" / "example" / "replicate_42"
    result_dir.mkdir(parents=True)
    (result_dir / "results.json").write_text(
        json.dumps(
            {
                "eval_complete": False,
                "num_eval_tasks": 100,
                "per_episode": [{"crashed": True, "solved": False} for _ in range(100)],
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


def test_evaluate_one_stages_saved_composed_config(tmp_path: Path, monkeypatch) -> None:
    """Protocol values absent from overrides survive via the composed config."""
    run = _make_run(tmp_path / "source" / "example__llm_genplan__none", "stamp", 42)
    config = "max_steps: 321\nnum_eval_tasks: 1\ncustom_protocol_value: true\n"
    (run / ".hydra" / "config.yaml").write_text(config, encoding="utf-8")
    evaluation = Evaluation(
        "example",
        42,
        run,
        run / "sandbox" / "impl0_candidate.py",
        ("num_eval_tasks=1",),
    )

    def _fake_run(command, **_kwargs):
        result_dir = Path(
            next(
                value for value in command if value.startswith("hydra.run.dir=")
            ).split("=", 1)[1]
        )
        (result_dir / "results.json").write_text(
            json.dumps({"num_eval_tasks": 1, "per_episode": [{}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    outcome = evaluate_one(evaluation, tmp_path / "output")

    assert outcome.status == "complete"
    staged = outcome.result_dir / "source_config" / "first_attempt_config.yaml"
    assert staged.read_text(encoding="utf-8") == config


def test_main_records_worker_exception(tmp_path: Path, monkeypatch) -> None:
    """One broken worker must not abort collection or suppress the summary."""
    evaluation = Evaluation("example", 42, tmp_path / "run", tmp_path / "code", ())
    output = tmp_path / "output"
    monkeypatch.setattr(evaluator, "discover_evaluations", lambda _roots: [evaluation])
    monkeypatch.setattr(evaluator, "evaluate_one", lambda *_args, **_kwargs: 1 / 0)
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_genplan_first_attempts.py", "--output-dir", str(output)],
    )

    assert evaluator.main() == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["status"] == "failed"
    assert "ZeroDivisionError" in summary[0]["message"]
