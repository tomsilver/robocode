"""Tests for collecting experiment results under the replicate-seed protocol."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "analyze_results.py"
)
_SPEC = importlib.util.spec_from_file_location("analyze_results", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
analyze_results: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze_results)


def test_collect_results_uses_replicate_seed_and_flattens_count_metrics(
    tmp_path: Path,
) -> None:
    """Nested protocol details stay out of the frame while headlines are flat."""
    run_dir = tmp_path / "run"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text(
        "replicate_seed: 24\n"
        "approach:\n  _target_: example.AgenticApproach\n"
        "environment:\n  _target_: example.VariableEnv\n",
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "solve_rate": 0.5,
                "design_count_solve_rate": 0.75,
                "held_out_count_solve_rate": 0.25,
                "by_count": {"1": {"solve_rate": 0.75}},
                "count_regimes": {
                    "design": {"counts": [1], "solve_rate": 0.75},
                    "held_out": {"counts": [5], "solve_rate": 0.25},
                },
                "per_episode": [],
            }
        ),
        encoding="utf-8",
    )

    dataframe = analyze_results.collect_results([tmp_path])

    assert dataframe.loc[0, "replicate_seed"] == 24
    assert dataframe.loc[0, "design_count_solve_rate"] == 0.75
    assert dataframe.loc[0, "held_out_count_solve_rate"] == 0.25
    assert dataframe.loc[0, "solve_rate@1"] == 0.75
    assert "count_regimes" not in dataframe.columns
    assert "per_episode" not in dataframe.columns


def test_collect_results_accepts_legacy_seed_config(tmp_path: Path) -> None:
    """Pre-migration Hydra runs are normalized to the replicate-seed column."""
    run_dir = tmp_path / "legacy-run"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text(
        "seed: 424\n"
        "approach:\n  _target_: example.AgenticApproach\n"
        "environment:\n  _target_: example.VariableEnv\n",
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps({"solve_rate": 0.5}), encoding="utf-8"
    )

    dataframe = analyze_results.collect_results([tmp_path])

    assert dataframe.loc[0, "replicate_seed"] == 424
    assert "seed" not in dataframe.columns


def test_aggregation_keeps_manual_rows_when_tracker_ids_are_present() -> None:
    """Optional experiment IDs do not drop manual or legacy runs from summaries."""
    dataframe = pd.DataFrame(
        [
            {
                "approach": "AgenticApproach",
                "environment": "ExampleEnv",
                "replicate_seed": 24,
                "experiment_id": None,
                "solve_rate": 0.5,
            },
            {
                "approach": "AgenticApproach",
                "environment": "ExampleEnv",
                "replicate_seed": 42,
                "experiment_id": "condition__abc12345",
                "solve_rate": 0.75,
            },
        ]
    )
    averaged = analyze_results.aggregate_results(dataframe)
    assert len(averaged) == 2
    assert set(averaged["replicate_seeds"]) == {"24", "42"}


def _write_run(run_dir: Path, replicate_seed: int, results: dict[str, Any]) -> None:
    """Lay out one Hydra job directory with a config and a results.json."""
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text(
        f"replicate_seed: {replicate_seed}\n"
        "approach:\n  _target_: example.AgenticApproach\n"
        "environment:\n  _target_: example.ExampleEnv\n",
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")


def test_boolean_flags_average_instead_of_splitting_a_condition(tmp_path: Path) -> None:
    """Replicates that differ only on a run flag still aggregate into one row.

    Left as bools these columns are not a numeric dtype, so aggregate_results treats
    them as grouping keys and emits one row per flag value -- two replicates of the
    same condition stop being averaged together precisely because one of them hit a
    limit or lost episodes to crashes.
    """
    _write_run(
        tmp_path / "a",
        24,
        {"solve_rate": 0.5, "eval_complete": True, "gen_turn_limit_hit": False},
    )
    _write_run(
        tmp_path / "b",
        42,
        {"solve_rate": 1.0, "eval_complete": False, "gen_turn_limit_hit": True},
    )

    dataframe = analyze_results.collect_results([tmp_path])
    averaged = analyze_results.aggregate_results(dataframe)

    assert len(averaged) == 1, "one condition must stay one row"
    assert averaged.loc[0, "n_replicates"] == 2
    assert averaged.loc[0, "solve_rate"] == 0.75
    # A fraction, so the reader can see that half the replicates are partial rather
    # than having to notice a missing row.
    assert averaged.loc[0, "eval_complete"] == 0.5
    assert averaged.loc[0, "gen_turn_limit_hit"] == 0.5


def test_collect_results_keeps_eval_complete_numeric(tmp_path: Path) -> None:
    """eval_complete survives collection as a number, not an object column."""
    _write_run(tmp_path / "a", 24, {"solve_rate": 0.5, "eval_complete": False})
    dataframe = analyze_results.collect_results([tmp_path])
    assert dataframe.loc[0, "eval_complete"] == 0.0
    assert "eval_complete" in dataframe.select_dtypes(include="number").columns
