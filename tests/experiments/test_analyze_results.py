"""Tests for collecting experiment results under the replicate-seed protocol."""

import importlib.util
import json
from pathlib import Path
from typing import Any

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

    dataframe = analyze_results._collect_results([tmp_path])

    assert dataframe.loc[0, "replicate_seed"] == 24
    assert dataframe.loc[0, "design_count_solve_rate"] == 0.75
    assert dataframe.loc[0, "held_out_count_solve_rate"] == 0.25
    assert dataframe.loc[0, "solve_rate@1"] == 0.75
    assert "count_regimes" not in dataframe.columns
    assert "per_episode" not in dataframe.columns
