"""Tests for final generated-program complexity analysis."""

import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "analyze_program_complexity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_program_complexity", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
analysis: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


SOURCE = """class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.phase = 0
    def reset(self, state, info):
        self.phase = 1
    def get_action(self, state):
        if self.phase and state is not None:
            return 1
        return 0
"""


def test_source_metrics_count_decisions_and_state() -> None:
    metrics = analysis._source_metrics(SOURCE)
    assert metrics["syntax_valid"] is True
    assert metrics["function_count"] == 3
    assert metrics["get_action_cyclomatic"] == 3
    assert metrics["persistent_state_fields"] == 1
    assert metrics["max_nesting_depth"] == 1


def test_collects_final_agentic_program_from_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "example__agentic__run.zip"
    root = "campaign/run/replicate_42"
    config = (
        "replicate_seed: 42\n"
        "approach:\n  _target_: robocode.approaches.agentic_approach.AgenticApproach\n"
        "environment:\n  _target_: example.ExampleEnv\n"
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{root}/sandbox/approach.py", SOURCE)
        archive.writestr(f"{root}/sandbox/impl1_candidate.py", SOURCE * 2)
        archive.writestr(f"{root}/.hydra/config.yaml", config)
        archive.writestr(
            f"{root}/.hydra/overrides.yaml",
            "- environment=example_generalized\n- replicate_seed=42\n",
        )
        archive.writestr(f"{root}/results.json", json.dumps({"solve_rate": 0.75}))

    frame = analysis.collect_complexity([archive_path])

    assert len(frame) == 1
    assert frame.loc[0, "approach"] == "agentic"
    assert frame.loc[0, "replicate_seed"] == 42
    assert frame.loc[0, "environment"] == "example_generalized"
    assert frame.loc[0, "result_solve_rate"] == 0.75
    assert "approach.py" in frame.loc[0, "source"]


def test_collects_final_genplan_program_from_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "example__llm_genplan__run.zip"
    root = "campaign/run/replicate_42"
    config = (
        "replicate_seed: 42\n"
        "approach:\n"
        "  _target_: robocode.approaches.llm_genplan_approach.LLMGenPlanApproach\n"
        "  completion:\n"
        "    model: claude-opus-5\n"
        "environment:\n  _target_: example.ExampleEnv\n"
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{root}/sandbox/approach.py", SOURCE)
        archive.writestr(f"{root}/.hydra/config.yaml", config)

    frame = analysis.collect_complexity([archive_path])

    assert len(frame) == 1
    assert frame.loc[0, "approach"] == "llm_genplan"
    assert frame.loc[0, "backend"] == "claude-opus-5"


def test_skips_malformed_archives_in_directory(tmp_path: Path) -> None:
    (tmp_path / "broken__agentic__run.zip").write_text("not a zip")

    frame = analysis.collect_complexity([tmp_path])

    assert frame.empty


def test_syntax_errors_are_reported_not_executed() -> None:
    metrics = analysis._source_metrics("raise RuntimeError('must not run')\nif:")
    assert metrics["syntax_valid"] is False
    assert metrics["syntax_error"]


def test_summary_has_mean_and_sample_std() -> None:
    frame = analysis.pd.DataFrame(
        [
            {
                "approach": "agentic",
                "environment": "env",
                "access": "whitebox",
                "backend": "model",
                "has_results": True,
                "source_loc": 10,
            },
            {
                "approach": "agentic",
                "environment": "env",
                "access": "whitebox",
                "backend": "model",
                "has_results": False,
                "source_loc": 14,
            },
        ]
    )
    summary = analysis.summarize_complexity(frame)
    assert summary.loc[0, "n_programs"] == 2
    assert summary.loc[0, "n_with_results"] == 1
    assert summary.loc[0, "source_loc_mean"] == 12
    assert summary.loc[0, "source_loc_std"] == 2**0.5 * 2
