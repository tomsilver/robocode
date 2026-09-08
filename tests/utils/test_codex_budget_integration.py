"""Actual subprocess/runner/parser/retry integration, with a deterministic CLI.

Only the model executable is replaced. These tests make zero paid calls and
exercise real session files, prompt rendering, git saves, and process stopping.
They test the protocol contract, not whether an LLM obeys the prompt.
"""

import json
import runpy
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from omegaconf import DictConfig

from robocode.utils.backends.codex import CodexBackend
from robocode.utils.rate_limit import run_with_rate_limit_retry
from robocode.utils.sandbox_types import SandboxConfig

POLICY = "def score():\n    return 1\n"


@pytest.fixture(autouse=True)
def fake_cli(tmp_path, monkeypatch):
    """Use a stdlib-only executable; never load host authentication."""
    source = Path(__file__).parent / "fixtures/fake_codex_cli.py"
    executable = tmp_path / "fake-codex"
    shutil.copyfile(source, executable)
    executable.chmod(0o755)
    monkeypatch.setenv("ROBOCODE_CODEX_CMD", str(executable))
    monkeypatch.setenv("CODEX_API_KEY", "fake-test-key-not-a-credential")
    return executable


def run_plan(directory, steps, budget=5):
    """Run the production local runner and retry loop against a fake CLI."""
    directory.mkdir(parents=True)
    plan = directory / "plan.json"
    plan.write_text(json.dumps(steps))
    config = SandboxConfig(
        sandbox_dir=directory / "sandbox",
        init_files={"fake_plan.json": plan},
        output_filename="approach.py",
        prompt="Improve the saved policy.",
        model="gpt-5.6-sol",
        max_budget_usd=budget,
    )
    backend = CodexBackend(
        DictConfig(
            {
                "input_usd_per_mtok": 4,
                "cached_input_usd_per_mtok": 0.4,
                "output_usd_per_mtok": 20,
                "usage_start_timeout_s": 5,
            }
        )
    )
    result = run_with_rate_limit_retry(None, config, backend)
    trace = [
        json.loads(line)
        for line in (config.sandbox_dir / "trace.jsonl").read_text().splitlines()
    ]
    return result, trace, config.sandbox_dir


def test_unchanged_policy_validation_can_continue(tmp_path):
    """Useful validation need not change policy or finish within two retries."""
    result, trace, sandbox = run_plan(
        tmp_path / "loop",
        [
            {"cost": 3.2, "policy": POLICY},
            {"cost": 0.1, "validate": True, "assert_finishing_contract": True},
            {"cost": 0.1, "validate": True, "assert_finishing_contract": True},
            {"cost": 0.1, "validate": True, "confident": True},
        ],
    )
    assert len(trace) == 4
    assert [t["resume"] for t in trace] == [False, True, True, True]
    assert [t["status"]["remaining_usd"] for t in trace] == pytest.approx(
        [5, 1.8, 1.7, 1.6]
    )
    assert result.total_cost_usd == pytest.approx(3.5)
    assert result.generation_metrics.stop_reason is None
    assert result.generation_metrics.unconfirmed_solution_retries == 3
    assert (sandbox / "validation.txt").read_text().splitlines() == ["passed"] * 3
    assert result.success
    assert runpy.run_path(str(result.output_file))["score"]() == 1


def test_warning_allows_targeted_work_then_confident_exit(tmp_path):
    """A resumed finishing pass can change the policy and genuinely confirm."""
    result, trace, _ = run_plan(
        tmp_path / "finish",
        [
            {"cost": 3.2, "policy": POLICY, "assert_finishing_contract": True},
            {
                "cost": 0.1,
                "policy": POLICY.replace("1", "2"),
                "assert_finishing_contract": True,
            },
            {"cost": 0.1, "confident": True, "assert_finishing_contract": True},
        ],
    )
    assert len(trace) == 3
    assert result.total_cost_usd == pytest.approx(3.4)
    assert result.generation_metrics.stop_reason is None
    assert runpy.run_path(str(result.output_file))["score"]() == 2


def test_low_confidence_iterations_continue_with_budget(tmp_path):
    """Resume low-confidence exits without an artificial iteration cap."""
    result, trace, _ = run_plan(
        tmp_path / "progress",
        [
            {"cost": 1, "policy": POLICY},
            {"cost": 0.1},
            {"cost": 0.1, "policy": POLICY.replace("1", "3")},
            {"cost": 0.1},
            {"cost": 0.1, "confident": True},
        ],
    )
    assert len(trace) == 5
    assert result.total_cost_usd == pytest.approx(1.4)
    assert result.generation_metrics.stop_reason is None


def test_tiny_remainder_stops_resuming(tmp_path):
    """Do not start another invocation with only fifty cents remaining."""
    result, trace, _ = run_plan(
        tmp_path / "tiny",
        [
            {"cost": 4.2, "policy": POLICY},
            {"cost": 0.2},
            {"cost": 0.1},
        ],
    )
    assert len(trace) == 3
    assert result.total_cost_usd == pytest.approx(4.5)
    assert result.generation_metrics.stop_reason == "error_max_budget_usd"


def test_real_process_hard_cutoff_includes_subagent_and_saved_code(tmp_path):
    """Kill while the CLI and setsid child are active, then load saved code."""
    result, trace, sandbox = run_plan(
        tmp_path / "kill",
        [
            {
                "cost": 3,
                "child_cost": 2,
                "policy": POLICY,
                "child": True,
                "wait_for_kill": True,
            },
        ],
    )
    assert len(trace) == 1
    assert result.total_cost_usd == pytest.approx(5)
    assert result.generation_metrics.stop_reason == "error_max_budget_usd"
    assert not (sandbox / "survived_cutoff").exists()
    assert runpy.run_path(str(result.output_file))["score"]() == 1


@pytest.mark.parametrize(
    "fault",
    ["delete_ledger", "malformed_ledger", "model_change", "status_write_failure"],
)
def test_mid_run_accounting_fault_kills_without_retry(tmp_path, fault):
    """A live monitoring failure cannot leave the model running unmetered."""
    result, trace, sandbox = run_plan(
        tmp_path / fault,
        [
            {"cost": 0.1, "policy": POLICY, "fault": fault},
        ],
    )
    assert len(trace) == 1
    assert result.total_cost_usd is None
    assert result.generation_metrics.stop_reason == "error_budget_accounting"
    assert not (sandbox / "survived_cutoff").exists()


def test_parallel_cutoff_and_success_are_isolated(tmp_path):
    """Stopping one experiment must not stop or contaminate its neighbor."""
    plans = [
        [
            {"cost": 3.2, "policy": POLICY},
            {"cost": 0.8},
            {"cost": 1, "wait_for_kill": True},
        ],
        [{"cost": 0.5, "policy": POLICY, "confident": True}],
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_plan, tmp_path / str(i), plan)
            for i, plan in enumerate(plans)
        ]
        results = [f.result(timeout=30) for f in futures]
    assert results[0][0].generation_metrics.stop_reason == "error_max_budget_usd"
    assert results[0][0].total_cost_usd == pytest.approx(5)
    assert results[1][0].generation_metrics.stop_reason is None
    assert results[1][0].total_cost_usd == 0.5
    assert len(results[1][1]) == 1
