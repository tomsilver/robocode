"""Actual subprocess/runner/parser/retry integration, with a deterministic CLI.

Only the model executable is replaced. These tests make zero paid calls and
exercise real session files, prompt rendering, git saves, and process stopping.
They test the protocol contract, not whether an LLM obeys the prompt.

Every scenario is parametrized over local and strict Docker runners. Docker
cases require the robocode-strict-blackbox image (built for the host UID/GID)
and a usable daemon; otherwise they skip. They keep the production entrypoint,
firewall, volume mounts and cleanup. No model credentials are required.
Run only Docker cases on a Docker-enabled machine with:
    pytest tests/utils/test_codex_budget_integration.py -k docker
"""

import json
import os
import runpy
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from omegaconf import DictConfig

from robocode.utils.backends.codex import CodexBackend
from robocode.utils.docker_sandbox import DockerSandboxConfig
from robocode.utils.rate_limit import run_with_rate_limit_retry
from robocode.utils.sandbox_types import SandboxConfig
from robocode.utils.strict_blackbox import STRICT_BLACKBOX_IMAGE

POLICY = "def score():\n    return 1\n"


@pytest.fixture(autouse=True, params=["local", "docker"])
def fake_cli(tmp_path, monkeypatch, request):
    """Use a stdlib-only executable; never load host authentication."""
    source = Path(__file__).parent / "fixtures/fake_codex_cli.py"
    executable = tmp_path / "fake-codex"
    shutil.copyfile(source, executable)
    executable.chmod(0o755)
    transport = request.param
    containers = []
    if transport == "docker":
        if shutil.which("docker") is None:
            pytest.skip("Docker CLI unavailable")
        available = subprocess.run(
            ["docker", "image", "inspect", STRICT_BLACKBOX_IMAGE],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if available.returncode:
            pytest.skip(f"Docker daemon/image {STRICT_BLACKBOX_IMAGE} unavailable")
        # Record actual production container names; do not replace the launcher,
        # entrypoint, firewall, mounts, CLI stream, parser, or budget monitor.
        from robocode.utils import (
            docker_sandbox,
        )  # pylint: disable=import-outside-toplevel

        original = docker_sandbox._docker_run_prefix  # pylint: disable=protected-access

        def record_container(*args, **kwargs):
            containers.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(docker_sandbox, "_docker_run_prefix", record_container)
    monkeypatch.setenv("ROBOCODE_TEST_CODEX_TRANSPORT", transport)
    monkeypatch.setenv("ROBOCODE_TEST_FAKE_CODEX", str(executable))
    monkeypatch.setenv(
        "ROBOCODE_CODEX_CMD",
        "/sandbox/fake-codex" if transport == "docker" else str(executable),
    )
    monkeypatch.setenv("CODEX_API_KEY", "fake-test-key-not-a-credential")
    yield
    for name in containers:
        result = subprocess.run(
            ["docker", "inspect", name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            subprocess.run(["docker", "rm", "-f", name], check=False, timeout=10)
            pytest.fail(f"Production runner failed to remove container {name}")
        assert "No such" in result.stderr, result.stderr


def run_plan(directory, steps, budget=5):
    """Run the production local/Docker runner and retry loop against a fake CLI."""
    directory.mkdir(parents=True)
    plan = directory / "plan.json"
    plan.write_text(json.dumps(steps))
    docker = os.environ["ROBOCODE_TEST_CODEX_TRANSPORT"] == "docker"
    files = {"fake_plan.json": plan}
    if docker:
        files["fake-codex"] = Path(os.environ["ROBOCODE_TEST_FAKE_CODEX"])
        # This protocol-only task never contacts an env server. Strict Docker's
        # production firewall still requires a specific allowed host port.
        metadata = directory / "env_spaces.json"
        metadata.write_text(json.dumps({"port": 9}))
        files["env_spaces.json"] = metadata
    cls = DockerSandboxConfig if docker else SandboxConfig
    config = cls(
        sandbox_dir=directory / "sandbox",
        init_files=files,
        output_filename="approach.py",
        prompt="Improve the saved policy.",
        model="gpt-5.6-sol",
        max_budget_usd=budget,
        **({"blackbox": True, "blackbox_strict": True} if docker else {}),
    )
    backend = CodexBackend(
        DictConfig(
            {
                "input_usd_per_mtok": 4,
                "cached_input_usd_per_mtok": 0.4,
                "output_usd_per_mtok": 20,
                "usage_start_timeout_s": 120 if docker else 5,
            }
        )
    )
    result = run_with_rate_limit_retry(
        config if docker else None, None if docker else config, backend
    )
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
