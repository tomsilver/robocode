"""Budget regressions using native response records, without paid API calls."""

# Directly exercise accounting internals and invocation-tree termination.
# pylint: disable=protected-access

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig

from robocode.utils import rate_limit
from robocode.utils.backends.codex import CodexBackend
from robocode.utils.sandbox import SandboxConfig, _stream_result_to_sandbox_result
from robocode.utils.sandbox_types import GenerationMetrics, SandboxResult


def record(
    response_id: str,
    input_tokens: int = 0,
    cached_input_tokens: int = 0,
    output_tokens: int = 0,
) -> str:
    """Encode one authoritative native response record."""
    return (
        json.dumps(
            {
                "type": "token_usage_record",
                "payload": {
                    "response_id": response_id,
                    "usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": cached_input_tokens,
                        "output_tokens": output_tokens,
                    },
                },
            }
        )
        + "\n"
    )


def setup(tmp_path, budget=20, **options):
    """Create an experiment-local backend and native ledger location."""
    backend = CodexBackend(
        DictConfig(
            {
                "input_usd_per_mtok": 4,
                "cached_input_usd_per_mtok": 0.4,
                "output_usd_per_mtok": 20,
                **options,
            }
        )
    )
    backend.build_cli_cmd(SandboxConfig(sandbox_dir=tmp_path, max_budget_usd=budget))
    return backend, tmp_path / ".agent_sessions/codex/root.jsonl"


def process(events=()):
    """Build a completed CLI stream without starting a model process."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.stdout = iter(json.dumps(e) + "\n" for e in events)
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""
    proc.returncode = 0
    return proc


def test_reset_duplicate_and_partial_record(tmp_path):
    """Ignore resetting UI counters; deduplicate and buffer native records."""
    backend, path = setup(tmp_path)
    first = record("one", 1000, 900, 100)
    second = record("two", 500, 400, 200)
    path.write_text(first + second[:20])
    assert backend._read_session_usage()["input_tokens"] == 1000
    with path.open("a") as f:
        f.write(second[20:])
        f.write(
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {"total_token_usage": {"input_tokens": 1}},
                    },
                }
            )
            + "\n"
        )
    path.with_name("copy.jsonl").write_text(first)
    assert backend._read_session_usage() == {
        "input_tokens": 1500,
        "cached_input_tokens": 1300,
        "output_tokens": 300,
    }
    assert backend._usage_cost(backend._read_session_usage()) == pytest.approx(0.00732)


def test_resume_charges_new_root_and_subagent_responses(tmp_path):
    """A continuation charges both root and child responses exactly once."""
    backend, path = setup(tmp_path)
    path.write_text(record("old", output_tokens=500_000))  # $10 before resume
    backend.build_cli_cmd(
        SandboxConfig(
            sandbox_dir=tmp_path, max_budget_usd=10, resume_previous_session=True
        )
    )
    with path.open("a") as f:
        f.write(record("new", output_tokens=250_000))
    path.with_name("child.jsonl").write_text(record("child", output_tokens=250_000))
    proc = process([{"type": "turn.completed", "usage": {"input_tokens": 1}}])
    result = backend.parse_stream(proc)
    assert result.total_cost == 10
    assert result.stop_reason == "error_max_budget_usd"
    assert not result.unconfirmed_solution
    proc.kill.assert_called_once()


@pytest.mark.parametrize("remaining", [0, 0.49, 0.50, 0.51])
def test_small_remainder_and_last_code_evaluation(tmp_path, remaining):
    """Tiny remainders finish with saved code rather than requiring confidence."""
    backend, path = setup(tmp_path)
    path.write_text(record("root", output_tokens=round((20 - remaining) / 20 * 1e6)))
    (tmp_path / "approach.py").write_text("# last saved code\n")
    result = backend.parse_stream(process())
    assert result.unconfirmed_solution == (remaining > 0.5)
    converted = _stream_result_to_sandbox_result(result, tmp_path, "approach.py")
    assert converted.success == (remaining <= 0.5)
    if remaining <= 0.5:
        assert converted.output_file == tmp_path / "approach.py"


def test_budget_overrides_confidence_and_retry_error(tmp_path):
    """The hard budget stop wins over confidence and otherwise retryable exits."""
    backend, path = setup(tmp_path)
    path.write_text(record("root", output_tokens=1_000_000))
    path.with_name("solution_confident").touch()
    result = backend.parse_stream(
        process([{"type": "error", "message": "maximum output token limit"}])
    )
    assert result.stop_reason == "error_max_budget_usd"
    assert not result.output_token_limit_hit
    assert not result.unconfirmed_solution


@pytest.mark.parametrize("contents", ["", "not json\n", record("bad", -1)])
def test_missing_or_invalid_ledger_fails_closed(tmp_path, contents):
    """Invalid native accounting cannot fall back to cheaper stdout counters."""
    backend, path = setup(tmp_path)
    path.write_text(contents)
    result = backend.parse_stream(
        process([{"type": "turn.completed", "usage": {"input_tokens": 1000}}])
    )
    assert result.stop_reason == "error_budget_accounting"
    assert result.total_cost is None
    assert not result.unconfirmed_solution


def test_no_usage_timeout_kills_instead_of_running_unmetered(tmp_path):
    """An invocation without timely authoritative usage is stopped."""
    backend, _ = setup(tmp_path, usage_start_timeout_s=0)
    proc = process()
    result = backend.parse_stream(proc)
    proc.kill.assert_called_once()
    assert result.stop_reason == "error_budget_accounting"


def test_conflicting_duplicate_and_truncation(tmp_path):
    """Reject mutated or truncated accounting history."""
    backend, path = setup(tmp_path)
    path.write_text(record("one", 100))
    backend._read_session_usage()
    with path.open("a") as f:
        f.write(record("one", 200))
    with pytest.raises(ValueError, match="Conflicting"):
        backend._read_session_usage()
    path.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        backend._read_session_usage()


@pytest.mark.parametrize(
    "spent,threshold", [(0, None), (17.99, None), (18, 2), (19, 1), (20, 1)]
)
def test_budget_warning_file(tmp_path, spent, threshold):
    """Expose the appropriate advisory threshold at each boundary."""
    backend, path = setup(tmp_path)
    backend._write_budget_status(spent)
    status = json.loads(path.with_name("budget_status.json").read_text())
    assert status["remaining_usd"] == 20 - spent
    assert status["max_budget_usd"] == 20
    assert status["spent_usd"] == spent
    assert status["wrap_up"] == (spent >= 18)
    assert status["warning_threshold_usd"] == threshold


def test_resumed_budget_file_and_prompt_preserve_original_maximum(tmp_path):
    """Resumption refreshes the allowance but not the experiment maximum."""
    backend, path = setup(tmp_path)
    path.write_text(record("first", output_tokens=400_000))  # $8 already spent
    config = SandboxConfig(
        sandbox_dir=tmp_path, max_budget_usd=12, resume_previous_session=True
    )
    backend.build_cli_cmd(config)
    backend.setup_sandbox_files(config)
    status = json.loads(path.with_name("budget_status.json").read_text())
    assert status["max_budget_usd"] == 20
    assert status["spent_usd"] == 8
    assert status["remaining_usd"] == 12
    instructions = (tmp_path / "AGENTS.md").read_text()
    assert "maximum experiment budget is $20.00" in instructions
    assert "remaining budget for this invocation is $12.00" in instructions
    backend._write_budget_status(11)
    status = json.loads(path.with_name("budget_status.json").read_text())
    assert status["max_budget_usd"] == 20
    assert status["spent_usd"] == 19
    assert status["remaining_usd"] == 1
    assert status["warning_threshold_usd"] == 1


def test_retryable_exit_with_fifty_cents_left_evaluates_without_resume(
    tmp_path, monkeypatch
):
    """Apply the tiny-remainder rule even to a quota interruption."""
    backend, _ = setup(tmp_path)
    output = tmp_path / "approach.py"
    output.write_text("# best saved code\n")
    calls = []

    def run(config, _agent):
        calls.append(config)
        return SandboxResult(
            success=False,
            output_file=None,
            error="quota",
            total_cost_usd=19.5,
            rate_limit_reset="3am",
            generation_metrics=GenerationMetrics(),
        )

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    final = rate_limit.run_with_rate_limit_retry(
        None,
        SandboxConfig(
            sandbox_dir=tmp_path, max_budget_usd=20, output_filename="approach.py"
        ),
        backend,
    )
    assert len(calls) == 1
    assert final.success
    assert final.output_file == output
    assert final.total_cost_usd == 19.5
    assert final.generation_metrics.stop_reason == "error_max_budget_usd"


def test_kill_includes_descendant_in_separate_session():
    """Kill detached descendants but preserve an unrelated sibling process."""
    # A real local process tree, no Codex or network usage.
    script = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
        "start_new_session=True); print(p.pid,flush=True); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None
    child = int(proc.stdout.readline())
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        CodexBackend._kill_process_tree(proc)
        proc.wait(timeout=5)
        assert sibling.poll() is None, "Stopping one experiment killed a parallel one"
        for _ in range(100):
            status = Path(f"/proc/{child}/stat")
            try:
                state = status.read_text(encoding="utf-8").split()[2]
            except (FileNotFoundError, ProcessLookupError):
                break
            if state == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("Descendant still running after budget stop")
    finally:
        proc.kill()
        proc.wait()
        sibling.kill()
        sibling.wait()
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_five_parallel_budgets_are_isolated(tmp_path):
    """Independent runs may reuse response IDs without sharing their budgets."""
    barrier = Barrier(5)

    def run(index):
        backend, path = setup(tmp_path / str(index))
        # Deliberately reuse a response ID across separate experiments: dedup
        # must be per experiment, never global across the campaign.
        cost = 20 if index == 0 else index
        path.write_text(record("same-id", output_tokens=cost * 50_000))
        proc = process()
        barrier.wait(timeout=5)
        result = backend.parse_stream(proc)
        status = json.loads(path.with_name("budget_status.json").read_text())
        return result, proc.kill.call_count, status

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(run, range(5)))
    assert results[0][0].stop_reason == "error_max_budget_usd"
    assert results[0][1] == 1
    for index, (result, kills, status) in enumerate(results[1:], 1):
        assert result.total_cost == index
        assert result.unconfirmed_solution
        assert kills == 0
        assert status["remaining_usd"] == 20 - index


def test_apptainer_budget_stop_without_model_calls(tmp_path):
    """Opt-in real-container cutoff test with a synthetic token producer."""
    image = os.environ.get("ROBOCODE_TEST_APPTAINER_IMAGE")
    if not image:
        pytest.skip("Set ROBOCODE_TEST_APPTAINER_IMAGE for real-container test")
    backend, path = setup(tmp_path, budget=1)
    code = (
        "import pathlib,sys,time; "
        "pathlib.Path('/sandbox/approach.py').write_text('# saved before cutoff\\n'); "
        "pathlib.Path('/sandbox/.agent_sessions/codex/root.jsonl')"
        ".write_text(sys.argv[1]); "
        "print('{}',flush=True); time.sleep(30); "
        "pathlib.Path('/sandbox/should_not_exist').touch()"
    )
    with subprocess.Popen(
        [
            "apptainer",
            "exec",
            "--containall",
            "--pid",
            "--no-home",
            "--cleanenv",
            "--bind",
            f"{tmp_path}:/sandbox",
            "--pwd",
            "/sandbox",
            image,
            "python3",
            "-u",
            "-c",
            code,
            record("fake", output_tokens=50_000),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        result = backend.parse_stream(proc)
    assert result.stop_reason == "error_max_budget_usd", result.error_text
    assert result.total_cost == 1
    assert not (tmp_path / "should_not_exist").exists()
    assert path.exists()
    converted = _stream_result_to_sandbox_result(result, tmp_path, "approach.py")
    assert converted.success
