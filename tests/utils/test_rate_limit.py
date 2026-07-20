"""Tests for rate_limit.py."""

from pathlib import Path

from robocode.utils import rate_limit
from robocode.utils.rate_limit import (
    _fold_retry_metrics,
    parse_reset_hour,
    run_with_rate_limit_retry,
)
from robocode.utils.sandbox_types import GenerationMetrics, SandboxConfig, SandboxResult


def _metrics(tokens: int, turns: int = 0) -> GenerationMetrics:
    return GenerationMetrics(input_tokens=tokens, num_turns=turns)


def _rate_limited(tokens: int, cost: float, turns: int = 0) -> SandboxResult:
    return SandboxResult(
        success=False,
        output_file=None,
        error="Rate-limited: resets 3am",
        total_cost_usd=cost,
        rate_limit_reset="3am",
        generation_metrics=_metrics(tokens, turns),
    )


def _succeeded(tokens: int, cost: float, path: Path, turns: int = 0) -> SandboxResult:
    return SandboxResult(
        success=True,
        output_file=path,
        error=None,
        total_cost_usd=cost,
        generation_metrics=_metrics(tokens, turns),
    )


def test_fold_retry_metrics_noop_when_no_retries() -> None:
    """With zero retries the result is returned untouched."""
    result = _succeeded(300, 2.0, Path("approach.py"))
    assert _fold_retry_metrics(result, 0, 0.0, 0) is result


def test_fold_retry_metrics_records_aborted_spend() -> None:
    """Retry count and aborted spend attach without disturbing final metrics."""
    result = _succeeded(300, 2.0, Path("approach.py"))
    folded = _fold_retry_metrics(
        result, aborted_tokens=300, aborted_cost=1.5, retries=2
    )
    assert folded.generation_metrics is not None
    assert folded.generation_metrics.rate_limit_retries == 2
    assert folded.generation_metrics.aborted_tokens == 300
    assert folded.generation_metrics.aborted_cost_usd == 1.5
    # The final attempt's own totals are preserved.
    assert folded.generation_metrics.input_tokens == 300
    assert folded.total_cost_usd == 2.0


def test_retry_loop_accumulates_aborted_attempts(tmp_path: Path, monkeypatch) -> None:
    """Two rate-limited attempts then success: their spend is summed and counted."""
    approach = tmp_path / "approach.py"
    approach.write_text("x = 1\n")
    results = iter(
        [
            _rate_limited(100, 0.5),
            _rate_limited(200, 1.0),
            _succeeded(300, 2.0, approach),
        ]
    )

    async def fake_run(*_args, **_kwargs) -> SandboxResult:
        return next(results)

    monkeypatch.setattr(rate_limit, "run_agent_in_sandbox", fake_run)
    monkeypatch.setattr(rate_limit.time, "sleep", lambda _s: None)

    config = SandboxConfig(sandbox_dir=tmp_path)
    final = run_with_rate_limit_retry(
        None, config, backend=None  # type: ignore[arg-type]
    )

    assert final.success
    assert final.generation_metrics is not None
    assert final.generation_metrics.rate_limit_retries == 2
    assert final.generation_metrics.aborted_tokens == 300  # 100 + 200
    assert final.generation_metrics.aborted_cost_usd == 1.5  # 0.5 + 1.0
    assert final.generation_metrics.input_tokens == 300  # final attempt only
    assert final.total_cost_usd == 2.0


def test_retry_loop_resumes_with_carried_budget(tmp_path: Path, monkeypatch) -> None:
    """Retries resume the session and continue with the budget/turns left over."""
    approach = tmp_path / "approach.py"
    approach.write_text("x = 1\n")
    results = iter(
        [
            _rate_limited(100, 0.5, turns=20),
            _rate_limited(200, 1.0, turns=10),
            _succeeded(300, 2.0, approach, turns=5),
        ]
    )
    captured: list[SandboxConfig] = []

    async def fake_run(config: SandboxConfig, _backend) -> SandboxResult:
        captured.append(config)
        return next(results)

    monkeypatch.setattr(rate_limit, "run_agent_in_sandbox", fake_run)
    monkeypatch.setattr(rate_limit.time, "sleep", lambda _s: None)

    config = SandboxConfig(sandbox_dir=tmp_path, max_budget_usd=5.0, max_turns=50)
    run_with_rate_limit_retry(None, config, backend=None)  # type: ignore[arg-type]

    # First attempt runs fresh; each retry resumes with the budget/turns left.
    assert captured[0].resume_previous_session is False
    assert captured[1].resume_previous_session is True
    assert captured[1].max_budget_usd == 4.5  # 5.0 - 0.5
    assert captured[1].max_turns == 30  # 50 - 20
    assert captured[2].resume_previous_session is True
    assert captured[2].max_budget_usd == 3.5  # 5.0 - (0.5 + 1.0)
    assert captured[2].max_turns == 20  # 50 - (20 + 10)


def test_parse_reset_hour_pm_utc() -> None:
    """A full usage message parses the reset hour and UTC flag."""
    hour, is_utc = parse_reset_hour("You've hit your session limit. resets 11pm (UTC)")
    assert hour == 23
    assert is_utc is True
