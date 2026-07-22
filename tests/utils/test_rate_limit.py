"""Tests for rate_limit.py."""

from pathlib import Path

import pytest

from robocode.utils import rate_limit
from robocode.utils.rate_limit import (
    _fold_retry_metrics,
    parse_reset_hour,
    parse_reset_time,
    run_with_rate_limit_retry,
    seconds_until_reset,
)
from robocode.utils.sandbox_types import GenerationMetrics, SandboxConfig, SandboxResult


def _metrics(tokens: int, turns: int = 0) -> GenerationMetrics:
    return GenerationMetrics(input_tokens=tokens, num_turns=turns)


def _rate_limited(tokens: int, cost: float | None, turns: int = 0) -> SandboxResult:
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


def _output_limited(tokens: int, cost: float | None, turns: int = 0) -> SandboxResult:
    return SandboxResult(
        success=False,
        output_file=None,
        error="Claude response exceeded the output token maximum",
        total_cost_usd=cost,
        output_token_limit_hit=True,
        generation_metrics=_metrics(tokens, turns),
    )


def _prompt_too_long(tokens: int, cost: float | None, turns: int = 0) -> SandboxResult:
    return SandboxResult(
        success=False,
        output_file=None,
        error="Prompt is too long",
        total_cost_usd=cost,
        prompt_too_long_hit=True,
        generation_metrics=_metrics(tokens, turns),
    )


def _compacted(tokens: int, cost: float | None, turns: int = 0) -> SandboxResult:
    return SandboxResult(
        success=False,
        output_file=None,
        error="output file not found",
        total_cost_usd=cost,
        generation_metrics=GenerationMetrics(
            input_tokens=tokens, num_turns=turns, num_autocompactions=1
        ),
    )


def test_fold_retry_metrics_noop_when_no_retries() -> None:
    """With zero retries the result is returned untouched."""
    result = _succeeded(300, 2.0, Path("approach.py"))
    assert _fold_retry_metrics(result, 0, 0.0, 0) is result


def test_fold_retry_metrics_records_aborted_spend() -> None:
    """Retry count and aborted spend attach without disturbing final metrics."""
    result = _succeeded(300, 2.0, Path("approach.py"))
    folded = _fold_retry_metrics(
        result, aborted_tokens=300, aborted_cost=1.5, rate_limit_retries=2
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


def test_retry_does_not_turn_exhausted_limits_into_unlimited(
    tmp_path: Path, monkeypatch
) -> None:
    """An exhausted positive limit stops instead of becoming unlimited zero."""
    results = iter([_rate_limited(100, 5.0, turns=50)])
    captured: list[SandboxConfig] = []
    slept: list[float] = []

    async def fake_run(config: SandboxConfig, _backend) -> SandboxResult:
        captured.append(config)
        return next(results)

    monkeypatch.setattr(rate_limit, "run_agent_in_sandbox", fake_run)
    monkeypatch.setattr(rate_limit.time, "sleep", slept.append)
    config = SandboxConfig(sandbox_dir=tmp_path, max_budget_usd=5.0, max_turns=50)

    final = run_with_rate_limit_retry(
        None, config, backend=None  # type: ignore[arg-type]
    )

    assert not final.success
    assert len(captured) == 1
    assert not slept
    assert final.generation_metrics is not None
    assert final.generation_metrics.rate_limit_retries == 1


def test_retry_stops_when_interrupted_cost_is_unknown(
    tmp_path: Path, monkeypatch
) -> None:
    """A finite budget cannot be carried safely without an attempt cost."""
    results = iter([_rate_limited(100, None)])
    captured: list[SandboxConfig] = []

    async def fake_run(config: SandboxConfig, _backend) -> SandboxResult:
        captured.append(config)
        return next(results)

    monkeypatch.setattr(rate_limit, "run_agent_in_sandbox", fake_run)
    monkeypatch.setattr(
        rate_limit.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )

    final = run_with_rate_limit_retry(
        None,
        SandboxConfig(sandbox_dir=tmp_path, max_budget_usd=5.0),
        backend=None,  # type: ignore[arg-type]
    )

    assert not final.success
    assert len(captured) == 1


def test_output_token_limit_resumes_with_remaining_budget(
    tmp_path: Path, monkeypatch
) -> None:
    """An oversized response resumes immediately with a concise prompt."""
    approach = tmp_path / "approach.py"
    approach.write_text("x = 1\n")
    results = iter(
        [
            _output_limited(100, 0.75, turns=4),
            _succeeded(200, 1.0, approach, turns=2),
        ]
    )
    captured: list[SandboxConfig] = []

    async def fake_run(config: SandboxConfig, _backend) -> SandboxResult:
        captured.append(config)
        return next(results)

    monkeypatch.setattr(rate_limit, "run_agent_in_sandbox", fake_run)
    monkeypatch.setattr(
        rate_limit.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )
    config = SandboxConfig(sandbox_dir=tmp_path, max_budget_usd=5.0, max_turns=20)

    final = run_with_rate_limit_retry(
        None, config, backend=None  # type: ignore[arg-type]
    )

    assert final.success
    assert len(captured) == 2
    assert captured[1].resume_previous_session
    assert captured[1].max_budget_usd == 4.25
    assert captured[1].max_turns == 16
    assert "Be concise" in captured[1].prompt
    assert final.generation_metrics is not None
    assert final.generation_metrics.output_token_retries == 1
    assert final.generation_metrics.aborted_tokens == 100
    assert final.generation_metrics.aborted_cost_usd == 0.75


def test_output_token_limit_retry_count_is_bounded(tmp_path: Path, monkeypatch) -> None:
    """Repeated oversized responses cannot loop forever with unlimited budgets."""
    results = iter([_output_limited(10, 0.1) for _ in range(3)])
    captured: list[SandboxConfig] = []

    async def fake_run(config: SandboxConfig, _backend) -> SandboxResult:
        captured.append(config)
        return next(results)

    monkeypatch.setattr(rate_limit, "run_agent_in_sandbox", fake_run)
    final = run_with_rate_limit_retry(
        None,
        SandboxConfig(sandbox_dir=tmp_path, max_budget_usd=0.0, max_turns=0),
        backend=None,  # type: ignore[arg-type]
    )

    assert not final.success
    assert len(captured) == 3  # initial attempt plus two resumptions
    assert final.generation_metrics is not None
    assert final.generation_metrics.output_token_retries == 2
    assert final.generation_metrics.aborted_cost_usd == pytest.approx(0.3)


def test_prompt_too_long_compacts_then_resumes_with_remaining_budget(
    tmp_path: Path, monkeypatch
) -> None:
    """An oversized context is compacted before work resumes."""
    approach = tmp_path / "approach.py"
    approach.write_text("x = 1\n")
    results = iter(
        [
            _prompt_too_long(100, 0.5, turns=4),
            _compacted(50, 0.25),
            _succeeded(200, 1.0, approach, turns=2),
        ]
    )
    captured: list[SandboxConfig] = []

    async def fake_run(config: SandboxConfig, _backend) -> SandboxResult:
        captured.append(config)
        return next(results)

    monkeypatch.setattr(rate_limit, "run_agent_in_sandbox", fake_run)
    config = SandboxConfig(sandbox_dir=tmp_path, max_budget_usd=5.0, max_turns=20)

    final = run_with_rate_limit_retry(
        None, config, backend=None  # type: ignore[arg-type]
    )

    assert final.success
    assert len(captured) == 3
    assert captured[1].resume_previous_session
    assert captured[1].prompt.startswith("/compact ")
    assert captured[1].max_budget_usd == 4.5
    assert captured[1].max_turns == 16
    assert captured[2].resume_previous_session
    assert "compacted conversation" in captured[2].prompt
    assert captured[2].max_budget_usd == 4.25
    assert captured[2].max_turns == 16
    assert final.generation_metrics is not None
    assert final.generation_metrics.prompt_too_long_retries == 1
    assert final.generation_metrics.aborted_tokens == 150
    assert final.generation_metrics.aborted_cost_usd == 0.75


def test_prompt_too_long_stops_if_compaction_is_not_confirmed(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed /compact is not mistaken for a safe continuation."""
    results = iter(
        [
            _prompt_too_long(100, 0.5),
            SandboxResult(
                success=False,
                output_file=None,
                error="Error during compaction: Conversation too long",
                total_cost_usd=0.25,
                generation_metrics=_metrics(50),
            ),
        ]
    )
    captured: list[SandboxConfig] = []

    async def fake_run(config: SandboxConfig, _backend) -> SandboxResult:
        captured.append(config)
        return next(results)

    monkeypatch.setattr(rate_limit, "run_agent_in_sandbox", fake_run)
    final = run_with_rate_limit_retry(
        None,
        SandboxConfig(sandbox_dir=tmp_path, max_budget_usd=5.0),
        backend=None,  # type: ignore[arg-type]
    )

    assert not final.success
    assert final.prompt_too_long_hit
    assert len(captured) == 2
    assert final.generation_metrics is not None
    assert final.generation_metrics.prompt_too_long_hit
    assert final.generation_metrics.prompt_too_long_retries == 0
    assert final.generation_metrics.aborted_cost_usd == 0.75


def test_prompt_too_long_compaction_retries_are_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    """Repeated context exhaustion cannot compact forever without limits."""
    results = iter(
        [
            _prompt_too_long(10, 0.1),
            _compacted(10, 0.1),
            _prompt_too_long(10, 0.1),
            _compacted(10, 0.1),
            _prompt_too_long(10, 0.1),
        ]
    )
    captured: list[SandboxConfig] = []

    async def fake_run(config: SandboxConfig, _backend) -> SandboxResult:
        captured.append(config)
        return next(results)

    monkeypatch.setattr(rate_limit, "run_agent_in_sandbox", fake_run)
    final = run_with_rate_limit_retry(
        None,
        SandboxConfig(sandbox_dir=tmp_path, max_budget_usd=0.0, max_turns=0),
        backend=None,  # type: ignore[arg-type]
    )

    assert not final.success
    assert len(captured) == 5
    assert final.generation_metrics is not None
    assert final.generation_metrics.prompt_too_long_retries == 2
    assert final.generation_metrics.aborted_cost_usd == pytest.approx(0.5)


def test_parse_reset_hour_pm_utc() -> None:
    """A full usage message parses the reset hour and UTC flag."""
    hour, is_utc = parse_reset_hour("You've hit your session limit. resets 11pm (UTC)")
    assert hour == 23
    assert is_utc is True


def test_parse_reset_time_with_minutes() -> None:
    """Minute-bearing session resets retain the minute and timezone."""
    parsed = parse_reset_time("You've hit your session limit · resets 1:30pm (UTC)")
    assert parsed == (13, 30, True)


def test_seconds_until_minute_bearing_reset(monkeypatch) -> None:
    """The sleep targets five minutes after the parsed reset minute."""

    class FixedDateTime(rate_limit.datetime):
        """Datetime with a stable current time for reset calculations."""

        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 22, 12, 0, tzinfo=tz)

    monkeypatch.setattr(rate_limit, "datetime", FixedDateTime)
    assert seconds_until_reset(13, is_utc=True, reset_minute=30) == 95 * 60
