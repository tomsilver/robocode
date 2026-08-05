"""Tests for the telemetry utility."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box

from robocode.utils import telemetry as tel
from robocode.utils.telemetry import (
    TelemetryNotInstrumentedError,
    enabled,
    fingerprint,
    instrument_class,
    instrument_env,
    log_event,
    require_instrumented,
)


class _ToyEnv(Env):
    """Env with get_state/set_state that terminates once position reaches 3.0.

    ``get_state`` counts its calls so a test can assert telemetry never invokes it
    (the fingerprint comes from the obs/arg, not a second get_state).
    """

    def __init__(self) -> None:
        self.observation_space = Box(0.0, 10.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._pos = 0.0
        self.get_state_calls = 0

    def reset(self, *, seed: Any = None, options: Any = None) -> Any:
        super().reset(seed=seed)
        self._pos = 0.0
        return np.array([self._pos], dtype=np.float32), {}

    def step(self, action: Any) -> Any:
        del action
        self._pos += 1.0
        return np.array([self._pos], np.float32), 0.0, self._pos >= 3.0, False, {}

    def get_state(self) -> Any:
        """Return the position; counts calls to detect telemetry side effects."""
        self.get_state_calls += 1
        return np.array([self._pos], dtype=np.float32)

    def set_state(self, state: Any) -> None:
        """Restore the position from a snapshot."""
        self._pos = float(state[0])

    def render(self) -> None:
        return None


@pytest.fixture()
def sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[dict]]:
    """Enable telemetry to a temp sink; return a reader for the emitted events."""
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("ROBOCODE_TELEMETRY", str(path))
    monkeypatch.setenv("ROBOCODE_RUN_ID", "test-run")

    def read() -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    return read


def test_disabled_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the sink env var unset, nothing is written and envs are not wrapped."""
    monkeypatch.delenv("ROBOCODE_TELEMETRY", raising=False)
    assert not enabled()
    log_event("reset", seed=1)
    env = instrument_env(_ToyEnv())
    env.reset(seed=1)
    env.step(env.action_space.sample())
    assert env.reset.__name__ == "reset" and not hasattr(env.reset, "__wrapped__")


def test_log_event_shape_and_run_id(sink: Callable[[], list[dict]]) -> None:
    """A logged event carries kind + reserved fields and the run id."""
    log_event("custom", foo=1)
    (event,) = sink()
    assert event["kind"] == "custom" and event["foo"] == 1
    assert event["run"] == "test-run" and event["schema"] == 1
    assert {"seq", "ts_ns", "pid"} <= event.keys()


def test_reserved_fields_cannot_be_shadowed(sink: Callable[[], list[dict]]) -> None:
    """A caller field named like a reserved key does not overwrite the real one.

    (``kind`` is a positional parameter, so Python already blocks shadowing it;
    this covers the ``**fields`` reserved keys.)
    """
    log_event("k", seq="evil", pid="evil", run="evil")
    (event,) = sink()
    assert event["kind"] == "k"
    assert isinstance(event["seq"], int) and isinstance(event["pid"], int)
    assert event["run"] == "test-run"  # the real run id, not the caller's


def test_log_event_never_raises_on_bad_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unwritable sink path is swallowed, not raised."""
    monkeypatch.setenv("ROBOCODE_TELEMETRY", "/nonexistent-dir/telemetry.jsonl")
    log_event("reset")  # must not raise


def test_fingerprint_array_is_canonical_and_shape_sensitive() -> None:
    """Array fingerprints include dtype+shape, so different shapes differ."""
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert fingerprint(a) == fingerprint(np.array([1.0, 2.0], dtype=np.float32))
    assert fingerprint(a) != fingerprint(np.array([[1.0, 2.0]], dtype=np.float32))
    assert fingerprint(None) is None


def test_instrument_env_emits_events(sink: Callable[[], list[dict]]) -> None:
    """Reset, set_state, and a terminal step each emit their event."""
    env = instrument_env(_ToyEnv(), label="x")
    env.reset(seed=4, options={"object_count": 2})
    env.set_state(np.array([1.0], dtype=np.float32))
    for _ in range(2):  # pos 1 -> 2 -> 3 terminates on the second step
        env.step(env.action_space.sample())
    kinds = [e["kind"] for e in sink()]
    assert kinds == ["reset", "set_state", "episode_end"]
    reset_event = sink()[0]
    assert reset_event["seed"] == 4 and reset_event["options"] == {"object_count": 2}


def test_episode_end_is_logged_once(sink: Callable[[], list[dict]]) -> None:
    """Stepping past termination does not log a second episode_end."""
    env = instrument_env(_ToyEnv())
    env.reset(seed=0)
    for _ in range(6):  # terminates at 3 steps; keep going
        env.step(env.action_space.sample())
    ends = [e for e in sink() if e["kind"] == "episode_end"]
    assert len(ends) == 1 and ends[0]["num_steps"] == 3


@pytest.mark.usefixtures("sink")
def test_telemetry_does_not_call_get_state() -> None:
    """Fingerprints come from obs/args, so telemetry adds no get_state call."""
    env = instrument_env(_ToyEnv())
    env.reset(seed=0)
    env.set_state(np.array([1.0], dtype=np.float32))
    env.step(env.action_space.sample())
    assert env.get_state_calls == 0


@pytest.mark.usefixtures("sink")
def test_instrument_env_preserves_return_values() -> None:
    """Wrapping does not alter what reset/step return."""
    env = instrument_env(_ToyEnv())
    obs, info = env.reset(seed=0)
    assert obs.tolist() == [0.0] and not info
    out = env.step(env.action_space.sample())
    assert out[0].tolist() == [1.0] and out[2] is False


def test_instrument_env_idempotent(sink: Callable[[], list[dict]]) -> None:
    """Instrumenting the same instance twice does not double-wrap it."""
    env = instrument_env(_ToyEnv())
    reset_after_first = env.reset
    instrument_env(env)
    assert env.reset is reset_after_first
    env.reset(seed=0)
    assert [e["kind"] for e in sink()] == ["reset"]  # one event, not two


def test_instrument_class_per_instance_counters(sink: Callable[[], list[dict]]) -> None:
    """Class-patched instances keep independent step counters."""

    class _C(_ToyEnv):
        pass

    instrument_class(_C)
    a, b = _C(), _C()
    a.reset(seed=1)
    b.reset(seed=2)
    b.step(b.action_space.sample())  # b: 1 step, not terminal
    for _ in range(3):  # a: 3 steps -> terminates with its own count
        a.step(a.action_space.sample())
    ends = [e for e in sink() if e["kind"] == "episode_end"]
    assert len(ends) == 1 and ends[0]["num_steps"] == 3  # a's count, not b's 1


@pytest.mark.usefixtures("sink")
def test_instance_guard_skips_class_wrapped() -> None:
    """instrument_env is a no-op on an env whose class is already instrumented."""

    class _C(_ToyEnv):
        pass

    instrument_class(_C)
    env = _C()
    reset_before = env.reset
    instrument_env(env)  # must not add a second wrapper
    assert env.reset == reset_before


@pytest.mark.usefixtures("sink")
def test_require_instrumented_raises_when_missing() -> None:
    """With telemetry on, an uninstrumented env fails loud."""
    with pytest.raises(TelemetryNotInstrumentedError, match="_ToyEnv"):
        require_instrumented(_ToyEnv())


@pytest.mark.usefixtures("sink")
def test_require_instrumented_passes_when_wrapped() -> None:
    """An instrumented env satisfies the guard (instance and class paths)."""
    require_instrumented(instrument_env(_ToyEnv()))

    class _C(_ToyEnv):
        pass

    instrument_class(_C)
    require_instrumented(_C())


def test_require_instrumented_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard never fires when telemetry is off."""
    monkeypatch.delenv("ROBOCODE_TELEMETRY", raising=False)
    require_instrumented(_ToyEnv())  # must not raise


@pytest.mark.usefixtures("sink")
def test_wraps_preserves_method_name() -> None:
    """Wrapped methods keep their name so instrumentation is not obvious."""
    env = instrument_env(_ToyEnv())
    assert env.reset.__name__ == "reset" and env.step.__name__ == "step"
    assert getattr(env.reset, tel._MARK) is True  # pylint: disable=protected-access
