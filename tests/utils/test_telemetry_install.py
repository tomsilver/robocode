"""Tests for the telemetry registry, installer, and the sandbox sitecustomize hook."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box

from robocode.utils import telemetry_install as ti
from robocode.utils.telemetry import TelemetryNotInstrumentedError

_HOOK = Path(__file__).resolve().parents[2] / "docker" / "telemetry_hook"


class _DummyEnv(Env):
    """Importable env used to exercise install() without a heavy real env."""

    def __init__(self) -> None:
        self.observation_space = Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Any = None, options: Any = None) -> Any:
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: Any) -> Any:
        del action
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}

    def render(self) -> None:
        return None


@pytest.fixture()
def enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Turn telemetry on, pointing the sink at a temp file; return the sink path."""
    sink = tmp_path / "events.jsonl"
    monkeypatch.setenv("ROBOCODE_TELEMETRY", str(sink))
    return sink


def test_require_registered_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With telemetry off, an unregistered env is not rejected."""
    monkeypatch.delenv("ROBOCODE_TELEMETRY", raising=False)
    ti.require_registered("some.unregistered.Env")  # must not raise


@pytest.mark.usefixtures("enabled")
def test_require_registered_accepts_registered_forms() -> None:
    """The registered env passes in both colon and dotted target forms."""
    target = ti.INSTRUMENTED_ENVS[0]
    ti.require_registered(target)
    ti.require_registered(target.replace(":", "."))


@pytest.mark.usefixtures("enabled")
def test_require_registered_rejects_unregistered() -> None:
    """An unregistered env fails loud when telemetry is on."""
    with pytest.raises(TelemetryNotInstrumentedError, match="not registered"):
        ti.require_registered("robocode.environments.maze_env:MazeEnv")


def test_install_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """install() does nothing (no instrumentation) when telemetry is off."""
    monkeypatch.delenv("ROBOCODE_TELEMETRY", raising=False)
    calls: list[Any] = []
    monkeypatch.setattr(ti, "instrument_class", calls.append)
    ti.install()
    assert not calls


def test_install_instruments_each_registered_env(
    enabled: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """install() instruments every registered class and logs the install event."""
    calls: list[Any] = []
    monkeypatch.setattr(ti, "instrument_class", calls.append)
    monkeypatch.setattr(ti, "INSTRUMENTED_ENVS", (f"{__name__}:_DummyEnv",))
    ti.install()
    assert calls == [_DummyEnv]
    events = [json.loads(line) for line in enabled.read_text().splitlines()]
    assert any(e["kind"] == "telemetry_install" for e in events)


def test_sitecustomize_installs_when_enabled(tmp_path: Path) -> None:
    """Running the real sitecustomize hook with telemetry on instruments + logs."""
    sink = tmp_path / "hook.jsonl"
    env = {
        **os.environ,
        "ROBOCODE_TELEMETRY": str(sink),
        "ROBOCODE_RUN_ID": "hook-test",
    }
    subprocess.run(
        [sys.executable, str(_HOOK / "sitecustomize.py")], env=env, check=True
    )
    events = [json.loads(line) for line in sink.read_text().splitlines()]
    assert any(e["kind"] == "telemetry_install" for e in events)


def test_sitecustomize_noop_when_disabled(tmp_path: Path) -> None:
    """With telemetry off the hook is a clean no-op and writes nothing."""
    sink = tmp_path / "none.jsonl"
    env = {k: v for k, v in os.environ.items() if k != "ROBOCODE_TELEMETRY"}
    subprocess.run(
        [sys.executable, str(_HOOK / "sitecustomize.py")], env=env, check=True
    )
    assert not sink.exists()
