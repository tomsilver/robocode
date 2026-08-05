"""Lightweight append-only event telemetry (JSONL).

Opt-in via ``ROBOCODE_TELEMETRY`` (a sink file path); when it is unset every call
here is a no-op, so importing and logging is always safe. One JSON object per
line, each carrying reserved fields (``kind``, ``seq``, ``ts_ns``, ``pid``,
``run``) that a caller's payload can never shadow -- they are written last.
``seq`` is per-process, so across the many short-lived processes an agent
spawns events are only PARTIALLY ordered: order by ``ts_ns`` and disambiguate
with ``pid``/``seq``; do not assume a total order. ``run`` (from
``ROBOCODE_RUN_ID``) groups a run's processes.

Producers wrap an env's reset/set_state/episode boundaries: ``instrument_env``
for a single instance (host/tests), ``instrument_class`` for every instance of a
class (the sandbox hook). Both share a wrap guard so an env is never
double-wrapped, and both preserve the wrapped method's identity via
``functools.wraps``. Telemetry must never change the instrumented run: writes and
fingerprints are guarded, and no extra ``get_state`` is called -- fingerprints
come from the obs ``reset`` already returns and the state handed to ``set_state``.
Episode boundaries, not individual steps, are logged.
"""

from __future__ import annotations

import functools
import importlib
import itertools
import json
import os
import time
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable
from weakref import WeakKeyDictionary

_SINK_ENV = "ROBOCODE_TELEMETRY"
_RUN_ENV = "ROBOCODE_RUN_ID"
_MARK = "_robocode_telemetry"  # sentinel attribute on instrumented reset wrappers
_seq = itertools.count()
# Per-instance episode counters, weak-keyed so instrumentation leaves nothing on
# the env and does not keep dead envs alive.
_EPISODE: WeakKeyDictionary[Any, dict[str, Any]] = WeakKeyDictionary()


class TelemetryNotInstrumentedError(RuntimeError):
    """Telemetry is on but an env class has no instrumentation wired up.

    Raised by :func:`require_registered` so a new env type never silently produces
    no telemetry: a run either records collection events or fails loud.
    """


def enabled() -> bool:
    """True when a telemetry sink is configured."""
    return bool(os.environ.get(_SINK_ENV))


def log_event(kind: str, **fields: Any) -> None:
    """Append one telemetry event; a no-op when ``ROBOCODE_TELEMETRY`` is unset."""
    path = os.environ.get(_SINK_ENV)
    if not path:
        return
    try:
        record = {
            **fields,
            # Reserved fields last so a caller's payload can never shadow them.
            "kind": kind,
            "seq": next(_seq),
            "ts_ns": time.time_ns(),
            "pid": os.getpid(),
            "run": os.environ.get(_RUN_ENV),
        }
        line = (json.dumps(record, default=str) + "\n").encode("utf-8")
        # A single O_APPEND write is atomic across the many processes an agent
        # spawns, so their lines never interleave.
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # telemetry must never break the instrumented run


def fingerprint(state: Any) -> str | None:
    """Short stable hash of a state, or None when it cannot be hashed.

    Canonical for numpy arrays (dtype + shape + bytes) and for objects exposing a
    ``vec`` array; best-effort ``repr`` otherwise. The authoritative identity of a
    reset instance is its ``(seed, options)`` -- this is a secondary dedup aid,
    mainly for ``set_state`` states that carry no seed.
    """
    if state is None:
        return None
    try:
        arr = state if hasattr(state, "tobytes") else getattr(state, "vec", None)
        if arr is not None and hasattr(arr, "tobytes"):
            raw = f"{arr.dtype}{arr.shape}".encode() + arr.tobytes()
        else:
            raw = repr(state).encode()
        return sha1(raw).hexdigest()[:12]
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _obs_of(reset_out: Any) -> Any:
    """The observation from a ``(obs, info)`` reset return, or None."""
    return reset_out[0] if isinstance(reset_out, tuple) and reset_out else None


def _episode(env: Any) -> dict[str, Any]:
    """Per-instance episode counters, kept off the env in a weak-keyed table.

    Storing ``{steps, ended}`` here (rather than as attributes on the env) means
    instrumentation leaves no visible trace on the env and cannot cross-
    contaminate instances that share a class-level wrapper.
    """
    state = _EPISODE.get(env)
    if state is None:
        state = {"steps": 0, "ended": False}
        _EPISODE[env] = state
    return state


def _reset_episode(env: Any) -> None:
    """Start a fresh episode's counters for ``env``."""
    _EPISODE[env] = {"steps": 0, "ended": False}


def _on_step(env: Any, step_out: Any, label: str | None) -> None:
    """Count a step and emit one ``episode_end`` on the first terminal step.

    The ``ended`` latch means stepping again after termination logs no second
    event. All bookkeeping is guarded by the callers so telemetry never breaks
    the wrapped run.
    """
    state = _episode(env)
    state["steps"] += 1
    if state["ended"]:
        return
    try:
        terminated, truncated = bool(step_out[2]), bool(step_out[3])
    except Exception:  # pylint: disable=broad-exception-caught
        terminated = truncated = False
    if not (terminated or truncated):
        return
    state["ended"] = True
    log_event(
        "episode_end",
        label=label,
        terminated=terminated,
        truncated=truncated,
        num_steps=state["steps"],
    )


def instrument_env(env: Any, *, label: str | None = None) -> Any:
    """Instrument a single env INSTANCE (host/test use).

    A no-op when telemetry is disabled or the env (via its instance or its class)
    is already instrumented -- detected by the marker on ``reset``.
    """
    if not enabled() or getattr(getattr(env, "reset", None), _MARK, False):
        return env
    orig_reset, orig_step = env.reset, env.step
    orig_set_state = getattr(env, "set_state", None)

    @functools.wraps(orig_reset)
    def reset(*args: Any, **kwargs: Any) -> Any:
        out = orig_reset(*args, **kwargs)
        _emit_reset(env, out, kwargs, label)
        return out

    @functools.wraps(orig_step)
    def step(*args: Any, **kwargs: Any) -> Any:
        out = orig_step(*args, **kwargs)
        _guard(_on_step, env, out, label)
        return out

    setattr(reset, _MARK, True)
    env.reset, env.step = reset, step
    if orig_set_state is not None:

        @functools.wraps(orig_set_state)
        def set_state(*args: Any, **kwargs: Any) -> Any:
            result = orig_set_state(*args, **kwargs)
            _emit_set_state(env, args, label)
            return result

        env.set_state = set_state
    return env


def instrument_class(cls: Any, *, label: str | None = None) -> Any:
    """Instrument EVERY instance of ``cls`` (the sandbox hook); idempotent.

    Per-instance counters are keyed on the instance, so two instances sharing the class
    wrapper never cross-contaminate each other's step counts.
    """
    if not enabled() or getattr(getattr(cls, "reset", None), _MARK, False):
        return cls
    lbl = label or cls.__name__
    orig_reset, orig_step = cls.reset, cls.step
    orig_set_state: Callable[..., Any] | None = getattr(cls, "set_state", None)

    @functools.wraps(orig_reset)
    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = orig_reset(self, *args, **kwargs)
        _emit_reset(self, out, kwargs, lbl)
        return out

    @functools.wraps(orig_step)
    def step(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = orig_step(self, *args, **kwargs)
        _guard(_on_step, self, out, lbl)
        return out

    setattr(reset, _MARK, True)
    cls.reset, cls.step = reset, step
    if orig_set_state is not None:

        @functools.wraps(orig_set_state)
        def set_state(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = orig_set_state(self, *args, **kwargs)
            _emit_set_state(self, args, lbl)
            return result

        cls.set_state = set_state
    return cls


def _guard(fn: Callable[..., None], *args: Any) -> None:
    """Run a telemetry side-effect, swallowing anything it raises."""
    try:
        fn(*args)
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # telemetry must never break the instrumented run


def _emit_reset(env: Any, out: Any, kwargs: dict[str, Any], label: str | None) -> None:
    """Reset the episode counters and log a ``reset`` event."""
    _guard(_reset_episode, env)
    log_event(
        "reset",
        label=label,
        seed=kwargs.get("seed"),
        options=kwargs.get("options"),
        state=fingerprint(_obs_of(out)),
    )


def _emit_set_state(env: Any, args: tuple[Any, ...], label: str | None) -> None:
    """Reset the episode counters and log a ``set_state`` event."""
    _guard(_reset_episode, env)
    log_event("set_state", label=label, state=fingerprint(args[0] if args else None))


# --- Env registry: which envs get telemetry, and the launch-time guard. ---

# "module:ClassName" for every env class instrumented for telemetry. Add a line
# when a new env type enters a telemetry experiment.
INSTRUMENTED_ENVS: tuple[str, ...] = (
    "robocode.environments.variable_object_count_env:VariableObjectCountEnv",
)


def instrument_registered_envs() -> None:
    """Wrap every registered env class in this process; a no-op when telemetry is off.

    Called by the sandbox ``sitecustomize`` hook and usable directly in local runs.
    Import/instrumentation errors propagate on purpose -- a broken telemetry setup
    should fail loud, not run an experiment blind.
    """
    if not enabled():
        return
    for target in INSTRUMENTED_ENVS:
        module_name, class_name = target.split(":")
        instrument_class(getattr(importlib.import_module(module_name), class_name))
    log_event("telemetry_ready", envs=list(INSTRUMENTED_ENVS))


def require_registered(env_target: str) -> None:
    """Fail loud if telemetry is on but ``env_target`` is not a registered env.

    A no-op when telemetry is off. ``env_target`` is a hydra ``_target_`` in
    ``module:Class`` or ``module.Class`` form.
    """
    if not enabled():
        return
    registered = {t.replace(":", ".") for t in INSTRUMENTED_ENVS}
    if env_target.replace(":", ".") not in registered:
        raise TelemetryNotInstrumentedError(
            f"Telemetry is on but env {env_target!r} is not registered for "
            "instrumentation; add it to INSTRUMENTED_ENVS in robocode.utils.telemetry."
        )


# --- Sandbox launch wiring: deliver the hook + sink to a sandboxed agent. ---

_HOOK_MOUNT = "/robocode/telemetry_hook"  # hook dir bind target inside a container
_SINK_MOUNT = "/telemetry"  # sink dir bind target inside a container
SINK_FILENAME = "events.jsonl"


def hook_dir() -> Path:
    """Host path of the ``sitecustomize`` hook directory."""
    return Path(__file__).resolve().parents[3] / "docker" / "telemetry_hook"


def _with_hook_pythonpath(hook: str) -> str:
    """Prepend the hook dir to any existing PYTHONPATH rather than clobbering it."""
    existing = os.environ.get("PYTHONPATH", "")
    return hook + (os.pathsep + existing if existing else "")


def container_launch(
    host_sink_dir: Path, run_id: str
) -> tuple[list[tuple[str, str, bool]], dict[str, str]]:
    """(mounts, env) enabling the whitebox sandbox hook in docker/apptainer.

    ``mounts`` are ``(host, container, read_only)`` tuples the caller renders as
    ``-v`` / ``--bind``. Not used for blackbox runs: their sandbox has no env
    source to instrument -- instrument the host env server there instead.
    """
    mounts = [
        (str(hook_dir()), _HOOK_MOUNT, True),
        (str(host_sink_dir), _SINK_MOUNT, False),
    ]
    env = {
        "PYTHONPATH": _HOOK_MOUNT,
        _SINK_ENV: f"{_SINK_MOUNT}/{SINK_FILENAME}",
        _RUN_ENV: run_id,
    }
    return mounts, env


def host_launch_env(sink_file: Path, run_id: str, *, with_hook: bool) -> dict[str, str]:
    """Env vars enabling telemetry for a host process (local agent or env server).

    ``with_hook`` prepends the hook dir to ``PYTHONPATH`` so a local agent's
    spawned python picks up ``sitecustomize``; the env server instruments its env
    in code and does not need the hook.
    """
    env = {_SINK_ENV: str(sink_file), _RUN_ENV: run_id}
    if with_hook:
        env["PYTHONPATH"] = _with_hook_pythonpath(str(hook_dir()))
    return env
