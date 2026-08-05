"""Registry of env classes wired for telemetry, and how to apply it.

Telemetry is opt-in and must never be silently missing. An env class is
instrumented only if it is listed in :data:`INSTRUMENTED_ENVS`, and a telemetry
run whose env is not listed is refused up front by :func:`require_registered` --
so a new env type forces a conscious one-line registration here instead of
quietly producing no data after an experiment.

The same registry drives both sides: the sandbox ``sitecustomize`` hook calls
:func:`instrument_registered_envs` to wrap every listed class in-process, and the
run harness calls :func:`require_registered` on the configured env before launch.
This is the project-specific layer over the generic mechanism in
:mod:`robocode.utils.telemetry`, which knows nothing about robocode's env classes.
"""

from __future__ import annotations

import importlib

from robocode.utils.telemetry import (
    TelemetryNotInstrumentedError,
    enabled,
    instrument_class,
    log_event,
)

# "module:ClassName" for every env class instrumented for telemetry. Add a line
# here when a new env type enters a telemetry experiment.
INSTRUMENTED_ENVS: tuple[str, ...] = (
    "robocode.environments.variable_object_count_env:VariableObjectCountEnv",
)


def _normalize(target: str) -> str:
    """Canonical dotted form of a hydra ``_target_`` (``mod:Cls`` or ``mod.Cls``)."""
    return target.replace(":", ".")


def instrument_registered_envs() -> None:
    """Wrap every registered env class in this process; a no-op when telemetry is off.

    Called by the sandbox ``sitecustomize`` hook and usable directly in local
    runs. Import or instrumentation errors propagate on purpose -- a broken
    telemetry setup should fail loud, not run an experiment blind.
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
    either ``module:Class`` or ``module.Class`` form.
    """
    if not enabled():
        return
    registered = {_normalize(t) for t in INSTRUMENTED_ENVS}
    if _normalize(env_target) not in registered:
        raise TelemetryNotInstrumentedError(
            f"Telemetry is on but env {env_target!r} is not registered for "
            "instrumentation; add it to INSTRUMENTED_ENVS in "
            "robocode.utils.telemetry_envs."
        )
