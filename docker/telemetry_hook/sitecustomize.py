"""Sandbox telemetry hook (whitebox), auto-imported by Python's ``site`` machinery.

Mounted read-only on the sandbox ``PYTHONPATH`` so it runs at the start of every
in-container ``python`` invocation. When ``ROBOCODE_TELEMETRY`` is set it
instruments the registered experiment-env classes, so the agent's own reset /
set_state calls are logged without any cooperation from its scripts.

A python invocation without robocode installed (not an experiment process) is a
no-op; but once robocode is importable, a broken install is allowed to fail loud
rather than let a telemetry run proceed with no data.
"""

import os


def _install() -> None:
    if not os.environ.get("ROBOCODE_TELEMETRY"):
        return
    try:
        # pylint: disable=import-outside-toplevel
        from robocode.utils.telemetry_envs import instrument_registered_envs
    except ImportError:
        return  # no robocode here; nothing to instrument
    instrument_registered_envs()  # errors propagate on purpose


_install()
