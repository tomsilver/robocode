"""Sandbox telemetry hook (whitebox), auto-imported by Python's ``site`` machinery.

Mounted read-only on the sandbox ``PYTHONPATH`` -- only when telemetry is enabled
-- so it runs at the start of every in-container ``python`` and instruments the
registered experiment-env classes. The agent's own reset/set_state calls are then
logged with no cooperation from its scripts.

``instrument_registered_envs`` is a no-op unless ``ROBOCODE_TELEMETRY`` is set, so
this is cheap when telemetry is off. robocode is always importable where this hook
runs, so a broken import is a real fault and is left to fail loud rather than
silently disabling telemetry.
"""

from robocode.utils.telemetry import instrument_registered_envs

instrument_registered_envs()
