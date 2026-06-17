"""Benchmark the per-call latency the blackbox env server adds.

Blackbox mode routes the agent's env calls (during ``train()`` exploration)
through a host-side TCP server that speaks JSON, instead of calling the env
in-process. This measures that wire overhead -- TCP loopback round-trip plus
JSON encode/decode -- by timing the same operations directly and through the
``env_client``. No LLM, no container.

Evaluation is NOT affected: the generated ``approach.py`` is scored against the
real env directly, so this overhead applies only to the agent's exploration.

Run from the repo root::

    python integration_tests/bench_blackbox_latency.py [N]   # N calls/op, default 300
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.utils.env_client import BlackboxEnv
from robocode.utils.env_server import env_server_running, serialize_space

ENV_ID = "kinder/Motion2D-p0-v0"
ENV_CONFIG = {
    "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
    "env_id": ENV_ID,
}


def _min_ms(fn: Callable[[], Any], n: int) -> float:
    """Return the fastest of *n* calls of *fn*, in milliseconds.

    The minimum is the least scheduler-perturbed sample, so it isolates the real per-
    call cost better than the mean/median for a sub-millisecond op.
    """
    best = float("inf")
    for _ in range(n):
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


def main() -> None:
    """Print absolute per-op latency (direct vs blackbox) and the round-trip."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    direct = KinderGeom2DEnv(ENV_ID)
    direct.reset(seed=0)
    direct.action_space.seed(0)
    action = direct.action_space.sample()
    state = direct.get_state()

    direct_ms = {
        "reset": _min_ms(lambda: direct.reset(seed=0), n),
        "step": _min_ms(lambda: direct.step(action), n),
        "get_state": _min_ms(direct.get_state, n),
        "set_state": _min_ms(lambda: direct.set_state(state), n),
    }

    with TemporaryDirectory() as tmp:
        sandbox = Path(tmp) / "sandbox"
        sandbox.mkdir()
        with env_server_running(json.dumps(ENV_CONFIG), sandbox) as (port, token):
            meta: dict[str, Any] = {
                "host": "127.0.0.1",
                "port": port,
                "token": token,
                "observation_space": serialize_space(direct.observation_space),
                "action_space": serialize_space(direct.action_space),
                "max_steps": 200,
            }
            with BlackboxEnv(meta) as client:
                client.reset(seed=0)
                blackbox_ms = {
                    "reset": _min_ms(lambda: client.reset(seed=0), n),
                    "step": _min_ms(lambda: client.step(action), n),
                    "get_state": _min_ms(client.get_state, n),
                    "set_state": _min_ms(lambda: client.set_state(state), n),
                }
                collision_ms = _min_ms(
                    lambda: client.check_action_collision(state, action), n
                )

    print(f"env={ENV_ID}  n={n} calls/op (min ms; what the agent experiences)\n")
    print(f"{'op':<16}{'direct':>10}{'blackbox':>12}")
    print("-" * 38)
    for op in ("reset", "step", "get_state", "set_state"):
        print(f"{op:<16}{direct_ms[op]:>10.3f}{blackbox_ms[op]:>12.3f}")
    print(f"{'check_collision':<16}{'n/a':>10}{collision_ms:>12.3f}")

    # get_state has ~zero direct cost (it returns a cached array), so its
    # blackbox time IS the pure wire round-trip; compute-heavy ops (step/reset)
    # add only payload marshaling on top, which stays in the noise of the env's
    # own ~1-2 ms compute, so a direct/blackbox subtraction there is unreliable.
    roundtrip = blackbox_ms["get_state"]
    print(f"\nwire round-trip latency (get_state, ~empty payload): ~{roundtrip:.3f} ms")
    print("so each agent env-call adds roughly one round-trip:")
    for calls in (1_000, 10_000, 100_000):
        print(
            f"  {calls:>7,} exploration env-calls -> +{roundtrip * calls / 1000:.1f} s"
        )
    print(
        "\nLoopback numbers (local/apptainer); docker adds a container->host hop. "
        "Eval is unaffected: it runs approach.py on the real env, not the server."
    )


if __name__ == "__main__":
    main()
