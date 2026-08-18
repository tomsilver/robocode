"""Check that eval episodes run and stay inside their budget on this platform.

Usage:
    python integration_tests/check_eval_timeout.py

The unit tests in ``tests/utils/test_episode.py`` drive both eval paths with a pure
Python env, which is why they never caught this: the failure needs a real MuJoCo env.

``run_episode_with_timeout`` used to always roll out in a forked worker. On macOS that
cannot work. Constructing a MuJoCo env initializes CGL/Metal in this process, and
Apple's GPU frameworks are undefined in a child that forks without exec'ing, so the
child dies with SIGSEGV (exit -11) inside ``mjr_freeContext`` as soon as it resets the
env. Every eval episode crashed and ``solve_rate`` came out NaN.

Two properties are checked against a real env:

1. A forked child really is unusable here (macOS only) -- the premise for taking the
   in-process path, re-verified rather than assumed, so this stops reporting a problem
   if a future mujoco or macOS release fixes it.
2. ``run_episode_with_timeout`` scores a working policy, and still stops a policy that
   overruns its budget both by hanging in one call and by being slow across steps.

Needs the kinder dynamic3D extras. Raises on any failure.
"""

import multiprocessing as mp
import sys
import time
from typing import Any

import numpy as np
from hydra.utils import instantiate

from robocode.approaches.base_approach import BaseApproach
from robocode.utils.episode import _EPISODE_FORK_SAFE, run_episode_with_timeout

# Mirrors experiments/conf/environment/tossing3d_generalized.yaml. Inlined so this
# check does not depend on a past run directory or on Hydra composition.
_ENV_CONFIG: dict[str, Any] = {
    "_target_": "robocode.environments.variable_object_count_env.VariableObjectCountEnv",
    "constant_object_env_path": "kinder.envs.dynamic3d.task_families:Tossing3DEnv",
    "count_kwarg": "num_objects",
    "count_object_prefix": "cube_",
    "design_counts": [1],
    "eval_counts": [1, 2],
    "constant_object_env_kwargs": {"scene_bg": False},
    "base_steps": 400,
    "steps_per_object": 400,
}


class _ZeroPolicy(BaseApproach[Any, Any]):
    """Holds still.

    Enough to step the env; it is not expected to solve anything.
    """

    def _get_action(self) -> Any:
        shape = self._action_space.shape
        assert shape is not None, "the tossing3d action space is a Box with a shape"
        return np.zeros(shape, dtype=np.float32)


class _HangingPolicy(_ZeroPolicy):
    """Never returns from a single call, so only the alarm can stop it."""

    def _get_action(self) -> Any:
        while True:
            pass


class _SlowPolicy(_ZeroPolicy):
    """Returns, but far too slowly, so the per-step deadline must stop it."""

    def _get_action(self) -> Any:
        time.sleep(0.5)
        return super()._get_action()


def _child_resets(result: Any) -> None:
    """Reset a freshly built env in a forked child and report what happened."""
    try:
        env = instantiate(_ENV_CONFIG)
        env.reset(seed=0, options={"object_count": 1})
        result["reset"] = "ok"
    except BaseException as exc:  # pylint: disable=broad-except
        result["error"] = repr(exc)


def _check_fork_is_still_broken() -> None:
    """Confirm the premise for avoiding fork on macOS still holds."""
    ctx = mp.get_context("fork")
    with ctx.Manager() as manager:
        result = manager.dict()
        proc = ctx.Process(target=_child_resets, args=(result,))
        proc.start()
        proc.join(300)
        if proc.exitcode == 0 and result.get("reset") == "ok":
            raise AssertionError(
                "a forked child reset a MuJoCo env successfully on macOS. The reason "
                "_EPISODE_FORK_SAFE excludes darwin no longer applies; re-check "
                "whether the in-process path is still needed."
            )
        print(f"OK  forked child still unusable here (exit={proc.exitcode})")


def _check_budget_is_enforced() -> None:
    """A working policy is scored; policies that overrun are stopped either way."""
    env = instantiate(_ENV_CONFIG)
    spaces: tuple[Any, Any, int, dict[str, Any]] = (
        env.action_space,
        env.observation_space,
        0,
        {},
    )

    metrics, _, _ = run_episode_with_timeout(
        env, _ZeroPolicy(*spaces), seed=0, max_steps=5, timeout=60, count=1
    )
    if metrics.get("timed_out"):
        raise AssertionError(f"a 5-step rollout hit a 60s budget: {metrics}")
    print(
        f"OK  rollout scored: steps={metrics['num_steps']} solved={metrics['solved']}"
    )

    for label, policy in (
        ("hangs in one call", _HangingPolicy(*spaces)),
        ("slow across steps", _SlowPolicy(*spaces)),
    ):
        started = time.monotonic()
        metrics, _, _ = run_episode_with_timeout(
            env, policy, seed=0, max_steps=1000, timeout=2.0, count=1
        )
        elapsed = time.monotonic() - started
        if not metrics.get("timed_out"):
            raise AssertionError(f"{label}: not stopped, scored {metrics}")
        if metrics["solved"]:
            raise AssertionError(f"{label}: timed-out episode scored as solved")
        if elapsed > 30:
            raise AssertionError(f"{label}: took {elapsed:.0f}s to stop a 2s budget")
        print(f"OK  {label}: stopped after {elapsed:.1f}s, scored unsolved")


def main() -> None:
    """Run the checks that apply to this platform."""
    print(f"platform={sys.platform} _EPISODE_FORK_SAFE={_EPISODE_FORK_SAFE}")
    if sys.platform == "darwin":
        _check_fork_is_still_broken()
    _check_budget_is_enforced()
    print("Eval episodes run and respect their wall-clock budget.")


if __name__ == "__main__":
    main()
