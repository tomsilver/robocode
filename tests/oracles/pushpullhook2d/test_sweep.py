"""Tests for Sweep behavior on PushPullHook2D."""

import kinder
import pytest
from gymnasium.wrappers import RecordVideo

from robocode.oracles.pushpullhook2d.behaviors import GraspRotate, Sweep
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/PushPullHook2D-v0"
GRASP_MAX_STEPS = 500
SWEEP_MAX_STEPS = 300


def _run_grasp_rotate(env, obs):
    """Run GraspRotate to set up the hook for sweeping."""
    behavior = GraspRotate()
    assert behavior.initializable(obs), "GraspRotate precondition not met."
    behavior.reset(obs)
    for _ in range(GRASP_MAX_STEPS):
        obs, _, _, _, _ = env.step(behavior.step(obs))
        if behavior.terminated(obs):
            return obs
    raise AssertionError(f"GraspRotate did not finish in {GRASP_MAX_STEPS} steps.")


@pytest.mark.parametrize("seed", list(range(50)))
def test_sweep(seed) -> None:
    """After Sweep, the movable button should be aligned with the target."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=seed)

    # Phase 1: GraspRotate to get the hook vertical.
    obs = _run_grasp_rotate(env, obs)

    # Phase 2: Sweep.
    behavior = Sweep()
    assert behavior.initializable(obs), "Sweep precondition not met."
    if behavior.terminated(obs):
        print("Sweep already achieved after GraspRotate.")
        env.close()
        return

    behavior.reset(obs)
    for s in range(SWEEP_MAX_STEPS):
        action = behavior.step(obs)
        obs, _, _, _, _ = env.step(action)
        if behavior.terminated(obs):
            print(f"Sweep achieved in {s + 1} steps.")
            break

    assert behavior.terminated(obs), f"Sweep not achieved within {SWEEP_MAX_STEPS} steps."
    env.close()
