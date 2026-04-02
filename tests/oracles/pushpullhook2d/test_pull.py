"""Tests for Pull behavior on PushPullHook2D."""

import kinder
import pytest
from gymnasium.wrappers import RecordVideo

from robocode.oracles.pushpullhook2d.behaviors import (
    GraspRotate,
    PrePushPull,
    Pull,
    Sweep,
)
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/PushPullHook2D-v0"
GRASP_MAX_STEPS = 500
SWEEP_MAX_STEPS = 1000
PREPUSHPULL_MAX_STEPS = 2000
PULL_MAX_STEPS = 2000


def _run_behavior(env, obs, behavior, max_steps, name):
    """Run a behavior until terminated or max_steps."""
    assert behavior.initializable(obs), f"{name} precondition not met."
    behavior.reset(obs)
    for _ in range(max_steps):
        obs, _, _, _, _ = env.step(behavior.step(obs))
        if behavior.terminated(obs):
            return obs
    raise AssertionError(f"{name} did not finish in {max_steps} steps.")


@pytest.mark.parametrize("seed", [3])
def test_pull(seed) -> None:
    """After Pull, both buttons should be pressed."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=seed)

    # Phase 1: GraspRotate.
    obs = _run_behavior(env, obs, GraspRotate(), GRASP_MAX_STEPS, "GraspRotate")

    # Phase 2: Sweep.
    obs = _run_behavior(env, obs, Sweep(), SWEEP_MAX_STEPS, "Sweep")

    # Phase 3: PrePushPull.
    obs = _run_behavior(
        env, obs, PrePushPull(), PREPUSHPULL_MAX_STEPS, "PrePushPull"
    )

    # Phase 4: Pull.
    behavior = Pull()
    assert behavior.initializable(obs), "Pull precondition not met."
    assert not behavior.terminated(obs), "Pull subgoal already satisfied."

    behavior.reset(obs)
    for s in range(PULL_MAX_STEPS):
        action = behavior.step(obs)
        obs, _, _, _, _ = env.step(action)
        if behavior.terminated(obs):
            print(f"Pull achieved in {s + 1} steps.")
            break

    assert behavior.terminated(obs), (
        f"Pull not achieved within {PULL_MAX_STEPS} steps."
    )
    env.close()
