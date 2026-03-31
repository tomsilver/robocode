"""Tests for GraspRotate behavior on PushPullHook2D."""

import kinder
import pytest
from gymnasium.wrappers import RecordVideo

from robocode.oracles.pushpullhook2d.behaviors import GraspRotate
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/PushPullHook2D-v0"
MAX_STEPS = 500


@pytest.mark.parametrize("seed", [42, 123, 636, 7, 0])
def test_grasp_rotate(seed) -> None:
    """After GraspRotate, the hook should be grasped and rotated to -π."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=seed)
    behavior = GraspRotate()

    assert behavior.initializable(obs), "Precondition should be satisfied at the start."
    assert not behavior.terminated(obs), "Subgoal should not be satisfied at the start."

    behavior.reset(obs)
    for s in range(MAX_STEPS):
        action = behavior.step(obs)
        obs, _, _, _, _ = env.step(action)
        if behavior.terminated(obs):
            print(f"Subgoal achieved in {s + 1} steps.")
            break

    assert behavior.terminated(obs), f"Subgoal not achieved within {MAX_STEPS} steps."
    env.close()
