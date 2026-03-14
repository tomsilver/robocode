"""Tests for PushPullHook2D pick skill."""

import numpy as np
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic2d.pushpullhook2d import ObjectCentricPushPullHook2DEnv
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace, get_suctioned_objects

from robocode.skills.pushpullhook2d.pick_skill import GroundPickController
from robocode.skills.utils import TrajectorySamplingFailure
from tests.conftest import MAKE_VIDEOS


def test_pick_skill_grasps_hook_seed0() -> None:
    """Verify that GroundPickController grasps the hook on seed 0."""
    env = ObjectCentricPushPullHook2DEnv()
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    state, _ = env.reset(seed=0)

    obj_map = {o.name: o for o in state}
    robot = obj_map["robot"]
    hook = obj_map["hook"]

    assert isinstance(env.action_space, CRVRobotActionSpace)
    controller = GroundPickController(
        objects=[robot, hook],
        action_space=env.action_space,
        init_constant_state=env.unwrapped.initial_constant_state,
    )

    rng = np.random.default_rng(0)
    # Try multiple parameter samples until one succeeds (some may collide).
    grasped = False
    for _ in range(50):
        params = controller.sample_parameters(state, rng)
        try:
            controller.reset(state, params)
            while not controller.terminated():
                action = controller.step()
                state, _, terminated, _, _ = env.step(action)
                controller.observe(state)
                if terminated:
                    break
        except TrajectorySamplingFailure:
            state, _ = env.reset(seed=0)
            continue

        # Check if the hook is suctioned by the robot.
        suctioned = get_suctioned_objects(state, robot)
        suctioned_names = [o.name for o, _ in suctioned]
        if "hook" in suctioned_names:
            grasped = True
            break

        # Reset env for next attempt.
        state, _ = env.reset(seed=0)

    env.close()
    assert grasped, "GroundPickController failed to grasp the hook after 50 attempts"
