"""Tests for PushPullHook2D skills (pick + push)."""

import numpy as np
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic2d.pushpullhook2d import ObjectCentricPushPullHook2DEnv
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace, get_suctioned_objects

from robocode.skills.pushpullhook2d.pick_skill import GroundPickController
from robocode.skills.pushpullhook2d.push_skill import GroundPushController
from robocode.skills.utils import TrajectorySamplingFailure
from tests.conftest import MAKE_VIDEOS


def _run_controller(controller, env, state, params):
    """Run a controller to completion, returning the final state."""
    controller.reset(state, params)
    while not controller.terminated():
        action = controller.step()
        state, _, terminated, _, _ = env.step(action)
        controller.observe(state)
        if terminated:
            break
    return state


def _pick_hook(env, state, rng, init_state=None, max_attempts=50):
    """Try to pick the hook, returning (state, success).

    Uses set_state to restore on failure so RecordVideo sees one
    continuous episode instead of splitting on each reset().
    """
    if init_state is None:
        init_state = state
    obj_map = {o.name: o for o in state}
    robot = obj_map["robot"]
    hook = obj_map["hook"]

    assert isinstance(env.action_space, CRVRobotActionSpace)
    controller = GroundPickController(
        objects=[robot, hook],
        action_space=env.action_space,
        init_constant_state=env.unwrapped.initial_constant_state,
    )

    for _ in range(max_attempts):
        params = controller.sample_parameters(state, rng)
        try:
            state = _run_controller(controller, env, state, params)
        except TrajectorySamplingFailure:
            env.unwrapped.set_state(init_state)
            state = init_state
            continue

        suctioned = get_suctioned_objects(state, robot)
        if any(o.name == "hook" for o, _ in suctioned):
            return state, True

        env.unwrapped.set_state(init_state)
        state = init_state

    return state, False


def test_pick_skill_grasps_hook_seed0() -> None:
    """Verify that GroundPickController grasps the hook on seed 0."""
    env = ObjectCentricPushPullHook2DEnv(allow_state_access=True)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    state, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    state, grasped = _pick_hook(env, state, rng, init_state=state)
    env.close()
    assert grasped, "GroundPickController failed to grasp the hook after 50 attempts"


def test_push_skill_moves_button_seed0() -> None:
    """Pick the hook, then push the movable button towards the target."""
    env = ObjectCentricPushPullHook2DEnv(allow_state_access=True)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    state, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)

    # Phase 1: pick the hook.
    state, grasped = _pick_hook(env, state, rng, init_state=state)
    assert grasped, "Pick phase failed — cannot test push"

    # Record initial button-target distance.
    obj_map = {o.name: o for o in state}
    robot = obj_map["robot"]
    hook = obj_map["hook"]
    movable = obj_map["movable_button"]
    target = obj_map["target_button"]
    init_dist = float(
        np.hypot(
            state.get(target, "x") - state.get(movable, "x"),
            state.get(target, "y") - state.get(movable, "y"),
        )
    )

    # Phase 2: push with the hook.
    assert isinstance(env.action_space, CRVRobotActionSpace)
    push_controller = GroundPushController(
        objects=[robot, hook, movable, target],
        action_space=env.action_space,
        init_constant_state=env.unwrapped.initial_constant_state,
    )

    pushed = False
    for _ in range(50):
        params = push_controller.sample_parameters(state, rng)
        try:
            state = _run_controller(push_controller, env, state, params)
        except TrajectorySamplingFailure:
            continue

        final_dist = float(
            np.hypot(
                state.get(target, "x") - state.get(movable, "x"),
                state.get(target, "y") - state.get(movable, "y"),
            )
        )
        if final_dist < init_dist:
            pushed = True
            break

    env.close()
    assert pushed, "GroundPushController failed to move button closer to target"
