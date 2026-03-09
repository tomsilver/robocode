"""Tests for pushpullhook2d_grasp_random.py."""

from collections.abc import Iterable

import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.geom2d.pushpullhook2d_grasp_random import (
    ObjectCentricPushPullHook2DGraspRandomEnv,
)
from kinder.envs.geom2d.structs import MultiBody2D, SE2Pose
from kinder.envs.geom2d.utils import CRVRobotActionSpace
from kinder.envs.utils import get_se2_pose, state_2d_has_collision
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs import Array, Object, ObjectCentricState
from robocode.primitives.motion_planning import BiRRT
from tests.conftest import MAKE_VIDEOS


def test_object_centric_pushpullhook2d_grasp_random_env():
    """Tests for ObjectCentricPushPullHook2DGraspRandomEnv()."""
    env = ObjectCentricPushPullHook2DGraspRandomEnv()
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")
    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_pushpullhook2d_grasp_random_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/PushPullHook2D-GraspRandom-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_pushpullhook2d_grasp_random_action_space():
    """Tests that actions are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/PushPullHook2D-GraspRandom-v0")
    assert isinstance(env.action_space, Box)
    for _ in range(5):
        action = env.action_space.sample()
        assert env.action_space.contains(action)


def test_hook_theta_varies():
    """Verify hook theta varies across seeds and stays in [pi/4, 3*pi/4]."""
    env = ObjectCentricPushPullHook2DGraspRandomEnv()
    thetas = []
    for seed in range(20):
        state, _ = env.reset(seed=seed)
        hook = next(o for o in state if o.name == "hook")
        theta = state.get(hook, "theta")
        assert np.pi / 4 - 1e-6 <= theta <= 3 * np.pi / 4 + 1e-6, (
            f"seed={seed}: hook theta={theta} out of bounds"
        )
        thetas.append(theta)
    env.close()
    # Check that not all thetas are the same (i.e. orientation is random).
    assert len(set(round(t, 4) for t in thetas)) > 1, (
        "Hook theta should vary across seeds"
    )


def _goal_config_is_collision_free(env, state, robot, gx, gy, gtheta, arm):
    """Check if robot at (gx, gy, gtheta, arm) is collision-free."""

    test = state.copy()
    test.set(robot, "x", gx)
    test.set(robot, "y", gy)
    test.set(robot, "theta", gtheta)
    test.set(robot, "arm_joint", arm)
    test.set(robot, "vacuum", 0.0)
    full = test.copy()
    full.data.update(env.initial_constant_state.data)
    return not state_2d_has_collision(
        full, {robot}, set(full) - {robot}, env._static_object_body_cache
    )


def run_motion_planning_for_crv_robot(
    state: ObjectCentricState,
    robot: Object,
    target_pose: SE2Pose,
    action_space: CRVRobotActionSpace,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    initial_constant_state: ObjectCentricState | None = None,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
) -> list[SE2Pose] | None:
    """Run motion planning in an environment with a CRV action space."""
    if static_object_body_cache is None:
        static_object_body_cache = {}

    rng = np.random.default_rng(seed)

    # Use the object positions in the state to create a rough room boundary.
    # Include the target pose in the bounds and add padding so the planner
    # can sample configurations around start/goal even when they sit at the
    # edge of the object layout.
    x_lb, x_ub, y_lb, y_ub = np.inf, -np.inf, np.inf, -np.inf
    for obj in state:
        pose = get_se2_pose(state, obj)
        x_lb = min(x_lb, pose.x)
        x_ub = max(x_ub, pose.x)
        y_lb = min(y_lb, pose.y)
        y_ub = max(y_ub, pose.y)
    x_lb = min(x_lb, target_pose.x)
    x_ub = max(x_ub, target_pose.x)
    y_lb = min(y_lb, target_pose.y)
    y_ub = max(y_ub, target_pose.y)
    padding = 0.5
    x_lb -= padding
    x_ub += padding
    y_lb -= padding
    y_ub += padding

    # Create a static version of the state so that the geoms only need to be
    # instantiated once during motion planning (except for the robot). Make
    # sure to not update the global cache because we don't want to carry over
    # static things that are not actually static.
    static_object_body_cache = static_object_body_cache.copy()
    moving_objects = {robot}
    static_state = state.copy()
    # Merge in constant objects (walls, table, etc.) so the collision
    # checker sees the full environment, matching what env.step() does.
    if initial_constant_state is not None:
        static_state.data.update(initial_constant_state.data)
    for o in static_state:
        if o in moving_objects:
            continue
        static_state.set(o, "static", 1.0)

    # Set up the RRT methods.
    def sample_fn(_: SE2Pose) -> SE2Pose:
        """Sample a robot pose."""
        x = rng.uniform(x_lb, x_ub)
        y = rng.uniform(y_lb, y_ub)
        theta = rng.uniform(-np.pi, np.pi)
        return SE2Pose(x, y, theta)

    def extend_fn(pt1: SE2Pose, pt2: SE2Pose) -> Iterable[SE2Pose]:
        """Interpolate between the two poses."""
        # Make sure that we obey the bounds on actions.
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        assert isinstance(action_space, CRVRobotActionSpace)
        abs_x = action_space.high[0] if dx > 0 else action_space.low[0]
        abs_y = action_space.high[1] if dy > 0 else action_space.low[1]
        abs_theta = action_space.high[2] if dtheta > 0 else action_space.low[2]
        x_num_steps = int(dx / abs_x) + 1
        assert x_num_steps > 0
        y_num_steps = int(dy / abs_y) + 1
        assert y_num_steps > 0
        theta_num_steps = int(dtheta / abs_theta) + 1
        assert theta_num_steps > 0
        num_steps = max(x_num_steps, y_num_steps, theta_num_steps)
        x = pt1.x
        y = pt1.y
        theta = pt1.theta
        yield SE2Pose(x, y, theta)
        for _ in range(num_steps):
            x += dx / num_steps
            y += dy / num_steps
            theta = wrap_angle(theta + dtheta / num_steps)
            yield SE2Pose(x, y, theta)

    def collision_fn(pt: SE2Pose) -> bool:
        """Check for collisions if the robot were at this pose."""

        # Update the static state with the robot's new hypothetical pose.
        static_state.set(robot, "x", pt.x)
        static_state.set(robot, "y", pt.y)
        static_state.set(robot, "theta", pt.theta)

        # Update the suctioned objects in the static state.
        obstacle_objects = set(static_state) - moving_objects

        return state_2d_has_collision(
            static_state, moving_objects, obstacle_objects, static_object_body_cache
        )

    def distance_fn(pt1: SE2Pose, pt2: SE2Pose) -> float:
        """Return a distance between the two points."""
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        return np.sqrt(dx**2 + dy**2) + abs(dtheta)

    birrt = BiRRT(
        sample_fn,
        extend_fn,
        collision_fn,
        distance_fn,
        rng,
        num_attempts,
        num_iters,
        smooth_amt,
    )

    initial_pose = get_se2_pose(state, robot)
    return birrt.query(initial_pose, target_pose)


def crv_pose_plan_to_action_plan(
    pose_plan: list[SE2Pose],
    action_space: CRVRobotActionSpace,
    vacuum_while_moving: bool = False,
) -> list[Array]:
    """Convert a CRV robot pose plan into corresponding actions."""
    action_plan: list[Array] = []
    for pt1, pt2 in zip(pose_plan[:-1], pose_plan[1:]):
        action = np.zeros_like(action_space.high)
        action[0] = pt2.x - pt1.x
        action[1] = pt2.y - pt1.y
        action[2] = get_signed_angle_distance(pt2.theta, pt1.theta)
        action[4] = 1.0 if vacuum_while_moving else 0.0
        action_plan.append(action)
    return action_plan


def _solve_grasp(env, state, max_steps=500, step_env=None):
    """Solver: compute a collision-free grasp pose, navigate there with arm
    retracted, extend arm, then vacuum.

    1. Compute goal config where suction zone overlaps the hook bar,
       trying both perpendicular approach directions and picking the one
       that is collision-free with arm fully extended.
    2. Navigate to goal (x, y, theta) with arm retracted.
    3. Extend arm at goal.
    4. Turn on vacuum.
    """
    if step_env is None:
        step_env = env
    obj_map = {o.name: o for o in state}
    robot = obj_map["robot"]
    hook = obj_map["hook"]

    hx = state.get(hook, "x")
    hy = state.get(hook, "y")
    ht = state.get(hook, "theta")
    hl1 = state.get(hook, "length_side1")
    arm_length = state.get(robot, "arm_length")
    gripper_w = state.get(robot, "gripper_width")
    base_r = state.get(robot, "base_radius")

    tcp2robot = SE2Pose(-arm_length - gripper_w - 0.01,
                        0.0, 0.0)  # TCP to robot base when arm extended
    hook_pose = SE2Pose(hx, hy, ht)
    assert isinstance(env.action_space, CRVRobotActionSpace)

    # Try each candidate goal; for each collision-free config, attempt motion
    # planning.  Use the first goal that is both collision-free and reachable.
    pose_plan = None
    for length_rt in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        for rel_theta in [np.pi / 2, -np.pi / 2]:
            hook2tcp = SE2Pose(
                -hl1 * length_rt, 0.0, rel_theta
            )
            tcp_pose = hook_pose * hook2tcp
            robot_pose = tcp_pose * tcp2robot
            gx, gy, ft = robot_pose.x, robot_pose.y, robot_pose.theta
            if not _goal_config_is_collision_free(
                env, state, robot, gx, gy, ft, arm_length
            ):
                continue
            target_pose = SE2Pose(gx, gy, ft)
            pose_plan = run_motion_planning_for_crv_robot(
                state,
                robot,
                target_pose,
                env.action_space,
                static_object_body_cache=env._static_object_body_cache,
                initial_constant_state=env.initial_constant_state,
                num_attempts=20,
                num_iters=500,
            )
            if pose_plan is not None:
                break
        if pose_plan is not None:
            break

    if pose_plan is None:
        return False, max_steps

    action_plan = crv_pose_plan_to_action_plan(pose_plan, env.action_space)
    step_count = 0
    for action in action_plan:
        state, _, terminated, _, _ = step_env.step(action)
        step_count += 1
        if terminated:
            return True, step_count

    # Phase 2: Extend arm.
    while state.get(robot, "arm_joint") < arm_length - 0.01:
        action = np.array(
            [0.0, 0.0, 0.0, env.config.max_darm, 0.0], dtype=np.float32
        )
        state, _, terminated, _, _ = step_env.step(action)
        step_count += 1
        if terminated:
            return True, step_count
        if step_count >= max_steps:
            return False, step_count

    # Phase 3: Turn on vacuum.
    action = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state, _, terminated, _, _ = step_env.step(action)
    step_count += 1
    if terminated:
        return True, step_count

    return False, step_count


def test_grasp_random_solvable_seed0():
    """Test that the scripted solver solves the environment with seed=0."""
    env = ObjectCentricPushPullHook2DGraspRandomEnv()
    step_env = env
    if MAKE_VIDEOS:
        step_env = RecordVideo(env, "unit_test_videos")
    state, _ = step_env.reset(seed=0)
    solved, steps = _solve_grasp(env, state, step_env=step_env)
    step_env.close()
    assert solved, f"Scripted solver failed on seed=0 after {steps} steps"
