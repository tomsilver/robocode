from collections.abc import Iterable

import numpy as np
from kinder.envs.kinematic2d.structs import MultiBody2D, SE2Pose
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace, 
    get_suctioned_objects, 
    snap_suctioned_objects
)
from kinder.envs.utils import get_se2_pose, state_2d_has_collision
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs import Array, Object, ObjectCentricState

from robocode.primitives.motion_planning import BiRRT

class TrajectorySamplingFailure(BaseException):
    """Raised when trajectory sampling fails."""


def run_motion_planning_for_crv_robot(
    state: ObjectCentricState,
    robot: Object,
    target_pose: SE2Pose,
    action_space: CRVRobotActionSpace,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    initial_constant_state: ObjectCentricState | None = None,
    seed: int = 0,
    num_attempts: int = 1,
    num_iters: int = 300,
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
    suctioned_objects = get_suctioned_objects(state, robot)
    moving_objects = {robot} | {o for o, _ in suctioned_objects}
    static_state = state.copy()
    # Merge in constant objects (walls, table, etc.) so the collision
    # checker sees the full environment, matching what env.step() does.
    if initial_constant_state is not None:
        static_state.data.update(initial_constant_state.data)
    for o in static_state:
        if o in moving_objects:
            continue
        static_state.set(o, "static", 1.0)

    # Uncomment to visualize the scene.
    # import matplotlib.pyplot as plt
    # import imageio.v2 as iio
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # render_state_on_ax(static_state, ax)
    # goal_state = static_state.copy()
    # goal_state.set(robot, "x", target_pose.x)
    # goal_state.set(robot, "y", target_pose.y)
    # goal_state.set(robot, "theta", target_pose.theta)
    # snap_suctioned_objects(goal_state, robot, suctioned_objects)
    # goal_robot_mb = _robot_to_multibody2d(robot, goal_state)
    # for body in goal_robot_mb.bodies:
    #     body.rendering_kwargs["facecolor"] = "pink"
    #     body.rendering_kwargs["alpha"] = 0.5
    # goal_robot_mb.plot(ax)
    # ax.set_xlim(-1, 11)
    # ax.set_ylim(-1, 11)
    # img = fig2data(fig)
    # import ipdb; ipdb.set_trace()

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
        snap_suctioned_objects(static_state, robot, suctioned_objects)
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
