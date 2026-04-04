"""Generic CRV motion planning based on exact geometric collision checking.

This module exposes a small public API for planning collision-free SE(2) motion
for the 2D CRV robot. Callers provide an ``ObjectCentricState`` together with a
goal base pose, and the planner returns a list of bounded CRV actions.

The public entry points are intentionally generic:

- ``plan_crv_actions()``: main interface for base motion planning.
- ``plan_crv_base_actions()``: compatibility wrapper for base-only planning.
- ``plan_crv_holding_actions()``: compatibility wrapper for holding-aware planning.
- ``crv_action_plan_to_pose_plan()`` / ``crv_pose_plan_to_action_plan()``:
  utility conversions between discrete pose plans and action plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from kinder.core import RobotActionSpace
from kinder.envs.geom2d.object_types import DoubleRectType, RectangleType
from kinder.envs.geom2d.structs import MultiBody2D, SE2Pose, ZOrder
from kinder.envs.utils import (
    BLACK,
    crv_robot_to_multibody2d,
    double_rectangle_object_to_part_geom,
    get_se2_pose,
    object_to_multibody2d,
    rectangle_object_to_geom,
    state_2d_has_collision,
)
from numpy.typing import NDArray
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs import Array, Object, ObjectCentricState
from tomsgeoms2d.structs import Rectangle
from tomsgeoms2d.utils import find_closest_points, geom2ds_intersect

from robocode.primitives.motion_planning import BiRRT


@dataclass(frozen=True)
class CRVConfig:
    """A minimal SE(2) configuration for the CRV robot base.

    This is the pose-level representation used by the public planner API.
    """

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class CRVActionLimits:
    """Relative action limits used to discretize CRV interpolation.

    These limits define the maximum per-step change in base translation and
    rotation when converting a continuous path into executable CRV actions.
    """

    max_dx: float
    max_dy: float
    max_dtheta: float


class CRVRobotActionSpace(RobotActionSpace):
    """Action bounds for the 2D CRV robot.

    The action layout is ``[dx, dy, dtheta, darm, vacuum]``.
    """

    def __init__(
        self,
        min_dx: float = -5e-1,
        max_dx: float = 5e-1,
        min_dy: float = -5e-1,
        max_dy: float = 5e-1,
        min_dtheta: float = -np.pi / 16,
        max_dtheta: float = np.pi / 16,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_vac: float = 0.0,
        max_vac: float = 1.0,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_vac])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_vac])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a markdown description of this space."""
        features = [
            ("dx", "Change in robot x position (positive is right)"),
            ("dy", "Change in robot y position (positive is up)"),
            ("dtheta", "Change in robot angle in radians (positive is ccw)"),
            ("darm", "Change in robot arm length (positive is out)"),
            ("vac", "Directly sets the vacuum (0.0 is off, 1.0 is on)"),
        ]
        md_table_str = (
            "| **Index** | **Feature** | **Description** | **Min** | **Max** |"
        )
        md_table_str += "\n| --- | --- | --- | --- | --- |"
        for idx, (feature, description) in enumerate(features):
            lb = self.low[idx]
            ub = self.high[idx]
            md_table_str += (
                f"\n| {idx} | {feature} | {description} | {lb:.3f} | {ub:.3f} |"
            )
        return (
            "The entries of an array in this Box space correspond to the following "
            f"action features:\n{md_table_str}\n"
        )


def create_walls_from_world_boundaries(
    world_min_x: float,
    world_max_x: float,
    world_min_y: float,
    world_max_y: float,
    min_dx: float,
    max_dx: float,
    min_dy: float,
    max_dy: float,
) -> dict[Object, dict[str, float]]:
    """Create synthetic wall objects from workspace boundaries.

    The returned objects can be inserted into an ``ObjectCentricState`` so the
    planner treats workspace limits as ordinary geometric obstacles.
    """
    state_dict: dict[Object, dict[str, float]] = {}
    right_wall = Object("right_wall", RectangleType)
    side_wall_height = world_max_y - world_min_y
    state_dict[right_wall] = {
        "x": world_max_x,
        "y": world_min_y,
        "width": 2 * max_dx,
        "height": side_wall_height,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    left_wall = Object("left_wall", RectangleType)
    state_dict[left_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_min_y,
        "width": 2 * abs(min_dx),
        "height": side_wall_height,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    top_wall = Object("top_wall", RectangleType)
    horiz_wall_width = 2 * 2 * abs(min_dx) + world_max_x - world_min_x
    state_dict[top_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_max_y,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    bottom_wall = Object("bottom_wall", RectangleType)
    state_dict[bottom_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_min_y + 2 * min_dy,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    return state_dict


def get_tool_tip_position(
    state: ObjectCentricState, robot: Object
) -> tuple[float, float]:
    """Return the gripper tool-tip position in world coordinates."""
    multibody = crv_robot_to_multibody2d(robot, state)
    gripper_geom = multibody.get_body("gripper").geom
    assert isinstance(gripper_geom, Rectangle)
    tool_tip = np.array([1.0, 0.5])
    scale_matrix = np.array([[gripper_geom.width, 0], [0, gripper_geom.height]])
    translate_vector = np.array([gripper_geom.x, gripper_geom.y])
    tool_tip = tool_tip @ scale_matrix.T
    tool_tip = tool_tip @ gripper_geom.rotation_matrix.T
    tool_tip = translate_vector + tool_tip
    return (float(tool_tip[0]), float(tool_tip[1]))


def get_suctioned_objects(
    state: ObjectCentricState, robot: Object
) -> list[tuple[Object, SE2Pose]]:
    """Return movable objects currently attached to the robot suction zone.

    Each result also includes the relative transform from the gripper tool-tip
    frame to the attached object pose.
    """
    if state.get(robot, "vacuum") <= 0.5:
        return []
    robot_multibody = crv_robot_to_multibody2d(robot, state)
    suction_body = robot_multibody.get_body("suction")
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    movable_objects = [o for o in state if o != robot and state.get(o, "static") < 0.5]
    suctioned_objects: list[tuple[Object, SE2Pose]] = []
    for obj in movable_objects:
        obj_multibody = object_to_multibody2d(obj, state, {})
        for obj_body in obj_multibody.bodies:
            if geom2ds_intersect(suction_body.geom, obj_body.geom):
                world_to_obj = get_se2_pose(state, obj)
                gripper_to_obj = world_to_gripper.inverse * world_to_obj
                suctioned_objects.append((obj, gripper_to_obj))
    return suctioned_objects


def snap_suctioned_objects(
    state: ObjectCentricState,
    robot: Object,
    suctioned_objs: list[tuple[Object, SE2Pose]],
) -> None:
    """Update attached-object poses so they rigidly follow the current gripper."""
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    for obj, gripper_to_obj in suctioned_objs:
        world_to_obj = world_to_gripper * gripper_to_obj
        state.set(obj, "x", world_to_obj.x)
        state.set(obj, "y", world_to_obj.y)
        state.set(obj, "theta", world_to_obj.theta)


def move_objects_in_contact(
    state: ObjectCentricState,
    robot: Object,
    suctioned_objs: list[tuple[Object, SE2Pose]],
) -> tuple[ObjectCentricState, set[tuple[Object, SE2Pose]]]:
    """Propagate contact from suctioned objects to nearby movable objects.

    This is a conservative approximation used during holding-aware planning so
    the planner can reject motions that would shove other movable objects.
    """
    moved_objects: list[tuple[Object, SE2Pose]] = []
    moving_objects = {robot} | {o for o, _ in suctioned_objs}
    nonstatic_objects = {
        o for o in state if (o not in moving_objects) and (not state.get(o, "static"))
    }

    for contact_obj in nonstatic_objects:
        for suctioned_obj, _ in suctioned_objs:
            suctioned_body = object_to_multibody2d(suctioned_obj, state, {})
            contact_body = object_to_multibody2d(contact_obj, state, {})
            for b1 in suctioned_body.bodies:
                for b2 in contact_body.bodies:
                    if geom2ds_intersect(b1.geom, b2.geom):
                        closest_points_b1, closest_points_b2, _ = find_closest_points(
                            b1.geom, b2.geom
                        )
                        contact_vec = np.array(closest_points_b2) - np.array(
                            closest_points_b1
                        )
                        state.set(
                            contact_obj,
                            "x",
                            state.get(contact_obj, "x") + contact_vec[0],
                        )
                        state.set(
                            contact_obj,
                            "y",
                            state.get(contact_obj, "y") + contact_vec[1],
                        )
                        moved_objects.append(
                            (contact_obj, get_se2_pose(state, contact_obj))
                        )
                        return state, set(moved_objects)
    return state, set(moved_objects)


def _default_action_space(
    action_limits: CRVActionLimits | None = None,
) -> CRVRobotActionSpace:
    if action_limits is None:
        return CRVRobotActionSpace()
    return CRVRobotActionSpace(
        min_dx=-abs(action_limits.max_dx),
        max_dx=abs(action_limits.max_dx),
        min_dy=-abs(action_limits.max_dy),
        max_dy=abs(action_limits.max_dy),
        min_dtheta=-abs(action_limits.max_dtheta),
        max_dtheta=abs(action_limits.max_dtheta),
    )


def _find_robot_object(state: ObjectCentricState) -> Object:
    robot = state.get_object_from_name("robot")
    assert robot is not None, "ObjectCentricState must contain object named 'robot'."
    return robot


def _run_motion_planning_for_crv_robot(
    state: ObjectCentricState,
    robot: Object,
    target_pose: SE2Pose,
    action_space: CRVRobotActionSpace,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    sample_goal_eps: float = 0.0,
    enable_contact_propagation: bool = True,
    ignored_obstacles: set[Object] | None = None,
) -> list[SE2Pose] | None:
    """Run BiRRT motion planning with exact geometric collision checks."""
    if static_object_body_cache is None:
        static_object_body_cache = {}

    rng = np.random.default_rng(seed)
    x_lb, x_ub, y_lb, y_ub = np.inf, -np.inf, np.inf, -np.inf
    for obj in state:
        pose = get_se2_pose(state, obj)
        x_lb = min(x_lb, pose.x)
        x_ub = max(x_ub, pose.x)
        y_lb = min(y_lb, pose.y)
        y_ub = max(y_ub, pose.y)

    static_object_body_cache = static_object_body_cache.copy()
    suctioned_objects = get_suctioned_objects(state, robot)
    moving_objects = {robot} | {o for o, _ in suctioned_objects}
    ignored = set() if ignored_obstacles is None else set(ignored_obstacles)
    static_state = state.copy()
    for obj in static_state:
        if obj in moving_objects:
            continue
        static_state.set(obj, "static", 1.0)

    def sample_fn(_: SE2Pose) -> SE2Pose:
        return SE2Pose(
            x=float(rng.uniform(x_lb, x_ub)),
            y=float(rng.uniform(y_lb, y_ub)),
            theta=float(rng.uniform(-np.pi, np.pi)),
        )

    def extend_fn(pt1: SE2Pose, pt2: SE2Pose) -> Iterable[SE2Pose]:
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        abs_x = action_space.high[0] if dx > 0 else abs(action_space.low[0])
        abs_y = action_space.high[1] if dy > 0 else abs(action_space.low[1])
        abs_theta = action_space.high[2] if dtheta > 0 else abs(action_space.low[2])
        x_num_steps = int(abs(dx) / abs_x) + 1
        y_num_steps = int(abs(dy) / abs_y) + 1
        theta_num_steps = int(abs(dtheta) / abs_theta) + 1
        num_steps = max(x_num_steps, y_num_steps, theta_num_steps)
        x = pt1.x
        y = pt1.y
        theta = pt1.theta
        yield SE2Pose(x, y, theta)
        for _ in range(num_steps):
            x += dx / num_steps
            y += dy / num_steps
            theta = wrap_angle(theta + dtheta / num_steps)
            yield SE2Pose(float(x), float(y), float(theta))

    def collision_fn(pt: SE2Pose) -> bool:
        state_for_check = (
            static_state.copy()
            if enable_contact_propagation and suctioned_objects
            else static_state
        )
        state_for_check.set(robot, "x", pt.x)
        state_for_check.set(robot, "y", pt.y)
        state_for_check.set(robot, "theta", pt.theta)

        snap_suctioned_objects(state_for_check, robot, suctioned_objects)
        if enable_contact_propagation and suctioned_objects:
            for _ in range(8):
                state_for_check, moved = move_objects_in_contact(
                    state_for_check, robot, suctioned_objects
                )
                if not moved:
                    break
        obstacle_objects = set(state_for_check) - moving_objects - ignored
        return state_2d_has_collision(
            state_for_check,
            moving_objects,
            obstacle_objects,
            static_object_body_cache,
        )

    def distance_fn(pt1: SE2Pose, pt2: SE2Pose) -> float:
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        return float(np.sqrt(dx**2 + dy**2) + abs(dtheta))

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
    return birrt.query(initial_pose, target_pose, sample_goal_eps=sample_goal_eps)


def crv_pose_plan_to_action_plan(
    pose_plan: list[SE2Pose],
    action_space: CRVRobotActionSpace,
    vacuum_while_moving: bool = False,
) -> list[Array]:
    """Convert a CRV pose plan into bounded CRV action deltas.

    Args:
        pose_plan: Discrete sequence of SE(2) robot poses.
        action_space: Action bounds used to shape the returned arrays.
        vacuum_while_moving: Whether motion actions should keep vacuum on.

    Returns:
        A list of CRV actions with layout ``[dx, dy, dtheta, darm, vacuum]``.
    """
    action_plan: list[Array] = []
    for pt1, pt2 in zip(pose_plan[:-1], pose_plan[1:]):
        action = np.zeros_like(action_space.high)
        action[0] = pt2.x - pt1.x
        action[1] = pt2.y - pt1.y
        action[2] = get_signed_angle_distance(pt2.theta, pt1.theta)
        action[4] = 1.0 if vacuum_while_moving else 0.0
        action_plan.append(action.astype(np.float32))
    return action_plan


def crv_action_plan_to_pose_plan(
    start: CRVConfig,
    actions: list[NDArray[np.float32]],
) -> list[CRVConfig]:
    """Integrate CRV base actions into a discrete pose path.

    Args:
        start: Starting base pose.
        actions: Sequence of CRV actions whose first three entries are interpreted
            as ``dx``, ``dy``, and ``dtheta``.

    Returns:
        The corresponding pose path, including the start configuration.
    """
    path = [start]
    current = start
    for action in actions:
        current = CRVConfig(
            x=float(current.x + float(action[0])),
            y=float(current.y + float(action[1])),
            theta=float(wrap_angle(current.theta + float(action[2]))),
        )
        path.append(current)
    return path


def plan_crv_base_actions(
    current_state: ObjectCentricState,
    target_pose: CRVConfig,
    *,
    action_limits: CRVActionLimits | None = None,
    ignore_object_names: set[str] | None = None,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    sample_goal_eps: float = 0.0,
    **_: Any,
) -> list[NDArray[np.float32]] | None:
    """Plan a collision-free base-only action sequence.

    This is a compatibility wrapper around :func:`plan_crv_actions` with
    ``carrying=False``.
    """
    return plan_crv_actions(
        current_state,
        target_pose,
        action_limits=action_limits,
        ignore_object_names=ignore_object_names,
        carrying=False,
        seed=seed,
        num_attempts=num_attempts,
        num_iters=num_iters,
        smooth_amt=smooth_amt,
        sample_goal_eps=sample_goal_eps,
    )


def plan_crv_holding_actions(
    current_state: ObjectCentricState,
    target_pose: CRVConfig,
    *,
    action_limits: CRVActionLimits | None = None,
    ignore_object_names: set[str] | None = None,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    sample_goal_eps: float = 0.0,
    **_: Any,
) -> list[NDArray[np.float32]] | None:
    """Plan a collision-free action sequence while transporting an attached object.

    This is a compatibility wrapper around :func:`plan_crv_actions` with
    ``carrying=True``.
    """
    return plan_crv_actions(
        current_state,
        target_pose,
        action_limits=action_limits,
        ignore_object_names=ignore_object_names,
        carrying=True,
        seed=seed,
        num_attempts=num_attempts,
        num_iters=num_iters,
        smooth_amt=smooth_amt,
        sample_goal_eps=sample_goal_eps,
    )


def plan_crv_actions(
    current_state: ObjectCentricState,
    target_pose: CRVConfig,
    *,
    action_limits: CRVActionLimits | None = None,
    ignore_object_names: set[str] | None = None,
    carrying: bool | None = None,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    sample_goal_eps: float = 0.0,
    **_: Any,
) -> list[NDArray[np.float32]] | None:
    """Plan a collision-free CRV action sequence to a target base pose.

    Args:
        current_state: Object-centric world state containing a CRV robot named
            ``"robot"``.
        target_pose: Desired robot base pose.
        action_limits: Optional per-step CRV motion limits.
        ignore_object_names: Optional object names that should be ignored as
            obstacles during planning.
        carrying: If ``True``, enable holding-aware collision checking and contact
            propagation. If ``None``, this is inferred from the current suctioned
            objects in the state.
        seed: Random seed for BiRRT sampling.
        num_attempts: Number of planner restarts.
        num_iters: Maximum BiRRT iterations per attempt.
        smooth_amt: Number of shortcut-smoothing passes.
        sample_goal_eps: Probability of directly sampling the goal during search.

    Returns:
        A list of executable CRV actions, or ``None`` if no collision-free plan
        is found.
    """
    robot = _find_robot_object(current_state)
    if carrying is None:
        carrying = bool(get_suctioned_objects(current_state, robot))
    ignored: set[Object] = set()
    if ignore_object_names is not None:
        ignored = {
            obj
            for obj in current_state
            if obj.name in ignore_object_names and obj != robot
        }
    action_space = _default_action_space(action_limits)
    pose_plan = _run_motion_planning_for_crv_robot(
        current_state,
        robot,
        SE2Pose(target_pose.x, target_pose.y, target_pose.theta),
        action_space,
        seed=seed,
        num_attempts=num_attempts,
        num_iters=num_iters,
        smooth_amt=smooth_amt,
        sample_goal_eps=sample_goal_eps,
        enable_contact_propagation=carrying,
        ignored_obstacles=ignored,
    )
    if pose_plan is None:
        return None
    return crv_pose_plan_to_action_plan(
        pose_plan, action_space, vacuum_while_moving=carrying
    )


def is_inside(
    state: ObjectCentricState,
    inner: Object,
    outer: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> bool:
    """Return ``True`` if one rectangle object is fully inside another."""
    inner_geom = rectangle_object_to_geom(state, inner, static_object_cache)
    outer_geom = rectangle_object_to_geom(state, outer, static_object_cache)
    for x, y in inner_geom.vertices:
        if not outer_geom.contains_point(x, y):
            return False
    return True


def is_inside_shelf(
    state: ObjectCentricState,
    inner: Object,
    outer: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> bool:
    """Return ``True`` if a rectangle is fully inside a shelf opening."""
    assert outer.is_instance(DoubleRectType)
    inner_geom = rectangle_object_to_geom(state, inner, static_object_cache)
    outer_geom = double_rectangle_object_to_part_geom(state, outer, static_object_cache)
    for x, y in inner_geom.vertices:
        if not outer_geom.contains_point(x, y):
            return False
    return True
