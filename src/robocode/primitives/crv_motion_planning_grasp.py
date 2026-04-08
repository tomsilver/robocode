"""CRV grasp planning built on top of geometric CRV motion planning.

This module provides a higher-level grasp primitive for 2D CRV environments.
Callers specify a target object, a relative grasp pose in the target frame, and
the arm length to use for suction. The planner then:

1. plans a collision-free base path to a pre-grasp pose,
2. checks the short final base approach,
3. checks arm extension to the requested grasp length, and
4. returns a full list of CRV waypoints ending with vacuum-on suction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from kinder.envs.geom2d.object_types import RectangleType
from kinder.envs.geom2d.structs import MultiBody2D, SE2Pose
from kinder.envs.utils import rectangle_object_to_geom, state_2d_has_collision
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs import Object, ObjectCentricState

from robocode.primitives.crv_motion_planning import (
    CRVActionLimits,
    CRVConfig,
    crv_action_plan_to_pose_plan,
    get_suctioned_objects,
    plan_crv_actions,
    snap_suctioned_objects,
)

_COLLISION_CACHE: dict[Object, MultiBody2D] = {}
_POSE_EPS = 1e-6
_ARM_EPS = 1e-6


@dataclass(frozen=True)
class RelativeGraspPose:
    """A grasp pose expressed in the target object's center frame.

    The pose describes where the robot base should be located relative to the
    target object immediately before the final arm extension and suction step.
    """

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class CRVGraspWaypoint:
    """A full CRV waypoint including base pose, arm setting, and vacuum command."""

    x: float
    y: float
    theta: float
    arm_joint: float
    vacuum: float


class SuctionFailedEmptySpaceError(RuntimeError):
    """Raised when the final suction attempt does not intersect the target."""


class SuctionFailedNoCollisionFreePathError(RuntimeError):
    """Raised when no collision-free grasp path can be constructed."""


def _find_robot_object(state: ObjectCentricState) -> Object:
    return state.get_object_from_name("robot")


def _resolve_target_object(state: ObjectCentricState, target: str | Object) -> Object:
    if isinstance(target, Object):
        return target
    resolved = state.get_object_from_name(target)
    if resolved is None:
        raise ValueError(f"ObjectCentricState has no object named '{target}'.")
    return resolved


def _compose_grasp_config(
    state: ObjectCentricState,
    target: Object,
    relative: RelativeGraspPose,
) -> CRVConfig:
    target_geom = rectangle_object_to_geom(state, target, _COLLISION_CACHE)
    center_x, center_y = target_geom.center
    target_theta = state.get(target, "theta")
    world_from_center = SE2Pose(center_x, center_y, target_theta)
    center_from_grasp = SE2Pose(relative.x, relative.y, relative.theta)
    world_from_grasp = world_from_center * center_from_grasp
    return CRVConfig(
        x=float(world_from_grasp.x),
        y=float(world_from_grasp.y),
        theta=float(world_from_grasp.theta),
    )


def _set_robot_pose(
    state: ObjectCentricState,
    robot: Object,
    pose: CRVConfig,
    arm_joint: float,
    vacuum: float,
) -> None:
    state.set(robot, "x", float(pose.x))
    state.set(robot, "y", float(pose.y))
    state.set(robot, "theta", float(pose.theta))
    state.set(robot, "arm_joint", float(arm_joint))
    state.set(robot, "vacuum", float(vacuum))


def _robot_state_in_collision(
    state: ObjectCentricState,
    robot: Object,
    pose: CRVConfig,
    arm_joint: float,
    vacuum: float,
    ignored_objects: set[Object] | None = None,
) -> bool:
    candidate = state.copy()
    _set_robot_pose(candidate, robot, pose, arm_joint, vacuum)
    suctioned = get_suctioned_objects(candidate, robot)
    moving_objects = {robot} | {obj for obj, _ in suctioned}
    snap_suctioned_objects(candidate, robot, suctioned)
    obstacle_objects = set(candidate) - moving_objects
    if ignored_objects is not None:
        obstacle_objects = obstacle_objects - ignored_objects
    return state_2d_has_collision(
        candidate,
        moving_objects,
        obstacle_objects,
        _COLLISION_CACHE,
    )


def _segment_collision_free(
    state: ObjectCentricState,
    robot: Object,
    start_pose: CRVConfig,
    end_pose: CRVConfig,
    arm_joint: float,
    action_limits: CRVActionLimits,
    ignored_objects: set[Object] | None = None,
) -> bool:
    dx = end_pose.x - start_pose.x
    dy = end_pose.y - start_pose.y
    dtheta = get_signed_angle_distance(end_pose.theta, start_pose.theta)
    steps = max(
        int(np.ceil(abs(dx) / action_limits.max_dx)),
        int(np.ceil(abs(dy) / action_limits.max_dy)),
        int(np.ceil(abs(dtheta) / action_limits.max_dtheta)),
        1,
    )
    for idx in range(1, steps + 1):
        ratio = idx / steps
        pose = CRVConfig(
            x=float(start_pose.x + ratio * dx),
            y=float(start_pose.y + ratio * dy),
            theta=float(wrap_angle(start_pose.theta + ratio * dtheta)),
        )
        if _robot_state_in_collision(
            state,
            robot,
            pose,
            arm_joint,
            vacuum=0.0,
            ignored_objects=ignored_objects,
        ):
            return False
    return True


def _extension_collision_free(
    state: ObjectCentricState,
    robot: Object,
    pose: CRVConfig,
    start_arm: float,
    end_arm: float,
    ignored_objects: set[Object] | None = None,
) -> bool:
    steps = max(int(np.ceil(abs(end_arm - start_arm) / 0.01)), 2)
    for arm in np.linspace(start_arm, end_arm, steps):
        if _robot_state_in_collision(
            state,
            robot,
            pose,
            float(arm),
            vacuum=0.0,
            ignored_objects=ignored_objects,
        ):
            return False
    return True


def _suction_hits_target(
    state: ObjectCentricState,
    robot: Object,
    target: Object,
    pose: CRVConfig,
    arm_joint: float,
    vacuum_on: float,
) -> bool:
    candidate = state.copy()
    _set_robot_pose(candidate, robot, pose, arm_joint, vacuum_on)
    suctioned = get_suctioned_objects(candidate, robot)
    return any(obj == target for obj, _ in suctioned)


def plan_crv_grasp(
    current_state: ObjectCentricState,
    grasp_target_object: str | Object,
    relative_grasp_pose: RelativeGraspPose,
    grasping_arm_length: float,
    *,
    pre_grasp_margin: float = 0.0,
    action_limits: CRVActionLimits | None = None,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    sample_goal_eps: float = 0.0,
    vacuum_on: float = 1.0,
    vacuum_off: float = 0.0,
) -> list[CRVGraspWaypoint]:
    """Plan a collision-free grasp and finish with extend-and-suction.

    Args:
        current_state: Object-centric world state containing a CRV robot named
            ``"robot"``.
        grasp_target_object: Object or object name to grasp. The target must be a
            rectangle object.
        relative_grasp_pose: Desired final base pose expressed in the target
            object's center frame.
        grasping_arm_length: Arm extension to use for the final suction step.
        pre_grasp_margin: Optional offset that backs the pre-grasp pose away from
            the final grasp pose along the grasp direction.
        action_limits: Optional per-step CRV base motion limits.
        seed: Random seed for BiRRT sampling.
        num_attempts: Number of planner restarts.
        num_iters: Maximum BiRRT iterations per attempt.
        smooth_amt: Number of shortcut-smoothing passes.
        sample_goal_eps: Probability of directly sampling the goal during search.
        vacuum_on: Vacuum value used for the final suction waypoint.
        vacuum_off: Vacuum value used before the final suction waypoint.

    Returns:
        A list of ``CRVGraspWaypoint`` objects describing the entire grasp
        execution, including any arm retraction, base motion, final extension,
        and the terminal vacuum-on waypoint.

    Raises:
        SuctionFailedEmptySpaceError: suction misses the target object.
        SuctionFailedNoCollisionFreePathError: no collision-free path exists.
    """
    robot = _find_robot_object(current_state)
    target = _resolve_target_object(current_state, grasp_target_object)
    if not target.is_instance(RectangleType):
        raise ValueError("grasp_target_object must be a rectangle object.")

    limits = action_limits or CRVActionLimits(
        max_dx=0.05,
        max_dy=0.05,
        max_dtheta=np.pi / 16,
    )

    start_pose = CRVConfig(
        x=float(current_state.get(robot, "x")),
        y=float(current_state.get(robot, "y")),
        theta=float(current_state.get(robot, "theta")),
    )
    current_arm = float(current_state.get(robot, "arm_joint"))
    retract_arm = float(current_state.get(robot, "base_radius"))

    grasp_pose = _compose_grasp_config(current_state, target, relative_grasp_pose)
    pre_grasp_pose = CRVConfig(
        x=float(grasp_pose.x - pre_grasp_margin * np.cos(grasp_pose.theta)),
        y=float(grasp_pose.y - pre_grasp_margin * np.sin(grasp_pose.theta)),
        theta=float(grasp_pose.theta),
    )

    actions = plan_crv_actions(
        current_state,
        pre_grasp_pose,
        action_limits=limits,
        carrying=False,
        seed=seed,
        num_attempts=num_attempts,
        num_iters=num_iters,
        smooth_amt=smooth_amt,
        sample_goal_eps=sample_goal_eps,
    )
    if actions is None:
        raise SuctionFailedNoCollisionFreePathError(
            "Sucting failed, no collision free path found."
        )

    if not _segment_collision_free(
        current_state,
        robot,
        pre_grasp_pose,
        grasp_pose,
        retract_arm,
        limits,
    ):
        raise SuctionFailedNoCollisionFreePathError(
            "Sucting failed, no collision free path found."
        )

    if not _extension_collision_free(
        current_state,
        robot,
        grasp_pose,
        retract_arm,
        float(grasping_arm_length),
        ignored_objects={target},
    ):
        raise SuctionFailedNoCollisionFreePathError(
            "Sucting failed, no collision free path found."
        )

    if _robot_state_in_collision(
        current_state,
        robot,
        grasp_pose,
        float(grasping_arm_length),
        vacuum_on,
        ignored_objects={target},
    ):
        raise SuctionFailedNoCollisionFreePathError(
            "Sucting failed, no collision free path found."
        )

    if not _suction_hits_target(
        current_state,
        robot,
        target,
        grasp_pose,
        float(grasping_arm_length),
        vacuum_on,
    ):
        raise SuctionFailedEmptySpaceError(
            "Sucting failed, trying to suction empty space."
        )

    path = crv_action_plan_to_pose_plan(start_pose, actions)
    waypoints = [
        CRVGraspWaypoint(
            x=float(start_pose.x),
            y=float(start_pose.y),
            theta=float(start_pose.theta),
            arm_joint=current_arm,
            vacuum=float(current_state.get(robot, "vacuum")),
        )
    ]
    if current_arm > retract_arm + _ARM_EPS:
        waypoints.append(
            CRVGraspWaypoint(
                x=float(start_pose.x),
                y=float(start_pose.y),
                theta=float(start_pose.theta),
                arm_joint=retract_arm,
                vacuum=vacuum_off,
            )
        )
    waypoints.extend(
        CRVGraspWaypoint(
            x=float(pose.x),
            y=float(pose.y),
            theta=float(pose.theta),
            arm_joint=retract_arm,
            vacuum=vacuum_off,
        )
        for pose in path[1:]
    )
    if (
        abs(pre_grasp_pose.x - grasp_pose.x) > _POSE_EPS
        or abs(pre_grasp_pose.y - grasp_pose.y) > _POSE_EPS
        or abs(get_signed_angle_distance(grasp_pose.theta, pre_grasp_pose.theta))
        > _POSE_EPS
    ):
        waypoints.append(
            CRVGraspWaypoint(
                x=float(grasp_pose.x),
                y=float(grasp_pose.y),
                theta=float(grasp_pose.theta),
                arm_joint=retract_arm,
                vacuum=vacuum_off,
            )
        )
    waypoints.extend(
        [
            CRVGraspWaypoint(
                x=float(grasp_pose.x),
                y=float(grasp_pose.y),
                theta=float(grasp_pose.theta),
                arm_joint=float(grasping_arm_length),
                vacuum=vacuum_off,
            ),
            CRVGraspWaypoint(
                x=float(grasp_pose.x),
                y=float(grasp_pose.y),
                theta=float(grasp_pose.theta),
                arm_joint=float(grasping_arm_length),
                vacuum=vacuum_on,
            ),
        ]
    )
    return waypoints
