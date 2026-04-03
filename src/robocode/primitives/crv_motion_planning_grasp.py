"""Generic CRV grasp-planning helpers built on top of CRV motion planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from robocode.primitives.crv_motion_planning import (
    CRVActionLimits,
    CRVConfig,
    PlannerBounds,
    plan_crv_base_path,
    wrap_angle,
)


@dataclass(frozen=True)
class RelativeGraspPose:
    """A grasp pose expressed in the target object's center frame."""

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class CRVGraspWaypoint:
    """A full CRV waypoint including arm and vacuum commands."""

    x: float
    y: float
    theta: float
    arm_joint: float
    vacuum: float


class SuctionFailedEmptySpaceError(RuntimeError):
    """Raised when the final suction attempt does not intersect the target."""


class SuctionFailedNoCollisionFreePathError(RuntimeError):
    """Raised when no collision-free grasp path can be constructed."""


SegmentCollisionFn = Callable[[tuple[float, float], tuple[float, float]], bool]
ExtensionCollisionFn = Callable[[str, CRVGraspWaypoint], bool]
SuctionSuccessFn = Callable[[str, CRVGraspWaypoint, float], bool]


def _compose_grasp_pose(block: Any, relative_grasp_pose: RelativeGraspPose) -> CRVConfig:
    """Compose a target-relative grasp pose into a world-frame base pose."""
    center_x, center_y = block.center
    c = float(np.cos(block.theta))
    s = float(np.sin(block.theta))
    dx = c * relative_grasp_pose.x - s * relative_grasp_pose.y
    dy = s * relative_grasp_pose.x + c * relative_grasp_pose.y
    return CRVConfig(
        x=center_x + dx,
        y=center_y + dy,
        theta=wrap_angle(block.theta + relative_grasp_pose.theta),
    )


def _point_in_bounds(point: tuple[float, float], bounds: PlannerBounds) -> bool:
    min_x, max_x, min_y, max_y = bounds
    return bool(min_x <= point[0] <= max_x and min_y <= point[1] <= max_y)


def plan_crv_grasp(
    current_state: Any,
    grasp_target_object: str,
    relative_grasp_pose: RelativeGraspPose,
    grasping_arm_length: float,
    *,
    action_limits: CRVActionLimits,
    bounds: PlannerBounds,
    collision_fn: Callable[[CRVConfig], bool],
    segment_collision_free_fn: SegmentCollisionFn,
    extension_collision_free_fn: ExtensionCollisionFn,
    suction_success_fn: SuctionSuccessFn,
    pre_grasp_margin: float = 0.0,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    sample_goal_eps: float = 0.0,
    vacuum_on: float = 1.0,
    vacuum_off: float = 0.0,
) -> list[CRVGraspWaypoint]:
    """Plan a collision-free base path and final suction sequence for one grasp."""
    robot = current_state.robot
    target = current_state.blocks[grasp_target_object]
    current_cfg = CRVConfig(robot.x, robot.y, robot.theta)
    grasp_cfg = _compose_grasp_pose(target, relative_grasp_pose)
    pre_grasp_cfg = CRVConfig(
        x=grasp_cfg.x - pre_grasp_margin * float(np.cos(grasp_cfg.theta)),
        y=grasp_cfg.y - pre_grasp_margin * float(np.sin(grasp_cfg.theta)),
        theta=grasp_cfg.theta,
    )
    if not _point_in_bounds((pre_grasp_cfg.x, pre_grasp_cfg.y), bounds):
        raise SuctionFailedNoCollisionFreePathError(
            "Sucting failed, no collision free path found."
        )

    path = plan_crv_base_path(
        current_cfg,
        pre_grasp_cfg,
        action_limits=action_limits,
        bounds=bounds,
        collision_fn=collision_fn,
        seed=seed,
        num_attempts=num_attempts,
        num_iters=num_iters,
        smooth_amt=smooth_amt,
        sample_goal_eps=sample_goal_eps,
    )
    if path is None:
        raise SuctionFailedNoCollisionFreePathError(
            "Sucting failed, no collision free path found."
        )
    if not segment_collision_free_fn(
        (pre_grasp_cfg.x, pre_grasp_cfg.y),
        (grasp_cfg.x, grasp_cfg.y),
    ):
        raise SuctionFailedNoCollisionFreePathError(
            "Sucting failed, no collision free path found."
        )

    retract_arm = robot.base_radius
    waypoints = [
        CRVGraspWaypoint(
            x=current_cfg.x,
            y=current_cfg.y,
            theta=current_cfg.theta,
            arm_joint=robot.arm_joint,
            vacuum=robot.vacuum,
        )
    ]
    if robot.arm_joint > retract_arm:
        waypoints.append(
            CRVGraspWaypoint(
                x=current_cfg.x,
                y=current_cfg.y,
                theta=current_cfg.theta,
                arm_joint=retract_arm,
                vacuum=vacuum_off,
            )
        )
    waypoints.extend(
        CRVGraspWaypoint(
            x=cfg.x,
            y=cfg.y,
            theta=cfg.theta,
            arm_joint=retract_arm,
            vacuum=vacuum_off,
        )
        for cfg in path[1:]
    )
    grasp_waypoint = CRVGraspWaypoint(
        x=grasp_cfg.x,
        y=grasp_cfg.y,
        theta=grasp_cfg.theta,
        arm_joint=retract_arm,
        vacuum=vacuum_off,
    )
    if not extension_collision_free_fn(grasp_target_object, grasp_waypoint):
        raise SuctionFailedNoCollisionFreePathError(
            "Sucting failed, no collision free path found."
        )
    if not suction_success_fn(grasp_target_object, grasp_waypoint, grasping_arm_length):
        raise SuctionFailedEmptySpaceError(
            "Sucting failed, trying to suction empty space."
        )

    waypoints.extend(
        [
            grasp_waypoint,
            CRVGraspWaypoint(
                x=grasp_cfg.x,
                y=grasp_cfg.y,
                theta=grasp_cfg.theta,
                arm_joint=grasping_arm_length,
                vacuum=vacuum_off,
            ),
            CRVGraspWaypoint(
                x=grasp_cfg.x,
                y=grasp_cfg.y,
                theta=grasp_cfg.theta,
                arm_joint=grasping_arm_length,
                vacuum=vacuum_on,
            ),
        ]
    )
    return waypoints
