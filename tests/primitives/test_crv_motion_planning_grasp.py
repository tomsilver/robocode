"""Tests for the CRV grasp motion-planning primitive."""

from __future__ import annotations

import math
from dataclasses import dataclass

from robocode.primitives.crv_motion_planning import CRVActionLimits, CRVConfig
from robocode.primitives.crv_motion_planning_grasp import (
    RelativeGraspPose,
    SuctionFailedEmptySpaceError,
    SuctionFailedNoCollisionFreePathError,
    plan_crv_grasp,
)


@dataclass(frozen=True)
class _Robot:
    x: float
    y: float
    theta: float
    base_radius: float
    arm_joint: float
    arm_length: float
    vacuum: float
    gripper_height: float
    gripper_width: float


@dataclass(frozen=True)
class _Block:
    theta: float
    width: float
    height: float
    center: tuple[float, float]


@dataclass(frozen=True)
class _State:
    robot: _Robot
    blocks: dict[str, _Block]


ACTION_LIMITS = CRVActionLimits(max_dx=0.05, max_dy=0.05, max_dtheta=math.pi / 16)
BOUNDS = (0.0, 1.0, 0.0, 1.0)


def _state() -> _State:
    return _State(
        robot=_Robot(
            x=0.1,
            y=0.1,
            theta=0.0,
            base_radius=0.2,
            arm_joint=0.2,
            arm_length=0.5,
            vacuum=0.0,
            gripper_height=0.1,
            gripper_width=0.1,
        ),
        blocks={"target": _Block(theta=0.0, width=0.28, height=0.04, center=(0.7, 0.3))},
    )


def test_plan_crv_grasp_success():
    """The grasp planner should produce a grasp sequence in free space."""
    state = _state()
    relative_pose = RelativeGraspPose(x=-0.45, y=0.0, theta=0.0)
    waypoints = plan_crv_grasp(
        state,
        "target",
        relative_pose,
        0.5,
        action_limits=ACTION_LIMITS,
        bounds=BOUNDS,
        collision_fn=lambda _: False,
        segment_collision_free_fn=lambda _start, _end: True,
        extension_collision_free_fn=lambda _name, _waypoint: True,
        suction_success_fn=lambda _name, _waypoint, _arm: True,
        seed=0,
    )
    assert waypoints
    assert waypoints[-1].vacuum == 1.0
    assert waypoints[-1].arm_joint == 0.5


def test_plan_crv_grasp_raises_empty_space():
    """The grasp planner should report empty-space suction explicitly."""
    state = _state()
    relative_pose = RelativeGraspPose(x=-0.45, y=0.0, theta=0.0)
    try:
        plan_crv_grasp(
            state,
            "target",
            relative_pose,
            0.5,
            action_limits=ACTION_LIMITS,
            bounds=BOUNDS,
            collision_fn=lambda _: False,
            segment_collision_free_fn=lambda _start, _end: True,
            extension_collision_free_fn=lambda _name, _waypoint: True,
            suction_success_fn=lambda _name, _waypoint, _arm: False,
            seed=1,
        )
    except SuctionFailedEmptySpaceError:
        return
    assert False, "Expected SuctionFailedEmptySpaceError"


def test_plan_crv_grasp_raises_no_path():
    """The grasp planner should fail when no collision-free path exists."""
    state = _state()
    relative_pose = RelativeGraspPose(x=-0.45, y=0.0, theta=0.0)
    try:
        plan_crv_grasp(
            state,
            "target",
            relative_pose,
            0.5,
            action_limits=ACTION_LIMITS,
            bounds=BOUNDS,
            collision_fn=lambda cfg: cfg.x > 0.2,
            segment_collision_free_fn=lambda _start, _end: True,
            extension_collision_free_fn=lambda _name, _waypoint: True,
            suction_success_fn=lambda _name, _waypoint, _arm: True,
            seed=2,
            num_iters=20,
        )
    except SuctionFailedNoCollisionFreePathError:
        return
    assert False, "Expected SuctionFailedNoCollisionFreePathError"


def test_plan_crv_grasp_obeys_segment_check():
    """The final short grasp approach should respect the segment validity check."""
    state = _state()
    relative_pose = RelativeGraspPose(x=-0.45, y=0.0, theta=0.0)
    try:
        plan_crv_grasp(
            state,
            "target",
            relative_pose,
            0.5,
            action_limits=ACTION_LIMITS,
            bounds=BOUNDS,
            collision_fn=lambda _cfg: False,
            segment_collision_free_fn=lambda _start, _end: False,
            extension_collision_free_fn=lambda _name, _waypoint: True,
            suction_success_fn=lambda _name, _waypoint, _arm: True,
            seed=3,
        )
    except SuctionFailedNoCollisionFreePathError:
        return
    assert False, "Expected SuctionFailedNoCollisionFreePathError"
