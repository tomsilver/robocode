"""Tests for the CRV geometric grasp-planning primitive."""

from __future__ import annotations

import numpy as np
from kinder.envs.geom2d.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from relational_structs import Object, ObjectCentricState

from robocode.primitives.crv_motion_planning import create_walls_from_world_boundaries
from robocode.primitives.crv_motion_planning_grasp import (
    RelativeGraspPose,
    SuctionFailedEmptySpaceError,
    SuctionFailedNoCollisionFreePathError,
    plan_crv_grasp,
)

WORLD_MIN_X = 0.0
WORLD_MAX_X = 1.0
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 1.0
DX_LIM = 0.05
DY_LIM = 0.05

_ROBOT = Object("robot", CRVRobotType)
_TYPE_FEATURES = {
    CRVRobotType: Geom2DRobotEnvTypeFeatures[CRVRobotType],
    RectangleType: Geom2DRobotEnvTypeFeatures[RectangleType],
}


def _rect_from_center(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> np.ndarray:
    return np.array(
        [
            center_x - width / 2,
            center_y - height / 2,
            0.0,
            0.0,
            0.2,
            0.2,
            0.2,
            100.0,
            width,
            height,
        ],
        dtype=np.float32,
    )


def _state(
    *,
    robot_xy: tuple[float, float] = (0.25, 0.2),
    target_center: tuple[float, float] = (0.78, 0.32),
    extra_blocks: dict[str, tuple[float, float, float, float]] | None = None,
) -> ObjectCentricState:
    data: dict[Object, np.ndarray] = {
        _ROBOT: np.array(
            [robot_xy[0], robot_xy[1], 0.0, 0.12, 0.12, 0.5, 0.0, 0.06, 0.04],
            dtype=np.float32,
        ),
        Object("target", RectangleType): _rect_from_center(
            target_center[0], target_center[1], 0.24, 0.06
        ),
    }
    for name, (cx, cy, width, height) in (extra_blocks or {}).items():
        data[Object(name, RectangleType)] = _rect_from_center(cx, cy, width, height)

    rect_features = Geom2DRobotEnvTypeFeatures[RectangleType]
    walls = create_walls_from_world_boundaries(
        WORLD_MIN_X,
        WORLD_MAX_X,
        WORLD_MIN_Y,
        WORLD_MAX_Y,
        min_dx=-DX_LIM,
        max_dx=DX_LIM,
        min_dy=-DY_LIM,
        max_dy=DY_LIM,
    )
    for wall_obj, wall_dict in walls.items():
        data[wall_obj] = np.array(
            [float(wall_dict[feature]) for feature in rect_features],
            dtype=np.float32,
        )
    return ObjectCentricState(data, _TYPE_FEATURES)


def test_plan_crv_grasp_success() -> None:
    """The grasp planner should produce a valid suction sequence."""
    state = _state()
    waypoints = plan_crv_grasp(
        state,
        "target",
        RelativeGraspPose(x=-0.56, y=0.0, theta=0.0),
        0.5,
        pre_grasp_margin=0.02,
        seed=0,
    )
    assert waypoints
    assert waypoints[-1].vacuum == 1.0
    assert np.isclose(waypoints[-1].arm_joint, 0.5)


def test_plan_crv_grasp_raises_empty_space() -> None:
    """The grasp planner should report empty-space suction explicitly."""
    state = _state()
    try:
        plan_crv_grasp(
            state,
            "target",
            RelativeGraspPose(x=-0.56, y=0.0, theta=np.pi / 2),
            0.5,
            pre_grasp_margin=0.02,
            seed=1,
        )
    except SuctionFailedEmptySpaceError:
        return
    assert False, "Expected SuctionFailedEmptySpaceError"


def test_plan_crv_grasp_raises_no_path() -> None:
    """The grasp planner should fail when no collision-free path exists."""
    state = _state(extra_blocks={"blocker": (0.22, 0.32, 0.20, 0.20)})
    try:
        plan_crv_grasp(
            state,
            "target",
            RelativeGraspPose(x=-0.56, y=0.0, theta=0.0),
            0.5,
            seed=2,
            num_iters=40,
        )
    except SuctionFailedNoCollisionFreePathError:
        return
    assert False, "Expected SuctionFailedNoCollisionFreePathError"


def test_plan_crv_grasp_raises_when_final_approach_blocked() -> None:
    """A blocked pre-grasp to grasp segment should raise no-path error."""
    state = _state(extra_blocks={"blocker": (0.15, 0.30, 0.03, 0.03)})
    try:
        plan_crv_grasp(
            state,
            "target",
            RelativeGraspPose(x=-0.56, y=0.0, theta=0.0),
            0.5,
            pre_grasp_margin=0.02,
            seed=3,
        )
    except SuctionFailedNoCollisionFreePathError:
        return
    assert False, "Expected SuctionFailedNoCollisionFreePathError"
