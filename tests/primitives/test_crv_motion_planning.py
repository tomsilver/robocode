"""Tests for the generic CRV geometric motion-planning primitive."""

from __future__ import annotations

import numpy as np
from kinder.envs.geom2d.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from relational_structs import Object, ObjectCentricState

from robocode.primitives.crv_motion_planning import (
    CRVConfig,
    create_walls_from_world_boundaries,
    crv_action_plan_to_pose_plan,
    plan_crv_actions,
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
    *,
    theta: float = 0.0,
    static: float = 0.0,
) -> np.ndarray:
    x = center_x - width / 2
    y = center_y - height / 2
    return np.array(
        [
            x,
            y,
            theta,
            static,
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
    robot_xy: tuple[float, float] = (0.25, 0.25),
    robot_theta: float = 0.0,
    arm_joint: float = 0.12,
    vacuum: float = 0.0,
    block_specs: dict[str, tuple[float, float, float, float]] | None = None,
) -> ObjectCentricState:
    data: dict[Object, np.ndarray] = {
        _ROBOT: np.array(
            [
                robot_xy[0],
                robot_xy[1],
                robot_theta,
                0.12,
                arm_joint,
                0.5,
                vacuum,
                0.06,
                0.04,
            ],
            dtype=np.float32,
        )
    }
    for name, (cx, cy, width, height) in (block_specs or {}).items():
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


def _integrate(state: ObjectCentricState, actions: list[np.ndarray]) -> list[CRVConfig]:
    start = CRVConfig(
        x=float(state.get(_ROBOT, "x")),
        y=float(state.get(_ROBOT, "y")),
        theta=float(state.get(_ROBOT, "theta")),
    )
    return crv_action_plan_to_pose_plan(start, actions)


def test_plan_crv_base_actions_direct() -> None:
    """The planner should return a direct action plan in free space."""
    state = _state()
    goal = CRVConfig(0.55, 0.55, np.pi / 4)
    actions = plan_crv_actions(state, goal, carrying=False, seed=0)
    assert actions is not None
    assert actions
    path = _integrate(state, actions)
    assert np.isclose(path[-1].x, goal.x)
    assert np.isclose(path[-1].y, goal.y)
    assert np.isclose(path[-1].theta, goal.theta)


def test_plan_crv_base_actions_avoid_obstacle() -> None:
    """The planner should route around a blocking object."""
    state = _state(block_specs={"blocker": (0.50, 0.50, 0.18, 0.18)})
    goal = CRVConfig(0.75, 0.75, 0.0)
    actions = plan_crv_actions(state, goal, carrying=False, seed=1, num_iters=200)
    assert actions is not None
    path = _integrate(state, actions)
    for cfg in path:
        assert not (0.41 <= cfg.x <= 0.59 and 0.41 <= cfg.y <= 0.59)


def test_plan_crv_base_actions_returns_none_when_goal_blocked() -> None:
    """The planner should fail cleanly when the goal region is blocked."""
    state = _state(block_specs={"wall": (0.8, 0.8, 0.30, 0.30)})
    goal = CRVConfig(0.8, 0.8, 0.0)
    actions = plan_crv_actions(state, goal, carrying=False, seed=2, num_iters=80)
    assert actions is None


def test_plan_crv_holding_actions_produces_vacuum_on_motion() -> None:
    """Holding-path planning should output motion with vacuum enabled."""
    state = _state(
        robot_xy=(0.25, 0.2),
        arm_joint=0.12,
        vacuum=1.0,
        block_specs={"held": (0.52, 0.2, 0.08, 0.08)},
    )
    goal = CRVConfig(0.7, 0.22, 0.0)
    actions = plan_crv_actions(state, goal, carrying=True, seed=3, num_iters=220)
    assert actions is not None
    assert actions
    assert all(float(action[4]) == 1.0 for action in actions)
