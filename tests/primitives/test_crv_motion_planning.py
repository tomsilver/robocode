"""Tests for the generic CRV motion-planning primitive."""

from __future__ import annotations

import math

from robocode.primitives.crv_motion_planning import (
    CRVActionLimits,
    CRVConfig,
    plan_crv_base_path,
    plan_crv_holding_path,
)

ACTION_LIMITS = CRVActionLimits(
    max_dx=0.05,
    max_dy=0.05,
    max_dtheta=math.pi / 16,
)
BOUNDS = (0.0, 1.0, 0.0, 1.0)


def test_plan_crv_base_path_direct():
    """The planner should return a direct path in free space."""
    start = CRVConfig(0.1, 0.1, 0.0)
    goal = CRVConfig(0.4, 0.4, math.pi / 4)
    path = plan_crv_base_path(
        start,
        goal,
        action_limits=ACTION_LIMITS,
        bounds=BOUNDS,
        collision_fn=lambda _: False,
        seed=0,
    )
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal


def test_plan_crv_base_path_avoids_obstacle():
    """The planner should route around a simple circular obstacle."""
    start = CRVConfig(0.1, 0.1, 0.0)
    goal = CRVConfig(0.9, 0.9, 0.0)

    def collision_fn(cfg: CRVConfig) -> bool:
        return math.hypot(cfg.x - 0.5, cfg.y - 0.5) < 0.15

    path = plan_crv_base_path(
        start,
        goal,
        action_limits=ACTION_LIMITS,
        bounds=BOUNDS,
        collision_fn=collision_fn,
        seed=1,
        num_iters=200,
    )
    assert path is not None
    assert all(not collision_fn(cfg) for cfg in path)


def test_plan_crv_base_path_returns_none_when_goal_blocked():
    """The planner should fail cleanly when the goal configuration is blocked."""
    start = CRVConfig(0.1, 0.1, 0.0)
    goal = CRVConfig(0.8, 0.8, 0.0)
    path = plan_crv_base_path(
        start,
        goal,
        action_limits=ACTION_LIMITS,
        bounds=BOUNDS,
        collision_fn=lambda cfg: cfg.x > 0.6 and cfg.y > 0.6,
        seed=2,
        num_iters=50,
    )
    assert path is None


def test_plan_crv_holding_path_obeys_collision_callback():
    """Holding-path planning should honor a stricter carrying collision model."""
    start = CRVConfig(0.1, 0.2, 0.0)
    goal = CRVConfig(0.9, 0.2, 0.0)

    def holding_collision(cfg: CRVConfig) -> bool:
        return math.hypot(cfg.x - 0.5, cfg.y - 0.2) < 0.18

    path = plan_crv_holding_path(
        start,
        goal,
        action_limits=ACTION_LIMITS,
        bounds=BOUNDS,
        collision_fn=holding_collision,
        seed=3,
        num_iters=200,
    )
    assert path is not None
    assert all(not holding_collision(cfg) for cfg in path)
