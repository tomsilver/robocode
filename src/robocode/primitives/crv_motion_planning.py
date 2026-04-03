"""Generic CRV motion planning helpers built on top of ``BiRRT``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from robocode.primitives.motion_planning import BiRRT


def wrap_angle(theta: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class CRVConfig:
    """A minimal SE(2) configuration for the CRV robot base."""

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class CRVActionLimits:
    """Relative action limits used to discretize CRV interpolation."""

    max_dx: float
    max_dy: float
    max_dtheta: float


PlannerBounds = tuple[float, float, float, float]
CollisionFn = Callable[[CRVConfig], bool]


def _extend_config(
    start: CRVConfig,
    goal: CRVConfig,
    action_limits: CRVActionLimits,
) -> list[CRVConfig]:
    dx = goal.x - start.x
    dy = goal.y - start.y
    dtheta = wrap_angle(goal.theta - start.theta)
    steps = max(
        1,
        int(np.ceil(abs(dx) / action_limits.max_dx)) if action_limits.max_dx > 0 else 1,
        int(np.ceil(abs(dy) / action_limits.max_dy)) if action_limits.max_dy > 0 else 1,
        (
            int(np.ceil(abs(dtheta) / action_limits.max_dtheta))
            if action_limits.max_dtheta > 0
            else 1
        ),
    )
    sequence: list[CRVConfig] = []
    for step in range(1, steps + 1):
        t = step / steps
        sequence.append(
            CRVConfig(
                x=start.x + t * dx,
                y=start.y + t * dy,
                theta=wrap_angle(start.theta + t * dtheta),
            )
        )
    return sequence


def _distance_fn(cfg1: CRVConfig, cfg2: CRVConfig) -> float:
    """Return a simple translation + rotation distance between two configs."""
    dx = cfg2.x - cfg1.x
    dy = cfg2.y - cfg1.y
    dtheta = wrap_angle(cfg2.theta - cfg1.theta)
    return float(np.hypot(dx, dy) + abs(dtheta))


def _sample_config(
    rng: np.random.Generator,
    bounds: PlannerBounds,
) -> CRVConfig:
    min_x, max_x, min_y, max_y = bounds
    return CRVConfig(
        x=float(rng.uniform(min_x, max_x)),
        y=float(rng.uniform(min_y, max_y)),
        theta=float(rng.uniform(-np.pi, np.pi)),
    )


def _plan_crv_path(
    start: CRVConfig,
    goal: CRVConfig,
    *,
    action_limits: CRVActionLimits,
    bounds: PlannerBounds,
    collision_fn: CollisionFn,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    sample_goal_eps: float = 0.0,
) -> list[CRVConfig] | None:
    rng = np.random.default_rng(seed)
    planner = BiRRT(
        sample_fn=lambda _: _sample_config(rng, bounds),
        extend_fn=lambda cfg1, cfg2: _extend_config(cfg1, cfg2, action_limits),
        collision_fn=collision_fn,
        distance_fn=_distance_fn,
        rng=rng,
        num_attempts=num_attempts,
        num_iters=num_iters,
        smooth_amt=smooth_amt,
    )
    return planner.query(start, goal, sample_goal_eps=sample_goal_eps)


def plan_crv_base_path(
    start: CRVConfig,
    goal: CRVConfig,
    *,
    action_limits: CRVActionLimits,
    bounds: PlannerBounds,
    collision_fn: CollisionFn,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    sample_goal_eps: float = 0.0,
) -> list[CRVConfig] | None:
    """Plan a collision-free CRV base path between two SE(2) configurations."""
    return _plan_crv_path(
        start,
        goal,
        action_limits=action_limits,
        bounds=bounds,
        collision_fn=collision_fn,
        seed=seed,
        num_attempts=num_attempts,
        num_iters=num_iters,
        smooth_amt=smooth_amt,
        sample_goal_eps=sample_goal_eps,
    )


def plan_crv_holding_path(
    start: CRVConfig,
    goal: CRVConfig,
    *,
    action_limits: CRVActionLimits,
    bounds: PlannerBounds,
    collision_fn: CollisionFn,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    sample_goal_eps: float = 0.0,
) -> list[CRVConfig] | None:
    """Plan a CRV path when the caller's collision model includes a held object."""
    return _plan_crv_path(
        start,
        goal,
        action_limits=action_limits,
        bounds=bounds,
        collision_fn=collision_fn,
        seed=seed,
        num_attempts=num_attempts,
        num_iters=num_iters,
        smooth_amt=smooth_amt,
        sample_goal_eps=sample_goal_eps,
    )
