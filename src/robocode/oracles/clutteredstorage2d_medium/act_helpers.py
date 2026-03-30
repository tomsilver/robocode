"""Action helpers for ClutteredStorage2D-b3 oracle behaviors."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from robocode.oracles.clutteredstorage2d_medium.obs_helpers import RobotPose

DX_LIM = 0.05
DY_LIM = 0.05
DTH_LIM = np.pi / 16
DARM_LIM = 0.1


def wrap_angle(theta: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def shortest_angle_diff(target: float, source: float) -> float:
    """Return the shortest signed angle from *source* to *target*."""
    return wrap_angle(target - source)


def connecting_waypoints(
    waypoints: list[RobotPose],
    action_limits: tuple[float, float, float, float] = (
        DX_LIM,
        DY_LIM,
        DTH_LIM,
        DARM_LIM,
    ),
) -> list[RobotPose]:
    """Linearly interpolate between key waypoints."""
    dx_lim, dy_lim, dth_lim, darm_lim = action_limits
    dense: list[RobotPose] = [waypoints[0]]

    for start, end in zip(waypoints[:-1], waypoints[1:]):
        dtheta = shortest_angle_diff(end.theta, start.theta)
        steps = max(
            1,
            math.ceil(abs(end.x - start.x) / dx_lim),
            math.ceil(abs(end.y - start.y) / dy_lim),
            math.ceil(abs(dtheta) / dth_lim) if dth_lim > 0 else 1,
            math.ceil(abs(end.arm_joint - start.arm_joint) / darm_lim),
        )
        for step in range(1, steps + 1):
            t = step / steps
            dense.append(
                RobotPose(
                    x=start.x + t * (end.x - start.x),
                    y=start.y + t * (end.y - start.y),
                    theta=wrap_angle(start.theta + t * dtheta),
                    base_radius=start.base_radius,
                    arm_joint=start.arm_joint + t * (end.arm_joint - start.arm_joint),
                    arm_length=start.arm_length,
                    vacuum=end.vacuum,
                    gripper_height=start.gripper_height,
                    gripper_width=start.gripper_width,
                )
            )

    return dense


def waypoints_to_actions(waypoints: list[RobotPose]) -> deque[NDArray]:
    """Convert dense waypoints into delta actions."""
    actions: deque[NDArray] = deque()
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        actions.append(
            np.array(
                [
                    end.x - start.x,
                    end.y - start.y,
                    shortest_angle_diff(end.theta, start.theta),
                    end.arm_joint - start.arm_joint,
                    end.vacuum,
                ],
                dtype=np.float32,
            )
        )
    return actions
