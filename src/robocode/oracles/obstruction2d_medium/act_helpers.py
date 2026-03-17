"""Action helpers for Obstruction2D-o2 oracle behaviors.

Converts sparse key-waypoints into dense action sequences that respect
the environment's action-space limits.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from robocode.oracles.obstruction2d_medium.obs_helpers import RobotPose

# Default per-step limits (matching the Obstruction2D action space).
DX_LIM = 0.05
DY_LIM = 0.05
DTH_LIM = np.pi / 16
DARM_LIM = 0.1


def connecting_waypoints(
    waypoints: list[RobotPose],
    action_limits: tuple[float, float, float, float] = (DX_LIM, DY_LIM, DTH_LIM, DARM_LIM),
) -> list[RobotPose]:
    """Linearly interpolate between consecutive key-waypoints.

    The number of intermediate steps between each pair is determined by the
    dimension that requires the most steps given *action_limits*.

    Vacuum is snapped to the target waypoint value (not interpolated).
    """
    dx_lim, dy_lim, dth_lim, darm_lim = action_limits
    dense: list[RobotPose] = [waypoints[0]]

    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        steps = max(
            1,
            math.ceil(abs(b.x - a.x) / dx_lim),
            math.ceil(abs(b.y - a.y) / dy_lim),
            math.ceil(abs(b.theta - a.theta) / dth_lim) if dth_lim > 0 else 1,
            math.ceil(abs(b.arm_joint - a.arm_joint) / darm_lim),
        )
        for s in range(1, steps + 1):
            t = s / steps
            dense.append(RobotPose(
                x=a.x + t * (b.x - a.x),
                y=a.y + t * (b.y - a.y),
                theta=a.theta + t * (b.theta - a.theta),
                base_radius=a.base_radius,
                arm_joint=a.arm_joint + t * (b.arm_joint - a.arm_joint),
                arm_length=a.arm_length,
                vacuum=b.vacuum,
                gripper_height=a.gripper_height,
                gripper_width=a.gripper_width,
            ))

    return dense


def waypoints_to_actions(waypoints: list[RobotPose]) -> deque[NDArray]:
    """Convert dense waypoints into a deque of delta-actions.

    Action format: [dx, dy, dtheta, darm, vacuum].
    The first four are deltas; vacuum is absolute (set directly).
    """
    actions: deque[NDArray] = deque()
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        actions.append(np.array([
            b.x - a.x,
            b.y - a.y,
            b.theta - a.theta,
            b.arm_joint - a.arm_joint,
            b.vacuum,
        ], dtype=np.float32))
    return actions
