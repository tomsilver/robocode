"""Action helpers for PushPullHook2D oracle behaviors.

Converts sparse key-waypoints into dense action sequences that respect the environment's
action-space limits.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from robocode.oracles.pushpullhook2d.obs_helpers import RobotPose

# Default per-step limits (matching the PushPullHook2D action space).
DX_LIM = 0.01
DY_LIM = 0.01
DTH_LIM = np.pi / 32
DARM_LIM = 0.1


def connecting_waypoints(
    waypoints: list[RobotPose],
    action_limits: tuple[float, float, float, float] = (
        DX_LIM,
        DY_LIM,
        DTH_LIM,
        DARM_LIM,
    ),
) -> list[RobotPose]:
    """Linearly interpolate between consecutive key-waypoints.

    The number of intermediate steps between each pair is determined by the dimension
    that requires the most steps given *action_limits*.

    Vacuum is snapped to the target waypoint value (not interpolated).
    """
    dx_lim, dy_lim, dth_lim, darm_lim = action_limits

    # Pre-adjust all waypoint thetas so that every consecutive pair
    # has a non-negative theta delta (visually-clockwise rotation).
    adjusted_thetas = [waypoints[0].theta]
    for i in range(1, len(waypoints)):
        prev = adjusted_thetas[-1]
        cur = waypoints[i].theta
        # Shift cur into [prev, prev + 2π) so the delta is always ≥ 0.
        while cur < prev:
            cur += 2 * math.pi
        adjusted_thetas.append(cur)

    dense: list[RobotPose] = [waypoints[0]]

    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        a_theta = adjusted_thetas[i]
        b_theta = adjusted_thetas[i + 1]

        dtheta = b_theta - a_theta
        steps = max(
            1,
            math.ceil(abs(b.x - a.x) / dx_lim),
            math.ceil(abs(b.y - a.y) / dy_lim),
            math.ceil(dtheta / dth_lim) if dth_lim > 0 else 1,
            math.ceil(abs(b.arm_joint - a.arm_joint) / darm_lim),
        )
        for s in range(1, steps + 1):
            t = s / steps
            dense.append(
                RobotPose(
                    x=a.x + t * (b.x - a.x),
                    y=a.y + t * (b.y - a.y),
                    theta=a_theta + t * (b_theta - a_theta),
                    base_radius=a.base_radius,
                    arm_joint=a.arm_joint + t * (b.arm_joint - a.arm_joint),
                    arm_length=a.arm_length,
                    vacuum=b.vacuum,
                    gripper_height=a.gripper_height,
                    gripper_width=a.gripper_width,
                )
            )

    return dense


def waypoints_to_actions(waypoints: list[RobotPose]) -> deque[NDArray]:
    """Convert dense waypoints into a deque of delta-actions.

    Action format: [dx, dy, dtheta, darm, vacuum].
    The first four are deltas; vacuum is absolute (set directly).
    """
    actions: deque[NDArray] = deque()
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        actions.append(
            np.array(
                [
                    b.x - a.x,
                    b.y - a.y,
                    b.theta - a.theta,
                    b.arm_joint - a.arm_joint,
                    b.vacuum,
                ],
                dtype=np.float32,
            )
        )
    return actions
