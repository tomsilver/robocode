"""Action helpers.

Converts sparse key-waypoints into dense action sequences that respect the environment's
action-space limits.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.typing import NDArray
from obs_helpers import RobotPose  # type: ignore[import-not-found]

# Default per-step limits (matching the ClutteredStorage2D action space).
DX_LIM = 0.05
DY_LIM = 0.05
DTH_LIM = np.pi / 16
DARM_LIM = 0.1

# Vacuum constants
VAC_ON = 1.0
VAC_OFF = 0.0

# Approach/placement constants
APPROACH_DIST = 0.4          # distance from block center to robot center for grasping
SUCTION_DIST_OFFSET = 0.03   # suction center = arm_joint + SUCTION_DIST_OFFSET
GRASP_ARM_JOINT = APPROACH_DIST - SUCTION_DIST_OFFSET  # arm_joint to put suction at block center

SHELF_APPROACH_X = 0.236     # robot x for shelf placement (center of slot)
SHELF_APPROACH_Y = 2.1       # robot y for shelf placement (below shelf)
SHELF_PLACE_THETA = math.pi / 2  # arm pointing up for placement
SHELF_PLACE_ARM = 0.55       # arm extension to place block inside shelf

# Navigation clearance
NAV_MARGIN = 0.22    # robot + arm margin for collision checking
BLOCK_CLEAR = 0.35   # minimum clearance between robot center and block center


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
            dense.append(
                RobotPose(
                    x=a.x + t * (b.x - a.x),
                    y=a.y + t * (b.y - a.y),
                    theta=a.theta + t * (b.theta - a.theta),
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


def make_action(dx: float = 0.0, dy: float = 0.0, dtheta: float = 0.0,
                darm: float = 0.0, vac: float = VAC_OFF) -> NDArray:
    """Create a single action array, clipped to limits."""
    return np.array([
        np.clip(dx, -DX_LIM, DX_LIM),
        np.clip(dy, -DY_LIM, DY_LIM),
        np.clip(dtheta, -DTH_LIM, DTH_LIM),
        np.clip(darm, -DARM_LIM, DARM_LIM),
        float(vac),
    ], dtype=np.float32)


def servo_theta(current_theta: float, target_theta: float) -> float:
    """Compute dtheta to servo toward target_theta (wrapped, clipped)."""
    from obs_helpers import wrap_angle
    err = wrap_angle(target_theta - current_theta)
    return float(np.clip(err, -DTH_LIM, DTH_LIM))


def servo_arm(current_arm: float, target_arm: float) -> float:
    """Compute darm to servo toward target_arm (clipped)."""
    err = target_arm - current_arm
    return float(np.clip(err, -DARM_LIM, DARM_LIM))


def path_to_actions(path: list, current_robot, vac: float = VAC_OFF) -> deque[NDArray]:
    """Convert a list of (x,y) waypoints to a deque of actions, also servoing theta.

    current_robot: RobotPose for the starting state.
    path: list of [x, y] or (x, y) waypoints from BiRRT.
    """
    actions: deque[NDArray] = deque()
    cx, cy, cth = current_robot.x, current_robot.y, current_robot.theta
    for wp in path:
        wx, wy = wp[0], wp[1]
        dx = wx - cx
        dy = wy - cy
        # Break into steps
        steps = max(1, math.ceil(abs(dx) / DX_LIM), math.ceil(abs(dy) / DY_LIM))
        for s in range(1, steps + 1):
            t = s / steps
            tx = cx + t * (wx - cx)
            ty = cy + t * (wy - cy)
            adx = np.clip(tx - cx, -DX_LIM, DX_LIM)
            ady = np.clip(ty - cy, -DY_LIM, DY_LIM)
            actions.append(np.array([adx, ady, 0.0, 0.0, vac], dtype=np.float32))
            cx += adx
            cy += ady
    return actions
