"""Action helpers."""

from __future__ import annotations
import math
from collections import deque
import numpy as np
from numpy.typing import NDArray
from obs_helpers import RobotPose

# Per-step action limits
DX_LIM = 0.05
DY_LIM = 0.05
DTH_LIM = np.pi / 16
DARM_LIM = 0.1

# Pick/place constants
PICK_ARM_DIST = 0.45       # arm_joint when gripper is at block center
PLACE_ARM_JOINT = 0.60     # arm_joint when placing block in shelf
SHELF_APPROACH_Y = 1.95    # robot base y when placing
RETRACT_ARM = 0.20         # fully retracted arm_joint (= base_radius)

# Navigation constants
PICK_APPROACH_OFFSET = 0.45  # approach from this far below the block
NAV_STEP = 0.048             # base navigation step (< DX_LIM for safety)

# Vacuum values
VAC_ON = 1.0
VAC_OFF = 0.0

# Angle for approach from below
THETA_UP = math.pi / 2


def connecting_waypoints(
    waypoints: list[RobotPose],
    action_limits: tuple[float, float, float, float] = (DX_LIM, DY_LIM, DTH_LIM, DARM_LIM),
) -> list[RobotPose]:
    dx_lim, dy_lim, dth_lim, darm_lim = action_limits
    dense: list[RobotPose] = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        steps = max(1,
            math.ceil(abs(b.x - a.x) / dx_lim),
            math.ceil(abs(b.y - a.y) / dy_lim),
            math.ceil(abs(b.theta - a.theta) / dth_lim) if dth_lim > 0 else 1,
            math.ceil(abs(b.arm_joint - a.arm_joint) / darm_lim),
        )
        for s in range(1, steps + 1):
            t = s / steps
            dense.append(RobotPose(
                x=a.x + t*(b.x - a.x), y=a.y + t*(b.y - a.y),
                theta=a.theta + t*(b.theta - a.theta),
                base_radius=a.base_radius,
                arm_joint=a.arm_joint + t*(b.arm_joint - a.arm_joint),
                arm_length=a.arm_length, vacuum=b.vacuum,
                gripper_height=a.gripper_height, gripper_width=a.gripper_width,
            ))
    return dense


def waypoints_to_actions(waypoints: list[RobotPose]) -> deque[NDArray]:
    actions: deque[NDArray] = deque()
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        actions.append(np.array([b.x-a.x, b.y-a.y, b.theta-a.theta,
                                  b.arm_joint-a.arm_joint, b.vacuum], dtype=np.float32))
    return actions


def path_to_actions(path: list[NDArray], robot: RobotPose, vacuum: float) -> deque[NDArray]:
    """Convert a (x,y) path to robot actions, keeping theta and arm fixed.
    Generates multiple steps per waypoint if the waypoint is far away."""
    actions: deque[NDArray] = deque()
    cur_x, cur_y = robot.x, robot.y
    for pt in path:
        nx, ny = float(pt[0]), float(pt[1])
        # Emit multiple steps until we reach this waypoint
        while True:
            dx = nx - cur_x
            dy = ny - cur_y
            if abs(dx) < 0.001 and abs(dy) < 0.001:
                break
            sdx = max(-DX_LIM, min(DX_LIM, dx))
            sdy = max(-DY_LIM, min(DY_LIM, dy))
            actions.append(np.array([sdx, sdy, 0.0, 0.0, vacuum], dtype=np.float32))
            cur_x += sdx
            cur_y += sdy
    return actions


def angle_diff(a: float, b: float) -> float:
    """Signed difference b - a, wrapped to (-pi, pi]."""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return d


def rotate_actions(current_theta: float, target_theta: float, vacuum: float) -> deque[NDArray]:
    """Actions to rotate from current_theta to target_theta."""
    actions: deque[NDArray] = deque()
    remaining = angle_diff(current_theta, target_theta)
    while abs(remaining) > 0.02:
        step = max(-DTH_LIM, min(DTH_LIM, remaining))
        actions.append(np.array([0.0, 0.0, step, 0.0, vacuum], dtype=np.float32))
        remaining -= step
        if abs(remaining) < 1e-9:
            break
    return actions


def arm_actions(current_arm: float, target_arm: float, vacuum: float) -> deque[NDArray]:
    """Actions to change arm_joint from current to target."""
    actions: deque[NDArray] = deque()
    remaining = target_arm - current_arm
    while abs(remaining) > 0.005:
        step = max(-DARM_LIM, min(DARM_LIM, remaining))
        actions.append(np.array([0.0, 0.0, 0.0, step, vacuum], dtype=np.float32))
        remaining -= step
        if abs(remaining) < 1e-9:
            break
    return actions


def hold_action(vacuum: float, n: int = 1) -> deque[NDArray]:
    """Hold still for n steps."""
    return deque([np.array([0.0, 0.0, 0.0, 0.0, vacuum], dtype=np.float32)] * n)
