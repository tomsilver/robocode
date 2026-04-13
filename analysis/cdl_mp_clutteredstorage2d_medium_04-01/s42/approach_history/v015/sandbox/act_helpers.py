"""Action helpers.

Converts sparse key-waypoints into dense action sequences that respect the environment's
action-space limits.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from obs_helpers import RobotPose

# Default per-step limits (matching the ClutteredStorage2D action space).
DX_LIM = 0.05
DY_LIM = 0.05
DTH_LIM = np.pi / 16   # ~0.196 rad
DARM_LIM = 0.1

# Arm constants
ARM_MIN_JOINT = 0.2     # == base_radius; fully retracted
ARM_MAX_JOINT = 0.8     # arm_length; fully extended

# Pick/place arm extension targets
PICK_ARM_EXTEND = 0.60   # arm_joint when grasping (must reach block from below)
PLACE_ARM_EXTEND = 0.65  # arm_joint when placing block inside shelf

# Vacuum threshold
VACUUM_ON = 1.0
VACUUM_OFF = 0.0
VACUUM_THRESH = 0.5

# Navigation tolerances
XY_TOL = 0.03      # accepted position error
THETA_TOL = 0.08   # accepted angle error
ARM_TOL = 0.02     # accepted arm joint error

# Clearance for BiRRT collision check
ROBOT_CLEARANCE = 0.02  # extra margin beyond base_radius

# Number of vacuum-on steps to ensure grasp
GRASP_STEPS = 20

# Number of vacuum-off steps to ensure release
RELEASE_STEPS = 10

# Arm retract steps (always retract to minimum)
RETRACT_STEPS = 12


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


def angle_diff(a: float, b: float) -> float:
    """Signed difference b - a, wrapped to [-pi, pi]."""
    d = b - a
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def rotate_actions(from_theta: float, to_theta: float, vacuum: float = VACUUM_OFF) -> list[NDArray]:
    """Generate actions to rotate from from_theta to to_theta."""
    diff = angle_diff(from_theta, to_theta)
    n_steps = max(1, math.ceil(abs(diff) / DTH_LIM))
    step = diff / n_steps
    return [np.array([0.0, 0.0, step, 0.0, vacuum], dtype=np.float32) for _ in range(n_steps)]


def navigate_actions(
    start_x: float, start_y: float,
    goal_x: float, goal_y: float,
    vacuum: float = VACUUM_OFF,
) -> list[NDArray]:
    """Generate actions to move from (start_x, start_y) to (goal_x, goal_y)."""
    dx = goal_x - start_x
    dy = goal_y - start_y
    n_steps = max(1, math.ceil(abs(dx) / DX_LIM), math.ceil(abs(dy) / DY_LIM))
    step_x = dx / n_steps
    step_y = dy / n_steps
    return [np.array([step_x, step_y, 0.0, 0.0, vacuum], dtype=np.float32) for _ in range(n_steps)]


def arm_actions(from_joint: float, to_joint: float, vacuum: float = VACUUM_OFF) -> list[NDArray]:
    """Generate actions to change arm_joint from from_joint to to_joint."""
    diff = to_joint - from_joint
    n_steps = max(1, math.ceil(abs(diff) / DARM_LIM))
    step = diff / n_steps
    return [np.array([0.0, 0.0, 0.0, step, vacuum], dtype=np.float32) for _ in range(n_steps)]


def hold_actions(n: int, vacuum: float = VACUUM_ON) -> list[NDArray]:
    """Generate n no-op actions with given vacuum setting."""
    return [np.array([0.0, 0.0, 0.0, 0.0, vacuum], dtype=np.float32) for _ in range(n)]


def birrt_xy_path(
    primitives: dict,
    start_xy: NDArray,
    goal_xy: NDArray,
    base_radius: float,
    shelf_floor_y: float,
    world_x_min: float = 0.0,
    world_x_max: float = 5.0,
    world_y_min: float = 0.0,
    rng=None,
) -> list[NDArray]:
    """Plan a collision-free XY path using BiRRT.

    Returns a list of [x, y] waypoints (including start and goal).
    Falls back to straight line if BiRRT fails.
    """
    import numpy as np

    clearance = base_radius + ROBOT_CLEARANCE
    x_lo = world_x_min + clearance
    x_hi = world_x_max - clearance
    y_lo = world_y_min + clearance
    y_hi = shelf_floor_y - clearance

    # Clip goal to valid region
    goal_xy = np.clip(goal_xy, [x_lo, y_lo], [x_hi, y_hi])

    if rng is None:
        rng = np.random.default_rng(0)

    def sample_fn(state):
        return rng.uniform([x_lo, y_lo], [x_hi, y_hi])

    extend_step = DX_LIM * 4  # larger steps for faster planning

    def extend_fn(s1, s2):
        diff = s2 - s1
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            return [s2]
        n = max(1, math.ceil(dist / extend_step))
        return [s1 + (i + 1) / n * diff for i in range(n)]

    def collision_fn(state):
        x, y = float(state[0]), float(state[1])
        return x < x_lo or x > x_hi or y < y_lo or y > y_hi

    def distance_fn(s1, s2):
        return float(np.linalg.norm(s2 - s1))

    BiRRT = primitives["BiRRT"]
    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng,
                  num_attempts=5, num_iters=300, smooth_amt=20)
    path = birrt.query(np.array(start_xy, dtype=float), np.array(goal_xy, dtype=float))

    if path is None:
        # Fallback: straight line
        path = [np.array(start_xy, dtype=float), np.array(goal_xy, dtype=float)]

    return path


def path_to_actions(path: list[NDArray], vacuum: float = VACUUM_OFF) -> list[NDArray]:
    """Convert a list of [x,y] waypoints to delta actions."""
    actions = []
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        n = max(1, math.ceil(abs(dx) / DX_LIM), math.ceil(abs(dy) / DY_LIM))
        for j in range(n):
            actions.append(np.array([dx / n, dy / n, 0.0, 0.0, vacuum], dtype=np.float32))
    return actions
