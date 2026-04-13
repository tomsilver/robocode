"""Action helpers.

Converts sparse key-waypoints into dense action sequences that respect the environment's
action-space limits.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from obs_helpers import RobotPose, WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, ROBOT_RADIUS, ARM_MIN_JOINT, ARM_MAX_JOINT

# ---------------------------------------------------------------------------
# Per-step action limits (matching ClutteredStorage2D action space)
# ---------------------------------------------------------------------------
DX_LIM = 0.05
DY_LIM = 0.05
DTH_LIM = np.pi / 16   # ~0.196 rad per step
DARM_LIM = 0.10

# ---------------------------------------------------------------------------
# Behaviour-level constants
# ---------------------------------------------------------------------------
APPROACH_DIST = 0.55      # arm_joint when picking (robot below block center)
DEPOSIT_ARM_JOINT = 0.72  # arm_joint when depositing block in shelf
THETA_UP = math.pi / 2   # robot angle for "arm pointing upward"
EXTEND_DARM = 0.02        # small arm step for EXTEND phase to enter suction window

# Tolerances
POS_TOL = 0.025           # distance at which a waypoint is considered reached
THETA_TOL = 0.05          # angle tolerance for alignment phase
ARM_TOL = 0.02            # arm_joint tolerance

# Shelf approach y-nav positions for two deposit slots
# (robot y when arm is fully extended at DEPOSIT_ARM_JOINT)
DEPOSIT_SLOT_Y_NAV = [2.02, 2.09]   # two slots → block deposited at y ≈ 2.77 / 2.84


def zero_action() -> NDArray:
    """Return a zero action vector."""
    return np.zeros(5, dtype=np.float32)


def build_action(
    dx: float = 0.0,
    dy: float = 0.0,
    dtheta: float = 0.0,
    darm: float = 0.0,
    vacuum: float = 0.0,
) -> NDArray:
    """Build a 5-element action array."""
    return np.array([dx, dy, dtheta, darm, vacuum], dtype=np.float32)


def clip_action(action: NDArray) -> NDArray:
    """Clip action to environment limits."""
    limits = np.array([DX_LIM, DY_LIM, DTH_LIM, DARM_LIM, 1.0])
    lo = np.array([-DX_LIM, -DY_LIM, -DTH_LIM, -DARM_LIM, 0.0])
    return np.clip(action, lo, limits)


def angle_diff(target: float, current: float) -> float:
    """Signed angle difference (target - current), wrapped to [-pi, pi]."""
    diff = target - current
    return (diff + math.pi) % (2 * math.pi) - math.pi


def prop_action_toward(
    robot: RobotPose,
    target_x: float,
    target_y: float,
    target_theta: float,
    target_arm: float,
    vacuum: float = 0.0,
) -> NDArray:
    """Proportional controller: one-step delta toward a target robot config."""
    dx = np.clip(target_x - robot.x, -DX_LIM, DX_LIM)
    dy = np.clip(target_y - robot.y, -DY_LIM, DY_LIM)
    dth = np.clip(angle_diff(target_theta, robot.theta), -DTH_LIM, DTH_LIM)
    darm = np.clip(target_arm - robot.arm_joint, -DARM_LIM, DARM_LIM)
    return np.array([dx, dy, dth, darm, vacuum], dtype=np.float32)


def make_birrt(primitives: dict, shelf_y: float, rng: np.random.Generator,
               y_ceiling_override: float | None = None,
               block_obstacles: list | None = None):
    """Construct a BiRRT planner for robot base (x, y) navigation.

    The robot circle (radius ROBOT_RADIUS) must stay within world bounds and
    below the shelf bottom edge (or y_ceiling_override if provided).

    block_obstacles: optional list of (bx, by, br) tuples — blocks treated as
    circular obstacles of radius br that the robot circle (radius ROBOT_RADIUS)
    must not penetrate.  Use BLOCK_HALF_DIAG + margin as br.
    """
    if y_ceiling_override is not None:
        y_ceiling = y_ceiling_override
    else:
        y_ceiling = shelf_y - ROBOT_RADIUS - 0.01   # maximum robot center y

    _obstacles = block_obstacles or []

    def sample_fn(state: tuple) -> tuple:
        x = rng.uniform(WORLD_MIN_X + ROBOT_RADIUS, WORLD_MAX_X - ROBOT_RADIUS)
        y = rng.uniform(WORLD_MIN_Y + ROBOT_RADIUS, y_ceiling)
        return (x, y)

    def extend_fn(s1: tuple, s2: tuple):
        x1, y1 = s1
        x2, y2 = s2
        n = max(1, math.ceil(abs(x2 - x1) / DX_LIM), math.ceil(abs(y2 - y1) / DY_LIM))
        for i in range(1, n + 1):
            t = i / n
            yield (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def collision_fn(state: tuple) -> bool:
        x, y = state
        r = ROBOT_RADIUS
        if (x - r < WORLD_MIN_X or x + r > WORLD_MAX_X
                or y - r < WORLD_MIN_Y or y + r > y_ceiling + r):
            return True
        for bx, by, br in _obstacles:
            if (x - bx) ** 2 + (y - by) ** 2 < (r + br) ** 2:
                return True
        return False

    def distance_fn(s1: tuple, s2: tuple) -> float:
        return math.sqrt((s2[0] - s1[0]) ** 2 + (s2[1] - s1[1]) ** 2)

    BiRRT = primitives["BiRRT"]
    return BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng, 5, 1000, 50)


# ---------------------------------------------------------------------------
# Legacy waypoint helpers (kept for compatibility)
# ---------------------------------------------------------------------------


def connecting_waypoints(
    waypoints: list[RobotPose],
    action_limits: tuple[float, float, float, float] = (
        DX_LIM,
        DY_LIM,
        DTH_LIM,
        DARM_LIM,
    ),
) -> list[RobotPose]:
    """Linearly interpolate between consecutive key-waypoints."""
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
    """Convert dense waypoints into a deque of delta-actions."""
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
