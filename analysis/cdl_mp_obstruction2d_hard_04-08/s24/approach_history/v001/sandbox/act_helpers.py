"""Action generation helpers. All magic numbers are named constants here."""
import numpy as np
from obs_helpers import (extract_robot, ARM_MIN_JOINT, ARM_MAX_JOINT,
                         NAV_XY_TOL, NAV_THETA_TOL, ARM_EXTEND_TOL,
                         TABLE_TOP_Y, ROBOT_RADIUS, NAV_CLEAR_Y)

# ── Action limits ─────────────────────────────────────────────────────────
DX_MAX     = 0.050
DY_MAX     = 0.050
DTHETA_MAX = 0.196
DARM_MAX   = 0.100

# ── Controller gains ──────────────────────────────────────────────────────
KP_XY    = 1.0
KP_THETA = 2.0
KP_ARM   = 1.0

# ── Grasping geometry ─────────────────────────────────────────────────────
GRIPPER_WIDTH     = 0.01
SUCTION_REACH_ADD = 1.5   # suction center at arm_joint + 1.5*gripper_width
GRASP_REACH       = ARM_MAX_JOINT + SUCTION_REACH_ADD * GRIPPER_WIDTH  # 0.215

# ── BiRRT planning ────────────────────────────────────────────────────────
BIRRT_ATTEMPTS = 5
BIRRT_ITERS    = 3000
BIRRT_SMOOTH   = 100
WORLD_MIN_X    = 0.12
WORLD_MAX_X    = 1.50
WORLD_MIN_Y    = NAV_CLEAR_Y  # keep arm clear of table
WORLD_MAX_Y    = 0.90

# ── Nav arm pose during navigation ───────────────────────────────────────
NAV_THETA = np.pi / 2  # arm pointing up during nav - avoids table objects

# ── Zero action ───────────────────────────────────────────────────────────
ZERO_ACTION = np.zeros(5, dtype=np.float32)


def clip_action(dx, dy, dtheta, darm, vac) -> np.ndarray:
    return np.array([
        np.clip(dx,     -DX_MAX,     DX_MAX),
        np.clip(dy,     -DY_MAX,     DY_MAX),
        np.clip(dtheta, -DTHETA_MAX, DTHETA_MAX),
        np.clip(darm,   -DARM_MAX,   DARM_MAX),
        float(np.clip(vac, 0.0, 1.0)),
    ], dtype=np.float32)


def angle_diff(a, b) -> float:
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def navigate_to_pose(obs, goal_x, goal_y, goal_theta=None, vac=0.0) -> np.ndarray:
    """Proportional controller toward (goal_x, goal_y[, goal_theta])."""
    r = extract_robot(obs)
    dx = (goal_x - r["x"]) * KP_XY
    dy = (goal_y - r["y"]) * KP_XY
    dtheta = angle_diff(goal_theta, r["theta"]) * KP_THETA if goal_theta is not None else 0.0
    darm   = (ARM_MIN_JOINT - r["arm_joint"]) * KP_ARM
    return clip_action(dx, dy, dtheta, darm, vac)


def robot_goal_for_grasp(tx, ty, theta) -> tuple:
    """Robot center position so suction zone reaches (tx,ty)."""
    rx = tx - np.cos(theta) * GRASP_REACH
    ry = ty - np.sin(theta) * GRASP_REACH
    return rx, ry


def follow_path(path, step_idx, obs, vac=0.0):
    """Follow a BiRRT path waypoint-by-waypoint."""
    r = extract_robot(obs)
    while step_idx < len(path) - 1:
        wp = path[step_idx]
        dist = np.hypot(r["x"] - wp[0], r["y"] - wp[1])
        ang  = abs(angle_diff(wp[2], r["theta"]))
        if dist < NAV_XY_TOL and ang < NAV_THETA_TOL:
            step_idx += 1
        else:
            break
    wp = path[step_idx]
    dx  = (wp[0] - r["x"])           * KP_XY
    dy  = (wp[1] - r["y"])           * KP_XY
    dth = angle_diff(wp[2], r["theta"]) * KP_THETA
    darm = (ARM_MIN_JOINT - r["arm_joint"]) * KP_ARM
    return clip_action(dx, dy, dth, darm, vac), step_idx


def make_birrt(primitives, obs, rects_to_avoid):
    """Build BiRRT for (x,y,theta) navigation.
    rects_to_avoid: list of (x1,x2,y1,y2) axis-aligned rects (from bottom-left convention)
    """
    BiRRT = primitives["BiRRT"]
    rng   = np.random.default_rng(42)
    r     = extract_robot(obs)
    base_r = r["base_radius"]
    # Add clearance to robot radius for collision check
    eff_radius = base_r + 0.01

    def sample_fn(state):
        x     = rng.uniform(WORLD_MIN_X, WORLD_MAX_X)
        y     = rng.uniform(WORLD_MIN_Y, WORLD_MAX_Y)
        theta = rng.uniform(-np.pi, np.pi)
        return np.array([x, y, theta])

    def extend_fn(s1, s2):
        dist  = np.hypot(s2[0]-s1[0], s2[1]-s1[1])
        steps = max(2, int(dist / (DX_MAX * 0.5)))
        pts   = []
        for k in range(1, steps+1):
            t     = k / steps
            x     = s1[0] + t*(s2[0]-s1[0])
            y     = s1[1] + t*(s2[1]-s1[1])
            theta = s1[2] + t*angle_diff(s2[2], s1[2])
            pts.append(np.array([x, y, theta]))
        return pts

    def collision_fn(state):
        x, y = state[0], state[1]
        if x < WORLD_MIN_X or x > WORLD_MAX_X or y < WORLD_MIN_Y or y > WORLD_MAX_Y:
            return True
        for (x1, x2, y1, y2) in rects_to_avoid:
            # Circle-AABB collision check
            nearest_x = max(x1, min(x, x2))
            nearest_y = max(y1, min(y, y2))
            if (x - nearest_x)**2 + (y - nearest_y)**2 < eff_radius**2:
                return True
        return False

    def distance_fn(s1, s2):
        return np.hypot(s2[0]-s1[0], s2[1]-s1[1]) + 0.3 * abs(angle_diff(s2[2], s1[2]))

    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn,
                  rng, BIRRT_ATTEMPTS, BIRRT_ITERS, BIRRT_SMOOTH)
    return birrt


def get_rects_for_nav(obs, skip_indices=None):
    """Return all rectangle obstacles (obstructions + block) that robot should avoid.
    Each rect is (x1, x2, y1, y2). skip_indices: obstruction indices to ignore."""
    from obs_helpers import extract_obstruction, extract_block, NUM_OBSTRUCTIONS
    skip = set(skip_indices or [])
    rects = []
    for i in range(NUM_OBSTRUCTIONS):
        if i not in skip:
            ob = extract_obstruction(obs, i)
            rects.append((ob["x1"], ob["x2"], ob["y1"], ob["y2"]))
    blk = extract_block(obs)
    rects.append((blk["x1"], blk["x2"], blk["y1"], blk["y2"]))
    return rects
