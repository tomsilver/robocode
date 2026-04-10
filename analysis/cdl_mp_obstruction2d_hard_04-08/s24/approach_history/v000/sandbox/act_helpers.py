"""Action generation helpers. All magic numbers are named constants here."""
import numpy as np
from obs_helpers import extract_robot, ARM_MIN_JOINT, ARM_MAX_JOINT

# ── Action limits ─────────────────────────────────────────────────────────
DX_MAX        = 0.050
DY_MAX        = 0.050
DTHETA_MAX    = 0.196
DARM_MAX      = 0.100

# ── Controller gains ──────────────────────────────────────────────────────
KP_XY         = 1.0   # proportional gain for xy navigation
KP_THETA      = 2.0   # proportional gain for angle control
KP_ARM        = 1.0   # proportional gain for arm extension

# ── Navigation / planning constants ──────────────────────────────────────
NAV_XY_TOL    = 0.012   # position tolerance (robot base)
NAV_THETA_TOL = 0.05    # angle tolerance
ARM_EXTEND_TOL= 0.008   # arm joint tolerance
STANDOFF_DIST = 0.005   # extra clearance beyond arm_max when positioning
BIRRT_ATTEMPTS= 5
BIRRT_ITERS   = 2000
BIRRT_SMOOTH  = 50
WORLD_MIN_X   = 0.12
WORLD_MAX_X   = 1.50
WORLD_MIN_Y   = 0.12
WORLD_MAX_Y   = 0.88

# ── Zero action ───────────────────────────────────────────────────────────
ZERO_ACTION   = np.zeros(5, dtype=np.float32)


def clip_action(dx, dy, dtheta, darm, vac) -> np.ndarray:
    act = np.array([
        np.clip(dx,     -DX_MAX,    DX_MAX),
        np.clip(dy,     -DY_MAX,    DY_MAX),
        np.clip(dtheta, -DTHETA_MAX, DTHETA_MAX),
        np.clip(darm,   -DARM_MAX,  DARM_MAX),
        float(vac),
    ], dtype=np.float32)
    return act


def angle_diff(a, b) -> float:
    """Signed difference a-b wrapped to [-pi, pi]."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def navigate_to_pose(obs, goal_x, goal_y, goal_theta=None, vac=0.0) -> np.ndarray:
    """Proportional controller: move robot base toward (goal_x, goal_y)
    and optionally rotate to goal_theta. Vacuum held constant."""
    r = extract_robot(obs)
    dx = (goal_x - r["x"]) * KP_XY
    dy = (goal_y - r["y"]) * KP_XY

    if goal_theta is not None:
        dtheta = angle_diff(goal_theta, r["theta"]) * KP_THETA
    else:
        dtheta = 0.0

    # Keep arm retracted during navigation
    darm = (ARM_MIN_JOINT - r["arm_joint"]) * KP_ARM

    return clip_action(dx, dy, dtheta, darm, vac)


def extend_arm_to(obs, target_joint, vac=0.0) -> np.ndarray:
    """Extend/retract arm toward target_joint length."""
    r = extract_robot(obs)
    darm = (target_joint - r["arm_joint"]) * KP_ARM
    return clip_action(0.0, 0.0, 0.0, darm, vac)


def face_target(obs, tx, ty) -> float:
    """Compute angle robot should face to point arm at (tx, ty)."""
    r = extract_robot(obs)
    return np.arctan2(ty - r["y"], tx - r["x"])


GRIPPER_WIDTH     = 0.01   # gripper_width feature value
SUCTION_REACH_ADD = 1.5    # suction center = arm_joint + 1.5*gripper_width from base
GRASP_REACH = ARM_MAX_JOINT + SUCTION_REACH_ADD * GRIPPER_WIDTH  # ≈ 0.215


def robot_goal_for_grasp(tx, ty, theta) -> tuple:
    """Robot center position so suction zone reaches (tx,ty) with arm at max extension."""
    rx = tx - np.cos(theta) * GRASP_REACH
    ry = ty - np.sin(theta) * GRASP_REACH
    return rx, ry


def follow_path(path, step_idx, obs, vac=0.0):
    """Follow a BiRRT path. Returns (action, next_step_idx)."""
    r = extract_robot(obs)
    # Advance past waypoints we've already reached
    while step_idx < len(path) - 1:
        wp = path[step_idx]
        dist = np.hypot(r["x"] - wp[0], r["y"] - wp[1])
        angle_err = abs(angle_diff(wp[2], r["theta"]))
        if dist < NAV_XY_TOL and angle_err < NAV_THETA_TOL:
            step_idx += 1
        else:
            break
    wp = path[step_idx]
    dx  = (wp[0] - r["x"])     * KP_XY
    dy  = (wp[1] - r["y"])     * KP_XY
    dth = angle_diff(wp[2], r["theta"]) * KP_THETA
    darm = (ARM_MIN_JOINT - r["arm_joint"]) * KP_ARM
    return clip_action(dx, dy, dth, darm, vac), step_idx


def make_birrt(primitives, obs, static_rects):
    """Build a BiRRT planner for (x,y,theta) robot navigation.
    static_rects: list of (cx,cy,hw,hh) axis-aligned rectangles to avoid."""
    BiRRT = primitives["BiRRT"]
    rng   = np.random.default_rng(42)
    r     = extract_robot(obs)
    base_r = r["base_radius"]

    def sample_fn(state):
        x     = rng.uniform(WORLD_MIN_X, WORLD_MAX_X)
        y     = rng.uniform(WORLD_MIN_Y, WORLD_MAX_Y)
        theta = rng.uniform(-np.pi, np.pi)
        return np.array([x, y, theta])

    def extend_fn(s1, s2):
        """Interpolate between s1 and s2 in small steps."""
        steps = max(2, int(np.hypot(s2[0]-s1[0], s2[1]-s1[1]) / (DX_MAX*0.5)))
        pts = []
        for k in range(1, steps+1):
            t = k / steps
            x     = s1[0] + t*(s2[0]-s1[0])
            y     = s1[1] + t*(s2[1]-s1[1])
            raw_dth = angle_diff(s2[2], s1[2])
            theta = s1[2] + t*raw_dth
            pts.append(np.array([x, y, theta]))
        return pts

    def collision_fn(state):
        x, y = state[0], state[1]
        # World bounds
        if x < WORLD_MIN_X or x > WORLD_MAX_X or y < WORLD_MIN_Y or y > WORLD_MAX_Y:
            return True
        # Check against static rectangles
        for (cx, cy, hw, hh) in static_rects:
            if (abs(x - cx) < hw + base_r and abs(y - cy) < hh + base_r):
                return True
        return False

    def distance_fn(s1, s2):
        return np.hypot(s2[0]-s1[0], s2[1]-s1[1]) + 0.3 * abs(angle_diff(s2[2], s1[2]))

    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn,
                  rng, BIRRT_ATTEMPTS, BIRRT_ITERS, BIRRT_SMOOTH)
    return birrt


def get_static_rects(obs):
    """Return list of (cx,cy,hw,hh) for the target surface (static obstacle)."""
    from obs_helpers import extract_surface
    surf = extract_surface(obs)
    # Target surface is static; robot shouldn't collide with it
    return [(surf["x"], surf["y"], surf["width"]/2, surf["height"]/2)]
