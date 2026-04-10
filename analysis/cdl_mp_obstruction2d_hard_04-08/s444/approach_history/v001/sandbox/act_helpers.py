"""Action generation helpers. All magic numbers are named constants here."""
import numpy as np

# ---- Action limits ----
MAX_DX = 0.050          # max base x displacement per step
MAX_DY = 0.050          # max base y displacement per step
MAX_DTHETA = 0.196      # max rotation per step (≈ π/16)
MAX_DARM = 0.100        # max arm extension change per step

# ---- Control gains ----
K_POS = 3.0             # proportional gain for position
K_THETA = 3.0           # proportional gain for theta
K_ARM = 5.0             # proportional gain for arm

# ---- Tolerances ----
POS_TOL = 0.008         # close enough for position
THETA_TOL = 0.05        # close enough for angle (radians)
ARM_TOL = 0.005         # close enough for arm joint

# ---- BiRRT parameters ----
BIRRT_NUM_ATTEMPTS = 5
BIRRT_NUM_ITERS = 2000
BIRRT_SMOOTH_AMT = 50
EXTEND_STEP = 0.04      # interpolation step size for BiRRT

# ---- World / robot constants (used in action helpers) ----
ROB_RADIUS = 0.1
X_MIN_NAV = 0.12        # min robot center x (with margin)
X_MAX_NAV = 1.50        # max robot center x
Y_MIN_NAV = 0.12        # min robot center y
Y_MAX_NAV = 0.88        # max robot center y

# Target arm joint for picking/placing
PICK_ARM_JOINT = 0.15
DROP_ARM_JOINT = 0.15
RETRACT_ARM = 0.10

# Navigation altitudes
NAV_HIGH_Y = 0.40       # high-altitude robot y for transit (clears all obstructions)
APPROACH_LOW_Y = 0.25   # low approach robot y (arm reaches y=0.10)

# Drop positions for obstructions (right side, away from surface)
DROP_X_BASE = 1.35      # x center for dropping obstruction
DROP_Y_OFFSET = 0.05    # x spacing between obstruction drops
DROP_ROBOT_Y = APPROACH_LOW_Y  # robot y when placing at table level

# Place approach
PLACE_ROBOT_Y = APPROACH_LOW_Y  # robot y when placing block on surface


def clip_action(dx=0.0, dy=0.0, dtheta=0.0, darm=0.0, vac=0.0):
    """Build and clip a 5D action array."""
    return np.array([
        np.clip(dx, -MAX_DX, MAX_DX),
        np.clip(dy, -MAX_DY, MAX_DY),
        np.clip(dtheta, -MAX_DTHETA, MAX_DTHETA),
        np.clip(darm, -MAX_DARM, MAX_DARM),
        float(vac),
    ], dtype=np.float32)


def toward_pos(cur_x, cur_y, tgt_x, tgt_y, vac=0.0, darm=0.0, cur_theta=None):
    """Proportional control action to move robot base toward (tgt_x, tgt_y).
    Optionally also corrects theta to -pi/2 if cur_theta is provided."""
    dx = np.clip((tgt_x - cur_x) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((tgt_y - cur_y) * K_POS, -MAX_DY, MAX_DY)
    dtheta = 0.0
    if cur_theta is not None:
        err_theta = normalize_angle(-np.pi / 2 - cur_theta)
        dtheta = np.clip(err_theta * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    return clip_action(dx=dx, dy=dy, dtheta=dtheta, darm=darm, vac=vac)


def control_arm(cur_arm, tgt_arm, vac=0.0):
    """Proportional arm extension action."""
    darm = np.clip((tgt_arm - cur_arm) * K_ARM, -MAX_DARM, MAX_DARM)
    return clip_action(darm=darm, vac=vac)


def normalize_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def pos_reached(cur_x, cur_y, tgt_x, tgt_y, tol=POS_TOL):
    return abs(cur_x - tgt_x) < tol and abs(cur_y - tgt_y) < tol


def theta_reached(cur_theta, tgt_theta=None, tol=THETA_TOL):
    if tgt_theta is None:
        tgt_theta = -np.pi / 2
    return abs(normalize_angle(tgt_theta - cur_theta)) < tol


def arm_reached(cur_arm, tgt_arm, tol=ARM_TOL):
    return abs(cur_arm - tgt_arm) < tol


def make_birrt_fns(obs_snapshot, excluded_obs_indices=None, rng=None):
    """Build (sample_fn, extend_fn, collision_fn, distance_fn) for robot base BiRRT.

    obs_snapshot: observation array at time of planning
    excluded_obs_indices: set of obstruction indices to ignore (e.g. currently held)
    """
    from obs_helpers import get_obstruction, NUM_OBSTRUCTIONS, TABLE_OBJ_Y

    if rng is None:
        rng = np.random.default_rng(0)
    if excluded_obs_indices is None:
        excluded_obs_indices = set()

    # Snapshot obstacle positions
    obstacles = []
    for i in range(NUM_OBSTRUCTIONS):
        if i in excluded_obs_indices:
            continue
        o = get_obstruction(obs_snapshot, i)
        obstacles.append((o['x'], o['y'], o['width'], o['height']))

    def sample_fn(s):
        return np.array([
            rng.uniform(X_MIN_NAV, X_MAX_NAV),
            rng.uniform(Y_MIN_NAV, Y_MAX_NAV),
        ])

    def extend_fn(s1, s2):
        diff = s2 - s1
        dist = float(np.linalg.norm(diff))
        if dist < 1e-9:
            return [s2]
        n = max(1, int(dist / EXTEND_STEP))
        return [s1 + diff * (t / n) for t in range(1, n + 1)]

    def collision_fn(s):
        x, y = float(s[0]), float(s[1])
        if x < X_MIN_NAV or x > X_MAX_NAV or y < Y_MIN_NAV or y > Y_MAX_NAV:
            return True
        r = ROB_RADIUS
        for (cx, cy, w, h) in obstacles:
            # Circle-AABB test
            dx = max(abs(x - cx) - w / 2.0, 0.0)
            dy = max(abs(y - cy) - h / 2.0, 0.0)
            if dx * dx + dy * dy < r * r:
                return True
        return False

    def distance_fn(s1, s2):
        return float(np.linalg.norm(s1 - s2))

    return sample_fn, extend_fn, collision_fn, distance_fn


def get_drop_position(i):
    """Return (drop_x, drop_robot_y) for obstruction i drop zone."""
    # Space drops out to the right
    drop_x = DROP_X_BASE - i * DROP_Y_OFFSET
    return drop_x, DROP_ROBOT_Y
