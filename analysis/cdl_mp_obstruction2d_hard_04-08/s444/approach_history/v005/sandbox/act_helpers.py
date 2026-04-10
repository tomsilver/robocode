"""Action generation helpers. All magic numbers are named constants here."""
import numpy as np

# ---- Action limits ----
MAX_DX = 0.050
MAX_DY = 0.050
MAX_DTHETA = 0.196
MAX_DARM = 0.100

# ---- Control gains ----
K_POS = 3.0
K_THETA = 3.0
K_ARM = 8.0

# ---- Tolerances ----
POS_TOL = 0.020         # position reached tolerance
THETA_TOL = 0.05        # angle tolerance (radians)
ARM_TOL = 0.005         # arm joint tolerance

# ---- BiRRT parameters ----
BIRRT_NUM_ATTEMPTS = 5
BIRRT_NUM_ITERS = 2000
BIRRT_SMOOTH_AMT = 50
EXTEND_STEP = 0.06

# ---- World / navigation limits ----
ROB_RADIUS = 0.10
X_MIN_NAV = 0.15
X_MAX_NAV = 1.50
Y_MIN_NAV = 0.15
Y_MAX_NAV = 0.88

# ---- Arm joint constants ----
PICK_ARM_JOINT = 0.13   # arm joint for picking (suction just inside obj top)
RETRACT_ARM = 0.10      # minimum arm joint (retracted)

# ---- Navigation altitudes ----
NAV_HIGH_Y = 0.50       # transit altitude (well above all objects)

# ---- Drop positions for obstructions ----
DROP_X_BASE = 1.35
DROP_X_SPACING = 0.08   # x spacing between drop positions


def clip_action(dx=0.0, dy=0.0, dtheta=0.0, darm=0.0, vac=0.0):
    return np.array([
        np.clip(dx, -MAX_DX, MAX_DX),
        np.clip(dy, -MAX_DY, MAX_DY),
        np.clip(dtheta, -MAX_DTHETA, MAX_DTHETA),
        np.clip(darm, -MAX_DARM, MAX_DARM),
        float(vac),
    ], dtype=np.float32)


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def toward_pos(cur_x, cur_y, tgt_x, tgt_y, vac=0.0, darm=0.0, cur_theta=None):
    dx = np.clip((tgt_x - cur_x) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((tgt_y - cur_y) * K_POS, -MAX_DY, MAX_DY)
    dtheta = 0.0
    if cur_theta is not None:
        err = normalize_angle(-np.pi / 2 - cur_theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    return clip_action(dx=dx, dy=dy, dtheta=dtheta, darm=darm, vac=vac)


def control_arm(cur_arm, tgt_arm, vac=0.0, cur_theta=None):
    darm = np.clip((tgt_arm - cur_arm) * K_ARM, -MAX_DARM, MAX_DARM)
    dtheta = 0.0
    if cur_theta is not None:
        err = normalize_angle(-np.pi / 2 - cur_theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    return clip_action(darm=darm, dtheta=dtheta, vac=vac)


def pos_reached(cur_x, cur_y, tgt_x, tgt_y, tol=POS_TOL):
    return abs(cur_x - tgt_x) < tol and abs(cur_y - tgt_y) < tol


def theta_reached(cur_theta, tgt_theta=None, tol=THETA_TOL):
    if tgt_theta is None:
        tgt_theta = -np.pi / 2
    return abs(normalize_angle(tgt_theta - cur_theta)) < tol


def arm_reached(cur_arm, tgt_arm, tol=ARM_TOL):
    return abs(cur_arm - tgt_arm) < tol


def make_birrt_fns(obs_snapshot, excluded_obs_indices=None, rng=None):
    """Build BiRRT callbacks for robot base navigation (world-bounds collision only)."""
    from obs_helpers import get_obstruction, NUM_OBSTRUCTIONS
    if rng is None:
        rng = np.random.default_rng(0)
    if excluded_obs_indices is None:
        excluded_obs_indices = set()

    # We exclude ALL obstructions since we navigate at high altitude above them.
    # Only world walls matter.

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
        return x < X_MIN_NAV or x > X_MAX_NAV or y < Y_MIN_NAV or y > Y_MAX_NAV

    def distance_fn(s1, s2):
        return float(np.linalg.norm(s1 - s2))

    return sample_fn, extend_fn, collision_fn, distance_fn


def plan_path(primitives, obs, start_xy, goal_xy, rng=None):
    """Plan a collision-free 2D path for robot base."""
    if rng is None:
        rng = np.random.default_rng(42)
    sample_fn, extend_fn, collision_fn, distance_fn = make_birrt_fns(obs, rng=rng)
    BiRRT = primitives['BiRRT']
    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn,
                  rng, BIRRT_NUM_ATTEMPTS, BIRRT_NUM_ITERS, BIRRT_SMOOTH_AMT)
    start = np.array(start_xy, dtype=float)
    goal = np.array(goal_xy, dtype=float)
    return birrt.query(start, goal)


def get_drop_xy(i):
    """Return (drop_x, drop_y) for placing obstruction i."""
    return DROP_X_BASE - i * DROP_X_SPACING, NAV_HIGH_Y
