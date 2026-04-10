"""Action generation helpers for Obstruction2D-o4-v0."""
import numpy as np
from obs_helpers import (
    WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, WORLD_MAX_Y,
    TABLE_HEIGHT, NUM_OBSTRUCTIONS,
    extract_robot, extract_target_surface,
)

# ─── Action indices ───────────────────────────────────────────────────────────
ACT_DX    = 0
ACT_DY    = 1
ACT_DTHETA = 2
ACT_DARM  = 3
ACT_VAC   = 4

# ─── Action limits ────────────────────────────────────────────────────────────
MAX_DX        = 0.050
MAX_DY        = 0.050
MAX_DTHETA    = 0.196   # ~11.2 degrees
MAX_DARM      = 0.100

# ─── Tolerances ───────────────────────────────────────────────────────────────
POS_TOL       = 0.02    # position tolerance for waypoint arrival
ANGLE_TOL     = 0.05    # angle tolerance (radians)
ARM_TOL       = 0.01    # arm extension tolerance

# ─── Navigation constants ────────────────────────────────────────────────────
NAV_HEIGHT    = 0.75    # safe cruising height above table
APPROACH_MARGIN = 0.05  # extra margin above pick position
COLLISION_MARGIN = 0.02 # extra clearance for BiRRT collision checks
BIRRT_ATTEMPTS = 8
BIRRT_ITERS    = 1000
BIRRT_SMOOTH   = 50

# ─── Drop zones (x positions, robot y = DROP_Y) ──────────────────────────────
DROP_Y     = NAV_HEIGHT       # robot y when dropping
DROP_XS    = [0.15, 0.80, 1.10, 1.40]  # x positions for 4 obstructions

# ─── Robot arm geometry ───────────────────────────────────────────────────────
THETA_DOWN = -np.pi / 2.0   # arm pointing down


def make_action(dx=0.0, dy=0.0, dtheta=0.0, darm=0.0, vac=0.0) -> np.ndarray:
    return np.array([
        np.clip(dx, -MAX_DX, MAX_DX),
        np.clip(dy, -MAX_DY, MAX_DY),
        np.clip(dtheta, -MAX_DTHETA, MAX_DTHETA),
        np.clip(darm, -MAX_DARM, MAX_DARM),
        float(vac),
    ], dtype=np.float32)


def angle_diff(a: float, b: float) -> float:
    """Signed difference a-b wrapped to [-pi, pi]."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def step_toward_xy(robot_x, robot_y, tx, ty, vacuum=0.0, darm=0.0) -> np.ndarray:
    """Action to move robot base toward (tx, ty)."""
    dx = np.clip(tx - robot_x, -MAX_DX, MAX_DX)
    dy = np.clip(ty - robot_y, -MAX_DY, MAX_DY)
    return make_action(dx=dx, dy=dy, darm=darm, vac=vacuum)


def rotate_toward(robot_theta, target_theta, vacuum=0.0) -> np.ndarray:
    """Action to rotate robot toward target_theta."""
    diff = angle_diff(target_theta, robot_theta)
    dtheta = np.clip(diff, -MAX_DTHETA, MAX_DTHETA)
    return make_action(dtheta=dtheta, vac=vacuum)


def extend_arm(current_joint, target_joint, vacuum=0.0) -> np.ndarray:
    """Action to set arm_joint toward target_joint."""
    darm = np.clip(target_joint - current_joint, -MAX_DARM, MAX_DARM)
    return make_action(darm=darm, vac=vacuum)


def circle_rect_collision(cx, cy, r, rx, ry, rw, rh) -> bool:
    """True if circle (cx, cy, r) overlaps rectangle (rx±rw/2, ry±rh/2)."""
    closest_x = max(rx - rw / 2, min(cx, rx + rw / 2))
    closest_y = max(ry - rh / 2, min(cy, ry + rh / 2))
    return (cx - closest_x) ** 2 + (cy - closest_y) ** 2 < r ** 2


def make_collision_fn(base_radius: float, obstacle_rects: list, min_y: float = TABLE_HEIGHT + 0.10):
    """Return a collision_fn(state) for robot base navigation.

    state is np.array([x, y]).
    obstacle_rects is list of (x, y, w, h).
    """
    r = base_radius + COLLISION_MARGIN

    def collision_fn(state) -> bool:
        x, y = float(state[0]), float(state[1])
        # World bounds
        if x - base_radius < WORLD_MIN_X or x + base_radius > WORLD_MAX_X:
            return True
        if y < min_y or y + base_radius > WORLD_MAX_Y:
            return True
        # Object rectangles
        for (rx, ry, rw, rh) in obstacle_rects:
            if circle_rect_collision(x, y, r, rx, ry, rw, rh):
                return True
        return False

    return collision_fn


def make_birrt_fns(base_radius: float, obstacle_rects: list,
                   min_y: float = TABLE_HEIGHT + 0.10, rng=None):
    """Build sample_fn, extend_fn, collision_fn, distance_fn for BiRRT."""
    collision_fn = make_collision_fn(base_radius, obstacle_rects, min_y)
    _rng = rng if rng is not None else np.random.default_rng()

    def sample_fn(state) -> np.ndarray:
        return np.array([
            _rng.uniform(WORLD_MIN_X + base_radius, WORLD_MAX_X - base_radius),
            _rng.uniform(min_y, WORLD_MAX_Y - base_radius),
        ])

    def extend_fn(s1, s2) -> list:
        d = np.linalg.norm(s2 - s1)
        if d < 1e-6:
            return []
        n_steps = max(1, int(d / (MAX_DX * 0.8)))
        return [s1 + (s2 - s1) * float(k) / n_steps for k in range(1, n_steps + 1)]

    def distance_fn(s1, s2) -> float:
        return float(np.linalg.norm(s1 - s2))

    return sample_fn, extend_fn, collision_fn, distance_fn


def plan_base_path(obs, primitives, target_x, target_y,
                   obstacle_rects=None, exclude_obs_idx=-1,
                   min_y=None) -> list | None:
    """Use BiRRT to plan robot base path from current pos to (target_x, target_y).

    Returns list of (x, y) waypoints or None.
    """
    from obs_helpers import get_obstacle_rects
    robot = extract_robot(obs)
    if obstacle_rects is None:
        obstacle_rects = get_obstacle_rects(obs, exclude_idx=exclude_obs_idx)
    if min_y is None:
        min_y = TABLE_HEIGHT + robot['base_radius']

    start = np.array([robot['x'], robot['y']])
    goal  = np.array([target_x, target_y])

    rng = np.random.default_rng()
    sample_fn, extend_fn, collision_fn, distance_fn = make_birrt_fns(
        robot['base_radius'], obstacle_rects, min_y, rng=rng)

    BiRRT = primitives['BiRRT']
    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng,
                  BIRRT_ATTEMPTS, BIRRT_ITERS, BIRRT_SMOOTH)
    path = birrt.query(start, goal)
    return path  # list of np.array([x, y]) or None


PLACE_EPSILON = 0.010   # extra margin above surface so block doesn't collide
DROP_Y_MARGIN = 0.030   # how far above pick_y to release obstruction at drop zone


def placement_robot_y(block_height: float, arm_length: float,
                      surf_y: float, surf_height: float) -> float:
    """Robot y so that gripper tip (arm extended) puts block just above surface.

    obs surf_y is BOTTOM-LEFT y; surf_top = surf_y + surf_height.
    With arm_joint = arm_length:
      gripper_tip_y = robot_y - arm_length - gripper_width/2 (≈ - 0.005)
    We want block_bottom = surf_top + PLACE_EPSILON:
      block_center_y = surf_top + PLACE_EPSILON + block_height/2
      gripper_tip_y = block_center_y
      robot_y = block_center_y + arm_length + 0.005
    """
    surf_top = surf_y + surf_height          # bottom-left → top
    block_center_y = surf_top + PLACE_EPSILON + block_height / 2.0
    return block_center_y + arm_length + 0.005
