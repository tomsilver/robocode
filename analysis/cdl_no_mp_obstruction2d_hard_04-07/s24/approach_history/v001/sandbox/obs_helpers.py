"""Observation parsing helpers. All magic numbers live here as named constants."""
import numpy as np

# ─── Observation indices ────────────────────────────────────────────────────
# Robot (9 features)
IDX_ROBOT_X = 0
IDX_ROBOT_Y = 1
IDX_ROBOT_THETA = 2
IDX_ROBOT_BASE_RADIUS = 3
IDX_ROBOT_ARM_JOINT = 4
IDX_ROBOT_ARM_LENGTH = 5
IDX_ROBOT_VACUUM = 6
IDX_ROBOT_GRIPPER_HEIGHT = 7
IDX_ROBOT_GRIPPER_WIDTH = 8

# Target surface (10 features, starting at 9)
IDX_SURF_X = 9
IDX_SURF_Y = 10
IDX_SURF_THETA = 11
IDX_SURF_STATIC = 12
IDX_SURF_WIDTH = 17
IDX_SURF_HEIGHT = 18

# Target block (10 features, starting at 19)
IDX_BLOCK_X = 19
IDX_BLOCK_Y = 20
IDX_BLOCK_THETA = 21
IDX_BLOCK_STATIC = 22
IDX_BLOCK_WIDTH = 27
IDX_BLOCK_HEIGHT = 28

# Obstructions: 4 × 10 features, starting at 29
IDX_OBS_BASES = [29, 39, 49, 59]
OBS_X_OFFSET = 0
OBS_Y_OFFSET = 1
OBS_THETA_OFFSET = 2
OBS_WIDTH_OFFSET = 8
OBS_HEIGHT_OFFSET = 9
NUM_OBSTRUCTIONS = 4

# ─── World / physics constants ───────────────────────────────────────────────
WORLD_MIN_X = 0.0
WORLD_MAX_X = 1.618   # golden ratio
TABLE_TOP_Y = 0.1     # y of table surface (= surface_y + surface_height)

# Gripper geometry (from obstruction2d config)
GRIPPER_WIDTH_ALONG_ARM = 0.01   # thin along arm
SUCTION_WIDTH = 0.01             # same as gripper_width
# Suction center offset from robot base along arm direction:
SUCTION_OFFSET = GRIPPER_WIDTH_ALONG_ARM + SUCTION_WIDTH / 2  # = 0.015

# Grasp / place tolerances
VACUUM_THRESHOLD = 0.5
IS_ON_TOL = 0.025      # same as the env's tol in is_on()
GRASP_DIST_TOL = 0.12  # block near gripper when carried (generous)

# Drop zone sizing
DROP_MARGIN = 0.18     # min clearance between drop zone and surface edge
DROP_SPACING = 0.15    # x spacing between consecutive drop zones


# ─── Extraction helpers ──────────────────────────────────────────────────────
def extract_robot(obs):
    return {
        'x':            float(obs[IDX_ROBOT_X]),
        'y':            float(obs[IDX_ROBOT_Y]),
        'theta':        float(obs[IDX_ROBOT_THETA]),
        'base_radius':  float(obs[IDX_ROBOT_BASE_RADIUS]),
        'arm_joint':    float(obs[IDX_ROBOT_ARM_JOINT]),
        'arm_length':   float(obs[IDX_ROBOT_ARM_LENGTH]),
        'vacuum':       float(obs[IDX_ROBOT_VACUUM]),
        'gripper_height': float(obs[IDX_ROBOT_GRIPPER_HEIGHT]),
        'gripper_width':  float(obs[IDX_ROBOT_GRIPPER_WIDTH]),
    }


def extract_target_surface(obs):
    return {
        'x':      float(obs[IDX_SURF_X]),
        'y':      float(obs[IDX_SURF_Y]),
        'width':  float(obs[IDX_SURF_WIDTH]),
        'height': float(obs[IDX_SURF_HEIGHT]),
        'theta':  float(obs[IDX_SURF_THETA]),
    }


def extract_target_block(obs):
    return {
        'x':      float(obs[IDX_BLOCK_X]),
        'y':      float(obs[IDX_BLOCK_Y]),
        'width':  float(obs[IDX_BLOCK_WIDTH]),
        'height': float(obs[IDX_BLOCK_HEIGHT]),
        'theta':  float(obs[IDX_BLOCK_THETA]),
    }


def extract_obstruction(obs, i):
    base = IDX_OBS_BASES[i]
    return {
        'x':      float(obs[base + OBS_X_OFFSET]),
        'y':      float(obs[base + OBS_Y_OFFSET]),
        'width':  float(obs[base + OBS_WIDTH_OFFSET]),
        'height': float(obs[base + OBS_HEIGHT_OFFSET]),
        'theta':  float(obs[base + OBS_THETA_OFFSET]),
    }


# ─── Geometric helpers ───────────────────────────────────────────────────────
def rect_center_x(rect):
    """Horizontal center of a rect (valid for theta=0)."""
    return rect['x'] + rect['width'] / 2.0


def rect_center_y(rect):
    """Vertical center of a rect (valid for theta=0)."""
    return rect['y'] + rect['height'] / 2.0


def rect_top(rect):
    """Top y of a rect (valid for theta=0)."""
    return rect['y'] + rect['height']


# Tolerance for determining "obstruction is at table/surface level"
OBS_ON_SURFACE_Y_TOL = 0.05  # obs y within this of surface top → at table level


def x_ranges_overlap(r1, r2):
    """True if r1's x-range overlaps r2's x-range."""
    return r1['x'] < r2['x'] + r2['width'] and r1['x'] + r1['width'] > r2['x']


def rects_overlap_aabb(r1, r2):
    """AABB overlap (ignores rotation — fine for theta≈0 objects)."""
    return (x_ranges_overlap(r1, r2) and
            r1['y'] < r2['y'] + r2['height'] and
            r1['y'] + r1['height'] > r2['y'])


def obstruction_on_surface(obs, i):
    """True if obstruction i is at table level and its x-range overlaps surface.

    Obstructions sit ON the surface (y = surface_top + 1e-6) so a strict AABB
    check fails.  We detect them by checking (a) x-range overlaps and (b) the
    obstruction bottom is within OBS_ON_SURFACE_Y_TOL of the surface top.
    """
    obs_rect  = extract_obstruction(obs, i)
    surf_rect = extract_target_surface(obs)
    surf_top  = surf_rect['y'] + surf_rect['height']
    # Y proximity: obstruction bottom must be near surface top
    if abs(obs_rect['y'] - surf_top) > OBS_ON_SURFACE_Y_TOL:
        return False
    # X overlap
    return x_ranges_overlap(obs_rect, surf_rect)


def any_obstruction_on_surface(obs):
    """True if any obstruction overlaps target surface."""
    return any(obstruction_on_surface(obs, i) for i in range(NUM_OBSTRUCTIONS))


# ─── Grasp / placement predicates ────────────────────────────────────────────
def gripper_center(obs):
    r = extract_robot(obs)
    gx = r['x'] + np.cos(r['theta']) * r['arm_joint']
    gy = r['y'] + np.sin(r['theta']) * r['arm_joint']
    return gx, gy


def is_block_grasped(obs):
    """Vacuum on AND block center is within GRASP_DIST_TOL of gripper center."""
    r = extract_robot(obs)
    if r['vacuum'] < VACUUM_THRESHOLD:
        return False
    gx, gy = gripper_center(obs)
    blk = extract_target_block(obs)
    bx = blk['x'] + blk['width'] / 2.0
    by = blk['y'] + blk['height'] / 2.0
    return np.hypot(bx - gx, by - gy) < GRASP_DIST_TOL


def block_is_on_surface(obs):
    """Replicate the env's is_on() check for theta=0 block/surface."""
    blk = extract_target_block(obs)
    surf = extract_target_surface(obs)
    bx = blk['x']
    by = blk['y']
    bw = blk['width']
    sx = surf['x']
    sy = surf['y']
    sw = surf['width']
    sh = surf['height']
    for vx, vy in [(bx, by), (bx + bw, by)]:
        offset_y = vy - IS_ON_TOL
        if not (sx <= vx <= sx + sw and sy <= offset_y <= sy + sh):
            return False
    return True


# ─── Drop zone computation ────────────────────────────────────────────────────
def get_drop_zones(obs):
    """Return 4 x-positions for dropping obstructions, away from target surface."""
    surf = extract_target_surface(obs)
    sx = surf['x']
    sw = surf['width']
    sx_max = sx + sw

    left_space  = sx - WORLD_MIN_X - DROP_MARGIN
    right_space = WORLD_MAX_X - sx_max - DROP_MARGIN

    zones = []
    if left_space >= 4 * DROP_SPACING:
        for i in range(4):
            zones.append(WORLD_MIN_X + DROP_MARGIN + i * DROP_SPACING)
    elif right_space >= 4 * DROP_SPACING:
        for i in range(4):
            zones.append(WORLD_MAX_X - DROP_MARGIN - i * DROP_SPACING)
    else:
        # Distribute: as many as fit on left, rest on right
        n_left = max(0, min(4, int(left_space / DROP_SPACING)))
        zones = [WORLD_MIN_X + DROP_MARGIN + i * DROP_SPACING for i in range(n_left)]
        n_right = 4 - n_left
        zones += [WORLD_MAX_X - DROP_MARGIN - i * DROP_SPACING for i in range(n_right)]

    # Fallback: ensure we have exactly 4
    while len(zones) < 4:
        zones.append(WORLD_MAX_X / 2.0)

    return zones[:4]
