"""Observation parsing helpers. All magic numbers are named constants here."""
import numpy as np

# ---- Robot feature indices ----
ROB_X = 0
ROB_Y = 1
ROB_THETA = 2
ROB_BASE_RADIUS = 3
ROB_ARM_JOINT = 4
ROB_ARM_LENGTH = 5
ROB_VACUUM = 6
ROB_GRIPPER_HEIGHT = 7
ROB_GRIPPER_WIDTH = 8

# ---- Target surface feature indices ----
SURF_X = 9
SURF_Y = 10
SURF_THETA = 11
SURF_STATIC = 12
SURF_COLOR_R = 13
SURF_COLOR_G = 14
SURF_COLOR_B = 15
SURF_Z_ORDER = 16
SURF_WIDTH = 17
SURF_HEIGHT = 18

# ---- Target block feature indices ----
BLOCK_X = 19
BLOCK_Y = 20
BLOCK_THETA = 21
BLOCK_STATIC = 22
BLOCK_COLOR_R = 23
BLOCK_COLOR_G = 24
BLOCK_COLOR_B = 25
BLOCK_Z_ORDER = 26
BLOCK_WIDTH = 27
BLOCK_HEIGHT = 28

# ---- Obstruction feature strides ----
NUM_OBSTRUCTIONS = 4
OBS0_START = 29         # index of obstruction0's first feature
OBS_STRIDE = 10         # features per obstruction

# Offsets within each obstruction block
OBS_OFF_X = 0
OBS_OFF_Y = 1
OBS_OFF_THETA = 2
OBS_OFF_STATIC = 3
OBS_OFF_WIDTH = 8
OBS_OFF_HEIGHT = 9

# ---- World geometry constants ----
WORLD_X_MIN = 0.0
WORLD_X_MAX = 1.618        # golden ratio
WORLD_Y_MIN = 0.0
WORLD_Y_MAX = 1.0
ROB_RADIUS = 0.1           # robot base radius
ARM_MIN = 0.1              # minimum arm_joint (= base_radius)
ARM_MAX = 0.2              # maximum arm_joint (= arm_length)

# Approach geometry
APPROACH_ARM_JOINT = 0.15  # arm_joint for picking (midpoint of [0.1, 0.2])
APPROACH_Y_ABOVE = 0.25    # robot y when arm (0.15) reaches table (y=0.1)
TABLE_OBJ_Y = 0.1          # objects sit at this y center

# is_on tolerance
IS_ON_TOL = 0.025

# Vacuum threshold
VAC_THRESHOLD = 0.5

# ---- Extraction helpers ----

def get_robot(obs):
    return {
        'x': float(obs[ROB_X]),
        'y': float(obs[ROB_Y]),
        'theta': float(obs[ROB_THETA]),
        'base_radius': float(obs[ROB_BASE_RADIUS]),
        'arm_joint': float(obs[ROB_ARM_JOINT]),
        'arm_length': float(obs[ROB_ARM_LENGTH]),
        'vacuum': float(obs[ROB_VACUUM]),
    }


def get_surface(obs):
    return {
        'x': float(obs[SURF_X]),
        'y': float(obs[SURF_Y]),
        'width': float(obs[SURF_WIDTH]),
        'height': float(obs[SURF_HEIGHT]),
    }


def get_block(obs):
    return {
        'x': float(obs[BLOCK_X]),
        'y': float(obs[BLOCK_Y]),
        'theta': float(obs[BLOCK_THETA]),
        'width': float(obs[BLOCK_WIDTH]),
        'height': float(obs[BLOCK_HEIGHT]),
    }


def get_obstruction(obs, i):
    """Get obstruction i (0-indexed) as a dict."""
    base = OBS0_START + i * OBS_STRIDE
    return {
        'x': float(obs[base + OBS_OFF_X]),
        'y': float(obs[base + OBS_OFF_Y]),
        'width': float(obs[base + OBS_OFF_WIDTH]),
        'height': float(obs[base + OBS_OFF_HEIGHT]),
    }


def gripper_pos(obs):
    """Returns (gx, gy) of gripper tip."""
    r = get_robot(obs)
    gx = r['x'] + np.cos(r['theta']) * r['arm_joint']
    gy = r['y'] + np.sin(r['theta']) * r['arm_joint']
    return gx, gy


def is_vacuum_on(obs):
    return float(obs[ROB_VACUUM]) > VAC_THRESHOLD


def is_holding_block(obs):
    """Heuristic: vacuum is on and block is near gripper tip."""
    if not is_vacuum_on(obs):
        return False
    gx, gy = gripper_pos(obs)
    b = get_block(obs)
    dist = np.hypot(b['x'] - gx, b['y'] - gy)
    return dist < HOLDING_DIST_THRESH


# How close block must be to gripper tip to be considered held
HOLDING_DIST_THRESH = 0.20


def rect_overlap_1d(c1, s1, c2, s2):
    """Check if two 1D intervals [c1-s1/2, c1+s1/2] and [c2-s2/2, c2+s2/2] overlap."""
    return abs(c1 - c2) < (s1 + s2) / 2.0


def obstruction_overlaps_surface(obs, i):
    """Check if obstruction i overlaps with target surface (x-axis only, robust check)."""
    s = get_surface(obs)
    o = get_obstruction(obs, i)
    # Use a slightly generous check: if the obstruction's x range overlaps surface x range
    x_overlap = rect_overlap_1d(s['x'], s['width'], o['x'], o['width'])
    # Also check y - obstruction should be near table level
    y_near_table = o['y'] < (TABLE_OBJ_Y + 0.15)
    return x_overlap and y_near_table


def any_obstruction_on_surface(obs):
    """Return True if any obstruction is overlapping the target surface."""
    return any(obstruction_overlaps_surface(obs, i) for i in range(NUM_OBSTRUCTIONS))


def block_on_surface(obs):
    """Check if target block's bottom vertices are within surface bounds (is_on condition)."""
    b = get_block(obs)
    s = get_surface(obs)
    bx, by = b['x'], b['y']
    bw, bh = b['width'], b['height']
    sw, sh = s['width'], s['height']
    sx, sy = s['x'], s['y']

    # Bottom two vertices of block (theta assumed ~0)
    half_h = bh / 2.0
    half_w = bw / 2.0
    bottom_y = by - half_h

    # After tol offset
    check_y = bottom_y - IS_ON_TOL

    # Surface bounds
    surf_x_min = sx - sw / 2.0
    surf_x_max = sx + sw / 2.0
    surf_y_min = sy - sh / 2.0
    surf_y_max = sy + sh / 2.0

    # Check both bottom vertices
    for bvx in [bx - half_w, bx + half_w]:
        if not (surf_x_min <= bvx <= surf_x_max):
            return False
        if not (surf_y_min <= check_y <= surf_y_max):
            return False
    return True
