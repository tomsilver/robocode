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
OBS_OFF_X = 0
OBS_OFF_Y = 1
OBS_OFF_WIDTH = 8
OBS_OFF_HEIGHT = 9

# ---- World geometry constants ----
WORLD_X_MIN = 0.0
WORLD_X_MAX = 1.618
WORLD_Y_MIN = 0.0
WORLD_Y_MAX = 1.0
ROB_RADIUS = 0.1
ARM_MIN = 0.10          # minimum arm_joint
ARM_MAX = 0.20          # maximum arm_joint

# Robot arm geometry (derived from source code analysis):
# Gripper center = robot_base + arm_joint * arm_dir
# Gripper extends ±gripper_width/2 along arm, ±gripper_height/2 perpendicular
# Suction center = robot_base + (arm_joint + gripper_width + suction_width/2) * arm_dir
#                = robot_base + (arm_joint + 0.015) * arm_dir  (suction_width = gripper_width = 0.01)
# Suction zone: ZOrder.NONE (no collision), height=gripper_height=0.07, width=gripper_width=0.01

GRIPPER_WIDTH = 0.01    # along arm direction
GRIPPER_HEIGHT = 0.07   # perpendicular to arm
SUCTION_WIDTH = 0.01    # same as gripper_width
SUCTION_EXTRA = 0.015   # arm_joint + suction_extra = suction center distance from base

# Pick approach constants:
# Robot approaches from above with theta = -pi/2.
# For arm_joint = PICK_ARM_JOINT:
#   arm_shaft_bottom_y = robot_y - arm_joint  (must be > obj_top)
#   suction_center_y = robot_y - arm_joint - SUCTION_EXTRA  (must be in obj_y range)
# Setting robot_y = obj_top + OBJ_TOP_OFFSET gives:
#   arm_shaft_bottom = obj_top + OBJ_TOP_OFFSET - PICK_ARM_JOINT = obj_top + 0.01 > obj_top ✓
#   suction_center = obj_top + 0.01 - SUCTION_EXTRA = obj_top - 0.005 (inside object top) ✓
PICK_ARM_JOINT = 0.13           # arm joint for picking
OBJ_TOP_OFFSET = 0.14           # robot_y = obj_top + OBJ_TOP_OFFSET for picking
# So: robot_y = obj_y + obj_h/2 + OBJ_TOP_OFFSET

# Place constants (with arm retracted to ARM_MIN = 0.10):
# block_center_y = robot_y - ARM_MIN - SUCTION_EXTRA = robot_y - 0.115
# block_bottom_y = block_center_y - block_h/2
# Target block_bottom_y = 0.025 (middle of is_on valid range)
# => robot_y = 0.025 + block_h/2 + 0.115 = PLACE_Y_OFFSET + block_h/2
PLACE_Y_OFFSET = 0.14           # robot_y = PLACE_Y_OFFSET + block_h/2 for placing

# Surface constants
SURF_TOP_Y = 0.05              # surface center y=0, height=0.1, top at y=0.05
IS_ON_TOL = 0.025              # tolerance in is_on check

# Navigation
TABLE_OBJ_Y = 0.1             # objects sit at this y center
VAC_THRESHOLD = 0.5
HOLDING_DIST_THRESH = 0.25    # max distance from gripper to block when held


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
    base = OBS0_START + i * OBS_STRIDE
    return {
        'x': float(obs[base + OBS_OFF_X]),
        'y': float(obs[base + OBS_OFF_Y]),
        'width': float(obs[base + OBS_OFF_WIDTH]),
        'height': float(obs[base + OBS_OFF_HEIGHT]),
    }


def gripper_pos(obs):
    """Returns (gx, gy) of gripper center (arm tip)."""
    r = get_robot(obs)
    gx = r['x'] + np.cos(r['theta']) * r['arm_joint']
    gy = r['y'] + np.sin(r['theta']) * r['arm_joint']
    return gx, gy


def suction_pos(obs):
    """Returns (sx, sy) of suction zone center."""
    r = get_robot(obs)
    d = r['arm_joint'] + SUCTION_EXTRA
    sx = r['x'] + np.cos(r['theta']) * d
    sy = r['y'] + np.sin(r['theta']) * d
    return sx, sy


def is_vacuum_on(obs):
    return float(obs[ROB_VACUUM]) > VAC_THRESHOLD


def is_holding_block(obs):
    """Check if robot is holding the target block."""
    if not is_vacuum_on(obs):
        return False
    gx, gy = gripper_pos(obs)
    b = get_block(obs)
    dist = np.hypot(b['x'] - gx, b['y'] - gy)
    return dist < HOLDING_DIST_THRESH


def pick_robot_y(obj_y, obj_h):
    """Compute robot y for approaching object from above."""
    obj_top = obj_y + obj_h / 2.0
    return obj_top + OBJ_TOP_OFFSET


def place_robot_y(block_h):
    """Compute robot y for placing block on surface (arm retracted)."""
    return PLACE_Y_OFFSET + block_h / 2.0


def rect_overlap_1d(c1, s1, c2, s2):
    """Check if [c1-s1/2, c1+s1/2] and [c2-s2/2, c2+s2/2] overlap."""
    return abs(c1 - c2) < (s1 + s2) / 2.0


def obstruction_overlaps_surface(obs, i):
    """Check if obstruction i overlaps with target surface (x-axis only)."""
    s = get_surface(obs)
    o = get_obstruction(obs, i)
    x_overlap = rect_overlap_1d(s['x'], s['width'], o['x'], o['width'])
    y_near_table = o['y'] < (TABLE_OBJ_Y + 0.20)  # still at table level
    return x_overlap and y_near_table


def any_obstruction_on_surface(obs):
    return any(obstruction_overlaps_surface(obs, i) for i in range(NUM_OBSTRUCTIONS))


def block_on_surface(obs):
    """Check if target block satisfies is_on(block, surface)."""
    b = get_block(obs)
    s = get_surface(obs)
    bx, by = b['x'], b['y']
    bw, bh = b['width'], b['height']
    sw, sh = s['width'], s['height']
    sx, sy = s['x'], s['y']

    half_h = bh / 2.0
    half_w = bw / 2.0
    bottom_y = by - half_h
    check_y = bottom_y - IS_ON_TOL

    surf_x_min = sx - sw / 2.0
    surf_x_max = sx + sw / 2.0
    surf_y_min = sy - sh / 2.0
    surf_y_max = sy + sh / 2.0

    for bvx in [bx - half_w, bx + half_w]:
        if not (surf_x_min <= bvx <= surf_x_max):
            return False
        if not (surf_y_min <= check_y <= surf_y_max):
            return False
    return True
