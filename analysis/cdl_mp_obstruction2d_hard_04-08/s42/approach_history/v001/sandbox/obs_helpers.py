"""Observation parsing helpers for Obstruction2D-o4-v0."""
import numpy as np

# ─── Robot feature indices ───────────────────────────────────────────────────
IDX_ROBOT_X       = 0
IDX_ROBOT_Y       = 1
IDX_ROBOT_THETA   = 2
IDX_ROBOT_BR      = 3   # base_radius
IDX_ROBOT_AJ      = 4   # arm_joint (current extension)
IDX_ROBOT_AL      = 5   # arm_length (max extension)
IDX_ROBOT_VAC     = 6   # vacuum on/off
IDX_ROBOT_GH      = 7   # gripper_height (lateral extent, 0.07)
IDX_ROBOT_GW      = 8   # gripper_width  (along-arm thickness, 0.01)

# ─── Target surface (indices 9-18) ───────────────────────────────────────────
IDX_SURF_X        = 9
IDX_SURF_Y        = 10
IDX_SURF_THETA    = 11
IDX_SURF_STATIC   = 12
IDX_SURF_W        = 17   # width
IDX_SURF_H        = 18   # height

# ─── Target block (indices 19-28) ────────────────────────────────────────────
IDX_BLOCK_X       = 19
IDX_BLOCK_Y       = 20
IDX_BLOCK_THETA   = 21
IDX_BLOCK_STATIC  = 22
IDX_BLOCK_W       = 27   # width
IDX_BLOCK_H       = 28   # height

# ─── Obstructions (10 features each, starting at 29) ─────────────────────────
NUM_OBSTRUCTIONS  = 4
OBS_STRIDE        = 10
IDX_OBS_BASE      = 29   # obstruction0 starts here
# obstruction i: IDX_OBS_BASE + i * OBS_STRIDE
#   +0=x, +1=y, +2=theta, +3=static, +4=r, +5=g, +6=b, +7=z, +8=width, +9=height

# ─── World constants ─────────────────────────────────────────────────────────
WORLD_MIN_X       = 0.0
WORLD_MAX_X       = 1.618
WORLD_MIN_Y       = 0.0
WORLD_MAX_Y       = 1.0
TABLE_HEIGHT      = 0.1    # table top y

# ─── is_on tolerance (from reward function) ──────────────────────────────────
IS_ON_TOL         = 0.025


def obs_i(obstruction_idx: int) -> int:
    """Start index for obstruction i in the observation vector."""
    return IDX_OBS_BASE + obstruction_idx * OBS_STRIDE


def extract_robot(obs: np.ndarray) -> dict:
    return {
        'x': float(obs[IDX_ROBOT_X]),
        'y': float(obs[IDX_ROBOT_Y]),
        'theta': float(obs[IDX_ROBOT_THETA]),
        'base_radius': float(obs[IDX_ROBOT_BR]),
        'arm_joint': float(obs[IDX_ROBOT_AJ]),
        'arm_length': float(obs[IDX_ROBOT_AL]),
        'vacuum': float(obs[IDX_ROBOT_VAC]),
        'gripper_height': float(obs[IDX_ROBOT_GH]),
        'gripper_width': float(obs[IDX_ROBOT_GW]),
    }


def extract_target_surface(obs: np.ndarray) -> dict:
    return {
        'x': float(obs[IDX_SURF_X]),
        'y': float(obs[IDX_SURF_Y]),
        'width': float(obs[IDX_SURF_W]),
        'height': float(obs[IDX_SURF_H]),
    }


def extract_target_block(obs: np.ndarray) -> dict:
    return {
        'x': float(obs[IDX_BLOCK_X]),
        'y': float(obs[IDX_BLOCK_Y]),
        'width': float(obs[IDX_BLOCK_W]),
        'height': float(obs[IDX_BLOCK_H]),
    }


def extract_obstruction(obs: np.ndarray, i: int) -> dict:
    base = obs_i(i)
    return {
        'x': float(obs[base + 0]),
        'y': float(obs[base + 1]),
        'width': float(obs[base + 8]),
        'height': float(obs[base + 9]),
    }


def gripper_tip_xy(obs: np.ndarray) -> tuple:
    """Compute gripper tip position from robot state.

    tool_tip_x = robot_x + (arm_joint + gripper_width/2) * cos(theta)
    tool_tip_y = robot_y + (arm_joint + gripper_width/2) * sin(theta)
    """
    r = extract_robot(obs)
    reach = r['arm_joint'] + r['gripper_width'] / 2.0
    tx = r['x'] + reach * np.cos(r['theta'])
    ty = r['y'] + reach * np.sin(r['theta'])
    return tx, ty


def _rect_contains_point_bl(rx, ry, rw, rh, px, py) -> bool:
    """Check if (px, py) is inside rectangle with BOTTOM-LEFT corner (rx, ry)."""
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def _rect_contains_point(rx, ry, rw, rh, px, py) -> bool:
    """Check if (px, py) is inside axis-aligned rectangle centered at (rx, ry)."""
    return (rx - rw / 2 <= px <= rx + rw / 2 and
            ry - rh / 2 <= py <= ry + rh / 2)


def is_rect_on_surface(obs: np.ndarray, rect_x_idx: int) -> bool:
    """True if the rectangle starting at rect_x_idx satisfies is_on(target_surface).

    Observation stores (x, y) as BOTTOM-LEFT corner of each rectangle.
    Implements the same check as the reward function: the two bottom-most
    vertices of the rect (offset down by IS_ON_TOL) must both lie inside the
    target surface rectangle.
    """
    surf = extract_target_surface(obs)
    # (x, y) is bottom-left corner; width/height follow
    bl_x = float(obs[rect_x_idx + 0])
    bl_y = float(obs[rect_x_idx + 1])
    rw   = float(obs[rect_x_idx + 8])

    # Bottom-two vertices at (bl_x, bl_y) and (bl_x+rw, bl_y), offset by tol
    offset_y = bl_y - IS_ON_TOL

    for vx in [bl_x, bl_x + rw]:
        if not _rect_contains_point_bl(surf['x'], surf['y'],
                                       surf['width'], surf['height'],
                                       vx, offset_y):
            return False
    return True


def is_block_on_surface(obs: np.ndarray) -> bool:
    return is_rect_on_surface(obs, IDX_BLOCK_X)


def is_obstruction_on_surface(obs: np.ndarray, i: int) -> bool:
    return is_rect_on_surface(obs, obs_i(i))


def any_obstruction_on_surface(obs: np.ndarray) -> bool:
    return any(is_obstruction_on_surface(obs, i) for i in range(NUM_OBSTRUCTIONS))


def is_holding(obs: np.ndarray) -> bool:
    """True if vacuum is active (robot is holding something)."""
    return obs[IDX_ROBOT_VAC] > 0.5


def approach_xy_for_pick(obj: dict, arm_length: float) -> tuple:
    """Compute robot (x, y) for approaching the object from above (theta=-pi/2).

    obs (x, y) is the BOTTOM-LEFT corner of the object.
    Object center: (x + w/2, y + h/2).
    When at (center_x, center_y + arm_length) with arm fully extended (theta=-pi/2),
    gripper tip is at object center.
    """
    center_x = obj['x'] + obj['width'] / 2.0
    center_y = obj['y'] + obj['height'] / 2.0
    return center_x, center_y + arm_length


def get_obstacle_rects(obs: np.ndarray, exclude_idx: int = -1) -> list:
    """Return list of (cx, cy, w, h) centered rects for BiRRT collision checking.

    obs stores (x, y) as BOTTOM-LEFT corner; we convert to center here.
    Excludes the obstruction at exclude_idx (the one being approached).
    """
    rects = []
    # Add obstructions (except the one being picked up)
    for i in range(NUM_OBSTRUCTIONS):
        if i == exclude_idx:
            continue
        ob = extract_obstruction(obs, i)
        cx = ob['x'] + ob['width'] / 2.0
        cy = ob['y'] + ob['height'] / 2.0
        rects.append((cx, cy, ob['width'], ob['height']))
    # Target block is also an obstacle during clearing phase
    # (but skip during PlaceTargetBlock when it's being carried)
    blk = extract_target_block(obs)
    cx = blk['x'] + blk['width'] / 2.0
    cy = blk['y'] + blk['height'] / 2.0
    rects.append((cx, cy, blk['width'], blk['height']))
    return rects
