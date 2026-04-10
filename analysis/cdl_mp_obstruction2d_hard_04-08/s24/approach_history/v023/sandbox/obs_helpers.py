"""Observation parsing helpers. All magic numbers are named constants here.

COORDINATE CONVENTION:
  - Robot: x,y = circle center
  - Rectangles (surface, block, obstructions): x,y = BOTTOM-LEFT CORNER
    actual center = (x + width/2, y + height/2)
"""
import numpy as np

# ── Observation indices ──────────────────────────────────────────────────────
IDX_ROBOT_X           = 0
IDX_ROBOT_Y           = 1
IDX_ROBOT_THETA       = 2
IDX_ROBOT_BASE_RADIUS = 3
IDX_ROBOT_ARM_JOINT   = 4
IDX_ROBOT_ARM_LENGTH  = 5
IDX_ROBOT_VACUUM      = 6
IDX_ROBOT_GRIP_HEIGHT = 7
IDX_ROBOT_GRIP_WIDTH  = 8

IDX_SURF_X      = 9
IDX_SURF_Y      = 10
IDX_SURF_THETA  = 11
IDX_SURF_WIDTH  = 17
IDX_SURF_HEIGHT = 18

IDX_BLOCK_X      = 19
IDX_BLOCK_Y      = 20
IDX_BLOCK_THETA  = 21
IDX_BLOCK_WIDTH  = 27
IDX_BLOCK_HEIGHT = 28

OBS_STRIDE   = 10
IDX_OBS0_BASE = 29

def _obs_base(i: int) -> int:
    return IDX_OBS0_BASE + i * OBS_STRIDE

# ── Named tolerances / physics constants ─────────────────────────────────────
ARM_MIN_JOINT    = 0.10   # base_radius
ARM_MAX_JOINT    = 0.20   # arm_length
GRASP_TOL        = 0.015
ON_TOL           = 0.025  # used by environment's is_on
SURF_OVERLAP_TOL = 0.010  # margin for obstruction-on-surface check
TABLE_TOP_Y      = 0.10   # table spans y∈[0, 0.1], top at y=0.1
ROBOT_RADIUS     = 0.10
NAV_CLEAR_Y      = TABLE_TOP_Y + ROBOT_RADIUS + 0.02  # minimum robot center y
NUM_OBSTRUCTIONS = 4

# re-exported tolerances (used by act_helpers too)
NAV_XY_TOL    = 0.015
NAV_THETA_TOL = 0.06
ARM_EXTEND_TOL = 0.008

# ── Extraction helpers (rectangles: x,y = bottom-left corner) ─────────────

def extract_robot(obs: np.ndarray) -> dict:
    return {
        "x":           obs[IDX_ROBOT_X],
        "y":           obs[IDX_ROBOT_Y],
        "theta":       obs[IDX_ROBOT_THETA],
        "base_radius": obs[IDX_ROBOT_BASE_RADIUS],
        "arm_joint":   obs[IDX_ROBOT_ARM_JOINT],
        "arm_length":  obs[IDX_ROBOT_ARM_LENGTH],
        "vacuum":      obs[IDX_ROBOT_VACUUM],
        "grip_height": obs[IDX_ROBOT_GRIP_HEIGHT],
        "grip_width":  obs[IDX_ROBOT_GRIP_WIDTH],
    }


def extract_surface(obs: np.ndarray) -> dict:
    """x,y = bottom-left corner; cx,cy = center."""
    x = obs[IDX_SURF_X];  y = obs[IDX_SURF_Y]
    w = obs[IDX_SURF_WIDTH]; h = obs[IDX_SURF_HEIGHT]
    return {"x": x, "y": y, "width": w, "height": h,
            "cx": x + w/2, "cy": y + h/2,
            "x1": x, "x2": x+w, "y1": y, "y2": y+h}


def extract_block(obs: np.ndarray) -> dict:
    """x,y = bottom-left corner; cx,cy = center."""
    x = obs[IDX_BLOCK_X]; y = obs[IDX_BLOCK_Y]
    w = obs[IDX_BLOCK_WIDTH]; h = obs[IDX_BLOCK_HEIGHT]
    return {"x": x, "y": y, "width": w, "height": h,
            "cx": x + w/2, "cy": y + h/2,
            "x1": x, "x2": x+w, "y1": y, "y2": y+h}


def extract_obstruction(obs: np.ndarray, i: int) -> dict:
    """x,y = bottom-left corner; cx,cy = center."""
    b = _obs_base(i)
    x = obs[b+0]; y = obs[b+1]
    w = obs[b+8]; h = obs[b+9]
    return {"x": x, "y": y, "width": w, "height": h,
            "cx": x + w/2, "cy": y + h/2,
            "x1": x, "x2": x+w, "y1": y, "y2": y+h,
            "static": obs[b+3]}


def gripper_tip_pos(obs: np.ndarray) -> np.ndarray:
    """Suction zone center (vacuum on condition)."""
    r = extract_robot(obs)
    gw = r["grip_width"]
    reach = r["arm_joint"] + gw + gw/2  # arm_joint + 1.5*gripper_width
    cx = r["x"] + np.cos(r["theta"]) * reach
    cy = r["y"] + np.sin(r["theta"]) * reach
    return np.array([cx, cy])


def rects_overlap(ax1, ax2, ay1, ay2, bx1, bx2, by1, by2, margin=0.0) -> bool:
    return (ax1 < bx2 - margin and ax2 > bx1 + margin and
            ay1 < by2 - margin and ay2 > by1 + margin)


def obstruction_overlaps_surface(obs: np.ndarray, i: int) -> bool:
    """True if obstruction i is resting on the target surface (horizontally overlaps)."""
    surf = extract_surface(obs)
    ob   = extract_obstruction(obs, i)
    # X ranges overlap
    x_ok = ob["x1"] < surf["x2"] - SURF_OVERLAP_TOL and ob["x2"] > surf["x1"] + SURF_OVERLAP_TOL
    # Obstruction bottom is near surface top (within vertical tolerance)
    ON_SURF_Y_TOL = 0.08
    y_ok = abs(ob["y1"] - surf["y2"]) < ON_SURF_Y_TOL
    return x_ok and y_ok


def block_is_on_surface(obs: np.ndarray) -> bool:
    """Approximate is_on: block bottom corners inside surface."""
    surf  = extract_surface(obs)
    block = extract_block(obs)
    # Bottom vertices of block (theta=0): (block_x, block_y) and (block_x+w, block_y)
    bx_l = block["x"]
    bx_r = block["x"] + block["width"]
    by   = block["y"] - ON_TOL   # offset down by tol
    return (surf["x1"] <= bx_l <= surf["x2"] and
            surf["x1"] <= bx_r <= surf["x2"] and
            surf["y1"] <= by   <= surf["y2"])


def block_held(obs: np.ndarray) -> bool:
    return obs[IDX_ROBOT_VACUUM] > 0.5
