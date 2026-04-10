"""Observation parsing helpers. All magic numbers are named constants here."""
import numpy as np

# ── Observation indices ──────────────────────────────────────────────────────
# Robot
IDX_ROBOT_X            = 0
IDX_ROBOT_Y            = 1
IDX_ROBOT_THETA        = 2
IDX_ROBOT_BASE_RADIUS  = 3
IDX_ROBOT_ARM_JOINT    = 4
IDX_ROBOT_ARM_LENGTH   = 5
IDX_ROBOT_VACUUM       = 6
IDX_ROBOT_GRIP_HEIGHT  = 7
IDX_ROBOT_GRIP_WIDTH   = 8

# Target surface
IDX_SURF_X       = 9
IDX_SURF_Y       = 10
IDX_SURF_THETA   = 11
IDX_SURF_STATIC  = 12
IDX_SURF_WIDTH   = 17
IDX_SURF_HEIGHT  = 18

# Target block
IDX_BLOCK_X      = 19
IDX_BLOCK_Y      = 20
IDX_BLOCK_THETA  = 21
IDX_BLOCK_STATIC = 22
IDX_BLOCK_WIDTH  = 27
IDX_BLOCK_HEIGHT = 28

# Obstructions: 4 obstructions, each 10 features starting at 29
OBS_STRIDE       = 10
IDX_OBS0_BASE    = 29

def _obs_base(i: int) -> int:
    return IDX_OBS0_BASE + i * OBS_STRIDE

# ── Named tolerances / physics constants ────────────────────────────────────
ARM_MIN_JOINT   = 0.10   # base_radius
ARM_MAX_JOINT   = 0.20   # arm_length
GRASP_TOL        = 0.015  # how close gripper center must be to object center
PLACE_TOL        = 0.020  # tolerance for "block center near surface center"
ON_TOL           = 0.025  # tolerance used by is_on()
SURF_OVERLAP_TOL = 0.005  # margin to declare obstruction overlaps surface
NAV_XY_TOL       = 0.012  # re-export for behaviors
NAV_THETA_TOL    = 0.05   # re-export for behaviors
ARM_EXTEND_TOL   = 0.008  # re-export for behaviors
NUM_OBSTRUCTIONS = 4

# ── Extraction helpers ───────────────────────────────────────────────────────

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
    return {
        "x":      obs[IDX_SURF_X],
        "y":      obs[IDX_SURF_Y],
        "theta":  obs[IDX_SURF_THETA],
        "width":  obs[IDX_SURF_WIDTH],
        "height": obs[IDX_SURF_HEIGHT],
    }


def extract_block(obs: np.ndarray) -> dict:
    return {
        "x":      obs[IDX_BLOCK_X],
        "y":      obs[IDX_BLOCK_Y],
        "theta":  obs[IDX_BLOCK_THETA],
        "width":  obs[IDX_BLOCK_WIDTH],
        "height": obs[IDX_BLOCK_HEIGHT],
    }


def extract_obstruction(obs: np.ndarray, i: int) -> dict:
    b = _obs_base(i)
    return {
        "x":      obs[b + 0],
        "y":      obs[b + 1],
        "theta":  obs[b + 2],
        "static": obs[b + 3],
        "width":  obs[b + 8],   # offset 8: width (after x,y,theta,static,r,g,b,z_order)
        "height": obs[b + 9],   # offset 9: height
    }


def gripper_tip_pos(obs: np.ndarray) -> np.ndarray:
    """Approximate gripper center position."""
    r = extract_robot(obs)
    cx = r["x"] + np.cos(r["theta"]) * r["arm_joint"]
    cy = r["y"] + np.sin(r["theta"]) * r["arm_joint"]
    return np.array([cx, cy])


def rects_overlap_1d(c1, half1, c2, half2, margin=0.0) -> bool:
    return abs(c1 - c2) < (half1 + half2 - margin)


def obstruction_overlaps_surface(obs: np.ndarray, i: int) -> bool:
    """Return True if obstruction i overlaps the target surface."""
    surf = extract_surface(obs)
    ob   = extract_obstruction(obs, i)
    return (rects_overlap_1d(ob["x"], ob["width"]/2,
                              surf["x"], surf["width"]/2, SURF_OVERLAP_TOL) and
            rects_overlap_1d(ob["y"], ob["height"]/2,
                              surf["y"], surf["height"]/2, SURF_OVERLAP_TOL))


def block_is_on_surface(obs: np.ndarray) -> bool:
    """Approximate check: block center inside surface with margin."""
    surf  = extract_surface(obs)
    block = extract_block(obs)
    half_sw = surf["width"]  / 2 - ON_TOL
    half_sh = surf["height"] / 2 - ON_TOL
    dx = abs(block["x"] - surf["x"])
    dy = abs(block["y"] - surf["y"])
    return dx <= half_sw and dy <= half_sh


def block_held(obs: np.ndarray) -> bool:
    """Return True if vacuum is on (block presumably held)."""
    return obs[IDX_ROBOT_VACUUM] > 0.5
