"""Action generation helpers. All magic numbers live here as named constants."""
import numpy as np

# ─── Action space limits ─────────────────────────────────────────────────────
MAX_DX     = 0.050   # max |dx| per step
MAX_DY     = 0.050   # max |dy| per step
MAX_DTHETA = 0.196   # max |dtheta| per step (rad)
MAX_DARM   = 0.100   # max |darm| per step

# ─── Arm range ───────────────────────────────────────────────────────────────
ARM_MIN  = 0.10   # base_radius (fully retracted)
ARM_MAX  = 0.20   # arm_length (fully extended)
ARM_PICK = 0.175  # arm extension used for picking objects

# ─── Suction offset ──────────────────────────────────────────────────────────
# suction center = base_pos + (arm_joint + SUCTION_OFFSET) in arm direction
# SUCTION_OFFSET = gripper_width (0.01) + suction_width/2 (0.005) = 0.015
SUCTION_OFFSET = 0.015

# ─── Navigation ──────────────────────────────────────────────────────────────
SAFE_NAV_HEIGHT = 0.80  # robot_y when moving horizontally (clears all objects)
NAV_KP     = 1.0        # proportional gain for x/y navigation (KP=1 → no overshoot)
THETA_KP   = 1.0        # proportional gain for theta alignment
ARM_KP     = 1.0        # proportional gain for arm extension

TARGET_THETA = -np.pi / 2  # arm pointing straight down

# ─── Tolerances ──────────────────────────────────────────────────────────────
POS_TOL   = 0.012   # position tolerance (m) for "arrived"
ARM_TOL   = 0.005   # arm extension tolerance
THETA_TOL = 0.04    # angle tolerance (rad)

# ─── Timing ──────────────────────────────────────────────────────────────────
GRASP_WAIT_STEPS  = 8   # steps to hold vacuum before retracting
MAX_PHASE_STEPS   = 200  # safety limit per phase before replanning


# ─── Core action helpers ─────────────────────────────────────────────────────
def clip_action(action):
    a = np.array(action, dtype=np.float32)
    a[0] = np.clip(a[0], -MAX_DX, MAX_DX)
    a[1] = np.clip(a[1], -MAX_DY, MAX_DY)
    a[2] = np.clip(a[2], -MAX_DTHETA, MAX_DTHETA)
    a[3] = np.clip(a[3], -MAX_DARM, MAX_DARM)
    a[4] = np.clip(a[4], 0.0, 1.0)
    return a


def make_action(dx=0.0, dy=0.0, dtheta=0.0, darm=0.0, vac=0.0):
    return clip_action([dx, dy, dtheta, darm, vac])


def delta_toward(cur, tgt, kp, limit):
    return float(np.clip((tgt - cur) * kp, -limit, limit))


def move_toward(cur_x, cur_y, tgt_x, tgt_y):
    dx = delta_toward(cur_x, tgt_x, NAV_KP, MAX_DX)
    dy = delta_toward(cur_y, tgt_y, NAV_KP, MAX_DY)
    return dx, dy


def rotate_toward(cur_theta, tgt_theta=None):
    if tgt_theta is None:
        tgt_theta = TARGET_THETA
    diff = tgt_theta - cur_theta
    # Normalise to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(diff * THETA_KP, -MAX_DTHETA, MAX_DTHETA))


def arm_toward(cur_arm, tgt_arm):
    return float(np.clip((tgt_arm - cur_arm) * ARM_KP, -MAX_DARM, MAX_DARM))


# ─── Geometry helpers ────────────────────────────────────────────────────────
def pick_robot_y(obj_top):
    """robot_y so that suction zone center is exactly at obj_top (ARM_PICK arm)."""
    return obj_top + ARM_PICK + SUCTION_OFFSET


def place_robot_y(surface_top, block_h):
    """robot_y so that block bottom is at surface_top when arm = ARM_MIN (retracted).

    PickAndPlace carries with ARM_MIN and releases at this height.
    block_bottom = robot_y - ARM_MIN - block_h - SUCTION_OFFSET = surface_top
    """
    return surface_top + block_h + ARM_MIN + SUCTION_OFFSET
