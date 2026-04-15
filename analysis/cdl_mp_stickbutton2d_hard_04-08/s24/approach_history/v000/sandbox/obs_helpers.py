"""All observation parsing helpers. All magic numbers live here."""
import numpy as np

# ── Robot feature indices (base 0) ──────────────────────────────────────────
ROB_X       = 0
ROB_Y       = 1
ROB_THETA   = 2
ROB_BASER   = 3   # base radius (always 0.1)
ROB_ARMJ    = 4   # arm_joint  – current arm extension from center
ROB_ARML    = 5   # arm_length – max extension (always 0.2)
ROB_VAC     = 6   # vacuum (0=off, 1=on)
ROB_GH      = 7   # gripper height (0.07, perpendicular to arm)
ROB_GW      = 8   # gripper width  (0.01, along arm direction)

# ── Stick feature indices (base 9) ──────────────────────────────────────────
STK_X      = 9
STK_Y      = 10
STK_THETA  = 11
STK_WIDTH  = 17
STK_HEIGHT = 18

# ── Button feature layout (9 features each, starting at 19) ─────────────────
BTN_BASE   = 19   # index of button0 x
BTN_STRIDE = 9    # features per button
BTN_X_OFF  = 0
BTN_Y_OFF  = 1
BTN_CR_OFF = 4   # color_r  (0.9 = unpressed, 0.0 = pressed)
BTN_CG_OFF = 5   # color_g  (0.0 = unpressed, 0.9 = pressed)
BTN_R_OFF  = 8   # radius

NUM_BUTTONS = 5

# ── Physical constants ────────────────────────────────────────────────────────
WORLD_MIN_X   = 0.0
WORLD_MAX_X   = 3.5
WORLD_MIN_Y   = 0.0
WORLD_MAX_Y   = 2.5

ROBOT_BASE_R  = 0.1
ARM_MAX       = 0.2
ARM_MIN       = 0.1   # arm_joint lower bound = base_radius
BUTTON_RADIUS = 0.05
GRIPPER_WIDTH = 0.01   # along arm direction
GRIPPER_HEIGHT = 0.07  # perpendicular to arm direction

# Z-order insight: buttons have ZOrder.NONE → they don't block robot movement.
# Only the stick (ZOrder.SURFACE) and world bounds block movement.

# Pressed colour threshold
BTN_UNPRESSED_CR = 0.9
COLOUR_TOL       = 0.1


# ── Extraction helpers ────────────────────────────────────────────────────────
def extract_robot(obs):
    """Return dict with named robot features."""
    return {
        "x":         float(obs[ROB_X]),
        "y":         float(obs[ROB_Y]),
        "theta":     float(obs[ROB_THETA]),
        "base_r":    float(obs[ROB_BASER]),
        "arm_joint": float(obs[ROB_ARMJ]),
        "arm_length":float(obs[ROB_ARML]),
        "vacuum":    float(obs[ROB_VAC]),
    }


def extract_stick(obs):
    return {
        "x":     float(obs[STK_X]),
        "y":     float(obs[STK_Y]),
        "theta": float(obs[STK_THETA]),
        "width": float(obs[STK_WIDTH]),
        "height":float(obs[STK_HEIGHT]),
    }


def extract_button(obs, i):
    base = BTN_BASE + i * BTN_STRIDE
    return {
        "x":       float(obs[base + BTN_X_OFF]),
        "y":       float(obs[base + BTN_Y_OFF]),
        "color_r": float(obs[base + BTN_CR_OFF]),
        "color_g": float(obs[base + BTN_CG_OFF]),
        "radius":  float(obs[base + BTN_R_OFF]),
    }


def is_button_pressed(obs, i):
    b = extract_button(obs, i)
    return b["color_r"] < BTN_UNPRESSED_CR - COLOUR_TOL


def get_unpressed_buttons(obs):
    """Return list of indices of unpressed buttons."""
    return [i for i in range(NUM_BUTTONS) if not is_button_pressed(obs, i)]


def robot_pos(obs):
    return np.array([obs[ROB_X], obs[ROB_Y]], dtype=float)


def robot_theta(obs):
    return float(obs[ROB_THETA])


def robot_arm_joint(obs):
    return float(obs[ROB_ARMJ])


def stick_pos(obs):
    return np.array([obs[STK_X], obs[STK_Y]], dtype=float)


def stick_dims(obs):
    """Return (width, height, theta) of stick."""
    return float(obs[STK_WIDTH]), float(obs[STK_HEIGHT]), float(obs[STK_THETA])


def button_pos(obs, i):
    base = BTN_BASE + i * BTN_STRIDE
    return np.array([obs[base + BTN_X_OFF], obs[base + BTN_Y_OFF]], dtype=float)


def gripper_pos(obs):
    """World position of the gripper tip."""
    rx, ry = obs[ROB_X], obs[ROB_Y]
    th = obs[ROB_THETA]
    aj = obs[ROB_ARMJ]
    return np.array([rx + aj * np.cos(th), ry + aj * np.sin(th)], dtype=float)


def all_buttons_pressed(obs):
    return all(is_button_pressed(obs, i) for i in range(NUM_BUTTONS))
