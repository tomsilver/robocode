"""All observation parsing helpers. All magic numbers live here."""
import numpy as np

# ── Robot feature indices (base 0) ──────────────────────────────────────────
ROB_X       = 0
ROB_Y       = 1
ROB_THETA   = 2
ROB_BASER   = 3
ROB_ARMJ    = 4   # current arm extension (arm_joint)
ROB_ARML    = 5   # max arm extension (arm_length)
ROB_VAC     = 6   # vacuum 0=off 1=on
ROB_GH      = 7   # gripper height (perpendicular to arm) = 0.07
ROB_GW      = 8   # gripper width  (along arm direction)  = 0.01

# ── Stick feature indices (base 9) ──────────────────────────────────────────
# Stick is RectangleType: (x,y) is BOTTOM-LEFT CORNER, theta=0 → vertical
STK_X      = 9
STK_Y      = 10
STK_THETA  = 11
STK_WIDTH  = 17   # 0.05
STK_HEIGHT = 18   # 1.25

# ── Button feature layout (9 features each, starting at 19) ─────────────────
BTN_BASE   = 19
BTN_STRIDE = 9
BTN_X_OFF  = 0
BTN_Y_OFF  = 1
BTN_CR_OFF = 4   # color_r: 0.9=unpressed, 0.0=pressed
BTN_R_OFF  = 8   # radius = 0.05

NUM_BUTTONS = 5

# ── World geometry ────────────────────────────────────────────────────────────
WORLD_MIN_X  = 0.0
WORLD_MAX_X  = 3.5
WORLD_MIN_Y  = 0.0
WORLD_MAX_Y  = 2.5

# TABLE occupies y ∈ [TABLE_Y, 2.5] and blocks robot movement (ZOrder.FLOOR)
TABLE_Y      = 1.25
# Maximum robot centre y (table_y - base_radius, with small safety margin)
FLOOR_MAX_Y  = 1.149

# ── Robot / object constants ─────────────────────────────────────────────────
ROBOT_BASE_R   = 0.1
ARM_MAX        = 0.2
ARM_MIN        = 0.1
BUTTON_RADIUS  = 0.05
GRIPPER_WIDTH  = 0.01   # along arm direction
GRIPPER_HEIGHT = 0.07   # perpendicular to arm

# Suction zone is at arm_joint + GRIPPER_WIDTH + SUCTION_WIDTH/2 from robot centre
SUCTION_WIDTH  = GRIPPER_WIDTH   # = 0.01
SUCTION_HEIGHT = GRIPPER_HEIGHT  # = 0.07
SUCTION_OFFSET = ARM_MIN + GRIPPER_WIDTH + SUCTION_WIDTH / 2  # ≈ 0.115 from robot centre

# Stick corner geometry (theta=0 initially)
STICK_WIDTH  = 0.05
STICK_HEIGHT = 1.25

# Colour threshold for pressed detection
BTN_UNPRESSED_CR = 0.9
COLOUR_TOL       = 0.1

# ── Extraction helpers ────────────────────────────────────────────────────────

def extract_robot(obs):
    return {
        "x":          float(obs[ROB_X]),
        "y":          float(obs[ROB_Y]),
        "theta":      float(obs[ROB_THETA]),
        "base_r":     float(obs[ROB_BASER]),
        "arm_joint":  float(obs[ROB_ARMJ]),
        "arm_length": float(obs[ROB_ARML]),
        "vacuum":     float(obs[ROB_VAC]),
    }


def extract_stick(obs):
    """Return stick corner position (bottom-left) and dims."""
    return {
        "x":      float(obs[STK_X]),     # bottom-left corner x
        "y":      float(obs[STK_Y]),     # bottom-left corner y
        "theta":  float(obs[STK_THETA]),
        "width":  float(obs[STK_WIDTH]),
        "height": float(obs[STK_HEIGHT]),
    }


def stick_center(obs):
    """Return centre of stick rectangle."""
    x = float(obs[STK_X])
    y = float(obs[STK_Y])
    w = float(obs[STK_WIDTH])
    h = float(obs[STK_HEIGHT])
    th = float(obs[STK_THETA])
    # For theta=0 (axis-aligned), center = corner + (w/2, h/2)
    # For rotated: need to rotate the half-extents
    cx = x + np.cos(th) * w / 2 - np.sin(th) * h / 2
    cy = y + np.sin(th) * w / 2 + np.cos(th) * h / 2
    return np.array([cx, cy])


def extract_button(obs, i):
    base = BTN_BASE + i * BTN_STRIDE
    return {
        "x":       float(obs[base + BTN_X_OFF]),
        "y":       float(obs[base + BTN_Y_OFF]),
        "color_r": float(obs[base + BTN_CR_OFF]),
        "radius":  float(obs[base + BTN_R_OFF]),
    }


def is_button_pressed(obs, i):
    b = extract_button(obs, i)
    return b["color_r"] < BTN_UNPRESSED_CR - COLOUR_TOL


def get_unpressed_buttons(obs):
    return [i for i in range(NUM_BUTTONS) if not is_button_pressed(obs, i)]


def is_button_on_table(obs, i):
    """True if button centre is on the table (y ≥ TABLE_Y)."""
    return float(obs[BTN_BASE + i * BTN_STRIDE + BTN_Y_OFF]) >= TABLE_Y


def robot_pos(obs):
    return np.array([obs[ROB_X], obs[ROB_Y]], dtype=float)


def robot_theta(obs):
    return float(obs[ROB_THETA])


def robot_arm_joint(obs):
    return float(obs[ROB_ARMJ])


def stick_corner(obs):
    return np.array([obs[STK_X], obs[STK_Y]], dtype=float)


def button_pos(obs, i):
    base = BTN_BASE + i * BTN_STRIDE
    return np.array([obs[base + BTN_X_OFF], obs[base + BTN_Y_OFF]], dtype=float)


def gripper_pos(obs):
    rx, ry = obs[ROB_X], obs[ROB_Y]
    th = obs[ROB_THETA]
    aj = obs[ROB_ARMJ]
    return np.array([rx + aj * np.cos(th), ry + aj * np.sin(th)], dtype=float)


def all_buttons_pressed(obs):
    return all(is_button_pressed(obs, i) for i in range(NUM_BUTTONS))


def vacuum_on(obs):
    return float(obs[ROB_VAC]) > 0.5
