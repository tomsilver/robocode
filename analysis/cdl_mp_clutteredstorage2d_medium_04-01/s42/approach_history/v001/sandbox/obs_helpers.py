"""Observation parsing and geometric predicates.

Provides named access to object features from the flat observation vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Feature name lists (must match kinder object_types.py)
# ---------------------------------------------------------------------------

ROBOT_FEATURES = [
    "x",
    "y",
    "theta",
    "base_radius",
    "arm_joint",
    "arm_length",
    "vacuum",
    "gripper_height",
    "gripper_width",
]

RECT_FEATURES = [
    "x",
    "y",
    "theta",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
    "width",
    "height",
]

SHELF_FEATURES = [
    "x",
    "y",
    "theta",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
    "width",
    "height",
    "x1",
    "y1",
    "theta1",
    "width1",
    "height1",
    "z_order1",
    "color_r1",
    "color_g1",
    "color_b1",
]

CIRCLE_FEATURES = [
    "x",
    "y",
    "theta",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
    "radius",
]

# ---------------------------------------------------------------------------
# Layout: object name -> (base_index, feature_list)
# ---------------------------------------------------------------------------

LAYOUT: dict[str, tuple[int, list[str]]] = {
    "robot":  (0,  ROBOT_FEATURES),   # indices 0–8
    "shelf":  (9,  SHELF_FEATURES),   # indices 9–27
    "block0": (28, RECT_FEATURES),    # indices 28–37
    "block1": (38, RECT_FEATURES),    # indices 38–47
    "block2": (48, RECT_FEATURES),    # indices 48–57
}

# Observation index shortcuts
SHELF_OUTER_X_IDX = 9
SHELF_OUTER_Y_IDX = 10    # bottom-left corner y of outer shelf
SHELF_OUTER_W_IDX = 17
SHELF_OUTER_H_IDX = 18
SHELF_INNER_X_IDX = 19   # bottom-left corner x of inner shelf part
SHELF_INNER_Y_IDX = 20   # bottom-left corner y of inner shelf part
SHELF_INNER_W_IDX = 22
SHELF_INNER_H_IDX = 23

# Block names and world constants
BLOCK_NAMES = ["block0", "block1", "block2"]
NUM_BLOCKS = 3

WORLD_MIN_X = 0.0
WORLD_MAX_X = 5.0
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 3.0

ROBOT_RADIUS = 0.2    # base_radius
ARM_MIN_JOINT = 0.2   # minimum arm_joint == base_radius
ARM_MAX_JOINT = 0.8   # maximum arm_joint == arm_length

BLOCK_HALF_WIDTH = 0.14   # half of block width (0.28)
BLOCK_HALF_HEIGHT = 0.02  # half of block height (0.04)
# Maximum distance from block center to any vertex (diagonal)
BLOCK_HALF_DIAG = math.sqrt(BLOCK_HALF_WIDTH**2 + BLOCK_HALF_HEIGHT**2)

# ---------------------------------------------------------------------------
# Generic feature access
# ---------------------------------------------------------------------------


def _base_and_features(name: str) -> tuple[int, list[str]]:
    return LAYOUT[name]


def get_feature(obs: NDArray, name: str, feature: str) -> float:
    """Get a single feature value for an object by name."""
    base, features = _base_and_features(name)
    return float(obs[base + features.index(feature)])


# ---------------------------------------------------------------------------
# Structured extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotPose:
    """Robot configuration extracted from the observation vector."""

    x: float
    y: float
    theta: float
    base_radius: float
    arm_joint: float
    arm_length: float
    vacuum: float
    gripper_height: float
    gripper_width: float


@dataclass(frozen=True)
class RectPose:
    """Rectangle pose extracted from the observation vector.

    NOTE: (x, y) is the bottom-left CORNER of the rectangle (not the centre).
    The centre is computed by block_center().
    """

    x: float
    y: float
    theta: float
    width: float
    height: float


@dataclass(frozen=True)
class CirclePose:
    """Circle pose (centre + radius) extracted from the observation vector."""

    x: float
    y: float
    radius: float


def extract_robot(obs: NDArray) -> RobotPose:
    """Extract robot pose from the observation."""
    base, _ = _base_and_features("robot")
    return RobotPose(
        x=float(obs[base + 0]),
        y=float(obs[base + 1]),
        theta=float(obs[base + 2]),
        base_radius=float(obs[base + 3]),
        arm_joint=float(obs[base + 4]),
        arm_length=float(obs[base + 5]),
        vacuum=float(obs[base + 6]),
        gripper_height=float(obs[base + 7]),
        gripper_width=float(obs[base + 8]),
    )


def extract_rect(obs: NDArray, name: str) -> RectPose:
    """Extract rectangle pose for a named object.

    (x, y) is the bottom-left corner.  Use block_center() for centre.
    """
    base, features = _base_and_features(name)
    return RectPose(
        x=float(obs[base + features.index("x")]),
        y=float(obs[base + features.index("y")]),
        theta=float(obs[base + features.index("theta")]),
        width=float(obs[base + features.index("width")]),
        height=float(obs[base + features.index("height")]),
    )


def extract_circle(obs: NDArray, name: str) -> CirclePose:
    """Extract circle pose for a named button."""
    base, features = _base_and_features(name)
    return CirclePose(
        x=float(obs[base + features.index("x")]),
        y=float(obs[base + features.index("y")]),
        radius=float(obs[base + features.index("radius")]),
    )


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------


def block_center(rect: RectPose) -> tuple[float, float]:
    """Return the (cx, cy) center of a block given its RectPose corner + dims.

    Rectangle convention: (x, y) is the bottom-left corner; the first edge
    runs along theta and has length `width`; the second edge is perpendicular
    with length `height`.
    """
    cx = rect.x + (rect.width / 2) * math.cos(rect.theta) - (rect.height / 2) * math.sin(rect.theta)
    cy = rect.y + (rect.width / 2) * math.sin(rect.theta) + (rect.height / 2) * math.cos(rect.theta)
    return cx, cy


def block_center_from_obs(obs: NDArray, block_name: str) -> tuple[float, float]:
    """Get block center (cx, cy) directly from the observation vector."""
    return block_center(extract_rect(obs, block_name))


def shelf_inner_bounds(obs: NDArray) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) of the shelf inner storage area."""
    sx1 = float(obs[SHELF_INNER_X_IDX])
    sy1 = float(obs[SHELF_INNER_Y_IDX])
    sw1 = float(obs[SHELF_INNER_W_IDX])
    sh1 = float(obs[SHELF_INNER_H_IDX])
    return sx1, sx1 + sw1, sy1, sy1 + sh1


def shelf_inner_center(obs: NDArray) -> tuple[float, float]:
    """Return (cx, cy) of the shelf inner storage area centre."""
    x_min, x_max, y_min, y_max = shelf_inner_bounds(obs)
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def shelf_y_bottom(obs: NDArray) -> float:
    """Return the y coordinate of the shelf bottom edge (robot collision boundary)."""
    return float(obs[SHELF_OUTER_Y_IDX])


def gripper_pos(robot: RobotPose) -> tuple[float, float]:
    """Return (gx, gy) — the gripper centre position."""
    gx = robot.x + robot.arm_joint * math.cos(robot.theta)
    gy = robot.y + robot.arm_joint * math.sin(robot.theta)
    return gx, gy


def is_block_in_shelf(obs: NDArray, block_name: str) -> bool:
    """Return True if ALL vertices of the block are inside the shelf inner region."""
    x_min, x_max, y_min, y_max = shelf_inner_bounds(obs)
    rect = extract_rect(obs, block_name)
    # Build the 4 corners using the standard Rectangle convention
    cx, cy = block_center(rect)
    hw = rect.width / 2.0
    hh = rect.height / 2.0
    cos_t = math.cos(rect.theta)
    sin_t = math.sin(rect.theta)
    # Corner offsets in world frame: ±hw along theta, ±hh perpendicular
    corners = [
        (cx + hw * cos_t - hh * sin_t,  cy + hw * sin_t + hh * cos_t),
        (cx - hw * cos_t - hh * sin_t,  cy - hw * sin_t + hh * cos_t),
        (cx + hw * cos_t + hh * sin_t,  cy + hw * sin_t - hh * cos_t),
        (cx - hw * cos_t + hh * sin_t,  cy - hw * sin_t - hh * cos_t),
    ]
    for vx, vy in corners:
        if vx < x_min or vx > x_max or vy < y_min or vy > y_max:
            return False
    return True


def get_outside_blocks(obs: NDArray) -> list[str]:
    """Return block names that are NOT inside the shelf."""
    return [name for name in BLOCK_NAMES if not is_block_in_shelf(obs, name)]
