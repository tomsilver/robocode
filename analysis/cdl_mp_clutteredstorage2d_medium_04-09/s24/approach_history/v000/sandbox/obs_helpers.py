"""Observation parsing and geometric predicates.

Provides named access to object features from the flat observation vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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

# Shelf uses DoubleRectType — two rectangles in one observation
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

# ---------------------------------------------------------------------------
# Layout: object name -> (base_index, feature_list)
# ---------------------------------------------------------------------------

ROBOT_BASE_IDX = 0
SHELF_BASE_IDX = 9
BLOCK0_BASE_IDX = 28
BLOCK1_BASE_IDX = 38
BLOCK2_BASE_IDX = 48

NUM_BLOCKS = 3
BLOCK_NAMES = ["block0", "block1", "block2"]
BLOCK_BASE_INDICES = [BLOCK0_BASE_IDX, BLOCK1_BASE_IDX, BLOCK2_BASE_IDX]

LAYOUT: dict[str, tuple[int, list[str]]] = {
    "robot": (ROBOT_BASE_IDX, ROBOT_FEATURES),
    "shelf": (SHELF_BASE_IDX, SHELF_FEATURES),
    "block0": (BLOCK0_BASE_IDX, RECT_FEATURES),
    "block1": (BLOCK1_BASE_IDX, RECT_FEATURES),
    "block2": (BLOCK2_BASE_IDX, RECT_FEATURES),
}

# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

WORLD_MIN_X = 0.0
WORLD_MAX_X = 5.0
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 3.0

ROBOT_BASE_RADIUS = 0.2
ARM_MIN = 0.2        # minimum arm_joint (cannot retract below this)
ARM_MAX = 0.8        # maximum arm_joint
GRIPPER_HEIGHT = 0.14   # perpendicular to arm
GRIPPER_WIDTH = 0.02    # along arm
SUCTION_OFFSET = GRIPPER_WIDTH + GRIPPER_WIDTH / 2  # arm_joint + this = suction center dist
# suction center = arm_joint + GRIPPER_WIDTH + SUCTION_WIDTH/2 = arm_joint + 0.02 + 0.01 = arm_joint + 0.03

BLOCK_WIDTH = 0.28
BLOCK_HEIGHT = 0.04

# Shelf y boundary: robot base center must stay below this
SHELF_FLOOR_Y = 2.625
ROBOT_MAX_Y_NAV = SHELF_FLOOR_Y - ROBOT_BASE_RADIUS  # 2.425

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
    """Axis-aligned rectangle pose extracted from the observation vector."""

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
    """Extract rectangle pose for a named object."""
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


def get_block_center(obs: NDArray, name: str) -> tuple[float, float]:
    """Compute the geometric center of a block from its first vertex + rotation."""
    p = extract_rect(obs, name)
    cx = p.x + (p.width / 2) * math.cos(p.theta) - (p.height / 2) * math.sin(p.theta)
    cy = p.y + (p.width / 2) * math.sin(p.theta) + (p.height / 2) * math.cos(p.theta)
    return cx, cy


def get_shelf_slot(obs: NDArray) -> tuple[float, float, float, float]:
    """Return (x, y, width, height) of the shelf storage slot (lower-left corner)."""
    base = SHELF_BASE_IDX
    feats = SHELF_FEATURES
    x1 = float(obs[base + feats.index("x1")])
    y1 = float(obs[base + feats.index("y1")])
    w1 = float(obs[base + feats.index("width1")])
    h1 = float(obs[base + feats.index("height1")])
    return x1, y1, w1, h1


def is_block_in_shelf(obs: NDArray, name: str) -> bool:
    """Check if block center is inside shelf slot (approximate)."""
    x1, y1, w1, h1 = get_shelf_slot(obs)
    cx, cy = get_block_center(obs, name)
    margin = BLOCK_HEIGHT  # small inset margin
    return (x1 + margin < cx < x1 + w1 - margin) and (y1 < cy < y1 + h1)


def get_blocks_outside_shelf(obs: NDArray) -> list[str]:
    """Return names of blocks not currently inside the shelf."""
    return [name for name in BLOCK_NAMES if not is_block_in_shelf(obs, name)]


def gripper_tip_pos(robot: RobotPose) -> tuple[float, float]:
    """Position of the gripper CENTER (= arm_joint distance from robot base in theta dir)."""
    x = robot.x + math.cos(robot.theta) * robot.arm_joint
    y = robot.y + math.sin(robot.theta) * robot.arm_joint
    return x, y


def suction_center_pos(robot: RobotPose) -> tuple[float, float]:
    """Position of the suction zone center (just beyond gripper tip)."""
    dist = robot.arm_joint + GRIPPER_WIDTH + GRIPPER_WIDTH / 2
    x = robot.x + math.cos(robot.theta) * dist
    y = robot.y + math.sin(robot.theta) * dist
    return x, y


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
