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

# Shelf has primary rect + secondary inner rect (DoubleRectType)
SHELF_FEATURES = [
    "x", "y", "theta", "static", "color_r", "color_g", "color_b",
    "z_order", "width", "height",
    "x1", "y1", "theta1", "width1", "height1",
    "z_order1", "color_r1", "color_g1", "color_b1",
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
# Observation index layout
# robot: 0-8 (9 features)
# shelf: 9-27 (19 features)
# block0: 28-37 (10 features)
# block1: 38-47 (10 features)
# block2: 48-57 (10 features)
# ---------------------------------------------------------------------------

LAYOUT: dict[str, tuple[int, list[str]]] = {
    "robot":  (0,  ROBOT_FEATURES),
    "shelf":  (9,  SHELF_FEATURES),
    "block0": (28, RECT_FEATURES),
    "block1": (38, RECT_FEATURES),
    "block2": (48, RECT_FEATURES),
}

NUM_BLOCKS = 3
BLOCK_NAMES = ["block0", "block1", "block2"]

# ---------------------------------------------------------------------------
# World constants (from environment config / observation inspection)
# ---------------------------------------------------------------------------
WORLD_X_MIN = 0.0
WORLD_X_MAX = 5.0
WORLD_Y_MIN = 0.0
WORLD_Y_MAX = 3.0
SHELF_FLOOR_Y = 2.625   # y-coordinate of shelf bottom face (open face)

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


@dataclass(frozen=True)
class ShelfInner:
    """Inner compartment of the shelf (where blocks must go)."""
    x: float   # left edge
    y: float   # bottom edge
    width: float
    height: float

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2

    @property
    def x_max(self) -> float:
        return self.x + self.width

    @property
    def y_max(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class BlockPose:
    x: float
    y: float
    theta: float
    width: float
    height: float


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


def extract_shelf_inner(obs: NDArray) -> ShelfInner:
    """Extract the inner compartment of the shelf."""
    base = LAYOUT["shelf"][0]
    feats = SHELF_FEATURES
    return ShelfInner(
        x=float(obs[base + feats.index("x1")]),
        y=float(obs[base + feats.index("y1")]),
        width=float(obs[base + feats.index("width1")]),
        height=float(obs[base + feats.index("height1")]),
    )


def extract_block(obs: NDArray, block_idx: int) -> BlockPose:
    """Extract block pose by index (0, 1, 2)."""
    name = BLOCK_NAMES[block_idx]
    base, features = _base_and_features(name)
    return BlockPose(
        x=float(obs[base + features.index("x")]),
        y=float(obs[base + features.index("y")]),
        theta=float(obs[base + features.index("theta")]),
        width=float(obs[base + features.index("width")]),
        height=float(obs[base + features.index("height")]),
    )


# ---------------------------------------------------------------------------
# Geometric predicates
# ---------------------------------------------------------------------------

BLOCK_IN_SHELF_MARGIN = 0.01  # small tolerance for containment check


def block_vertices(block: BlockPose):
    """Return the 4 corners of the block in world frame.

    NOTE: block.x/y is the world-frame position of the LOCAL (0,0) corner,
    i.e. the block's bottom-left corner before rotation is applied at origin.
    The 4 local corners are (0,0), (w,0), (w,h), (0,h).
    """
    x, y, th = block.x, block.y, block.theta
    w, h = block.width, block.height
    cos_t, sin_t = math.cos(th), math.sin(th)
    local_corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    corners = []
    for lx, ly in local_corners:
        wx = x + cos_t * lx - sin_t * ly
        wy = y + sin_t * lx + cos_t * ly
        corners.append((wx, wy))
    return corners


def block_center(block: BlockPose) -> tuple[float, float]:
    """Return the world-frame center of the block."""
    x, y, th = block.x, block.y, block.theta
    w, h = block.width, block.height
    cos_t, sin_t = math.cos(th), math.sin(th)
    cx = x + cos_t * w / 2 - sin_t * h / 2
    cy = y + sin_t * w / 2 + cos_t * h / 2
    return cx, cy


def is_block_in_shelf(obs: NDArray, block_idx: int) -> bool:
    """Return True if all vertices of block_idx are inside the shelf inner rect."""
    shelf = extract_shelf_inner(obs)
    block = extract_block(obs, block_idx)
    for vx, vy in block_vertices(block):
        if vx < shelf.x - BLOCK_IN_SHELF_MARGIN:
            return False
        if vx > shelf.x_max + BLOCK_IN_SHELF_MARGIN:
            return False
        if vy < shelf.y - BLOCK_IN_SHELF_MARGIN:
            return False
        if vy > shelf.y_max + BLOCK_IN_SHELF_MARGIN:
            return False
    return True


def get_outside_block_indices(obs: NDArray) -> list[int]:
    """Return indices of blocks that are NOT inside the shelf."""
    return [i for i in range(NUM_BLOCKS) if not is_block_in_shelf(obs, i)]


def gripper_tip_position(robot: RobotPose) -> tuple[float, float]:
    """World-frame position of the end of the arm (past the gripper)."""
    tip_dist = robot.arm_joint + robot.gripper_width
    gx = robot.x + math.cos(robot.theta) * tip_dist
    gy = robot.y + math.sin(robot.theta) * tip_dist
    return gx, gy


def is_holding_block(obs: NDArray, block_idx: int) -> bool:
    """Heuristic: vacuum on and block CENTER is near gripper tip."""
    robot = extract_robot(obs)
    if robot.vacuum < 0.5:
        return False
    block = extract_block(obs, block_idx)
    bcx, bcy = block_center(block)
    gx, gy = gripper_tip_position(robot)
    dist = math.sqrt((gx - bcx) ** 2 + (gy - bcy) ** 2)
    return dist < HOLDING_DIST_THRESHOLD


HOLDING_DIST_THRESHOLD = 0.20  # tip-to-block-center: max when picking a face = w/2=0.14 or h/2=0.02
