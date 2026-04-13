"""Observation parsing and geometric predicates."""

from __future__ import annotations
import math
from dataclasses import dataclass
from numpy.typing import NDArray

ROBOT_FEATURES = ["x","y","theta","base_radius","arm_joint","arm_length","vacuum","gripper_height","gripper_width"]
RECT_FEATURES = ["x","y","theta","static","color_r","color_g","color_b","z_order","width","height"]
CIRCLE_FEATURES = ["x","y","theta","static","color_r","color_g","color_b","z_order","radius"]
SHELF_FEATURES = ["x","y","theta","static","color_r","color_g","color_b","z_order","width","height",
                  "x1","y1","theta1","width1","height1","z_order1","color_r1","color_g1","color_b1"]

# Layout: object name -> (base_index, feature_list)
LAYOUT: dict[str, tuple[int, list[str]]] = {
    "robot":  (0,  ROBOT_FEATURES),
    "shelf":  (9,  SHELF_FEATURES),
    "block0": (28, RECT_FEATURES),
    "block1": (38, RECT_FEATURES),
    "block2": (48, RECT_FEATURES),
}

# World constants
WORLD_MIN_X = 0.0
WORLD_MAX_X = 5.0
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 3.0

# Shelf interior (part rect) - from env observation at seed=0
SHELF_INNER_X_MIN = 0.077
SHELF_INNER_X_MAX = 0.395
SHELF_INNER_Y_MIN = 2.625
SHELF_INNER_Y_MAX = 3.0
SHELF_CENTER_X = (SHELF_INNER_X_MIN + SHELF_INNER_X_MAX) / 2  # ~0.236

# Block constants
BLOCK_WIDTH = 0.28
BLOCK_HEIGHT = 0.04
BLOCK_HALF_W = BLOCK_WIDTH / 2   # 0.14
BLOCK_HALF_H = BLOCK_HEIGHT / 2  # 0.02

# Safe block center x range inside shelf (center ± half_w must be within inner)
SHELF_BLOCK_X_MIN = SHELF_INNER_X_MIN + BLOCK_HALF_W  # 0.217
SHELF_BLOCK_X_MAX = SHELF_INNER_X_MAX - BLOCK_HALF_W  # 0.255
SHELF_BLOCK_Y_MIN = SHELF_INNER_Y_MIN + BLOCK_HALF_H  # 2.645
SHELF_BLOCK_Y_MAX = SHELF_INNER_Y_MAX - BLOCK_HALF_H  # 2.980

# Robot constants
ROBOT_BASE_RADIUS = 0.2
ARM_MIN = 0.2   # = base_radius
ARM_MAX = 0.8

# All block names
BLOCK_NAMES = ["block0", "block1", "block2"]


def _base_and_features(name: str) -> tuple[int, list[str]]:
    return LAYOUT[name]


def get_feature(obs: NDArray, name: str, feature: str) -> float:
    base, features = _base_and_features(name)
    return float(obs[base + features.index(feature)])


@dataclass(frozen=True)
class RobotPose:
    x: float; y: float; theta: float; base_radius: float
    arm_joint: float; arm_length: float; vacuum: float
    gripper_height: float; gripper_width: float


@dataclass(frozen=True)
class RectPose:
    x: float; y: float; theta: float; width: float; height: float


@dataclass(frozen=True)
class CirclePose:
    x: float; y: float; radius: float


def extract_robot(obs: NDArray) -> RobotPose:
    base, _ = _base_and_features("robot")
    return RobotPose(x=float(obs[base+0]), y=float(obs[base+1]), theta=float(obs[base+2]),
                     base_radius=float(obs[base+3]), arm_joint=float(obs[base+4]),
                     arm_length=float(obs[base+5]), vacuum=float(obs[base+6]),
                     gripper_height=float(obs[base+7]), gripper_width=float(obs[base+8]))


def extract_rect(obs: NDArray, name: str) -> RectPose:
    base, features = _base_and_features(name)
    return RectPose(x=float(obs[base+features.index("x")]), y=float(obs[base+features.index("y")]),
                    theta=float(obs[base+features.index("theta")]),
                    width=float(obs[base+features.index("width")]),
                    height=float(obs[base+features.index("height")]))


def extract_circle(obs: NDArray, name: str) -> CirclePose:
    base, features = _base_and_features(name)
    return CirclePose(x=float(obs[base+features.index("x")]), y=float(obs[base+features.index("y")]),
                      radius=float(obs[base+features.index("radius")]))


def get_gripper_pos(robot: RobotPose) -> tuple[float, float]:
    """Return (gx, gy) of gripper center."""
    return (robot.x + robot.arm_joint * math.cos(robot.theta),
            robot.y + robot.arm_joint * math.sin(robot.theta))


def block_center(block: RectPose) -> tuple[float, float]:
    """Return (cx, cy) center of block. Block (x,y) is bottom-left corner."""
    ct, st = math.cos(block.theta), math.sin(block.theta)
    cx = block.x + ct * block.width / 2 - st * block.height / 2
    cy = block.y + st * block.width / 2 + ct * block.height / 2
    return cx, cy


def block_corners(block: RectPose) -> list[tuple[float, float]]:
    """Return 4 corners of a block rectangle. Block (x,y) is bottom-left corner."""
    w, h = block.width, block.height
    ct, st = math.cos(block.theta), math.sin(block.theta)
    x, y = block.x, block.y
    return [
        (x,              y),               # bottom-left
        (x - st*h,       y + ct*h),        # top-left
        (x + ct*w - st*h, y + st*w + ct*h), # top-right
        (x + ct*w,       y + st*w),         # bottom-right
    ]


def is_block_in_shelf(obs: NDArray, block_name: str) -> bool:
    """Return True if all corners of block are inside shelf inner rect."""
    block = extract_rect(obs, block_name)
    corners = block_corners(block)
    return all(SHELF_INNER_X_MIN <= cx <= SHELF_INNER_X_MAX and
               SHELF_INNER_Y_MIN <= cy <= SHELF_INNER_Y_MAX
               for cx, cy in corners)


def gripper_near_block(robot: RobotPose, block: RectPose, tol: float = 0.15) -> bool:
    """Return True if gripper center is within tol of block center."""
    gx, gy = get_gripper_pos(robot)
    cx, cy = block_center(block)
    return math.sqrt((gx - cx)**2 + (gy - cy)**2) < tol


def blocks_outside_shelf(obs: NDArray) -> list[str]:
    """Return names of blocks not yet in the shelf."""
    return [name for name in BLOCK_NAMES if not is_block_in_shelf(obs, name)]
