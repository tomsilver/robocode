"""Observation parsing and geometry helpers for ClutteredStorage2D-b3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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

LAYOUT: dict[str, tuple[int, list[str]]] = {
    "robot": (0, ROBOT_FEATURES),
    "shelf": (9, SHELF_FEATURES),
    "block0": (28, RECT_FEATURES),
    "block1": (38, RECT_FEATURES),
    "block2": (48, RECT_FEATURES),
}

BLOCK_NAMES = ["block0", "block1", "block2"]
NUM_STORAGE_SLOTS = 3
VACUUM_ON_THRESHOLD = 0.5
TOOLTIP_OFFSET_SCALE = 0.5
HELD_BLOCK_Y_TOL = 0.02
HELD_BLOCK_DISTANCE_MAX = 0.5
SLOT_OCCUPANCY_TOL = 0.06
APPROACH_MARGIN = 0.01
WORLD_MIN_X = 0.0
WORLD_MAX_X = 5.0
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 3.0
POSE_EPS = 1e-4
STAGING_OCCUPANCY_TOL = 0.08
STAGING_AXIS_SEPARATION_TOL = 0.18
STAGING_RELEASE_SWEEP_X_TOL = 0.32
STAGING_RELEASE_SWEEP_Y_TOL = 0.32


def wrap_angle(theta: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class Pose2D:
    """An SE(2) pose."""

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class RobotPose:
    """Robot configuration extracted from the observation."""

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
    """Rectangle pose using the env's lower-left convention."""

    x: float
    y: float
    theta: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        """Return the rectangle center."""
        offset = rotate_vector(self.width / 2, self.height / 2, self.theta)
        return (self.x + offset[0], self.y + offset[1])


@dataclass(frozen=True)
class ShelfPose:
    """Shelf pose with the open storage region separated out."""

    x: float
    y: float
    theta: float
    width: float
    height: float
    x1: float
    y1: float
    theta1: float
    width1: float
    height1: float

    @property
    def opening_center_x(self) -> float:
        """Center x of the storage opening."""
        return self.x1 + self.width1 / 2


def _base_and_features(name: str) -> tuple[int, list[str]]:
    return LAYOUT[name]


def get_feature(obs: NDArray, name: str, feature: str) -> float:
    """Get a feature value by object name and feature name."""
    base, features = _base_and_features(name)
    return float(obs[base + features.index(feature)])


def rotate_vector(x: float, y: float, theta: float) -> tuple[float, float]:
    """Rotate a vector by *theta*."""
    return (
        x * float(np.cos(theta)) - y * float(np.sin(theta)),
        x * float(np.sin(theta)) + y * float(np.cos(theta)),
    )


def compose_pose(a: Pose2D, b: Pose2D) -> Pose2D:
    """Compose two SE(2) poses."""
    dx, dy = rotate_vector(b.x, b.y, a.theta)
    return Pose2D(a.x + dx, a.y + dy, wrap_angle(a.theta + b.theta))


def invert_pose(pose: Pose2D) -> Pose2D:
    """Invert an SE(2) pose."""
    c = float(np.cos(pose.theta))
    s = float(np.sin(pose.theta))
    x = -(c * pose.x + s * pose.y)
    y = -(-s * pose.x + c * pose.y)
    return Pose2D(x, y, wrap_angle(-pose.theta))


def extract_robot(obs: NDArray) -> RobotPose:
    """Extract the robot pose."""
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


def extract_block(obs: NDArray, name: str) -> RectPose:
    """Extract a block pose."""
    base, features = _base_and_features(name)
    return RectPose(
        x=float(obs[base + features.index("x")]),
        y=float(obs[base + features.index("y")]),
        theta=float(obs[base + features.index("theta")]),
        width=float(obs[base + features.index("width")]),
        height=float(obs[base + features.index("height")]),
    )


def extract_shelf(obs: NDArray) -> ShelfPose:
    """Extract the shelf pose."""
    base, features = _base_and_features("shelf")
    return ShelfPose(
        x=float(obs[base + features.index("x")]),
        y=float(obs[base + features.index("y")]),
        theta=float(obs[base + features.index("theta")]),
        width=float(obs[base + features.index("width")]),
        height=float(obs[base + features.index("height")]),
        x1=float(obs[base + features.index("x1")]),
        y1=float(obs[base + features.index("y1")]),
        theta1=float(obs[base + features.index("theta1")]),
        width1=float(obs[base + features.index("width1")]),
        height1=float(obs[base + features.index("height1")]),
    )


def rect_vertices(rect: RectPose) -> list[tuple[float, float]]:
    """Return the four rectangle vertices."""
    offsets = [
        (0.0, 0.0),
        (rect.width, 0.0),
        (rect.width, rect.height),
        (0.0, rect.height),
    ]
    return [
        (
            rect.x + rotate_vector(dx, dy, rect.theta)[0],
            rect.y + rotate_vector(dx, dy, rect.theta)[1],
        )
        for dx, dy in offsets
    ]


def is_block_inside_shelf(obs: NDArray, name: str) -> bool:
    """Return True when the block is entirely inside the shelf opening."""
    shelf = extract_shelf(obs)
    block = extract_block(obs, name)
    return all(
        shelf.x1 - POSE_EPS <= x <= shelf.x1 + shelf.width1 + POSE_EPS
        and shelf.y1 - POSE_EPS <= y <= shelf.y1 + shelf.height1 + POSE_EPS
        for x, y in rect_vertices(block)
    )


def all_blocks_inside_shelf(obs: NDArray) -> bool:
    """Return True when all blocks are inside the shelf."""
    return all(is_block_inside_shelf(obs, name) for name in BLOCK_NAMES)


def outside_blocks(obs: NDArray) -> list[str]:
    """Return the block names that are still outside the shelf."""
    return [name for name in BLOCK_NAMES if not is_block_inside_shelf(obs, name)]


def inside_blocks(obs: NDArray) -> list[str]:
    """Return the block names that are currently inside the shelf."""
    return [name for name in BLOCK_NAMES if is_block_inside_shelf(obs, name)]


def tool_tip_pose(robot: RobotPose) -> Pose2D:
    """Return the robot tool-tip pose."""
    offset = robot.arm_joint + TOOLTIP_OFFSET_SCALE * robot.gripper_width
    dx = offset * float(np.cos(robot.theta))
    dy = offset * float(np.sin(robot.theta))
    return Pose2D(robot.x + dx, robot.y + dy, robot.theta)


def holding_any_block(obs: NDArray) -> bool:
    """Return True when the robot is vacuuming and one block is lifted."""
    return held_block_name(obs) is not None


def held_block_name(obs: NDArray) -> str | None:
    """Return the most likely currently held block."""
    robot = extract_robot(obs)
    if robot.vacuum <= VACUUM_ON_THRESHOLD:
        return None
    tip = tool_tip_pose(robot)
    threshold = robot.base_radius + HELD_BLOCK_Y_TOL
    candidates = []
    for name in BLOCK_NAMES:
        block = extract_block(obs, name)
        center_x, center_y = block.center
        distance = float(np.hypot(center_x - tip.x, center_y - tip.y))
        if center_y > threshold:
            candidates.append((distance, name))
    if not candidates:
        return None
    distance, name = min(candidates)
    if distance > HELD_BLOCK_DISTANCE_MAX:
        return None
    return name


def choose_next_block(obs: NDArray) -> str | None:
    """Choose the next outside block to store."""
    candidates = outside_blocks(obs)
    if not candidates:
        return None
    robot = extract_robot(obs)
    return min(
        candidates,
        key=lambda name: (
            extract_block(obs, name).center[1],
            abs(extract_block(obs, name).center[0] - robot.x),
        ),
    )


def slot_centers(obs: NDArray) -> list[tuple[float, float]]:
    """Return deep shelf slot centers ordered from deepest to shallowest."""
    shelf = extract_shelf(obs)
    block = extract_block(obs, BLOCK_NAMES[0])
    x = shelf.opening_center_x
    half_height = 0.5 * block.height
    edge_margin = max(0.01, 0.25 * block.height)
    low = shelf.y1 + half_height + edge_margin
    high = shelf.y1 + shelf.height1 - half_height - edge_margin
    if NUM_STORAGE_SLOTS == 1:
        return [(x, 0.5 * (low + high))]
    spacing = (high - low) / (NUM_STORAGE_SLOTS - 1)
    return [(x, high - idx * spacing) for idx in range(NUM_STORAGE_SLOTS)]


def next_free_slot_center(obs: NDArray) -> tuple[float, float]:
    """Return the next free shelf slot."""
    occupied = []
    for name in BLOCK_NAMES:
        if is_block_inside_shelf(obs, name):
            occupied.append(extract_block(obs, name).center)
    for center in slot_centers(obs):
        if all(abs(center[1] - occ[1]) > SLOT_OCCUPANCY_TOL for occ in occupied):
            return center
    return slot_centers(obs)[-1]


def staging_slot_centers() -> list[tuple[float, float]]:
    """Return temporary staging slot centers outside the shelf."""
    return [(1.20, 1.20), (2.30, 1.05), (3.45, 1.25)]


def next_free_staging_center(obs: NDArray) -> tuple[float, float]:
    """Return the next free staging slot."""
    occupied = [extract_block(obs, name).center for name in outside_blocks(obs)]
    scored = sorted(
        staging_slot_centers(),
        key=lambda center: _staging_center_penalty(center, occupied),
    )
    return scored[0]


def farthest_free_staging_center(obs: NDArray) -> tuple[float, float]:
    """Return the free staging slot farthest from the shelf opening."""
    occupied = [extract_block(obs, name).center for name in outside_blocks(obs)]
    shelf = extract_shelf(obs)
    return min(
        staging_slot_centers(),
        key=lambda center: (
            _staging_center_penalty(center, occupied),
            -float(np.hypot(center[0] - shelf.opening_center_x, center[1] - shelf.y1)),
        ),
    )


def _staging_center_clear_of_occ(
    center: tuple[float, float],
    occ: tuple[float, float],
) -> bool:
    """Return True if a staging center stays clear of an occupied staging pose."""
    dx = abs(center[0] - occ[0])
    dy = abs(center[1] - occ[1])
    return (
        np.hypot(dx, dy) > STAGING_OCCUPANCY_TOL
        and dx > STAGING_AXIS_SEPARATION_TOL
        and dy > STAGING_AXIS_SEPARATION_TOL
    )


def _staging_center_penalty(
    center: tuple[float, float],
    occupied: list[tuple[float, float]],
) -> float:
    """Return a penalty for choosing a staging center near occupied blockers."""
    penalty = 0.0
    for occ in occupied:
        dx = abs(center[0] - occ[0])
        dy = abs(center[1] - occ[1])
        distance = float(np.hypot(dx, dy))
        if distance < STAGING_OCCUPANCY_TOL:
            penalty += 1000.0
        else:
            penalty += 1.0 / max(distance, 1e-3)
        if dx < STAGING_AXIS_SEPARATION_TOL:
            penalty += 200.0
        if dy < STAGING_AXIS_SEPARATION_TOL:
            penalty += 120.0
        # Avoid choosing a slot whose release sweep would run into an occupied blocker.
        if dx < STAGING_RELEASE_SWEEP_X_TOL and dy < STAGING_RELEASE_SWEEP_Y_TOL:
            penalty += 250.0
    return penalty


def rect_pose_from_center(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    theta: float,
) -> RectPose:
    """Construct a rectangle pose from center coordinates."""
    dx, dy = rotate_vector(width / 2, height / 2, theta)
    return RectPose(center_x - dx, center_y - dy, theta, width, height)


def in_world_bounds(robot: RobotPose, x: float, y: float, margin: float) -> bool:
    """Check conservative bounds for the robot base."""
    return (
        WORLD_MIN_X + robot.base_radius + margin
        <= x
        <= WORLD_MAX_X - robot.base_radius - margin
        and WORLD_MIN_Y + robot.base_radius + margin
        <= y
        <= WORLD_MAX_Y - robot.base_radius - margin
    )


def pick_base_pose_candidates(obs: NDArray, block_name: str) -> list[Pose2D]:
    """Return feasible pick poses on the two broad-face normals of a block."""
    robot = extract_robot(obs)
    block = extract_block(obs, block_name)
    center_x, center_y = block.center
    face_offset = block.height / 2 + APPROACH_MARGIN
    radial_offset = (
        robot.arm_length + TOOLTIP_OFFSET_SCALE * robot.gripper_width + face_offset
    )
    candidates: list[Pose2D] = []
    for sign in (1.0, -1.0):
        theta = wrap_angle(block.theta + sign * (np.pi / 2))
        dx = radial_offset * float(np.cos(theta))
        dy = radial_offset * float(np.sin(theta))
        pose = Pose2D(center_x - dx, center_y - dy, theta)
        if in_world_bounds(robot, pose.x, pose.y, APPROACH_MARGIN):
            candidates.append(pose)
    return candidates


def desired_place_block_pose(obs: NDArray, block_name: str) -> RectPose:
    """Return the target pose for the held block inside the shelf."""
    robot = extract_robot(obs)
    held = extract_block(obs, block_name)
    slot_x, slot_y = next_free_slot_center(obs)
    relative_theta = wrap_angle(held.theta - tool_tip_pose(robot).theta)
    target_theta = wrap_angle((np.pi / 2) + relative_theta)
    return rect_pose_from_center(slot_x, slot_y, held.width, held.height, target_theta)
