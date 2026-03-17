"""Observation parsing and geometric predicates for Obstruction2D-o2.

Provides named access to object features from the flat observation vector.

Object names and feature layout:
  robot            [0:9]   x y theta base_radius arm_joint arm_length vacuum gripper_height gripper_width
  target_surface   [9:19]  x y theta static cr cg cb z_order width height
  target_block     [19:29] x y theta static cr cg cb z_order width height
  obstruction0     [29:39] x y theta static cr cg cb z_order width height
  obstruction1     [39:49] x y theta static cr cg cb z_order width height

Position convention: (x, y) is the bottom-left corner of each rectangle.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Layout: object name -> (base_index, feature_list)
# ---------------------------------------------------------------------------

LAYOUT: dict[str, tuple[int, list[str]]] = {
    "robot": (0, ROBOT_FEATURES),
    "target_surface": (9, RECT_FEATURES),
    "target_block": (19, RECT_FEATURES),
    "obstruction0": (29, RECT_FEATURES),
    "obstruction1": (39, RECT_FEATURES),
}

NUM_OBSTRUCTIONS = 2

# Physics constants
TABLE_TOP = 0.1
GRIP_OFFSET = 0.015
# Clearance so the gripper clears the block top when arm is extended.
# gripper_width/2 (collision extent along arm) + small margin.
GRIPPER_CLEARANCE = 0.01


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
    x: float
    y: float
    theta: float
    width: float
    height: float

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2

    @property
    def top(self) -> float:
        return self.y + self.height

    @property
    def right(self) -> float:
        return self.x + self.width


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
    """Extract rectangle pose for a named object (surface, block, obstruction)."""
    base, features = _base_and_features(name)
    return RectPose(
        x=float(obs[base + features.index("x")]),
        y=float(obs[base + features.index("y")]),
        theta=float(obs[base + features.index("theta")]),
        width=float(obs[base + features.index("width")]),
        height=float(obs[base + features.index("height")]),
    )


# ---------------------------------------------------------------------------
# Geometric predicates
# ---------------------------------------------------------------------------


def overlaps_surface_h(obs: NDArray, obstruction_name: str) -> bool:
    """True if the named obstruction overlaps the target surface horizontally and sits
    at table level."""
    obj = extract_rect(obs, obstruction_name)
    surf = extract_rect(obs, "target_surface")
    h_overlap = min(obj.right, surf.right) - max(obj.x, surf.x)
    at_table = abs(obj.y - surf.top) < 0.05
    return h_overlap > 1e-4 and at_table


def goal_region_clear(obs: NDArray) -> bool:
    """True when no obstruction overlaps the target surface."""
    for i in range(NUM_OBSTRUCTIONS):
        if overlaps_surface_h(obs, f"obstruction{i}"):
            return False
    return True


def is_on_surface(obs: NDArray, obj_name: str) -> bool:
    """True if the bottom edge of *obj_name* is contained within the target surface."""
    obj = extract_rect(obs, obj_name)
    surf = extract_rect(obs, "target_surface")
    tol = 0.025
    py = obj.y - tol
    return (
        obj.x >= surf.x - 1e-4
        and obj.right <= surf.right + 1e-4
        and py >= surf.y - 1e-4
        and py <= surf.top + 1e-4
    )


WORLD_WIDTH = 1.618  # golden ratio


def occupied_intervals(obs: NDArray) -> list[tuple[float, float]]:
    """Return sorted list of (left, right) intervals occupied on the table.

    Includes: target_surface, target_block, and all obstructions that are
    at table level (y close to TABLE_TOP).
    """
    intervals: list[tuple[float, float]] = []
    for name in ["target_surface", "target_block"] + [
        f"obstruction{i}" for i in range(NUM_OBSTRUCTIONS)
    ]:
        rect = extract_rect(obs, name)
        # Only count objects sitting on or near the table
        if rect.y < TABLE_TOP + 0.1:
            intervals.append((rect.x, rect.right))
    intervals.sort()
    return intervals


def find_largest_gap(obs: NDArray) -> float:
    """Return the center-x of the largest free gap on the table."""
    intervals = occupied_intervals(obs)
    # Merge overlapping intervals
    merged: list[tuple[float, float]] = []
    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))

    # Find largest gap between [0, WORLD_WIDTH]
    best_cx = 0.0
    best_gap = 0.0
    # Gap before first interval
    if merged:
        gap = merged[0][0]
        if gap > best_gap:
            best_gap = gap
            best_cx = gap / 2
    # Gaps between intervals
    for i in range(len(merged) - 1):
        gap = merged[i + 1][0] - merged[i][1]
        if gap > best_gap:
            best_gap = gap
            best_cx = (merged[i][1] + merged[i + 1][0]) / 2
    # Gap after last interval
    if merged:
        gap = WORLD_WIDTH - merged[-1][1]
        if gap > best_gap:
            best_gap = gap
            best_cx = (merged[-1][1] + WORLD_WIDTH) / 2

    return best_cx


def pickup_y(block: RectPose, robot: RobotPose) -> float:
    """Robot y that positions the fully-extended gripper just above *block*.top.

    At this height the suction zone (which extends past the gripper) overlaps the block
    for pickup, but the gripper itself doesn't collide.
    """
    return block.top + robot.arm_length + GRIPPER_CLEARANCE


def place_y(target_top: float, block: RectPose, robot: RobotPose) -> float:
    """Robot y that places the held block so its bottom lands at *target_top*.

    The suction zone holds the block at GRIP_OFFSET below the gripper tip.
    With arm fully extended, gripper tip is at robot_y - arm_length.
    Block top ≈ robot_y - arm_length - GRIP_OFFSET, so
    block bottom ≈ robot_y - arm_length - GRIP_OFFSET - block.height.
    Setting block bottom = target_top and solving for robot_y:
    """
    return target_top + block.height + robot.arm_length + GRIP_OFFSET


def holding_block(obs: NDArray) -> bool:
    """True when vacuum is on and the target block is lifted off the table."""
    robot = extract_robot(obs)
    block = extract_rect(obs, "target_block")
    return robot.vacuum > 0.5 and block.y > TABLE_TOP + 0.04


def holding_obstruction(obs: NDArray) -> bool:
    """True when vacuum is on and any obstruction is lifted off the table."""
    robot = extract_robot(obs)
    if robot.vacuum <= 0.5:
        return False
    for i in range(NUM_OBSTRUCTIONS):
        obj = extract_rect(obs, f"obstruction{i}")
        if obj.y > TABLE_TOP + 0.04:
            return True
    return False
