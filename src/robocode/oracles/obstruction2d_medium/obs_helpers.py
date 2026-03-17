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
    "x", "y", "theta", "base_radius", "arm_joint",
    "arm_length", "vacuum", "gripper_height", "gripper_width",
]

RECT_FEATURES = [
    "x", "y", "theta", "static", "color_r", "color_g",
    "color_b", "z_order", "width", "height",
]

# ---------------------------------------------------------------------------
# Layout: object name -> (base_index, feature_list)
# ---------------------------------------------------------------------------

LAYOUT: dict[str, tuple[int, list[str]]] = {
    "robot":          (0,  ROBOT_FEATURES),
    "target_surface": (9,  RECT_FEATURES),
    "target_block":   (19, RECT_FEATURES),
    "obstruction0":   (29, RECT_FEATURES),
    "obstruction1":   (39, RECT_FEATURES),
}

NUM_OBSTRUCTIONS = 2

# Physics constants
TABLE_TOP = 0.1
GRIP_OFFSET = 0.015


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
    """True if the named obstruction overlaps the target surface horizontally
    and sits at table level."""
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
    return (obj.x >= surf.x - 1e-4
            and obj.right <= surf.right + 1e-4
            and py >= surf.y - 1e-4
            and py <= surf.top + 1e-4)


def holding_block(obs: NDArray) -> bool:
    """True when vacuum is on and the target block is lifted off the table."""
    robot = extract_robot(obs)
    block = extract_rect(obs, "target_block")
    return robot.vacuum > 0.5 and block.y > TABLE_TOP + 0.04
