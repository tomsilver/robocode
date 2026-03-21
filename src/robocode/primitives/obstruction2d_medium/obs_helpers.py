"""Observation parsing and geometric predicates for Obstruction2D-o2.

Provides named access to object features from the flat observation vector.

Object names and feature layout:
  robot            [0:9]   x y theta base_radius arm_joint arm_length
                           vacuum gripper_height gripper_width
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

    @property
    def cx(self) -> float:
        """Centre x."""
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        """Centre y."""
        return self.y + self.height / 2

    @property
    def top(self) -> float:
        """Top edge y."""
        return self.y + self.height

    @property
    def right(self) -> float:
        """Right edge x."""
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
