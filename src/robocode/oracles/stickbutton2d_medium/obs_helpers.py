"""Observation parsing and geometric predicates for StickButton2D-b3.

Provides named access to object features from the flat observation vector.

Object names and feature layout:
  robot    [0:9]   x y theta base_radius arm_joint arm_length
                    vacuum gripper_height gripper_width
  stick    [9:19]  x y theta static cr cg cb z_order width height
  button0  [19:28] x y theta static cr cg cb z_order radius
  button1  [28:37] x y theta static cr cg cb z_order radius
  button2  [37:46] x y theta static cr cg cb z_order radius

Position convention:
  Robot: (x, y) is the centre of the base circle.
  Stick: (x, y) is the bottom-left corner.
  Buttons: (x, y) is the centre of the circle.
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

# ---------------------------------------------------------------------------
# Layout: object name -> (base_index, feature_list)
# ---------------------------------------------------------------------------

LAYOUT: dict[str, tuple[int, list[str]]] = {
    "robot": (0, ROBOT_FEATURES),
    "stick": (9, RECT_FEATURES),
    "button0": (19, CIRCLE_FEATURES),
    "button1": (28, CIRCLE_FEATURES),
    "button2": (37, CIRCLE_FEATURES),
}

NUM_BUTTONS = 3

# World / physics constants
TABLE_Y = 1.25  # bottom edge of the table
WORLD_WIDTH = 3.5
WORLD_HEIGHT = 2.5
MIN_GRASP_CLEARANCE = 0.05


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
# Geometric predicates
# ---------------------------------------------------------------------------


def has_space_stick_bottom(obs: NDArray) -> bool:
    """True if the robot can position below the stick bottom to grasp it.

    The robot centre must fit at ``stick.y - arm_length`` with enough
    clearance above the floor (robot.base_radius + margin).
    """
    robot = extract_robot(obs)
    stick = extract_rect(obs, "stick")
    required_y = stick.y - robot.arm_length
    return required_y >= robot.base_radius + MIN_GRASP_CLEARANCE


def no_space_stick_bottom(obs: NDArray) -> bool:
    """Negation of :func:`has_space_stick_bottom`."""
    return not has_space_stick_bottom(obs)


def holding_stick(obs: NDArray) -> bool:
    """True when the vacuum is on (the only movable object is the stick)."""
    robot = extract_robot(obs)
    return robot.vacuum > 0.5


def stick_bottom_grasped(obs: NDArray) -> bool:
    """True if the stick is held and the grip is near the stick bottom.

    When grasped at the bottom with the arm pointing up, ``stick.y`` is
    above ``robot.y`` (the stick sits above the gripper).  When grasped
    higher up the stick length, ``stick.y`` drops well below ``robot.y``
    because the bottom hangs down.
    """
    robot = extract_robot(obs)
    stick = extract_rect(obs, "stick")
    if robot.vacuum <= 0.5:
        return False
    return stick.y > robot.y - 0.05


def is_button_pressed(obs: NDArray, button_name: str) -> bool:
    """True if the button colour is green (pressed)."""
    return get_feature(obs, button_name, "color_g") > 0.5


def all_buttons_pressed(obs: NDArray) -> bool:
    """True when every button has been pressed."""
    return all(is_button_pressed(obs, f"button{i}") for i in range(NUM_BUTTONS))


def unpressed_buttons(obs: NDArray) -> list[str]:
    """Return names of all unpressed buttons."""
    return [
        f"button{i}"
        for i in range(NUM_BUTTONS)
        if not is_button_pressed(obs, f"button{i}")
    ]


def nearest_reachable_y_on_stick(obs: NDArray) -> float:
    """Return the y-coordinate on the stick nearest to the robot's reach.

    Used by :class:`RePositionStick` when the bottom is not reachable.
    The robot positions at its lowest safe y with the arm fully extended
    upward; the suction centre y is the point that contacts the stick.
    """
    robot = extract_robot(obs)
    stick = extract_rect(obs, "stick")
    min_robot_y = robot.base_radius + 0.02
    suction_y = min_robot_y + robot.arm_length + 1.5 * robot.gripper_width
    # Clamp to the stick's vertical extent
    return max(stick.y, min(suction_y, stick.top))


def pickup_y_bottom(stick: RectPose, robot: RobotPose) -> float:
    """Robot y that positions the suction zone at the stick bottom.

    With the arm fully extended upward (``theta = pi/2``), the suction
    centre is at ``robot.y + arm_length + 1.5 * gripper_width``.
    Setting that equal to ``stick.y`` and solving for ``robot.y``:
    """
    return stick.y - robot.arm_length - 1.5 * robot.gripper_width
