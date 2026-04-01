"""Observation parsing and geometric predicates for PushPullHook2D.

Provides named access to object features from the flat observation vector.

Object names and feature layout:
  robot            [0:9]   x y theta base_radius arm_joint arm_length
                           vacuum gripper_height gripper_width
  hook             [9:20]  x y theta static cr cg cb z_order
                           width length_side1 length_side2
  movable_button   [20:29] x y theta static cr cg cb z_order radius
  target_button    [29:38] x y theta static cr cg cb z_order radius

Position convention:
  Robot: (x, y) is the centre of the base circle.
  Hook (L-object): (x, y) is the top-right vertex at theta=0.
    At theta=0 the shape looks like:
        --------|
                |
                |
    The horizontal arm extends LEFT by length_side1.
    The vertical arm extends DOWN by length_side2.
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

LOBJECT_FEATURES = [
    "x",
    "y",
    "theta",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
    "width",
    "length_side1",
    "length_side2",
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
    "hook": (9, LOBJECT_FEATURES),
    "movable_button": (20, CIRCLE_FEATURES),
    "target_button": (29, CIRCLE_FEATURES),
}

# World / physics constants
WORLD_WIDTH = 3.5
WORLD_HEIGHT = 2.5
TABLE_Y = WORLD_HEIGHT / 2  # bottom edge of the table (1.25)

# Target configuration for the GraspRotate behavior.
# "Vertical" means hook theta = -π (short side at bottom).
HOOK_TARGET_THETA = -math.pi / 2


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
class HookPose:
    """L-shaped hook pose extracted from the observation vector.

    (x, y) is the top-right vertex at theta=0.
    """

    x: float
    y: float
    theta: float
    width: float
    length_side1: float  # horizontal arm length (long)
    length_side2: float  # vertical arm length (short)


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


def extract_hook(obs: NDArray) -> HookPose:
    """Extract hook pose from the observation."""
    base, features = _base_and_features("hook")
    return HookPose(
        x=float(obs[base + features.index("x")]),
        y=float(obs[base + features.index("y")]),
        theta=float(obs[base + features.index("theta")]),
        width=float(obs[base + features.index("width")]),
        length_side1=float(obs[base + features.index("length_side1")]),
        length_side2=float(obs[base + features.index("length_side2")]),
    )


# ---------------------------------------------------------------------------
# Geometric predicates
# ---------------------------------------------------------------------------


def holding_hook(obs: NDArray) -> bool:
    """True when the vacuum is on (the only movable object the robot grabs)."""
    robot = extract_robot(obs)
    return robot.vacuum > 0.5


def hook_at_target_theta(obs: NDArray, tol: float = 0.05) -> bool:
    """True when the hook's theta is close to the target (-π)."""
    hook = extract_hook(obs)
    diff = math.remainder(hook.theta - HOOK_TARGET_THETA, 2 * math.pi)
    return abs(diff) < tol


def hook_grasped_and_rotated(obs: NDArray) -> bool:
    """True when the hook is held and at the target rotation (-π/2)."""
    return holding_hook(obs) and hook_at_target_theta(obs)


def hook_at_pushpull_theta(obs: NDArray, tol: float = 0.05) -> bool:
    """True when the hook's theta is close to π/2 (ready for push/pull)."""
    hook = extract_hook(obs)
    diff = math.remainder(hook.theta - math.pi / 2, 2 * math.pi)
    return abs(diff) < tol


def buttons_vertically_aligned(obs: NDArray, tol: float = 0.15) -> bool:
    """True when the movable button's x is close to the target button's x."""
    mx = get_feature(obs, "movable_button", "x")
    tx = get_feature(obs, "target_button", "x")
    return abs(mx - tx) < tol


def both_buttons_pressed(obs: NDArray) -> bool:
    """True when both buttons have turned green (task success)."""
    return (
        get_feature(obs, "movable_button", "color_g") > 0.5
        and get_feature(obs, "target_button", "color_g") > 0.5
    )
