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
# "Center of the bottom half" means bounding-box centre at (1.75, 0.625).
HOOK_TARGET_THETA = -math.pi / 2
HOOK_TARGET_CX = WORLD_WIDTH / 2      # 1.75
HOOK_TARGET_CY = TABLE_Y / 2          # 0.625


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


def extract_circle(obs: NDArray, name: str) -> CirclePose:
    """Extract circle pose for a named button."""
    base, features = _base_and_features(name)
    return CirclePose(
        x=float(obs[base + features.index("x")]),
        y=float(obs[base + features.index("y")]),
        radius=float(obs[base + features.index("radius")]),
    )


# ---------------------------------------------------------------------------
# Geometric utilities for the L-shaped hook
# ---------------------------------------------------------------------------


def hook_vertices(hook: HookPose) -> NDArray:
    """Compute the 6 main vertices of the L-shaped hook in world coordinates.

    At theta=0 the local vertices (relative to origin at top-right) are::

        v1=(-l1, 0) ---- v0=(0, 0)       <-- origin
        v2=(-l1,-w) ---- v3=(-w,-w)              |
                          v4=(-w,-l2) -- v5=(0,-l2)
    """
    w, l1, l2 = hook.width, hook.length_side1, hook.length_side2
    cos_t = math.cos(hook.theta)
    sin_t = math.sin(hook.theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    local = np.array(
        [
            [0, 0],
            [-l1, 0],
            [-l1, -w],
            [-w, -w],
            [-w, -l2],
            [0, -l2],
        ]
    )

    return local @ rot.T + np.array([hook.x, hook.y])


def hook_center(hook: HookPose) -> tuple[float, float]:
    """Bounding-box centre of the hook in world coordinates."""
    verts = hook_vertices(hook)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    return (float((mn[0] + mx[0]) / 2), float((mn[1] + mx[1]) / 2))


def hook_bbox(hook: HookPose) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box (min_x, min_y, max_x, max_y)."""
    verts = hook_vertices(hook)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))


def _seg_x_at_y(
    x0: float, y0: float, x1: float, y1: float, y: float
) -> float | None:
    """X where segment (x0,y0)-(x1,y1) crosses the horizontal line at *y*."""
    if y0 == y1:
        return None  # horizontal segment
    t = (y - y0) / (y1 - y0)
    if 0.0 <= t <= 1.0:
        return x0 + t * (x1 - x0)
    return None


def hook_x_extent_at_y(
    hook: HookPose, y: float
) -> tuple[float, float] | None:
    """Left and right x-extent of the hook polygon at the given *y*.

    Returns ``(min_x, max_x)`` or *None* when the horizontal line at *y*
    does not intersect the hook.
    """
    verts = hook_vertices(hook)
    n = len(verts)
    crossings: list[float] = []
    for i in range(n):
        j = (i + 1) % n
        xc = _seg_x_at_y(
            float(verts[i][0]),
            float(verts[i][1]),
            float(verts[j][0]),
            float(verts[j][1]),
            y,
        )
        if xc is not None:
            crossings.append(xc)
    if len(crossings) < 2:
        return None
    return (min(crossings), max(crossings))


def hook_long_arm_center(hook: HookPose) -> tuple[float, float]:
    """Centre of the long (horizontal) arm of the hook in world coords.

    At theta=0 the long arm occupies x in [-l1, 0], y in [-w, 0].
    Its local centre is (-l1/2, -w/2).
    """
    local_x = -hook.length_side1 / 2
    local_y = -hook.width / 2
    cos_t = math.cos(hook.theta)
    sin_t = math.sin(hook.theta)
    wx = hook.x + local_x * cos_t - local_y * sin_t
    wy = hook.y + local_x * sin_t + local_y * cos_t
    return (wx, wy)


# ---------------------------------------------------------------------------
# Geometric predicates
# ---------------------------------------------------------------------------


def holding_hook(obs: NDArray) -> bool:
    """True when the vacuum is on (the only movable object the robot grabs)."""
    robot = extract_robot(obs)
    return robot.vacuum > 0.5


def hook_is_horizontal(obs: NDArray, tol: float = 0.15) -> bool:
    """True when the hook's theta is close to 0 (horizontal)."""
    hook = extract_hook(obs)
    theta = math.remainder(hook.theta, 2 * math.pi)
    return abs(theta) < tol


def hook_at_target_theta(obs: NDArray, tol: float = 0.15) -> bool:
    """True when the hook's theta is close to the target (-π)."""
    hook = extract_hook(obs)
    diff = math.remainder(hook.theta - HOOK_TARGET_THETA, 2 * math.pi)
    return abs(diff) < tol


def hook_at_center(obs: NDArray, pos_tol: float = 0.3) -> bool:
    """True when the hook's bounding-box centre is near the target centre."""
    hook = extract_hook(obs)
    cx, cy = hook_center(hook)
    return abs(cx - HOOK_TARGET_CX) < pos_tol and abs(cy - HOOK_TARGET_CY) < pos_tol


def hook_grasped_and_horizontal(obs: NDArray) -> bool:
    """True when the hook is held, horizontal, and roughly centred."""
    return holding_hook(obs) and hook_is_horizontal(obs) and hook_at_center(obs)


def hook_grasped_and_rotated(obs: NDArray) -> bool:
    """True when the hook is held and at the target rotation (-π)."""
    return holding_hook(obs) and hook_at_target_theta(obs)


def both_buttons_pressed(obs: NDArray) -> bool:
    """True when both buttons have turned green (task success)."""
    return (
        get_feature(obs, "movable_button", "color_g") > 0.5
        and get_feature(obs, "target_button", "color_g") > 0.5
    )
