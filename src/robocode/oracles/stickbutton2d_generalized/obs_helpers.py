"""Observation parsing and geometric predicates for variable-count StickButton2D.

The object-centric counterpart of :mod:`robocode.oracles.stickbutton2d_medium`.
Observations are ``ObjectCentricState``s holding any number of buttons, so
buttons are discovered from the state rather than read at fixed indices. The
pose dataclasses, world constants and geometric predicates are shared with the
fixed-count oracle; only the feature access differs.

Position convention:
  Robot: (x, y) is the centre of the base circle.
  Stick: (x, y) is the bottom-left corner.
  Buttons: (x, y) is the centre of the circle.
"""

from __future__ import annotations

import math

from relational_structs import ObjectCentricState

from robocode.oracles.stickbutton2d_medium.obs_helpers import (
    MIN_GRASP_CLEARANCE,
    WORLD_WIDTH,
    CirclePose,
    RectPose,
    RobotPose,
)

BUTTON_PREFIX = "button"


def get_feature(state: ObjectCentricState, name: str, feature: str) -> float:
    """Get a single feature value for an object by name."""
    return float(state.get(state.get_object_from_name(name), feature))


def extract_robot(state: ObjectCentricState) -> RobotPose:
    """Extract robot pose from the observation."""
    robot = state.get_object_from_name("robot")
    return RobotPose(
        x=float(state.get(robot, "x")),
        y=float(state.get(robot, "y")),
        theta=float(state.get(robot, "theta")),
        base_radius=float(state.get(robot, "base_radius")),
        arm_joint=float(state.get(robot, "arm_joint")),
        arm_length=float(state.get(robot, "arm_length")),
        vacuum=float(state.get(robot, "vacuum")),
        gripper_height=float(state.get(robot, "gripper_height")),
        gripper_width=float(state.get(robot, "gripper_width")),
    )


def extract_rect(state: ObjectCentricState, name: str) -> RectPose:
    """Extract rectangle pose for a named object."""
    obj = state.get_object_from_name(name)
    return RectPose(
        x=float(state.get(obj, "x")),
        y=float(state.get(obj, "y")),
        theta=float(state.get(obj, "theta")),
        width=float(state.get(obj, "width")),
        height=float(state.get(obj, "height")),
    )


def extract_circle(state: ObjectCentricState, name: str) -> CirclePose:
    """Extract circle pose for a named button."""
    obj = state.get_object_from_name(name)
    return CirclePose(
        x=float(state.get(obj, "x")),
        y=float(state.get(obj, "y")),
        radius=float(state.get(obj, "radius")),
    )


def button_names(state: ObjectCentricState) -> list[str]:
    """Every button in this instance, ordered by index."""
    names = [n for n in state.get_object_names() if n.startswith(BUTTON_PREFIX)]
    return sorted(names, key=lambda n: int(n[len(BUTTON_PREFIX) :]))


def has_space_stick_bottom(state: ObjectCentricState) -> bool:
    """True if the robot can position below the stick bottom to grasp it.

    The robot centre must fit at ``stick.y - arm_length`` with enough clearance above
    the floor (robot.base_radius + margin), and the stick centre must be far enough
    from either side wall for the base to sit under it.
    """
    robot = extract_robot(state)
    stick = extract_rect(state, "stick")
    required_y = stick.y - robot.arm_length - robot.gripper_width
    y_ok = required_y >= robot.base_radius + MIN_GRASP_CLEARANCE
    x_ok = stick.cx > robot.base_radius + MIN_GRASP_CLEARANCE and (
        WORLD_WIDTH - stick.cx > robot.base_radius + MIN_GRASP_CLEARANCE
    )
    return x_ok and y_ok


def no_space_stick_bottom(state: ObjectCentricState) -> bool:
    """Negation of :func:`has_space_stick_bottom`."""
    return not has_space_stick_bottom(state)


def holding_stick(state: ObjectCentricState) -> bool:
    """True when the vacuum is on (the only movable object is the stick)."""
    return extract_robot(state).vacuum > 0.5


def stick_bottom_grasped(state: ObjectCentricState) -> bool:
    """True if the stick is held at its bottom with the arm pointing up.

    The arm must point up (a side grasp uses theta near 0 or pi) and the stick bottom
    must sit above the robot centre, which is the geometry produced by suctioning the
    very bottom of the stick with the arm extended upward.
    """
    robot = extract_robot(state)
    stick = extract_rect(state, "stick")
    if robot.vacuum <= 0.5:
        return False
    arm_up = abs(robot.theta - math.pi / 2) < 0.3
    bottom_above = stick.y > robot.y - 0.05
    return arm_up and bottom_above


def is_button_pressed(state: ObjectCentricState, button_name: str) -> bool:
    """True if the button colour is green (pressed)."""
    return get_feature(state, button_name, "color_g") > 0.5


def all_buttons_pressed(state: ObjectCentricState) -> bool:
    """True when every button has been pressed."""
    return all(is_button_pressed(state, n) for n in button_names(state))


def unpressed_buttons(state: ObjectCentricState) -> list[str]:
    """Return names of all unpressed buttons."""
    return [n for n in button_names(state) if not is_button_pressed(state, n)]


def pickup_y_bottom(stick: RectPose, robot: RobotPose) -> float:
    """Robot y that positions the suction zone at the stick bottom.

    With the arm fully extended upward (``theta = pi/2``), the suction centre is at
    ``robot.y + arm_length + 1.5 * gripper_width``; this solves that for ``robot.y``.
    """
    return stick.y - robot.arm_length - 1.5 * robot.gripper_width
