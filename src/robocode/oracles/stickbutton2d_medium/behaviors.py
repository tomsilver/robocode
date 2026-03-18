"""Oracle behaviors for StickButton2D-b3 (medium, 3 buttons).

Three sequential behaviors that solve the task:
  RePositionStick   -> HasSpaceStickBottom
  GraspStickBottom  -> StickBottomGrasped
  TouchAllButtons   -> AllButtonsPressed (GoalAchieved)

Observation layout (46 features):
  Robot    [0:9]   x y theta base_r arm_j arm_l vac grip_h grip_w
  Stick    [9:19]  x y theta static cr cg cb z w h
  Button 0 [19:28] x y theta static cr cg cb z radius
  Button 1 [28:37] x y theta static cr cg cb z radius
  Button 2 [37:46] x y theta static cr cg cb z radius

Position convention:
  Robot/buttons (x, y) = centre.  Stick (x, y) = bottom-left corner.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from robocode.oracles.stickbutton2d_medium.act_helpers import (
    connecting_waypoints,
    waypoints_to_actions,
)
from robocode.oracles.stickbutton2d_medium.obs_helpers import (
    TABLE_Y,
    WORLD_WIDTH,
    RobotPose,
    all_buttons_pressed,
    extract_circle,
    extract_rect,
    extract_robot,
    has_space_stick_bottom,
    holding_stick,
    no_space_stick_bottom,
    pickup_y_bottom,
    stick_bottom_grasped,
    unpressed_buttons,
)
from robocode.primitives.behavior import Behavior

UP = np.pi / 2
SAFE_Y = 0.5


def _current_pose(robot: RobotPose) -> RobotPose:
    """Return a copy of *robot* as-is (used as the first waypoint)."""
    return RobotPose(
        x=robot.x,
        y=robot.y,
        theta=robot.theta,
        base_radius=robot.base_radius,
        arm_joint=robot.arm_joint,
        arm_length=robot.arm_length,
        vacuum=robot.vacuum,
        gripper_height=robot.gripper_height,
        gripper_width=robot.gripper_width,
    )


# ---------------------------------------------------------------------------
# RePositionStick
# ---------------------------------------------------------------------------


class RePositionStick(Behavior[NDArray, NDArray]):
    """Lift the stick so the robot can later grasp it at the bottom.

    Used when the stick bottom is too close to the floor for the robot
    to fit beneath it.  The robot grabs the stick at the nearest
    reachable point, lifts straight up, then releases.

    Subgoal  (HasSpaceStickBottom): enough room below the stick bottom.
    Precond  (NoSpaceStickBottom):  not enough room.
    """

    def __init__(self) -> None:
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step
        self._actions: deque[NDArray] = deque()

    def reset(self, x: NDArray) -> None:
        self._generate_waypoints(x)

    def _generate_waypoints(self, x: NDArray) -> None:
        robot = extract_robot(x)
        stick = extract_rect(x, "stick")

        def wp(
            px: float, py: float, arm_joint: float, vacuum: float
        ) -> RobotPose:
            return RobotPose(
                x=px,
                y=py,
                theta=UP,
                base_radius=robot.base_radius,
                arm_joint=arm_joint,
                arm_length=robot.arm_length,
                vacuum=vacuum,
                gripper_height=robot.gripper_height,
                gripper_width=robot.gripper_width,
            )

        current = _current_pose(robot)
        min_robot_y = robot.base_radius + 0.02
        # Grab y: lowest the robot can go, arm extended → suction hits stick
        grab_y = min_robot_y
        lift_y = SAFE_Y + 0.3  # lift target

        key_waypoints = [
            current,
            # (1) Retract arm, point up, safe height
            wp(robot.x, SAFE_Y, robot.base_radius, 0.0),
            # (2) Move over stick centre-x
            wp(stick.cx, SAFE_Y, robot.base_radius, 0.0),
            # (3) Lower to minimum y
            wp(stick.cx, grab_y, robot.base_radius, 0.0),
            # (4) Extend arm to reach stick
            wp(stick.cx, grab_y, robot.arm_length, 0.0),
            # (5) Vacuum on
            wp(stick.cx, grab_y, robot.arm_length, 1.0),
            # (6) Lift straight up (stick follows)
            wp(stick.cx, lift_y, robot.arm_length, 1.0),
            # (7) Release vacuum
            wp(stick.cx, lift_y, robot.arm_length, 0.0),
            # (8) Retract arm, safe height
            wp(stick.cx, SAFE_Y, robot.base_radius, 0.0),
        ]

        dense = connecting_waypoints(key_waypoints)
        self._actions = waypoints_to_actions(dense)

    def initializable(self, x: NDArray) -> bool:
        return no_space_stick_bottom(x)

    def terminated(self, x: NDArray) -> bool:
        return has_space_stick_bottom(x)

    def step(self, x: NDArray) -> NDArray:
        if not self._actions:
            self._generate_waypoints(x)
        return self._actions.popleft()


# ---------------------------------------------------------------------------
# GraspStickBottom
# ---------------------------------------------------------------------------


class GraspStickBottom(Behavior[NDArray, NDArray]):
    """Navigate to the stick bottom and grasp it with the arm pointing up.

    If the robot is already holding the stick (e.g. from RePositionStick),
    it first releases and then re-grabs at the bottom.

    Subgoal  (StickBottomGrasped): vacuum on with grip near stick bottom.
    Precond  (HasSpaceStickBottom): enough room below the stick.
    """

    def __init__(self) -> None:
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step
        self._actions: deque[NDArray] = deque()

    def reset(self, x: NDArray) -> None:
        self._generate_waypoints(x)

    def _generate_waypoints(self, x: NDArray) -> None:
        robot = extract_robot(x)
        stick = extract_rect(x, "stick")

        def wp(
            px: float, py: float, arm_joint: float, vacuum: float
        ) -> RobotPose:
            return RobotPose(
                x=px,
                y=py,
                theta=UP,
                base_radius=robot.base_radius,
                arm_joint=arm_joint,
                arm_length=robot.arm_length,
                vacuum=vacuum,
                gripper_height=robot.gripper_height,
                gripper_width=robot.gripper_width,
            )

        current = _current_pose(robot)
        grab_y = pickup_y_bottom(stick, robot)

        waypoints: list[RobotPose] = [current]

        # If already holding (from RePositionStick), release first.
        if holding_stick(x):
            waypoints += [
                wp(robot.x, robot.y, robot.arm_joint, 0.0),  # vacuum off
                wp(robot.x, SAFE_Y, robot.base_radius, 0.0),  # retract + lift
            ]

        waypoints += [
            # Move over stick centre-x
            wp(stick.cx, SAFE_Y, robot.base_radius, 0.0),
            # Lower to grab position
            wp(stick.cx, grab_y, robot.base_radius, 0.0),
            # Extend arm (suction zone at stick bottom)
            wp(stick.cx, grab_y, robot.arm_length, 0.0),
            # Vacuum on
            wp(stick.cx, grab_y, robot.arm_length, 1.0),
            # Retract arm and lift
            wp(stick.cx, SAFE_Y, robot.base_radius, 1.0),
        ]

        dense = connecting_waypoints(waypoints)
        self._actions = waypoints_to_actions(dense)

    def initializable(self, x: NDArray) -> bool:
        return has_space_stick_bottom(x) and not stick_bottom_grasped(x)

    def terminated(self, x: NDArray) -> bool:
        return stick_bottom_grasped(x)

    def step(self, x: NDArray) -> NDArray:
        if not self._actions:
            self._generate_waypoints(x)
        return self._actions.popleft()


# ---------------------------------------------------------------------------
# TouchAllButtons
# ---------------------------------------------------------------------------


class TouchAllButtons(Behavior[NDArray, NDArray]):
    """Press every unpressed button by sweeping left-to-right.

    Buttons are ranked by increasing *x*.  For each button the robot
    aligns on the x-axis (adjusting for the stick offset) and then:

    * If the button is **below the robot** → move down so the robot body
      overlaps the button.
    * If the button is **above the stick top** → move up so the stick
      sweeps over the button.
    * Otherwise the button is already within the stick/robot coverage and
      is pressed during the horizontal sweep.

    Subgoal  (AllButtonsPressed): every button has turned green.
    Precond  (StickBottomGrasped): stick is held at its bottom.
    """

    def __init__(self) -> None:
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step
        self._actions: deque[NDArray] = deque()
        self._buttons_to_press: deque[str] = deque()

    def reset(self, x: NDArray) -> None:
        self._populate_buttons(x)
        self._generate_waypoints(x)

    def _populate_buttons(self, x: NDArray) -> None:
        """Queue unpressed buttons sorted by increasing x (left → right)."""
        names = unpressed_buttons(x)
        names.sort(key=lambda n: extract_circle(x, n).x)
        self._buttons_to_press = deque(names)

    def _generate_waypoints(self, x: NDArray) -> None:
        if not self._buttons_to_press:
            self._populate_buttons(x)
        if not self._buttons_to_press:
            return

        button_name = self._buttons_to_press.popleft()
        button = extract_circle(x, button_name)
        robot = extract_robot(x)
        stick = extract_rect(x, "stick")

        max_y = TABLE_Y - robot.base_radius - 0.01
        min_y = robot.base_radius + 0.01

        def wp(
            px: float, py: float, arm_joint: float, vacuum: float
        ) -> RobotPose:
            return RobotPose(
                x=px,
                y=py,
                theta=UP,
                base_radius=robot.base_radius,
                arm_joint=arm_joint,
                arm_length=robot.arm_length,
                vacuum=vacuum,
                gripper_height=robot.gripper_height,
                gripper_width=robot.gripper_width,
            )

        current = _current_pose(robot)

        # Horizontal alignment: compensate for stick offset from robot centre
        stick_offset_x = stick.cx - robot.x
        target_x = button.x - stick_offset_x
        target_x = max(
            robot.base_radius + 0.01,
            min(target_x, WORLD_WIDTH - robot.base_radius - 0.01),
        )

        # Vertical targeting
        if button.y < robot.y:
            # Button is below the robot → move down (robot body presses it)
            target_y = max(min_y, min(button.y, max_y))
        elif button.y > stick.top:
            # Button is above the stick top → move up to raise the stick
            delta = button.y - stick.top + 0.02
            target_y = min(robot.y + delta, max_y)
        else:
            # Button is within the current coverage → stay at current y
            target_y = robot.y

        key_waypoints = [
            current,
            # Move horizontally at current y
            wp(target_x, robot.y, robot.base_radius, 1.0),
            # Move to target y (press the button)
            wp(target_x, target_y, robot.base_radius, 1.0),
            # Return to safe-ish y for next button
            wp(target_x, SAFE_Y, robot.base_radius, 1.0),
        ]

        dense = connecting_waypoints(key_waypoints)
        self._actions = waypoints_to_actions(dense)

    def initializable(self, x: NDArray) -> bool:
        return stick_bottom_grasped(x) and not all_buttons_pressed(x)

    def terminated(self, x: NDArray) -> bool:
        return all_buttons_pressed(x)

    def step(self, x: NDArray) -> NDArray:
        if not self._actions:
            self._generate_waypoints(x)
        return self._actions.popleft()
