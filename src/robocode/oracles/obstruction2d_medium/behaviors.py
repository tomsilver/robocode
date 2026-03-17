"""Oracle behaviors for Obstruction2D-o2 (medium, 2 obstructions).

Three sequential behaviors that solve the task:
  RemoveObstruction  -> GoalRegionClear
  PickTargetBlock    -> HoldingTarget
  PlaceTargetBlock   -> GoalAchieved

Observation layout (49 features):
  Robot          [0:9]   x y theta base_r arm_j arm_l vac grip_h grip_w
  Target surface [9:19]  x y theta static cr cg cb z w h
  Target block   [19:29] x y theta static cr cg cb z w h
  Obstruction 0  [29:39] x y theta static cr cg cb z w h
  Obstruction 1  [39:49] x y theta static cr cg cb z w h

Position convention: (x, y) is the bottom-left corner of each rectangle.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from robocode.utils.structs import Behavior
from robocode.oracles.obstruction2d_medium.obs_helpers import (
    GRIPPER_CLEARANCE,
    RobotPose,
    TABLE_TOP,
    extract_robot,
    extract_rect,
    find_largest_gap,
    goal_region_clear,
    holding_obstruction,
    holding_block,
    overlaps_surface_h,
    pickup_y,
)
from robocode.oracles.obstruction2d_medium.act_helpers import (
    connecting_waypoints,
    waypoints_to_actions,
)

DOWN = -np.pi / 2
LIFT_Y = 0.8

# ---------------------------------------------------------------------------
# PickTargetBlock
# ---------------------------------------------------------------------------

class PickTargetBlock(Behavior[NDArray, NDArray]):
    """Pick up the target block.

    Subgoal  (HoldingTarget): vacuum on and block lifted off the table.
    Precond: goal region is clear and robot is not holding any block.
    """

    def __init__(self, num_obstructions: int = 2) -> None:
        self._num_obs = num_obstructions
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step
        self._actions: deque[NDArray] = deque()

    def reset(self, x: NDArray) -> None:
        """Reset and generate the initial action plan."""
        self._generate_waypoints(x)

    def _generate_waypoints(self, x: NDArray) -> None:
        """Generate key waypoints for picking the target block, then
        interpolate and convert to an action plan."""
        robot = extract_robot(x)
        block = extract_rect(x, "target_block")

        def wp(x: float, y: float, arm_joint: float, vacuum: float) -> RobotPose:
            return RobotPose(
                x=x, y=y, theta=DOWN,
                base_radius=robot.base_radius,
                arm_joint=arm_joint,
                arm_length=robot.arm_length,
                vacuum=vacuum,
                gripper_height=robot.gripper_height,
                gripper_width=robot.gripper_width,
            )

        # Start from the actual current robot state
        current = RobotPose(
            x=robot.x, y=robot.y, theta=robot.theta,
            base_radius=robot.base_radius,
            arm_joint=robot.arm_joint,
            arm_length=robot.arm_length,
            vacuum=robot.vacuum,
            gripper_height=robot.gripper_height,
            gripper_width=robot.gripper_width,
        )

        grab_y = pickup_y(block, robot)

        key_waypoints = [
            # (0) Current state
            current,
            # (1) Retract arm, lift to safe height at current x
            wp(robot.x, LIFT_Y, robot.base_radius, 0.0),
            # (2) Move horizontally over the target block
            wp(block.cx, LIFT_Y, robot.base_radius, 0.0),
            # (3) Descend so gripper clears block top when arm extended
            wp(block.cx, grab_y, robot.base_radius, 0.0),
            # (4) Extend arm fully (gripper just above block top)
            wp(block.cx, grab_y, robot.arm_length, 0.0),
            # (5) Turn vacuum on (suction zone reaches into block)
            wp(block.cx, grab_y, robot.arm_length, 1.0),
            # (6) Retract arm and lift to safe height, vacuum stays on
            wp(block.cx, LIFT_Y, robot.base_radius, 1.0),
        ]

        dense = connecting_waypoints(key_waypoints)
        self._actions = waypoints_to_actions(dense)

    def initializable(self, x: NDArray) -> bool:
        """Check that the goal region is clear and robot is not already holding any block."""
        return goal_region_clear(x) and not holding_block(x)

    def terminated(self, x: NDArray) -> bool:
        """Check if the robot is holding the target block."""
        return holding_block(x)

    def step(self, x: NDArray) -> NDArray:
        """Return the next action to execute.

        If the action plan is exhausted but the subgoal is not yet reached,
        re-generate waypoints from the current observation.
        """
        if not self._actions:
            self._generate_waypoints(x)
        return self._actions.popleft()


# ---------------------------------------------------------------------------
# ClearTargetRegion
# ---------------------------------------------------------------------------

class ClearTargetRegion(Behavior[NDArray, NDArray]):
    """Remove all obstructions from the target surface, one at a time.

    Subgoal  (GoalRegionClear): no obstruction overlaps the surface and
        the robot is not holding anything.
    Precond: at least one obstruction overlaps the surface.
    """

    def __init__(self, num_obstructions: int = 2) -> None:
        self._num_obs = num_obstructions
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step
        self._actions: deque[NDArray] = deque()
        self._obstructions_uncleared: deque[str] = deque()

    def reset(self, x: NDArray) -> None:
        """Reset and generate the initial action plan."""
        self._populate_obstructions(x)
        self._generate_waypoints(x)

    def _populate_obstructions(self, x: NDArray) -> None:
        """Identify which obstructions overlap the surface and queue them
        tallest-first for removal."""
        overlapping: list[tuple[float, str]] = []
        for i in range(self._num_obs):
            name = f"obstruction{i}"
            if overlaps_surface_h(x, name):
                rect = extract_rect(x, name)
                overlapping.append((rect.height, name))
        # Sort tallest first (descending height)
        overlapping.sort(reverse=True)
        self._obstructions_uncleared = deque(name for _, name in overlapping)

    def _find_safe_location(self, x: NDArray) -> float:
        """Return the center-x of the largest free gap on the table."""
        return find_largest_gap(x)

    def _generate_waypoints(self, x: NDArray) -> None:
        """Generate a full pick-travel-place waypoint plan for one obstruction."""
        # If the queue is empty, re-scan (world state may have changed)
        if not self._obstructions_uncleared:
            self._populate_obstructions(x)
        if not self._obstructions_uncleared:
            return  # nothing left to clear

        current_obstruction = self._obstructions_uncleared.popleft()
        robot = extract_robot(x)
        block = extract_rect(x, current_obstruction)
        safe_x = self._find_safe_location(x)
        grab_y = pickup_y(block, robot)
        # y for placing: arm fully extended puts gripper just above table
        place_y = TABLE_TOP + block.height + robot.arm_length + GRIPPER_CLEARANCE

        def wp(x: float, y: float, arm_joint: float, vacuum: float) -> RobotPose:
            return RobotPose(
                x=x, y=y, theta=DOWN,
                base_radius=robot.base_radius,
                arm_joint=arm_joint,
                arm_length=robot.arm_length,
                vacuum=vacuum,
                gripper_height=robot.gripper_height,
                gripper_width=robot.gripper_width,
            )

        current = RobotPose(
            x=robot.x, y=robot.y, theta=robot.theta,
            base_radius=robot.base_radius,
            arm_joint=robot.arm_joint,
            arm_length=robot.arm_length,
            vacuum=robot.vacuum,
            gripper_height=robot.gripper_height,
            gripper_width=robot.gripper_width,
        )

        key_waypoints = [
            # (0) Current state
            current,
            # (1) Retract arm, lift to safe height
            wp(robot.x, LIFT_Y, robot.base_radius, 0.0),
            # (2) Move over the obstruction
            wp(block.cx, LIFT_Y, robot.base_radius, 0.0),
            # (3) Descend so gripper clears block top when arm extended
            wp(block.cx, grab_y, robot.base_radius, 0.0),
            # (4) Extend arm (gripper just above block top)
            wp(block.cx, grab_y, robot.arm_length, 0.0),
            # (5) Turn vacuum on (suction zone reaches block)
            wp(block.cx, grab_y, robot.arm_length, 1.0),
            # (6) Retract arm to lift
            wp(block.cx, LIFT_Y, robot.base_radius, 1.0),
            # (7) Travel to safe drop location
            wp(safe_x, LIFT_Y, robot.base_radius, 1.0),
            # (8) Descend to table level
            wp(safe_x, place_y, robot.base_radius, 1.0),
            # (9) Extend arm to place
            wp(safe_x, place_y, robot.arm_length, 1.0),
            # (10) Release vacuum
            wp(safe_x, place_y, robot.arm_length, 0.0),
            # (11) Retract arm and lift
            wp(safe_x, LIFT_Y, robot.base_radius, 0.0),
        ]

        dense = connecting_waypoints(key_waypoints)
        self._actions = waypoints_to_actions(dense)

    def initializable(self, x: NDArray) -> bool:
        """True when at least one obstruction overlaps the surface."""
        return not goal_region_clear(x)

    def terminated(self, x: NDArray) -> bool:
        """True when no obstruction overlaps and robot is not holding anything."""
        return goal_region_clear(x) and not holding_obstruction(x)

    def step(self, x: NDArray) -> NDArray:
        """Pop next action; re-plan if exhausted but not done."""
        if not self._actions:
            self._generate_waypoints(x)
        return self._actions.popleft()