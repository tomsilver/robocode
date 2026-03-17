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
    RobotPose,
    extract_robot,
    extract_rect,
    goal_region_clear,
    holding_block,
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

        key_waypoints = [
            # (0) Current state
            current,
            # (1) Retract arm, lift to safe height at current x
            wp(robot.x, LIFT_Y, robot.base_radius, 0.0),
            # (2) Move horizontally over the target block
            wp(block.cx, LIFT_Y, robot.base_radius, 0.0),
            # (3) Descend so that fully extending the arm reaches the block center
            wp(block.cx, block.cy + robot.arm_length, robot.base_radius, 0.0),
            # (4) Extend arm fully, turn vacuum on
            wp(block.cx, block.cy + robot.arm_length, robot.arm_length, 1.0),
            # (5) Retract arm and lift to safe height, vacuum stays on
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
