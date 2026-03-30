"""Oracle behaviors for ClutteredStorage2D-b3."""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from robocode.oracles.clutteredstorage2d_medium.act_helpers import (
    DARM_LIM,
    DTH_LIM,
    DX_LIM,
    DY_LIM,
    connecting_waypoints,
    waypoints_to_actions,
)
from robocode.oracles.clutteredstorage2d_medium.obs_helpers import (
    APPROACH_MARGIN,
    RobotPose,
    all_blocks_inside_shelf,
    choose_next_block,
    extract_block,
    extract_robot,
    extract_shelf,
    held_block_name,
    inside_blocks,
    is_block_inside_shelf,
    next_free_slot_center,
    pick_base_pose_candidates,
    slot_centers,
    wrap_angle,
)
from robocode.primitives.behavior import Behavior

ANGLE_TOL = 0.05
ARM_TOL = 0.02
POS_TOL = 0.02
DEEP_PLACE_Y_TOL = 0.03
COMPACT_Y_TOL = 0.02
TRANSPORT_Y_MARGIN = 0.08
SAFE_VACUUM = 1.0
VACUUM_OFF = 0.0
CARRY_ARM_FRACTION = 0.35
PLACE_ARM_FRACTION = 1.0
NOOP_ACTION = np.zeros(5, dtype=np.float32)
UP = np.pi / 2
SHELF_TARGET = "shelf"


def _robot_pose(robot: RobotPose) -> RobotPose:
    """Return a copy of the robot pose."""
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


def _waypoint(
    robot: RobotPose,
    x: float,
    y: float,
    theta: float,
    arm_joint: float,
    vacuum: float,
) -> RobotPose:
    """Build a waypoint using the robot's fixed geometry."""
    return RobotPose(
        x=x,
        y=y,
        theta=theta,
        base_radius=robot.base_radius,
        arm_joint=arm_joint,
        arm_length=robot.arm_length,
        vacuum=vacuum,
        gripper_height=robot.gripper_height,
        gripper_width=robot.gripper_width,
    )


class StoreRemainingBlocks(Behavior[NDArray, NDArray]):
    """Store all remaining outside blocks one at a time."""

    def __init__(self) -> None:
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step
        self._actions: deque[NDArray] = deque()
        self._phase = "compact"
        self._target_kind = SHELF_TARGET
        self._target_center: tuple[float, float] | None = None
        self._active_block: str | None = None

    def reset(self, x: NDArray) -> None:
        self._phase = "compact"
        self._target_kind = SHELF_TARGET
        self._target_center = None
        self._active_block = None
        self._generate_pick_plan(x)

    def initializable(self, x: NDArray) -> bool:
        return not all_blocks_inside_shelf(x)

    def terminated(self, x: NDArray) -> bool:
        return all_blocks_inside_shelf(x)

    def _choose_compact_block(self, x: NDArray) -> str | None:
        candidates = inside_blocks(x)
        if not candidates:
            return None
        target_x, target_y = slot_centers(x)[0]
        for name in candidates:
            block_x, block_y = extract_block(x, name).center
            if abs(block_x - target_x) > POS_TOL or block_y < target_y - COMPACT_Y_TOL:
                return name
        return None

    def _sync_phase(self, x: NDArray) -> None:
        if (
            self._phase == "compact"
            and self._active_block is None
            and self._choose_compact_block(x) is None
        ):
            self._phase = "store"

    def _generate_pick_plan(self, x: NDArray) -> None:
        self._sync_phase(x)
        robot = extract_robot(x)
        current = _robot_pose(robot)
        carry_arm = max(robot.base_radius, CARRY_ARM_FRACTION * robot.arm_length)
        if self._phase == "compact":
            next_block = self._choose_compact_block(x)
            self._target_kind = SHELF_TARGET
            self._target_center = slot_centers(x)[0]
        else:
            next_block = choose_next_block(x)
            self._target_kind = SHELF_TARGET
            self._target_center = next_free_slot_center(x)
        if next_block is None:
            self._active_block = None
            self._actions = deque([NOOP_ACTION.copy()])
            return
        self._active_block = next_block

        if self._phase == "compact":
            block = extract_block(x, next_block)
            target = _waypoint(
                robot,
                block.center[0],
                block.center[1]
                - (robot.arm_length + 1.5 * robot.gripper_width + APPROACH_MARGIN),
                UP,
                robot.base_radius,
                VACUUM_OFF,
            )
            candidates = [target]
        else:
            candidates = pick_base_pose_candidates(x, next_block)
        if not candidates:
            self._actions = deque([NOOP_ACTION.copy()])
            return

        target = min(
            candidates,
            key=lambda pose: abs(pose.x - robot.x) + abs(pose.y - robot.y),
        )

        key_waypoints = [
            current,
            _waypoint(
                robot,
                robot.x,
                robot.y,
                target.theta,
                robot.base_radius,
                VACUUM_OFF,
            ),
            _waypoint(
                robot,
                target.x,
                target.y,
                target.theta,
                robot.base_radius,
                VACUUM_OFF,
            ),
            _waypoint(
                robot,
                target.x,
                target.y,
                target.theta,
                robot.arm_length,
                VACUUM_OFF,
            ),
            _waypoint(
                robot,
                target.x,
                target.y,
                target.theta,
                robot.arm_length,
                SAFE_VACUUM,
            ),
            _waypoint(
                robot,
                target.x,
                target.y,
                target.theta,
                carry_arm,
                SAFE_VACUUM,
            ),
        ]

        dense = connecting_waypoints(key_waypoints)
        self._actions = waypoints_to_actions(dense)

    def _queue_retreat(self, x: NDArray) -> None:
        robot = extract_robot(x)
        shelf = extract_shelf(x)
        retreat_y = shelf.y1 - 0.75
        key_waypoints = [
            _robot_pose(robot),
            _waypoint(
                robot,
                robot.x,
                retreat_y,
                UP,
                robot.arm_joint,
                VACUUM_OFF,
            ),
            _waypoint(
                robot,
                robot.x,
                retreat_y,
                UP,
                robot.base_radius,
                VACUUM_OFF,
            ),
        ]
        dense = connecting_waypoints(key_waypoints)
        self._actions = waypoints_to_actions(dense)

    def _place_action(self, x: NDArray, block_name: str) -> NDArray:
        robot = extract_robot(x)
        block = extract_block(x, block_name)
        assert self._target_center is not None
        target_x, target_y = self._target_center
        block_x, block_y = block.center
        carry_arm = max(robot.base_radius, CARRY_ARM_FRACTION * robot.arm_length)
        place_arm = max(robot.base_radius, PLACE_ARM_FRACTION * robot.arm_length)
        action = NOOP_ACTION.copy()
        action[4] = SAFE_VACUUM

        if (
            self._target_kind == SHELF_TARGET
            and not is_block_inside_shelf(x, block_name)
            and robot.arm_joint > carry_arm + ARM_TOL
            and (abs(block_x - target_x) > POS_TOL or block_y < target_y - POS_TOL)
        ):
            action[3] = float(np.clip(carry_arm - robot.arm_joint, -DARM_LIM, DARM_LIM))
            return action

        angle_error = wrap_angle(UP - robot.theta)
        if abs(angle_error) > ANGLE_TOL:
            action[2] = float(np.clip(angle_error, -DTH_LIM, DTH_LIM))
            return action

        shelf = extract_shelf(x)
        target_y_tol = COMPACT_Y_TOL if self._phase == "compact" else DEEP_PLACE_Y_TOL
        at_target_x = abs(block_x - target_x) <= POS_TOL
        deep_enough = block_y >= target_y - target_y_tol

        if not is_block_inside_shelf(x, block_name):
            if not at_target_x:
                action[0] = float(np.clip(target_x - block_x, -DX_LIM, DX_LIM))
                return action
            action[1] = float(np.clip(target_y - block_y, -DY_LIM, DY_LIM))
            return action

        if at_target_x and deep_enough:
            action[4] = VACUUM_OFF
            self._queue_retreat(x)
            return action

        if block_y < target_y - target_y_tol and robot.arm_joint < place_arm - ARM_TOL:
            action[3] = float(np.clip(place_arm - robot.arm_joint, -DARM_LIM, DARM_LIM))
            return action

        transport_y = shelf.y1 - TRANSPORT_Y_MARGIN
        if block_y > transport_y + POS_TOL and not at_target_x:
            target_robot_x = robot.x
            target_robot_y = robot.y + (transport_y - block_y)
        elif not at_target_x:
            target_robot_x = robot.x + (target_x - block_x)
            target_robot_y = robot.y
        else:
            target_robot_x = robot.x
            target_robot_y = robot.y + (target_y - block_y)

        dx = target_robot_x - robot.x
        dy = target_robot_y - robot.y
        if abs(dx) > POS_TOL or abs(dy) > POS_TOL:
            action[0] = float(np.clip(dx, -DX_LIM, DX_LIM))
            action[1] = float(np.clip(dy, -DY_LIM, DY_LIM))
            return action

        action[0] = float(np.clip(target_x - block_x, -DX_LIM, DX_LIM))
        action[1] = float(np.clip(target_y - block_y, -DY_LIM, DY_LIM))
        if at_target_x and deep_enough:
            action[4] = VACUUM_OFF
            self._queue_retreat(x)
        return action

    def step(self, x: NDArray) -> NDArray:
        observed_held = held_block_name(x)
        robot = extract_robot(x)
        if observed_held is None and robot.vacuum <= VACUUM_OFF and not self._actions:
            self._active_block = None
        self._sync_phase(x)
        held_name = None
        if observed_held is not None and observed_held == self._active_block:
            held_name = observed_held
        if held_name is not None:
            self._actions.clear()
            return self._place_action(x, held_name)
        if not self._actions:
            self._generate_pick_plan(x)
        return self._actions.popleft()
