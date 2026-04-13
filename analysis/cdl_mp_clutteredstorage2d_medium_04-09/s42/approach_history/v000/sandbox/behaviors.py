"""Behavior classes for ClutteredStorage2D."""

from __future__ import annotations
import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from behavior import Behavior
from obs_helpers import (
    extract_robot, extract_rect, is_block_in_shelf, get_gripper_pos, gripper_near_block,
    blocks_outside_shelf, SHELF_CENTER_X, SHELF_BLOCK_Y_MIN, SHELF_BLOCK_Y_MAX,
    WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, WORLD_MAX_Y,
    SHELF_INNER_Y_MIN, ROBOT_BASE_RADIUS, ARM_MIN, ARM_MAX, block_center,
)
from act_helpers import (
    DX_LIM, DY_LIM, DTH_LIM, DARM_LIM,
    PICK_ARM_DIST, PLACE_ARM_JOINT, SHELF_APPROACH_Y, RETRACT_ARM,
    THETA_UP, VAC_ON, VAC_OFF,
    path_to_actions, rotate_actions, arm_actions, hold_action, angle_diff,
    connecting_waypoints, waypoints_to_actions,
)
from obs_helpers import RobotPose

# Target GRIPPER y for placement - block center = gripper pos, must be in [2.645, 2.98]
# Use center of valid range with some spread for stacking
PLACE_Y_SLOTS = [2.72, 2.78, 2.84]
# Max steps before replanning
MAX_PHASE_STEPS = 300


def _make_birrt(primitives, obs, block_names_to_avoid: list[str], rng):
    """Create a BiRRT planner for 2D base navigation."""
    from obs_helpers import extract_rect as _er

    blocks_to_avoid = [_er(obs, bn) for bn in block_names_to_avoid]

    def sample_fn(state):
        x = rng.uniform(WORLD_MIN_X + ROBOT_BASE_RADIUS + 0.01,
                        WORLD_MAX_X - ROBOT_BASE_RADIUS - 0.01)
        y = rng.uniform(WORLD_MIN_Y + ROBOT_BASE_RADIUS + 0.01,
                        SHELF_INNER_Y_MIN - ROBOT_BASE_RADIUS - 0.05)
        return np.array([x, y], dtype=np.float64)

    def extend_fn(s1, s2):
        diff = s2 - s1
        dist = np.linalg.norm(diff)
        n_steps = max(1, int(math.ceil(dist / (DX_LIM * 0.9))))
        pts = []
        for i in range(1, n_steps + 1):
            pts.append(s1 + diff * (i / n_steps))
        return pts

    def collision_fn(state):
        x, y = float(state[0]), float(state[1])
        r = ROBOT_BASE_RADIUS
        if x < WORLD_MIN_X + r or x > WORLD_MAX_X - r:
            return True
        if y < WORLD_MIN_Y + r or y > SHELF_INNER_Y_MIN - r:
            return True
        # Avoid running over blocks (they'd collide anyway)
        for b in blocks_to_avoid:
            if math.sqrt((x - b.x)**2 + (y - b.y)**2) < r + 0.18:
                return True
        return False

    def distance_fn(s1, s2):
        return float(np.linalg.norm(s2 - s1))

    BiRRT = primitives['BiRRT']
    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng,
                  num_attempts=10, num_iters=2000, smooth_amt=50)
    return birrt


class PickBlock(Behavior):
    """Navigate to a block, extend arm, activate vacuum to grasp it."""

    def __init__(self, block_name: str, primitives: dict, place_slot: int = 0):
        self.block_name = block_name
        self._primitives = primitives
        self._place_slot = place_slot
        self._actions: deque[NDArray] = deque()
        self._phase = 0
        self._rng = np.random.default_rng(42)

    def initializable(self, obs: NDArray) -> bool:
        return not is_block_in_shelf(obs, self.block_name) and extract_robot(obs).vacuum < 0.5

    def terminated(self, obs: NDArray) -> bool:
        robot = extract_robot(obs)
        block = extract_rect(obs, self.block_name)
        return robot.vacuum > 0.5 and gripper_near_block(robot, block, tol=0.12)

    def reset(self, obs: NDArray) -> None:
        self._phase = 0
        self._actions = deque()
        self._plan_phase(obs)

    def _plan_phase(self, obs: NDArray) -> None:
        robot = extract_robot(obs)
        block = extract_rect(obs, self.block_name)

        if self._phase == 0:
            # Navigate base to approach position below block CENTER
            bx, by = block_center(block)
            goal_x = float(np.clip(bx, WORLD_MIN_X + ROBOT_BASE_RADIUS + 0.01,
                                   WORLD_MAX_X - ROBOT_BASE_RADIUS - 0.01))
            goal_y = float(np.clip(by - PICK_ARM_DIST,
                                   WORLD_MIN_Y + ROBOT_BASE_RADIUS + 0.01,
                                   SHELF_INNER_Y_MIN - ROBOT_BASE_RADIUS - 0.1))

            # First retract arm to min
            if robot.arm_joint > ARM_MIN + 0.05:
                self._actions = arm_actions(robot.arm_joint, ARM_MIN, VAC_OFF)
                return

            # Use BiRRT to plan base path
            other_blocks = [n for n in ['block0','block1','block2'] if n != self.block_name]
            birrt = _make_birrt(self._primitives, obs, other_blocks, self._rng)
            start = np.array([robot.x, robot.y])
            goal = np.array([goal_x, goal_y])
            path = birrt.query(start, goal)
            if path is None or len(path) < 1:
                # Fallback: direct movement
                path = [goal]
            self._actions = path_to_actions(path, robot, VAC_OFF)

        elif self._phase == 1:
            # Rotate to face block center
            bx, by = block_center(block)
            target_theta = math.atan2(by - robot.y, bx - robot.x)
            self._actions = rotate_actions(robot.theta, target_theta, VAC_OFF)
            if not self._actions:
                self._phase = 2
                self._plan_phase(obs)

        elif self._phase == 2:
            # Extend arm to reach block center
            bx, by = block_center(block)
            dist = math.sqrt((robot.x - bx)**2 + (robot.y - by)**2)
            target_arm = float(np.clip(dist, ARM_MIN, ARM_MAX))
            self._actions = arm_actions(robot.arm_joint, target_arm, VAC_OFF)

        elif self._phase == 3:
            # Activate vacuum for several steps
            self._actions = hold_action(VAC_ON, n=10)

    def step(self, obs: NDArray) -> NDArray:
        if not self._actions:
            self._phase += 1
            if self._phase > 3:
                self._phase = 3
            self._plan_phase(obs)
        if self._actions:
            return self._actions.popleft()
        return np.array([0.0, 0.0, 0.0, 0.0, VAC_ON], dtype=np.float32)


class PlaceInShelf(Behavior):
    """Carry block to shelf and release it."""

    def __init__(self, block_name: str, primitives: dict, slot_index: int = 0):
        self.block_name = block_name
        self._primitives = primitives
        self._slot_index = slot_index
        self._actions: deque[NDArray] = deque()
        self._phase = 0
        self._rng = np.random.default_rng(43)

    def initializable(self, obs: NDArray) -> bool:
        robot = extract_robot(obs)
        block = extract_rect(obs, self.block_name)
        return robot.vacuum > 0.5 and gripper_near_block(robot, block, tol=0.15)

    def terminated(self, obs: NDArray) -> bool:
        return is_block_in_shelf(obs, self.block_name)

    def reset(self, obs: NDArray) -> None:
        self._phase = 0
        self._actions = deque()
        self._plan_phase(obs)

    def _target_place_y(self) -> float:
        idx = self._slot_index % len(PLACE_Y_SLOTS)
        return PLACE_Y_SLOTS[idx]

    def _plan_phase(self, obs: NDArray) -> None:
        robot = extract_robot(obs)

        if self._phase == 0:
            # Navigate base to shelf approach position
            goal_x = SHELF_CENTER_X
            goal_y = SHELF_APPROACH_Y

            other_blocks = [n for n in ['block0','block1','block2'] if n != self.block_name]
            birrt = _make_birrt(self._primitives, obs, other_blocks, self._rng)
            start = np.array([robot.x, robot.y])
            goal = np.array([goal_x, goal_y])
            path = birrt.query(start, goal)
            if path is None or len(path) < 1:
                path = [goal]
            # Keep vacuum on while navigating
            self._actions = path_to_actions(path, robot, VAC_ON)

        elif self._phase == 1:
            # Rotate to face up (theta = pi/2)
            self._actions = rotate_actions(robot.theta, THETA_UP, VAC_ON)
            if not self._actions:
                self._phase = 2
                self._plan_phase(obs)

        elif self._phase == 2:
            # Extend arm to place block inside shelf
            target_y = self._target_place_y()
            target_arm = float(np.clip(target_y - robot.y, ARM_MIN, ARM_MAX))
            self._actions = arm_actions(robot.arm_joint, target_arm, VAC_ON)

        elif self._phase == 3:
            # Release vacuum
            self._actions = hold_action(VAC_OFF, n=5)

        elif self._phase == 4:
            # Retract arm
            self._actions = arm_actions(robot.arm_joint, ARM_MIN, VAC_OFF)

    def step(self, obs: NDArray) -> NDArray:
        if not self._actions:
            self._phase += 1
            if self._phase > 4:
                self._phase = 4
            self._plan_phase(obs)
        if self._actions:
            return self._actions.popleft()
        return np.array([0.0, 0.0, 0.0, 0.0, VAC_OFF], dtype=np.float32)
