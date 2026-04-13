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
    SHELF_INNER_Y_MIN, SHELF_INNER_Y_MAX, ROBOT_BASE_RADIUS, ARM_MIN, ARM_MAX, block_center,
    get_shelf_inner_rect,
)
from act_helpers import (
    DX_LIM, DY_LIM, DTH_LIM, DARM_LIM,
    PICK_ARM_DIST, PLACE_ARM_JOINT, SHELF_APPROACH_Y, RETRACT_ARM,
    THETA_UP, VAC_ON, VAC_OFF,
    path_to_actions, rotate_actions, arm_actions, hold_action, angle_diff,
)
from obs_helpers import RobotPose

# Target BLOCK CENTER y positions inside shelf (stacked)
PLACE_BLOCK_Y_SLOTS = [2.72, 2.78, 2.84]

# Navigation clearance
NAV_SHELF_MARGIN = 0.25  # keep robot this far below shelf floor when navigating


# How close robot base center can get to a block center (robot_radius + block_half_extent)
BIRRT_BLOCK_CLEARANCE = ROBOT_BASE_RADIUS + 0.12  # conservative: 0.32


def _make_birrt(primitives, obs, block_names_to_avoid: list[str], rng,
                extra_clearance: float = 0.0):
    """Create a BiRRT planner for 2D base navigation."""
    from obs_helpers import extract_rect as _er
    blocks_to_avoid = [(_er(obs, bn), bn) for bn in block_names_to_avoid]
    clearance = BIRRT_BLOCK_CLEARANCE + extra_clearance

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
        return [s1 + diff * (i / n_steps) for i in range(1, n_steps + 1)]

    def collision_fn(state):
        x, y = float(state[0]), float(state[1])
        r = ROBOT_BASE_RADIUS
        if x < WORLD_MIN_X + r or x > WORLD_MAX_X - r:
            return True
        if y < WORLD_MIN_Y + r or y > SHELF_INNER_Y_MIN - r:
            return True
        for b, _ in blocks_to_avoid:
            bx, by = block_center(b)
            if math.sqrt((x - bx)**2 + (y - by)**2) < clearance:
                return True
        return False

    def distance_fn(s1, s2):
        return float(np.linalg.norm(s2 - s1))

    BiRRT = primitives['BiRRT']
    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng,
                  num_attempts=10, num_iters=2000, smooth_amt=50)
    return birrt


def _block_final_theta(block_theta: float, robot_theta_pick: float) -> float:
    """Compute block world-frame theta after robot rotates from theta_pick to pi/2."""
    return block_theta + (math.pi / 2 - robot_theta_pick)


def _block_y_half(block, robot_theta_pick: float) -> float:
    """y half-extent of block when robot is at theta_pick and then rotates to pi/2."""
    ft = _block_final_theta(block.theta, robot_theta_pick)
    a, b = block.width / 2, block.height / 2
    return a * abs(math.sin(ft)) + b * abs(math.cos(ft))


def _horizontal_approach_thetas(block) -> list[float]:
    """Return pick thetas that make block horizontal (y_half≈h/2) after rotating to pi/2."""
    B = block.theta
    # block_final_theta = B + (pi/2 - theta_pick) should be 0 or pi:
    # 0:  theta_pick = B + pi/2      (may be large)
    # pi: theta_pick = B + pi/2 - pi = B - pi/2
    t1 = B - math.pi / 2  # preferred (usually smaller rotation needed)
    t2 = B + math.pi / 2
    # Normalize to (-pi, pi]
    def norm(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
    return [norm(t1), norm(t2)]


def _find_approach(block, other_blocks, arm_dist=PICK_ARM_DIST) -> tuple[float, float, float]:
    """Find approach position and arm theta that avoids other blocks.
    Prefers angles that result in block being horizontal when placed in shelf."""
    bx, by = block_center(block)
    gy_max = SHELF_INNER_Y_MIN - ROBOT_BASE_RADIUS - 0.15
    gy_min = WORLD_MIN_Y + ROBOT_BASE_RADIUS + 0.05
    gx_min = WORLD_MIN_X + ROBOT_BASE_RADIUS + 0.05
    gx_max = WORLD_MAX_X - ROBOT_BASE_RADIUS - 0.05
    # Try horizontal-placement angles first, then fallback angles
    preferred = _horizontal_approach_thetas(block)
    fallback = [math.pi/2, 0.0, math.pi, -math.pi/2,
                math.pi/4, 3*math.pi/4, -math.pi/4, -3*math.pi/4]
    candidates = preferred + [t for t in fallback if t not in preferred]
    for theta in candidates:
        # Robot is behind the block (arm points from robot toward block at angle theta)
        gx = float(np.clip(bx - math.cos(theta) * arm_dist, gx_min, gx_max))
        gy = float(np.clip(by - math.sin(theta) * arm_dist, gy_min, gy_max))
        # Validate: from clipped position, arm in direction theta reaches block
        reach_x = gx + arm_dist * math.cos(theta)
        reach_y = gy + arm_dist * math.sin(theta)
        if abs(reach_x - bx) > 0.15 or abs(reach_y - by) > 0.15:
            continue  # clipping moved robot too far; this theta won't work
        # Check collision with other blocks
        ok = True
        for b in other_blocks:
            cx, cy = block_center(b)
            if math.sqrt((gx - cx)**2 + (gy - cy)**2) < ROBOT_BASE_RADIUS + 0.15:
                ok = False
                break
        if ok:
            return gx, gy, theta
    # Fallback: approach from below
    gy = float(np.clip(by - arm_dist, gy_min, gy_max))
    return bx, gy, math.pi/2


def _find_shelf_y_slot(obs: NDArray, block_name: str, final_y_half: float) -> float:
    """Target floor-level placement: just above shelf inner floor."""
    # Place block just inside the shelf floor — arm can reliably reach this
    # with DARM_LIM/2 steps, and blocks already in shelf sit higher (no collision)
    target = SHELF_INNER_Y_MIN + final_y_half + 0.003
    return float(np.clip(target, SHELF_INNER_Y_MIN + final_y_half + 0.003,
                         SHELF_INNER_Y_MAX - final_y_half - 0.01))


class PickBlock(Behavior):
    """Navigate to a block and grasp it with vacuum."""

    # Phase 0: retract arm, 1: navigate, 2: rotate, 3: vacuum_on, 4: extend_arm
    N_PHASES = 5

    def __init__(self, block_name: str, primitives: dict, place_slot: int = 0):
        self.block_name = block_name
        self._primitives = primitives
        self._actions: deque[NDArray] = deque()
        self._phase = 0
        self._rng = np.random.default_rng(42 + place_slot)
        self._approach_theta: float = math.pi / 2  # direction arm points to reach block

    def initializable(self, obs: NDArray) -> bool:
        return not is_block_in_shelf(obs, self.block_name) and extract_robot(obs).vacuum < 0.5

    def terminated(self, obs: NDArray) -> bool:
        robot = extract_robot(obs)
        # Require arm extended past suction-attachment threshold (ARM_MIN+0.22=0.42)
        # to ensure suction zone has reached block before we declare success
        if robot.vacuum < 0.5 or self._phase < 4 or robot.arm_joint < ARM_MIN + 0.22:
            return False
        block = extract_rect(obs, self.block_name)
        return gripper_near_block(robot, block, tol=0.22)

    def reset(self, obs: NDArray) -> None:
        self._phase = 0
        self._actions = deque()
        self._plan_phase(obs)

    def _plan_phase(self, obs: NDArray) -> None:
        robot = extract_robot(obs)
        block = extract_rect(obs, self.block_name)

        if self._phase == 0:
            # Retract arm (or skip if already retracted)
            if robot.arm_joint > ARM_MIN + 0.02:
                self._actions = arm_actions(robot.arm_joint, ARM_MIN, VAC_OFF)
            else:
                # Already retracted → skip straight to navigate
                self._phase = 1
                self._plan_phase(obs)

        elif self._phase == 1:
            # Navigate base to approach position
            other_block_names = [n for n in ['block0','block1','block2'] if n != self.block_name]
            other_blocks = [extract_rect(obs, n) for n in other_block_names]
            goal_x, goal_y, self._approach_theta = _find_approach(block, other_blocks)
            birrt = _make_birrt(self._primitives, obs, other_block_names, self._rng)
            start = np.array([robot.x, robot.y])
            goal = np.array([goal_x, goal_y])
            path = birrt.query(start, goal)
            if path is None or len(path) < 1:
                path = [goal]
            self._actions = path_to_actions(path, robot, VAC_OFF)
            if not self._actions:
                self._actions = hold_action(VAC_OFF, n=1)

        elif self._phase == 2:
            # Rotate to face block
            self._actions = rotate_actions(robot.theta, self._approach_theta, VAC_OFF)
            if not self._actions:
                self._phase = 3
                self._plan_phase(obs)

        elif self._phase == 3:
            # Activate vacuum before arm extension
            self._actions = hold_action(VAC_ON, n=3)

        elif self._phase == 4:
            # Extend arm so suction zone center reaches block center (dist - 0.030)
            # This stops arm right where overlap is maximal → ensures attachment
            bx, by = block_center(block)
            dist = math.sqrt((robot.x - bx)**2 + (robot.y - by)**2)
            target_arm = float(np.clip(dist - 0.030, ARM_MIN + 0.22, ARM_MAX))
            self._actions = arm_actions(robot.arm_joint, target_arm, VAC_ON)
            if not self._actions:
                self._actions = hold_action(VAC_ON, n=3)

    def step(self, obs: NDArray) -> NDArray:
        if not self._actions:
            self._phase = min(self._phase + 1, self.N_PHASES - 1)
            self._plan_phase(obs)
        if self._actions:
            return self._actions.popleft()
        return np.array([0.0, 0.0, 0.0, 0.0, VAC_ON], dtype=np.float32)


class PlaceInShelf(Behavior):
    """Carry attached block to shelf and release it.

    Phase order: 0=retract arm, 1=rotate to pi/2 at pick position,
                 2=navigate to shelf, 3=extend arm to place,
                 4=release vacuum, 5=retract arm.
    """

    N_PHASES = 6

    def __init__(self, block_name: str, primitives: dict, slot_index: int = 0):
        self.block_name = block_name
        self._primitives = primitives
        self._slot_index = slot_index
        self._actions: deque[NDArray] = deque()
        self._phase = 0
        self._rng = np.random.default_rng(43 + slot_index)
        self._arm_offset: float = 0.15
        self._perp_offset: float = 0.0
        self._target_y: float = 2.78  # block center y in shelf

    def initializable(self, obs: NDArray) -> bool:
        robot = extract_robot(obs)
        block = extract_rect(obs, self.block_name)
        return robot.vacuum > 0.5 and gripper_near_block(robot, block, tol=0.22)

    def terminated(self, obs: NDArray) -> bool:
        # Require vacuum off so next behavior's arm retraction doesn't drag block out
        robot = extract_robot(obs)
        return is_block_in_shelf(obs, self.block_name) and robot.vacuum < 0.5

    def reset(self, obs: NDArray) -> None:
        self._phase = 0
        self._actions = deque()
        robot = extract_robot(obs)
        block = extract_rect(obs, self.block_name)
        gx, gy = get_gripper_pos(robot)
        cx, cy = block_center(block)
        dx, dy = cx - gx, cy - gy
        arm_cos = math.cos(robot.theta)
        arm_sin = math.sin(robot.theta)
        self._arm_offset = dx * arm_cos + dy * arm_sin
        self._perp_offset = -dx * arm_sin + dy * arm_cos
        # Compute final block y_half (at theta=pi/2 after rotation from current robot.theta)
        final_block_theta = block.theta + (THETA_UP - robot.theta)
        a, b = block.width / 2, block.height / 2
        final_y_half = a * abs(math.sin(final_block_theta)) + b * abs(math.cos(final_block_theta))
        # Use dynamic slot: place above all existing shelf blocks
        self._target_y = _find_shelf_y_slot(obs, self.block_name, final_y_half)
        self._plan_phase(obs)

    def _plan_phase(self, obs: NDArray) -> None:
        robot = extract_robot(obs)

        if self._phase == 0:
            # Retract arm fully so carried block clears shelf blocks during navigation
            target_arm = ARM_MIN  # 0.200 keeps block at robot.y+0.350 max
            self._actions = arm_actions(robot.arm_joint, target_arm, VAC_ON)
            if not self._actions:
                self._phase = 1
                self._plan_phase(obs)

        elif self._phase == 1:
            # Rotate to theta=pi/2 at pick position (far from shelf → no wall collision)
            self._actions = rotate_actions(robot.theta, THETA_UP, VAC_ON)
            if not self._actions:
                self._phase = 2
                self._plan_phase(obs)

        elif self._phase == 2:
            # Navigate to (SHELF_CENTER_X + perp_offset, SHELF_APPROACH_Y)
            # With theta=pi/2: block.x = robot.x - perp_offset → robot.x = shelf_center_x + perp_offset
            sx_min, sx_max, _, _ = get_shelf_inner_rect(obs)
            shelf_center_x = (sx_min + sx_max) / 2
            goal_x = float(np.clip(shelf_center_x + self._perp_offset,
                                   WORLD_MIN_X + ROBOT_BASE_RADIUS + 0.01,
                                   WORLD_MAX_X - ROBOT_BASE_RADIUS - 0.01))
            goal_y = SHELF_APPROACH_Y
            other_blocks = [n for n in ['block0','block1','block2'] if n != self.block_name]
            birrt = _make_birrt(self._primitives, obs, other_blocks, self._rng)
            start = np.array([robot.x, robot.y])
            goal_arr = np.array([goal_x, goal_y])
            path = birrt.query(start, goal_arr)
            if path is None or len(path) < 1:
                path = [goal_arr]
            self._actions = path_to_actions(path, robot, VAC_ON)
            if not self._actions:
                self._actions = hold_action(VAC_ON, n=1)

        elif self._phase == 3:
            # Extend arm so block center is at self._target_y
            # Use DARM_LIM/2 steps to avoid jumping over the narrow shelf floor opening
            if is_block_in_shelf(obs, self.block_name):
                self._actions = hold_action(VAC_ON, n=1)
            else:
                target_arm = float(np.clip(
                    self._target_y - self._arm_offset - robot.y, ARM_MIN, ARM_MAX))
                fine_lim = DARM_LIM / 2  # 0.050
                remaining = target_arm - robot.arm_joint
                acts: deque = deque()
                while abs(remaining) > 0.005:
                    s = max(-fine_lim, min(fine_lim, remaining))
                    acts.append(np.array([0.0, 0.0, 0.0, s, VAC_ON], dtype=np.float32))
                    remaining -= s
                    if abs(remaining) < 1e-9:
                        break
                self._actions = acts if acts else hold_action(VAC_ON, n=3)

        elif self._phase == 4:
            # Release vacuum
            self._actions = hold_action(VAC_OFF, n=10)

        elif self._phase == 5:
            # Retract arm
            if robot.arm_joint > ARM_MIN + 0.01:
                self._actions = arm_actions(robot.arm_joint, ARM_MIN, VAC_OFF)
            else:
                self._actions = hold_action(VAC_OFF, n=1)

    def step(self, obs: NDArray) -> NDArray:
        # Early-exit phase 3: block is already in shelf → release vacuum
        if self._phase == 3 and is_block_in_shelf(obs, self.block_name):
            self._actions = deque()
        if not self._actions:
            self._phase = min(self._phase + 1, self.N_PHASES - 1)
            self._plan_phase(obs)
        if self._actions:
            return self._actions.popleft()
        return np.array([0.0, 0.0, 0.0, 0.0, VAC_OFF], dtype=np.float32)
