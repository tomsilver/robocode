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

# Top-down slot y positions (block center y inside shelf).
# Slot 0 = highest, slot 2 = lowest. 0.080 center-to-center spacing → 0.040 physical gap.
SHELF_TOP_DOWN_SLOTS = [2.930, 2.850, 2.770]

# Place approach y: robot base y during placement (closer to shelf = less arm extension needed)
SHELF_APPROACH_Y_PLACE = 2.150  # must be < SHELF_INNER_Y_MIN - ROBOT_BASE_RADIUS = 2.425

# Navigation clearance
NAV_SHELF_MARGIN = 0.25  # keep robot this far below shelf floor when navigating


# How close robot base center can get to a block center (robot_radius + block_half_extent)
BIRRT_BLOCK_CLEARANCE = ROBOT_BASE_RADIUS + 0.16  # robot_radius + block_half_diag + margin ≈ 0.36


def _make_birrt(primitives, obs, block_names_to_avoid: list[str], rng,
                extra_clearance: float = 0.0,
                carried_block_info: tuple | None = None):
    """Create a BiRRT planner for 2D base navigation.

    carried_block_info: (perp_offset, arm_offset, arm_joint) — when set, also checks
    that the carried block (held at theta=pi/2) doesn't collide with avoided blocks.
    Carried block center = (robot.x - perp_offset, robot.y + arm_joint + arm_offset).
    """
    from obs_helpers import extract_rect as _er
    blocks_to_avoid = [(_er(obs, bn), bn) for bn in block_names_to_avoid]
    clearance = BIRRT_BLOCK_CLEARANCE + extra_clearance
    # carried block half-extents (block is horizontal: width=0.28, height=0.04)
    CB_HALF_X = 0.14 + 0.05  # carried block x-half + margin
    CB_HALF_Y = 0.14 + 0.05  # use x-half for y too (conservative; block might be rotated)

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
            # Also check carried block (if any) doesn't collide with this block
            if carried_block_info is not None:
                perp_off, arm_off, arm_jt = carried_block_info
                cbx = x - perp_off   # carried block center x at theta=pi/2
                cby = y + arm_jt + arm_off  # carried block center y
                # Conservative rectangle overlap check
                if abs(cbx - bx) < CB_HALF_X and abs(cby - by) < CB_HALF_Y:
                    return True
        return False

    def distance_fn(s1, s2):
        return float(np.linalg.norm(s2 - s1))

    BiRRT = primitives['BiRRT']
    birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng,
                  num_attempts=20, num_iters=5000, smooth_amt=50)
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

    def __init__(self, block_name: str, primitives: dict, place_slot: int = 0,
                 allow_in_shelf: bool = False, nav_ignore: list | None = None):
        self.block_name = block_name
        self._primitives = primitives
        self._allow_in_shelf = allow_in_shelf
        self._nav_ignore: list[str] = nav_ignore or []  # blocks to skip in nav collision
        self._actions: deque[NDArray] = deque()
        self._phase = 0
        self._rng = np.random.default_rng(42 + place_slot)
        self._approach_theta: float = math.pi / 2  # direction arm points to reach block

    def initializable(self, obs: NDArray) -> bool:
        if not self._allow_in_shelf and is_block_in_shelf(obs, self.block_name):
            return False
        return extract_robot(obs).vacuum < 0.5

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
            # Avoid ALL blocks including target (target block is approached from side, not ran into)
            all_block_names = ['block0', 'block1', 'block2']
            other_block_names = [n for n in all_block_names if n != self.block_name]
            other_blocks = [extract_rect(obs, n) for n in other_block_names
                            if n not in self._nav_ignore]
            goal_x, goal_y, self._approach_theta = _find_approach(block, other_blocks)
            # BiRRT avoids all blocks except nav_ignore (temp-placed blocks near robot)
            nav_avoid = [n for n in all_block_names if n not in self._nav_ignore]
            birrt = _make_birrt(self._primitives, obs, nav_avoid, self._rng)
            start = np.array([robot.x, robot.y])
            goal = np.array([goal_x, goal_y])
            path = birrt.query(start, goal)
            if path is None or len(path) < 1:
                path = [goal]
            self._actions = path_to_actions(path, robot, VAC_OFF)
            if not self._actions:
                self._actions = hold_action(VAC_OFF, n=1)

        elif self._phase == 2:
            # Rotate to face block — use positive (CCW) direction to avoid sweeping arm through blocks
            diff = (self._approach_theta - robot.theta) % (2 * math.pi)  # positive ccw
            acts: deque = deque()
            remaining = diff
            while abs(remaining) > 0.02:
                s = min(DTH_LIM, remaining)
                acts.append(np.array([0.0, 0.0, s, 0.0, VAC_OFF], dtype=np.float32))
                remaining -= s
                if abs(remaining) < 1e-9:
                    break
            self._actions = acts
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
            next_phase = min(self._phase + 1, self.N_PHASES - 1)
            # If stuck at phase 4 (arm can't reach block), retry from navigation
            if self._phase == 4 and next_phase == 4:
                robot = extract_robot(obs)
                block = extract_rect(obs, self.block_name)
                if not gripper_near_block(robot, block, tol=0.22):
                    # Retry: go back to navigation
                    self._phase = 1
                    self._plan_phase(obs)
                    if self._actions:
                        return self._actions.popleft()
                    return np.array([0.0, 0.0, 0.0, 0.0, VAC_ON], dtype=np.float32)
            self._phase = next_phase
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
        # Use fixed top-down slot target
        slot = min(self._slot_index, len(SHELF_TOP_DOWN_SLOTS) - 1)
        self._target_y = SHELF_TOP_DOWN_SLOTS[slot]
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
            goal_y = SHELF_APPROACH_Y_PLACE
            other_blocks = [n for n in ['block0','block1','block2'] if n != self.block_name]
            # Pass carried block info so BiRRT avoids collisions between carried block and env blocks
            carried = (self._perp_offset, self._arm_offset, ARM_MIN)
            birrt = _make_birrt(self._primitives, obs, other_blocks, self._rng,
                                carried_block_info=carried)
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
            # Always extend to target regardless of whether block has entered shelf
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
        if not self._actions:
            self._phase = min(self._phase + 1, self.N_PHASES - 1)
            self._plan_phase(obs)
        if self._actions:
            return self._actions.popleft()
        return np.array([0.0, 0.0, 0.0, 0.0, VAC_OFF], dtype=np.float32)


class MoveBlockToTemp(Behavior):
    """Carry held block to a temp floor location and release it.

    Phase 0: retract arm, 1: rotate to THETA_UP,
    2: navigate to temp position (vacuum on), 3: release vacuum.
    """

    N_PHASES = 4

    def __init__(self, block_name: str, primitives: dict,
                 temp_x: float = 2.5, temp_y: float = 1.0):
        self.block_name = block_name
        self._primitives = primitives
        self._temp_x = temp_x
        self._temp_y = temp_y
        self._actions: deque[NDArray] = deque()
        self._phase = 0
        self._rng = np.random.default_rng(99)

    def initializable(self, obs: NDArray) -> bool:
        robot = extract_robot(obs)
        block = extract_rect(obs, self.block_name)
        return robot.vacuum > 0.5 and gripper_near_block(robot, block, tol=0.30)

    def terminated(self, obs: NDArray) -> bool:
        return extract_robot(obs).vacuum < 0.5

    def reset(self, obs: NDArray) -> None:
        self._phase = 0
        self._actions = deque()
        self._plan_phase(obs)

    def _plan_phase(self, obs: NDArray) -> None:
        robot = extract_robot(obs)

        if self._phase == 0:
            self._actions = arm_actions(robot.arm_joint, ARM_MIN, VAC_ON)
            if not self._actions:
                self._phase = 1
                self._plan_phase(obs)

        elif self._phase == 1:
            self._actions = rotate_actions(robot.theta, THETA_UP, VAC_ON)
            if not self._actions:
                self._phase = 2
                self._plan_phase(obs)

        elif self._phase == 2:
            other_blocks = [n for n in ['block0', 'block1', 'block2'] if n != self.block_name]
            birrt = _make_birrt(self._primitives, obs, other_blocks, self._rng)
            start = np.array([robot.x, robot.y])
            goal = np.array([self._temp_x, self._temp_y])
            path = birrt.query(start, goal)
            if path is None or len(path) < 1:
                path = [goal]
            self._actions = path_to_actions(path, robot, VAC_ON)
            if not self._actions:
                self._actions = hold_action(VAC_ON, n=1)

        elif self._phase == 3:
            self._actions = hold_action(VAC_OFF, n=10)

    def step(self, obs: NDArray) -> NDArray:
        if not self._actions:
            self._phase = min(self._phase + 1, self.N_PHASES - 1)
            self._plan_phase(obs)
        if self._actions:
            return self._actions.popleft()
        return np.array([0.0, 0.0, 0.0, 0.0, VAC_OFF], dtype=np.float32)
