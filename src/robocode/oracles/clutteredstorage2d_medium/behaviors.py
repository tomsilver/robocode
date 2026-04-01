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
    inflate_block_radius,
    path_length,
    plan_base_path,
    plan_holding_base_path,
    segment_collision_free,
    waypoints_to_actions,
)
from robocode.oracles.clutteredstorage2d_medium.obs_helpers import (
    APPROACH_MARGIN,
    RobotPose,
    WORLD_MAX_X,
    WORLD_MAX_Y,
    WORLD_MIN_X,
    WORLD_MIN_Y,
    all_blocks_inside_shelf,
    extract_block,
    extract_robot,
    extract_shelf,
    farthest_free_staging_center,
    held_block_name,
    inside_blocks,
    is_block_inside_shelf,
    next_free_slot_center,
    next_free_staging_center,
    outside_blocks,
    pick_base_pose_candidates,
    slot_centers,
    wrap_angle,
)
from robocode.primitives.behavior import Behavior

ANGLE_TOL = 0.05
ARM_TOL = 0.02
POS_TOL = 0.02
INSERT_X_TOL = 0.005
DEEP_PLACE_Y_TOL = 0.03
COMPACT_Y_TOL = 0.02
TRANSPORT_Y_MARGIN = 0.08
PREINSERT_Y_MARGIN = 0.25
ROTATION_STAGE_MARGIN = 0.18
PRE_PICK_RING_MARGIN = 0.12
SAFE_VACUUM = 1.0
VACUUM_OFF = 0.0
CARRY_ARM_FRACTION = 0.35
PLACE_ARM_FRACTION = 1.0
STAGING_RELEASE_ARM_FRACTION = 0.55
NOOP_ACTION = np.zeros(5, dtype=np.float32)
UP = np.pi / 2
SHELF_TARGET = "shelf"
STAGING_TARGET = "staging"
STUCK_WINDOW = 8
STUCK_POSITION_EPS = 1e-3
STUCK_COMMAND_EPS = 1e-3
INSIDE_PUSH_STUCK_WINDOW = 6
INSIDE_PUSH_STUCK_POSITION_EPS = 5e-3
INSIDE_PUSH_COMMAND_EPS = 1e-2
HOLD_LOSS_PATIENCE = 1
HOLD_RECOVERY_MOVE_EPS = 0.01


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
        self._holding_actions: deque[NDArray] = deque()
        self._phase = "compact"
        self._target_kind = SHELF_TARGET
        self._target_center: tuple[float, float] | None = None
        self._active_block: str | None = None
        self._chosen_pick_pose: tuple[float, float, float] | None = None
        self._planned_path_len: float | None = None
        self._staging_release_active = False
        self._recent_positions: deque[tuple[float, float]] = deque(maxlen=STUCK_WINDOW)
        self._recent_base_commands: deque[float] = deque(maxlen=STUCK_WINDOW)
        self._recent_inside_push_block_centers: deque[tuple[float, float]] = deque(
            maxlen=INSIDE_PUSH_STUCK_WINDOW
        )
        self._recent_inside_push_commands: deque[float] = deque(
            maxlen=INSIDE_PUSH_STUCK_WINDOW
        )
        self._hold_loss_steps = 0
        self._last_active_block_center: tuple[float, float] | None = None

    def reset(self, x: NDArray) -> None:
        self._phase = "compact"
        self._holding_actions = deque()
        self._target_kind = SHELF_TARGET
        self._target_center = None
        self._active_block = None
        self._chosen_pick_pose = None
        self._planned_path_len = None
        self._staging_release_active = False
        self._clear_motion_monitor()
        self._clear_inside_push_monitor()
        self._hold_loss_steps = 0
        self._last_active_block_center = None
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

    def _clear_motion_monitor(self) -> None:
        self._recent_positions.clear()
        self._recent_base_commands.clear()

    def _clear_inside_push_monitor(self) -> None:
        self._recent_inside_push_block_centers.clear()
        self._recent_inside_push_commands.clear()

    def _path_is_stuck(self) -> bool:
        if (
            len(self._recent_positions) < STUCK_WINDOW
            or len(self._recent_base_commands) < STUCK_WINDOW
        ):
            return False
        if not all(command > STUCK_COMMAND_EPS for command in self._recent_base_commands):
            return False
        first_x, first_y = self._recent_positions[0]
        max_displacement = max(
            np.hypot(x - first_x, y - first_y) for x, y in self._recent_positions
        )
        return bool(max_displacement < STUCK_POSITION_EPS)

    def _inside_push_is_stuck(
        self,
        x: NDArray,
        block_name: str,
    ) -> bool:
        if not is_block_inside_shelf(x, block_name):
            return False
        if (
            len(self._recent_inside_push_block_centers) < INSIDE_PUSH_STUCK_WINDOW
            or len(self._recent_inside_push_commands) < INSIDE_PUSH_STUCK_WINDOW
        ):
            return False
        if not all(
            command > INSIDE_PUSH_COMMAND_EPS for command in self._recent_inside_push_commands
        ):
            return False
        first_x, first_y = self._recent_inside_push_block_centers[0]
        max_displacement = max(
            np.hypot(cx - first_x, cy - first_y)
            for cx, cy in self._recent_inside_push_block_centers
        )
        return bool(max_displacement < INSIDE_PUSH_STUCK_POSITION_EPS)

    def _store_sort_key(self, x: NDArray, block_name: str) -> tuple[float, float]:
        robot = extract_robot(x)
        block = extract_block(x, block_name)
        return (
            block.center[1],
            abs(block.center[0] - robot.x),
        )

    def _base_path_bounds(
        self,
        x: NDArray,
        robot: RobotPose,
        mover_radius: float | None = None,
    ) -> tuple[float, float, float, float]:
        effective_radius = robot.base_radius if mover_radius is None else mover_radius
        margin = effective_radius + APPROACH_MARGIN
        shelf = extract_shelf(x)
        max_y = min(
            WORLD_MAX_Y - margin,
            shelf.y1 - effective_radius - 0.02,
        )
        return (
            WORLD_MIN_X + margin,
            WORLD_MAX_X - margin,
            WORLD_MIN_Y + margin,
            max_y,
        )

    def _base_path_obstacles(
        self,
        x: NDArray,
        ignore_block: str,
        robot: RobotPose,
        mover_radius: float | None = None,
    ) -> list[tuple[float, float, float]]:
        obstacles: list[tuple[float, float, float]] = []
        effective_radius = robot.base_radius if mover_radius is None else mover_radius
        for name in outside_blocks(x):
            if name == ignore_block:
                continue
            block = extract_block(x, name)
            center_x, center_y = block.center
            obstacles.append(
                (
                    center_x,
                    center_y,
                    inflate_block_radius(block.width, block.height, effective_radius),
                )
            )
        return obstacles

    def _queue_holding_transport(
        self,
        x: NDArray,
        block_name: str,
    ) -> bool:
        robot = extract_robot(x)
        block = extract_block(x, block_name)
        assert self._target_center is not None
        target_x, target_y = self._target_center
        held_radius = 0.5 * float(np.hypot(block.width, block.height))
        mover_radius = max(robot.base_radius, held_radius)
        held_offset = (block.center[0] - robot.x, block.center[1] - robot.y)
        if self._target_kind == STAGING_TARGET:
            transport_block_y = target_y
        else:
            shelf = extract_shelf(x)
            transport_block_y = max(
                block.center[1],
                shelf.y1 - 0.5 * block.height - PREINSERT_Y_MARGIN,
            )
        goal = (
            target_x - held_offset[0],
            transport_block_y - held_offset[1],
        )
        bounds = self._base_path_bounds(x, robot)
        if not (
            bounds[0] <= goal[0] <= bounds[1]
            and bounds[2] <= goal[1] <= bounds[3]
        ):
            return False
        obstacles = self._base_path_obstacles(
            x, ignore_block=block_name, robot=robot, mover_radius=mover_radius
        )
        path = plan_holding_base_path(
            (robot.x, robot.y),
            goal,
            held_offset,
            held_radius,
            obstacles,
            bounds,
        )
        if path is None or len(path) < 2:
            return False

        key_waypoints = [_robot_pose(robot)]
        for path_x, path_y in path[1:]:
            key_waypoints.append(
                _waypoint(
                    robot,
                    path_x,
                    path_y,
                    robot.theta,
                    robot.arm_joint,
                    SAFE_VACUUM,
                )
            )
        dense = connecting_waypoints(key_waypoints)
        self._holding_actions = waypoints_to_actions(dense)
        self._planned_path_len = path_length(path)
        return bool(self._holding_actions)

    def _queue_holding_rotation_to_up(
        self,
        x: NDArray,
        block_name: str,
    ) -> bool:
        robot = extract_robot(x)
        block = extract_block(x, block_name)
        held_radius = 0.5 * float(np.hypot(block.width, block.height))
        held_offset = (block.center[0] - robot.x, block.center[1] - robot.y)
        held_reach = float(np.hypot(held_offset[0], held_offset[1])) + held_radius
        transport_bounds = self._base_path_bounds(x, robot)
        rotate_bounds = self._base_path_bounds(x, robot, mover_radius=held_reach)
        transport_obstacles = self._base_path_obstacles(
            x,
            ignore_block=block_name,
            robot=robot,
            mover_radius=max(robot.base_radius, held_radius),
        )
        rotate_obstacles = self._base_path_obstacles(
            x,
            ignore_block=block_name,
            robot=robot,
            mover_radius=held_reach,
        )

        def rotation_safe(goal: tuple[float, float]) -> bool:
            gx, gy = goal
            if not (
                rotate_bounds[0] <= gx <= rotate_bounds[1]
                and rotate_bounds[2] <= gy <= rotate_bounds[3]
            ):
                return False
            return all(
                np.hypot(gx - cx, gy - cy) >= radius - 1e-6
                for cx, cy, radius in rotate_obstacles
            )

        shelf = extract_shelf(x)
        max_safe_base_y = rotate_bounds[3] - POS_TOL
        stage_base_y = min(robot.y, max_safe_base_y)
        if not (transport_bounds[2] <= stage_base_y <= transport_bounds[3]):
            return False
        candidate_base_xs = [
            robot.x,
            shelf.opening_center_x - held_offset[0],
        ]
        candidate_goals: list[tuple[float, float]] = []
        current_goal = (robot.x, robot.y)
        if rotation_safe(current_goal):
            candidate_goals.append(current_goal)
        for base_x in candidate_base_xs:
            goal = (base_x, stage_base_y)
            if (
                transport_bounds[0] <= goal[0] <= transport_bounds[1]
                and transport_bounds[2] <= goal[1] <= transport_bounds[3]
                and rotation_safe(goal)
            ):
                candidate_goals.append(goal)

        best_path: list[tuple[float, float]] | None = None
        best_length = float("inf")
        for goal in candidate_goals:
            if abs(goal[0] - robot.x) <= POS_TOL and abs(goal[1] - robot.y) <= POS_TOL:
                path = [(robot.x, robot.y)]
                length = 0.0
            else:
                path = plan_holding_base_path(
                    (robot.x, robot.y),
                    goal,
                    held_offset,
                    held_radius,
                    transport_obstacles,
                    transport_bounds,
                )
                if path is None:
                    continue
                length = path_length(path)
            if length < best_length:
                best_length = length
                best_path = path

        if best_path is None:
            return False

        key_waypoints = [_robot_pose(robot)]
        for path_x, path_y in best_path[1:]:
            key_waypoints.append(
                _waypoint(
                    robot,
                    path_x,
                    path_y,
                    robot.theta,
                    robot.arm_joint,
                    SAFE_VACUUM,
                )
            )
        key_waypoints.append(
            _waypoint(
                robot,
                key_waypoints[-1].x,
                key_waypoints[-1].y,
                UP,
                robot.arm_joint,
                SAFE_VACUUM,
            )
        )
        dense = connecting_waypoints(key_waypoints)
        self._holding_actions = waypoints_to_actions(dense)
        self._planned_path_len = best_length
        return bool(self._holding_actions)

    def _select_store_pick(
        self,
        x: NDArray,
    ) -> tuple[str, RobotPose, RobotPose, list[tuple[float, float]]] | None:
        robot = extract_robot(x)
        bounds = self._base_path_bounds(x, robot)
        reachable: list[
            tuple[
                tuple[float, float],
                float,
                str,
                RobotPose,
                RobotPose,
                list[tuple[float, float]],
            ]
        ] = []

        for block_name in outside_blocks(x):
            path_obstacles = self._base_path_obstacles(x, ignore_block="", robot=robot)
            local_obstacles = self._base_path_obstacles(x, block_name, robot)
            best_candidate: tuple[
                float, RobotPose, RobotPose, list[tuple[float, float]]
            ] | None = None
            for candidate in pick_base_pose_candidates(x, block_name):
                pre_pick_x = candidate.x - PRE_PICK_RING_MARGIN * float(np.cos(candidate.theta))
                pre_pick_y = candidate.y - PRE_PICK_RING_MARGIN * float(np.sin(candidate.theta))
                if not (
                    bounds[0] <= pre_pick_x <= bounds[1]
                    and bounds[2] <= pre_pick_y <= bounds[3]
                ):
                    continue
                pre_pick = _waypoint(
                    robot,
                    pre_pick_x,
                    pre_pick_y,
                    candidate.theta,
                    robot.base_radius,
                    VACUUM_OFF,
                )
                path = plan_base_path(
                    (robot.x, robot.y),
                    (pre_pick.x, pre_pick.y),
                    path_obstacles,
                    bounds,
                )
                if path is None:
                    continue
                if not segment_collision_free(
                    (pre_pick.x, pre_pick.y),
                    (candidate.x, candidate.y),
                    local_obstacles,
                    bounds,
                ):
                    continue
                candidate_length = path_length(path) + float(
                    np.hypot(candidate.x - pre_pick.x, candidate.y - pre_pick.y)
                )
                target = _waypoint(
                    robot,
                    candidate.x,
                    candidate.y,
                    candidate.theta,
                    robot.base_radius,
                    VACUUM_OFF,
                )
                if best_candidate is None or candidate_length < best_candidate[0]:
                    best_candidate = (candidate_length, target, pre_pick, path)
            if best_candidate is None:
                continue
            path_len, target, pre_pick, path = best_candidate
            reachable.append(
                (
                    self._store_sort_key(x, block_name),
                    path_len,
                    block_name,
                    target,
                    pre_pick,
                    path,
                )
            )

        if not reachable:
            return None

        _, _, block_name, target, pre_pick, path = min(
            reachable,
            key=lambda item: (item[0][0], item[0][1], item[1]),
        )
        return (block_name, target, pre_pick, path)

    def _select_pick_for_block(
        self,
        x: NDArray,
        block_name: str,
    ) -> tuple[str, RobotPose, RobotPose, list[tuple[float, float]]] | None:
        robot = extract_robot(x)
        bounds = self._base_path_bounds(x, robot)
        path_obstacles = self._base_path_obstacles(x, ignore_block="", robot=robot)
        local_obstacles = self._base_path_obstacles(x, block_name, robot)
        best_candidate: tuple[
            float, RobotPose, RobotPose, list[tuple[float, float]]
        ] | None = None
        for candidate in pick_base_pose_candidates(x, block_name):
            pre_pick_x = candidate.x - PRE_PICK_RING_MARGIN * float(np.cos(candidate.theta))
            pre_pick_y = candidate.y - PRE_PICK_RING_MARGIN * float(np.sin(candidate.theta))
            if not (
                bounds[0] <= pre_pick_x <= bounds[1]
                and bounds[2] <= pre_pick_y <= bounds[3]
            ):
                continue
            pre_pick = _waypoint(
                robot,
                pre_pick_x,
                pre_pick_y,
                candidate.theta,
                robot.base_radius,
                VACUUM_OFF,
            )
            path = plan_base_path(
                (robot.x, robot.y),
                (pre_pick.x, pre_pick.y),
                path_obstacles,
                bounds,
            )
            if path is None:
                continue
            if not segment_collision_free(
                (pre_pick.x, pre_pick.y),
                (candidate.x, candidate.y),
                local_obstacles,
                bounds,
            ):
                continue
            candidate_length = path_length(path) + float(
                np.hypot(candidate.x - pre_pick.x, candidate.y - pre_pick.y)
            )
            target = _waypoint(
                robot,
                candidate.x,
                candidate.y,
                candidate.theta,
                robot.base_radius,
                VACUUM_OFF,
            )
            if best_candidate is None or candidate_length < best_candidate[0]:
                best_candidate = (candidate_length, target, pre_pick, path)
        if best_candidate is None:
            return None
        _, target, pre_pick, path = best_candidate
        return (block_name, target, pre_pick, path)

    def _compact_pick_target(self, x: NDArray, block_name: str) -> RobotPose:
        robot = extract_robot(x)
        block = extract_block(x, block_name)
        return _waypoint(
            robot,
            block.center[0],
            block.center[1]
            - (robot.arm_length + 1.8 * robot.gripper_width + APPROACH_MARGIN),
            UP,
            robot.base_radius,
            VACUUM_OFF,
        )

    def _select_compact_blocker_to_clear(
        self,
        x: NDArray,
        compact_block: str,
    ) -> tuple[str, RobotPose, RobotPose, list[tuple[float, float]]] | None:
        shelf = extract_shelf(x)
        blocker_names = sorted(
            outside_blocks(x),
            key=lambda name: float(
                np.hypot(
                    extract_block(x, name).center[0] - shelf.opening_center_x,
                    extract_block(x, name).center[1] - shelf.y1,
                )
            ),
        )
        for blocker_name in blocker_names:
            selection = self._select_pick_for_block(x, blocker_name)
            if selection is not None:
                return selection
        return None

    def _plan_compact_pick(
        self,
        x: NDArray,
        block_name: str,
    ) -> tuple[RobotPose, list[tuple[float, float]]] | None:
        robot = extract_robot(x)
        target = self._compact_pick_target(x, block_name)
        obstacles = self._base_path_obstacles(x, ignore_block="", robot=robot)
        path = plan_base_path(
            (robot.x, robot.y),
            (target.x, target.y),
            obstacles,
            self._base_path_bounds(x, robot),
        )
        if path is None:
            return None
        return (target, path)

    def _generate_pick_plan(self, x: NDArray) -> None:
        self._sync_phase(x)
        robot = extract_robot(x)
        current = _robot_pose(robot)
        carry_arm = max(robot.base_radius, CARRY_ARM_FRACTION * robot.arm_length)
        self._chosen_pick_pose = None
        self._planned_path_len = None
        if self._phase == "compact":
            next_block = self._choose_compact_block(x)
            self._target_kind = SHELF_TARGET
            self._target_center = slot_centers(x)[0]
            if next_block is None:
                pre_pick = None
                target = None
                path_points = None
            else:
                selection = self._plan_compact_pick(x, next_block)
                if selection is None:
                    self._phase = "clear_compact"
                    self._target_kind = STAGING_TARGET
                    self._target_center = farthest_free_staging_center(x)
                    clear_selection = self._select_compact_blocker_to_clear(
                        x, next_block
                    )
                    if clear_selection is None:
                        next_block = None
                        pre_pick = None
                        target = None
                        path_points = None
                    else:
                        next_block, target, pre_pick, path_points = clear_selection
                else:
                    pre_pick = None
                    target, path_points = selection
        else:
            if self._phase == "clear_compact":
                self._target_kind = STAGING_TARGET
                self._target_center = farthest_free_staging_center(x)
            else:
                self._target_kind = SHELF_TARGET
                self._target_center = next_free_slot_center(x)
            selection = self._select_store_pick(x)
            if selection is None:
                next_block = None
                target = None
                pre_pick = None
                path_points = None
            else:
                next_block, target, pre_pick, path_points = selection
        if next_block is None:
            self._active_block = None
            self._holding_actions.clear()
            self._clear_inside_push_monitor()
            self._hold_loss_steps = 0
            self._last_active_block_center = None
            self._actions = deque()
            return
        self._active_block = next_block
        self._holding_actions.clear()
        self._clear_inside_push_monitor()
        self._hold_loss_steps = 0
        self._last_active_block_center = None
        self._staging_release_active = False

        if target is None:
            self._actions = deque()
            return

        self._chosen_pick_pose = (target.x, target.y, target.theta)

        key_waypoints = [current]
        if robot.arm_joint > robot.base_radius + ARM_TOL:
            key_waypoints.append(
                _waypoint(
                    robot,
                    robot.x,
                    robot.y,
                    robot.theta,
                    robot.base_radius,
                    VACUUM_OFF,
                )
            )
        key_waypoints.append(
            _waypoint(
                robot,
                key_waypoints[-1].x,
                key_waypoints[-1].y,
                target.theta,
                robot.base_radius,
                VACUUM_OFF,
            )
        )
        if path_points is not None:
            self._planned_path_len = path_length(path_points)
            for path_x, path_y in path_points[1:]:
                key_waypoints.append(
                    _waypoint(
                        robot,
                        path_x,
                        path_y,
                        target.theta,
                        robot.base_radius,
                        VACUUM_OFF,
                    )
                )
            if self._phase != "compact" and pre_pick is not None:
                key_waypoints.append(pre_pick)
        else:
            self._planned_path_len = float(
                np.hypot(target.x - robot.x, target.y - robot.y)
            )
            key_waypoints.append(
                _waypoint(
                    robot,
                    target.x,
                    target.y,
                    target.theta,
                    robot.base_radius,
                    VACUUM_OFF,
                )
            )
        key_waypoints.extend(
            [
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
        )

        dense = connecting_waypoints(key_waypoints)
        self._actions = waypoints_to_actions(dense)
        self._clear_motion_monitor()

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

        if (
            self._target_kind == STAGING_TARGET
            and not self._staging_release_active
            and robot.arm_joint > carry_arm + ARM_TOL
        ):
            action[3] = float(np.clip(carry_arm - robot.arm_joint, -DARM_LIM, DARM_LIM))
            return action

        angle_error = wrap_angle(UP - robot.theta)
        if abs(angle_error) > ANGLE_TOL:
            if not self._holding_actions and self._queue_holding_rotation_to_up(x, block_name):
                return self._holding_actions.popleft()
            action[2] = float(np.clip(angle_error, -DTH_LIM, DTH_LIM))
            return action

        shelf = extract_shelf(x)
        target_y_tol = COMPACT_Y_TOL if self._phase == "compact" else DEEP_PLACE_Y_TOL
        at_target_x = abs(block_x - target_x) <= POS_TOL
        insert_aligned_x = abs(block_x - target_x) <= INSERT_X_TOL
        deep_enough = block_y >= target_y - target_y_tol

        if self._target_kind == STAGING_TARGET:
            if (
                not self._staging_release_active
                and at_target_x
                and abs(block_y - target_y) <= POS_TOL
            ):
                self._staging_release_active = True
            if robot.arm_joint > carry_arm + ARM_TOL and (
                not self._staging_release_active
                and (
                abs(block_x - target_x) > POS_TOL or abs(block_y - target_y) > POS_TOL
                )
            ):
                action[3] = float(np.clip(carry_arm - robot.arm_joint, -DARM_LIM, DARM_LIM))
                return action
            if (
                not self._staging_release_active
                and (not at_target_x or abs(block_y - target_y) > POS_TOL)
            ):
                if not self._holding_actions and self._queue_holding_transport(x, block_name):
                    return self._holding_actions.popleft()
                action[0] = float(np.clip(target_x - block_x, -DX_LIM, DX_LIM))
                action[1] = float(np.clip(target_y - block_y, -DY_LIM, DY_LIM))
                return action
            staging_release_arm = max(
                carry_arm, STAGING_RELEASE_ARM_FRACTION * robot.arm_length
            )
            if robot.arm_joint < staging_release_arm - ARM_TOL:
                action[3] = float(
                    np.clip(staging_release_arm - robot.arm_joint, -DARM_LIM, DARM_LIM)
                )
                return action
            action[4] = VACUUM_OFF
            self._staging_release_active = False
            if self._phase == "clear_compact":
                self._phase = "compact"
            self._queue_retreat(x)
            return action

        if not is_block_inside_shelf(x, block_name):
            transport_block_y = max(
                block_y,
                shelf.y1 - 0.5 * block.height - PREINSERT_Y_MARGIN,
            )
            at_preinsert = (
                abs(block_x - target_x) <= POS_TOL
                and abs(block_y - transport_block_y) <= POS_TOL
            )
            if not at_preinsert:
                if not self._holding_actions:
                    if self._queue_holding_transport(x, block_name):
                        return self._holding_actions.popleft()
            if not insert_aligned_x:
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
        self._recent_positions.append((robot.x, robot.y))
        if self._path_is_stuck():
            self._actions.clear()
            self._holding_actions.clear()
            self._chosen_pick_pose = None
            self._planned_path_len = None
            self._clear_motion_monitor()
            self._clear_inside_push_monitor()
            self._recent_positions.append((robot.x, robot.y))
        if (
            observed_held is None
            and robot.vacuum <= VACUUM_OFF
            and not self._actions
            and not self._holding_actions
        ):
            self._active_block = None
        self._sync_phase(x)
        held_name = None
        if observed_held is not None and observed_held == self._active_block:
            self._hold_loss_steps = 0
            self._last_active_block_center = extract_block(x, observed_held).center
            held_name = observed_held
        elif self._active_block is not None and robot.vacuum > VACUUM_OFF:
            current_center = extract_block(x, self._active_block).center
            moved_since_last_hold = False
            if self._last_active_block_center is not None:
                moved_since_last_hold = (
                    np.hypot(
                        current_center[0] - self._last_active_block_center[0],
                        current_center[1] - self._last_active_block_center[1],
                    )
                    > HOLD_RECOVERY_MOVE_EPS
                )
            if moved_since_last_hold:
                self._hold_loss_steps = 0
                self._last_active_block_center = current_center
                held_name = self._active_block
            elif self._last_active_block_center is not None and (
                self._hold_loss_steps < HOLD_LOSS_PATIENCE
            ):
                self._hold_loss_steps += 1
                held_name = self._active_block
            else:
                self._last_active_block_center = None
        else:
            self._hold_loss_steps = 0
            self._last_active_block_center = None
        if held_name is not None:
            self._actions.clear()
            self._clear_motion_monitor()
            if self._holding_actions:
                action = self._holding_actions.popleft()
                self._recent_base_commands.append(float(np.hypot(action[0], action[1])))
                return action
            action = self._place_action(x, held_name)
            if is_block_inside_shelf(x, held_name):
                block_center = extract_block(x, held_name).center
                push_command = max(float(action[1]), float(action[3]), 0.0)
                self._recent_inside_push_block_centers.append(block_center)
                self._recent_inside_push_commands.append(push_command)
                if self._inside_push_is_stuck(x, held_name):
                    if self._phase == "compact":
                        self._phase = "store"
                    self._active_block = None
                    self._holding_actions.clear()
                    self._clear_inside_push_monitor()
                    release_action = NOOP_ACTION.copy()
                    release_action[4] = VACUUM_OFF
                    self._queue_retreat(x)
                    self._recent_base_commands.append(0.0)
                    return release_action
            else:
                self._clear_inside_push_monitor()
            self._recent_base_commands.append(float(np.hypot(action[0], action[1])))
            return action
        if not self._actions:
            self._generate_pick_plan(x)
        if not self._actions:
            self._recent_base_commands.append(0.0)
            return NOOP_ACTION.copy()
        action = self._actions.popleft()
        self._recent_base_commands.append(float(np.hypot(action[0], action[1])))
        return action

    def debug_snapshot(self) -> dict[str, object]:
        """Return a compact snapshot of the current behavior state."""
        return {
            "phase": self._phase,
            "active_block": self._active_block,
            "target_center": self._target_center,
            "chosen_pick_pose": self._chosen_pick_pose,
            "path_len": self._planned_path_len,
            "queued_actions": len(self._actions),
            "queued_holding_actions": len(self._holding_actions),
        }
