"""Planner-driven oracle behaviors for ClutteredRetrieval2D-o10."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robocode.oracles.clutteredretrieval2d_medium.act_helpers import (
    append_grasp_action,
    append_micro_approach_actions,
    append_release_action,
    config_path_to_actions,
)
from robocode.oracles.clutteredretrieval2d_medium.obs_helpers import (
    AttemptedTransition,
    PlanningMode,
    RobotConfig,
    candidate_pick_poses,
    candidate_place_poses,
    config_distance,
    current_config,
    extract_robot,
    filter_feasible_grasp_poses,
    held_object_transform,
    held_object_changed,
    holding_obstruction_named,
    holding_target_block,
    infer_blocking_obstruction,
    interpolate_configs,
    is_blocker_cleared_from_pick_corridor,
    is_blocker_cleared_from_place_corridor,
    iter_obstruction_names,
    robot_config_changed,
    staging_robot_configs,
    target_inside_region,
    with_config_applied,
    wrap_angle,
    WORLD_MAX_X,
    WORLD_MAX_Y,
    WORLD_MIN_X,
    WORLD_MIN_Y,
)
from robocode.primitives.behavior import Behavior


class _PlannerBehavior(Behavior[NDArray, NDArray]):
    """Common planner helpers shared by the oracle behaviors."""

    def __init__(self, primitives: dict[str, Any], seed: int = 0) -> None:
        self._primitives = primitives
        self._rng = np.random.default_rng(seed)
        self._actions: deque[NDArray] = deque()
        self.required_blocker: str | None = None
        self.last_failed_transition: AttemptedTransition | None = None

    def _require_primitives(self) -> tuple[Callable[[NDArray, NDArray], bool], Any]:
        if "check_action_collision" not in self._primitives or "BiRRT" not in self._primitives:
            raise RuntimeError(
                "ClutteredRetrieval2D oracle requires primitives "
                "'check_action_collision' and 'BiRRT'."
            )
        return self._primitives["check_action_collision"], self._primitives["BiRRT"]

    def _sample_config(self, start: RobotConfig) -> RobotConfig:
        del start
        return RobotConfig(
            x=float(self._rng.uniform(0.15, 2.35)),
            y=float(self._rng.uniform(0.15, 2.35)),
            theta=float(self._rng.uniform(-math.pi, math.pi)),
            arm_joint=float(self._rng.uniform(0.1, 0.2)),
            vacuum=0.0,
        )

    def _plan_to_candidates(
        self,
        obs: NDArray,
        candidates: list[RobotConfig],
        *,
        carry_vacuum: float,
        planning_mode: PlanningMode,
        held_name: str | None = None,
    ) -> list[RobotConfig] | None:
        """Plan to the first reachable candidate; on failure remember the blocker."""
        check_action_collision, birrt_cls = self._require_primitives()
        start = current_config(obs)
        start = RobotConfig(
            x=start.x,
            y=start.y,
            theta=start.theta,
            arm_joint=start.arm_joint,
            vacuum=carry_vacuum,
        )
        held_transform = None
        if held_name is not None:
            held_transform = held_object_transform(obs, held_name)

        for goal in candidates:
            goal = RobotConfig(
                x=goal.x,
                y=goal.y,
                theta=goal.theta,
                arm_joint=goal.arm_joint,
                vacuum=carry_vacuum,
            )

            last_edge: AttemptedTransition | None = None

            def extend_fn(q1: RobotConfig, q2: RobotConfig) -> Iterable[RobotConfig]:
                nonlocal last_edge
                prev = q1
                for nxt in interpolate_configs(q1, q2):
                    last_edge = AttemptedTransition(prev, nxt)
                    yield nxt
                    prev = nxt

            def sample_fn(_: RobotConfig) -> RobotConfig:
                return RobotConfig(
                    x=float(self._rng.uniform(0.15, 2.35)),
                    y=float(self._rng.uniform(0.15, 2.35)),
                    theta=float(self._rng.uniform(-math.pi, math.pi)),
                    arm_joint=float(self._rng.uniform(0.1, 0.2)),
                    vacuum=carry_vacuum,
                )

            def collision_fn(q: RobotConfig) -> bool:
                if last_edge is None:
                    # BiRRT.query() probes the start / goal states before any edge
                    # expansion. Our collision model is edge-based, so endpoint
                    # checks are deferred to direct-path probing and extend_fn edges.
                    del q
                    return False
                pred_obs = with_config_applied(
                    obs,
                    last_edge.start,
                    held_name=held_name,
                    held_transform=held_transform,
                )
                action = np.array(
                    [
                        q.x - last_edge.start.x,
                        q.y - last_edge.start.y,
                        wrap_angle(q.theta - last_edge.start.theta),
                        q.arm_joint - last_edge.start.arm_joint,
                        q.vacuum,
                    ],
                    dtype=np.float32,
                )
                return bool(check_action_collision(pred_obs, action))

            birrt = birrt_cls(
                sample_fn=sample_fn,
                extend_fn=extend_fn,
                collision_fn=collision_fn,
                distance_fn=config_distance,
                rng=self._rng,
                num_attempts=5,
                num_iters=150,
                # The collision check is edge-based rather than state-based, so
                # BiRRT's post-hoc smoothing would probe states out of the original
                # edge order and create invalid delta-actions. Keep the raw path.
                smooth_amt=0,
            )
            path = birrt.query(start, goal)
            if path is not None:
                self.required_blocker = None
                self.last_failed_transition = None
                return path

            fallback_failure = self._direct_failure(
                obs,
                start,
                goal,
                check_action_collision,
                held_name=held_name,
                held_transform=held_transform,
            )
            if fallback_failure is not None:
                self.last_failed_transition = fallback_failure
                self.required_blocker = infer_blocking_obstruction(
                    obs,
                    fallback_failure,
                    planning_mode,
                    held_name=held_name,
                    held_transform=held_transform,
                )

        return None

    def _direct_failure(
        self,
        obs: NDArray,
        start: RobotConfig,
        goal: RobotConfig,
        check_action_collision: Callable[[NDArray, NDArray], bool],
        *,
        held_name: str | None = None,
        held_transform: Any = None,
    ) -> AttemptedTransition | None:
        prev = start
        for nxt in interpolate_configs(start, goal):
            pred_obs = with_config_applied(
                obs,
                prev,
                held_name=held_name,
                held_transform=held_transform,
            )
            action = np.array(
                [
                    nxt.x - prev.x,
                    nxt.y - prev.y,
                    wrap_angle(nxt.theta - prev.theta),
                    nxt.arm_joint - prev.arm_joint,
                    nxt.vacuum,
                ],
                dtype=np.float32,
            )
            if check_action_collision(pred_obs, action):
                return AttemptedTransition(prev, nxt)
            prev = nxt
        return None

    def _stop_or_replan(self, obs: NDArray) -> NDArray:
        if not self._actions:
            self.reset(obs)
        if not self._actions:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self._actions.popleft()


class AcquireTargetBlock(_PlannerBehavior):
    """Plan to a target grasp pose, or request blocker removal."""

    def __init__(self, primitives: dict[str, Any], seed: int = 0) -> None:
        super().__init__(primitives, seed)
        self._recent_configs: deque[RobotConfig] = deque(maxlen=6)
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step

    def reset(self, x: NDArray) -> None:
        self._actions.clear()
        self._recent_configs.clear()
        robot = current_config(x)
        if (
            robot.vacuum > 0.5
            and not holding_target_block(x)
            and not any(holding_obstruction_named(x, name) for name in iter_obstruction_names())
        ):
            self._actions.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            return
        path = self._plan_to_candidates(
            x,
            candidate_pick_poses(x, "target_block"),
            carry_vacuum=0.0,
            planning_mode="robot_only",
            held_name=None,
        )
        if path is None:
            return
        self._actions = config_path_to_actions(path)
        append_grasp_action(self._actions, path[-1])

    def initializable(self, x: NDArray) -> bool:
        return not holding_target_block(x) and not target_inside_region(x)

    def terminated(self, x: NDArray) -> bool:
        return holding_target_block(x)

    def step(self, x: NDArray) -> NDArray:
        cfg = current_config(x)
        self._recent_configs.append(cfg)
        if len(self._recent_configs) >= 4:
            a = self._recent_configs[-1]
            b = self._recent_configs[-2]
            c = self._recent_configs[-3]
            d = self._recent_configs[-4]
            if (
                config_distance(a, c) < 0.03
                and config_distance(b, d) < 0.03
                and config_distance(a, b) > 0.05
            ):
                self._actions.clear()
                self._recent_configs.clear()
                if (
                    cfg.vacuum > 0.5
                    and not holding_target_block(x)
                    and not any(
                        holding_obstruction_named(x, name) for name in iter_obstruction_names()
                    )
                ):
                    return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self._stop_or_replan(x)


class RemoveSingleBlockingObstruction(_PlannerBehavior):
    """Pick one named obstruction and relocate it to a staging position."""

    def __init__(self, primitives: dict[str, Any], seed: int = 0) -> None:
        super().__init__(primitives, seed)
        self.blocker_name: str | None = None
        self._active_blocker_name: str | None = None
        self._has_grasped_blocker = False
        self._phase = "grasp_planning"
        self._staging_candidate_index = 0
        self._transport_retry_count = 0
        self._grasp_retry_count = 0
        self._consecutive_stuck_steps = 0
        self._last_staging_count = 0
        self._last_transport_goal_index = -1
        self._last_transport_event = "init"
        self._failed_grasp_goal_counts: dict[int, int] = {}
        self._active_grasp_goal_key: int | None = None
        self.blocker_grasp_infeasible = False
        self.remove_context = "acquire"
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step

    def _reset_execution_state(self) -> None:
        self._actions.clear()
        self._has_grasped_blocker = False
        self._phase = "grasp_planning"
        self._staging_candidate_index = 0
        self._transport_retry_count = 0
        self._grasp_retry_count = 0
        self._consecutive_stuck_steps = 0
        self._last_staging_count = 0
        self._last_transport_goal_index = -1
        self._last_transport_event = "reset"
        self._failed_grasp_goal_counts.clear()
        self._active_grasp_goal_key = None
        self.blocker_grasp_infeasible = False

    def _grasp_goal_key(self, cfg: RobotConfig) -> int:
        return int(round(wrap_angle(cfg.theta) * 100))

    def _grasp_variants(self, grasp_cfg: RobotConfig) -> list[RobotConfig]:
        """Generate local same-side grasp variants before switching sides."""
        lateral_offsets = [0.0, -0.02, 0.02]
        forward_offsets = [0.0, 0.01]
        variants: list[RobotConfig] = []
        for forward in forward_offsets:
            for lateral in lateral_offsets:
                dx = math.cos(grasp_cfg.theta) * forward - math.sin(grasp_cfg.theta) * lateral
                dy = math.sin(grasp_cfg.theta) * forward + math.cos(grasp_cfg.theta) * lateral
                variants.append(
                    RobotConfig(
                        x=grasp_cfg.x + dx,
                        y=grasp_cfg.y + dy,
                        theta=grasp_cfg.theta,
                        arm_joint=grasp_cfg.arm_joint,
                        vacuum=0.0,
                    )
                )
        return variants

    def clear(self) -> None:
        """Clear all blocker-specific state after a completed removal."""
        self.blocker_name = None
        self._active_blocker_name = None
        self._reset_execution_state()

    def _blocker_cleared(self, obs: NDArray) -> bool:
        assert self.blocker_name is not None
        if self.remove_context == "place":
            return is_blocker_cleared_from_place_corridor(obs, self.blocker_name)
        return is_blocker_cleared_from_pick_corridor(obs, self.blocker_name)

    def _retreat_transport_path(
        self,
        obs: NDArray,
        staging_goal: RobotConfig,
    ) -> list[RobotConfig] | None:
        """Try a short straight retreat before the main transport motion."""
        assert self.blocker_name is not None
        cfg = current_config(obs)
        robot = extract_robot(obs)
        held_transform = held_object_transform(obs, self.blocker_name)
        retreat_distances = (0.03, 0.06)
        for distance in retreat_distances:
            retreat = RobotConfig(
                x=cfg.x - math.cos(cfg.theta) * distance,
                y=cfg.y - math.sin(cfg.theta) * distance,
                theta=cfg.theta,
                arm_joint=cfg.arm_joint,
                vacuum=1.0,
            )
            if not (
                WORLD_MIN_X + robot.base_radius <= retreat.x <= WORLD_MAX_X - robot.base_radius
                and WORLD_MIN_Y + robot.base_radius <= retreat.y <= WORLD_MAX_Y - robot.base_radius
            ):
                continue
            retreat_path = self._plan_to_candidates(
                obs,
                [retreat],
                carry_vacuum=1.0,
                planning_mode="carrying_blocker",
                held_name=self.blocker_name,
            )
            if retreat_path is None:
                continue
            retreat_obs = with_config_applied(
                obs,
                retreat_path[-1],
                held_name=self.blocker_name,
                held_transform=held_transform,
            )
            main_path = self._plan_to_candidates(
                retreat_obs,
                [staging_goal],
                carry_vacuum=1.0,
                planning_mode="carrying_blocker",
                held_name=self.blocker_name,
            )
            if main_path is None:
                continue
            self._last_transport_event = "transport_plan_ready_with_retreat"
            return retreat_path + main_path[1:]
        return None

    def reset(self, x: NDArray) -> None:
        if self.blocker_name is None:
            self._active_blocker_name = None
            self._reset_execution_state()
            return
        if self.blocker_name != self._active_blocker_name:
            self._active_blocker_name = self.blocker_name
            self._reset_execution_state()

        if holding_obstruction_named(x, self.blocker_name):
            self._has_grasped_blocker = True
            if "transport" not in self._phase:
                self._phase = "transport_planning"

        # Keep executing an existing plan until it is exhausted.
        if self._actions:
            return

        cfg = current_config(x)
        if (
            "grasp" in self._phase
            and cfg.vacuum > 0.5
            and not holding_obstruction_named(x, self.blocker_name)
        ):
            if self._active_grasp_goal_key is not None:
                self._failed_grasp_goal_counts[self._active_grasp_goal_key] = (
                    self._failed_grasp_goal_counts.get(self._active_grasp_goal_key, 0) + 1
                )
            self._active_grasp_goal_key = None
            self._phase = "grasp_planning"
            self._grasp_retry_count += 1
            self._actions.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            return

        if "grasp" in self._phase and not holding_obstruction_named(x, self.blocker_name):
            grasp_candidates = filter_feasible_grasp_poses(
                x,
                self.blocker_name,
                candidate_pick_poses(x, self.blocker_name),
            )
            if not grasp_candidates:
                self.blocker_grasp_infeasible = True
                self._last_transport_event = "no_feasible_grasp_pose"
                return
            if self._active_grasp_goal_key is not None:
                grasp_candidates.sort(
                    key=lambda grasp_cfg: (
                        self._grasp_goal_key(grasp_cfg) != self._active_grasp_goal_key,
                        self._failed_grasp_goal_counts.get(self._grasp_goal_key(grasp_cfg), 0),
                        config_distance(current_config(x), grasp_cfg),
                    )
                )
            for grasp_cfg in grasp_candidates:
                key = self._grasp_goal_key(grasp_cfg)
                if self._failed_grasp_goal_counts.get(key, 0) >= 8:
                    continue
                for variant_cfg in self._grasp_variants(grasp_cfg):
                    backoff = 0.05
                    pregrasp_cfg = RobotConfig(
                        x=variant_cfg.x - math.cos(variant_cfg.theta) * backoff,
                        y=variant_cfg.y - math.sin(variant_cfg.theta) * backoff,
                        theta=variant_cfg.theta,
                        arm_joint=max(0.1, variant_cfg.arm_joint - 0.05),
                        vacuum=0.0,
                    )
                    pick_path = self._plan_to_candidates(
                        x,
                        [pregrasp_cfg],
                        carry_vacuum=0.0,
                        planning_mode="robot_only",
                        held_name=None,
                    )
                    if pick_path is None:
                        continue
                    self._actions = config_path_to_actions(pick_path)
                    final_config = append_micro_approach_actions(
                        self._actions,
                        pick_path[-1],
                        variant_cfg,
                    )
                    append_grasp_action(self._actions, final_config)
                    self._active_grasp_goal_key = key
                    self._phase = "grasp_executing"
                    return
            self._grasp_retry_count += 1
            exhausted_keys = {
                self._grasp_goal_key(grasp_cfg)
                for grasp_cfg in grasp_candidates
                if self._failed_grasp_goal_counts.get(self._grasp_goal_key(grasp_cfg), 0) >= 8
            }
            candidate_keys = {self._grasp_goal_key(grasp_cfg) for grasp_cfg in grasp_candidates}
            if candidate_keys and exhausted_keys == candidate_keys:
                self.blocker_grasp_infeasible = True
                self._last_transport_event = "all_feasible_grasp_sides_failed"
            return

        staging_configs = staging_robot_configs(x, self.blocker_name)
        self._last_staging_count = len(staging_configs)
        if not staging_configs:
            self._last_transport_goal_index = -1
            self._last_transport_event = "no_staging_configs"
            self._transport_retry_count += 1
            return
        if self._staging_candidate_index >= len(staging_configs):
            self._last_transport_goal_index = self._staging_candidate_index
            self._last_transport_event = "staging_exhausted"
            self._staging_candidate_index = 0
            self._transport_retry_count += 1
            return
        self._last_transport_goal_index = self._staging_candidate_index
        chosen_goal = staging_configs[self._staging_candidate_index]
        place_path = self._retreat_transport_path(x, chosen_goal)
        if place_path is None:
            place_path = self._plan_to_candidates(
                x,
                [chosen_goal],
                carry_vacuum=1.0,
                planning_mode="carrying_blocker",
                held_name=self.blocker_name,
            )
        if place_path is None:
            self._last_transport_event = "transport_plan_failed"
            self._staging_candidate_index += 1
            self._transport_retry_count += 1
            return
        self._actions = config_path_to_actions(place_path)
        append_release_action(self._actions, place_path[-1])
        self._phase = "transport_executing"
        if self._last_transport_event != "transport_plan_ready_with_retreat":
            self._last_transport_event = "transport_plan_ready"

    def observe_transition(self, prev_obs: NDArray, next_obs: NDArray) -> None:
        """Update stuck counters from the actual executed transition."""
        if self.blocker_name is None:
            return
        if "executing" not in self._phase:
            return

        if "transport" in self._phase and not holding_obstruction_named(next_obs, self.blocker_name):
            if self._has_grasped_blocker and self._blocker_cleared(next_obs):
                self._actions.clear()
                self._phase = "transport_executing"
                self._consecutive_stuck_steps = 0
                self._last_transport_event = "transport_payload_lost_but_cleared"
                self._active_grasp_goal_key = None
                return
            self._actions.clear()
            self._has_grasped_blocker = False
            self._phase = "grasp_planning"
            self._staging_candidate_index += 1
            self._transport_retry_count += 1
            self._consecutive_stuck_steps = 0
            self._last_transport_event = "transport_payload_lost"
            self._active_grasp_goal_key = None
            return

        robot_moved = robot_config_changed(prev_obs, next_obs)
        blocker_moved = False
        if holding_obstruction_named(prev_obs, self.blocker_name) or holding_obstruction_named(
            next_obs, self.blocker_name
        ):
            blocker_moved = held_object_changed(prev_obs, next_obs, self.blocker_name)

        if "transport" in self._phase:
            stuck = (not robot_moved) and (not blocker_moved)
        else:
            stuck = (not robot_moved) and (
                not holding_obstruction_named(next_obs, self.blocker_name)
            )

        if stuck:
            self._consecutive_stuck_steps += 1
        else:
            self._consecutive_stuck_steps = 0

        if self._consecutive_stuck_steps >= 2:
            self._actions.clear()
            self._consecutive_stuck_steps = 0
            if "transport" in self._phase:
                self._phase = "transport_planning"
                if holding_obstruction_named(next_obs, self.blocker_name):
                    self._staging_candidate_index += 1
                    self._last_transport_event = "transport_stuck_while_holding"
                else:
                    self._has_grasped_blocker = False
                    self._phase = "grasp_planning"
                    self._staging_candidate_index = 0
                    self._last_transport_event = "transport_stuck_after_drop"
                    self._active_grasp_goal_key = None
                self._transport_retry_count += 1
            else:
                if self._active_grasp_goal_key is not None:
                    self._failed_grasp_goal_counts[self._active_grasp_goal_key] = (
                        self._failed_grasp_goal_counts.get(self._active_grasp_goal_key, 0) + 1
                    )
                self._active_grasp_goal_key = None
                self._phase = "grasp_planning"
                self._grasp_retry_count += 1

    def initializable(self, x: NDArray) -> bool:
        if self.blocker_name is None:
            return False
        return self.blocker_name in set(iter_obstruction_names())

    def terminated(self, x: NDArray) -> bool:
        if self.blocker_name is None:
            return True
        if self._actions:
            return False
        if holding_obstruction_named(x, self.blocker_name):
            return False
        if not self._has_grasped_blocker:
            return False
        return self._blocker_cleared(x)

    def step(self, x: NDArray) -> NDArray:
        return self._stop_or_replan(x)


class PlaceTargetBlockInRegion(_PlannerBehavior):
    """Plan a collision-free transfer for the held target block into the region."""

    def __init__(self, primitives: dict[str, Any], seed: int = 0) -> None:
        super().__init__(primitives, seed)
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step

    def reset(self, x: NDArray) -> None:
        self._actions.clear()
        path = self._plan_to_candidates(
            x,
            candidate_place_poses(x),
            carry_vacuum=1.0,
            planning_mode="carrying_target",
            held_name="target_block",
        )
        if path is None:
            return
        self._actions = config_path_to_actions(path)
        append_release_action(self._actions, path[-1])

    def initializable(self, x: NDArray) -> bool:
        return holding_target_block(x)

    def terminated(self, x: NDArray) -> bool:
        return target_inside_region(x)

    def step(self, x: NDArray) -> NDArray:
        return self._stop_or_replan(x)
