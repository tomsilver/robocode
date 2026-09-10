"""Per-instance PDDLStream planning baseline.

Plans each evaluation seed once with PDDLStream, then executes the plan open-loop
against the evaluated environment. Two families have an upstream PDDLStream domain
and each reaches it differently, so the baseline dispatches on the environment and
refuses any other:

* **Packing3D** plans on a twin simulator held in this process and replays the plan
  through it in lockstep (:mod:`robocode.planners.pddlstream_packing3d`).
* **PR2Packed** is stock PDDLStream's own ``packed`` benchmark, so it plans with the
  upstream domain and streams in a subprocess and servos the returned joint-space
  waypoints (:mod:`robocode.planners.pddlstream_pr2packed`).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robocode.approaches.base_approach import BaseApproach, InstanceResult
from robocode.environments.pr2_tamp_blocked_variable_count_env import (
    PR2BlockedVariableCountEnv,
)
from robocode.environments.pr2_tamp_variable_count_env import (
    PR2PackedVariableCountEnv,
)
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.planners.pddlstream_packing3d import (
    PACKING3D_ENV_PATH,
    Packing3DPDDLStreamPlanner,
    StopExecution,
)
from robocode.planners.pddlstream_pr2packed import (
    PlanningFailure,
    PR2PackedPDDLStreamPlanner,
)

logger = logging.getLogger(__name__)

# Tolerance for the twin sim to agree with the evaluated env after each action.
_TWIN_ATOL = 1e-5


class PDDLStreamPlanningApproach(BaseApproach[Any, Any]):
    """Solve each eval seed with a fresh PDDLStream plan, executed open-loop."""

    per_instance = True

    def __init__(
        self,
        action_space: Any,
        observation_space: Any,
        seed: int,
        primitives: dict[str, Any],
        *,
        max_steps: int,
        eval_timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(action_space, observation_space, seed, primitives, **kwargs)
        self._max_steps = max_steps
        self._eval_timeout = eval_timeout

    def train(self) -> None:
        # Per-instance approaches solve each seed via solve_instance; the runner
        # branches on approach.per_instance and never calls train().
        raise NotImplementedError(
            "PDDLStreamPlanningApproach solves each seed via solve_instance; "
            "train() is not used (the runner branches on approach.per_instance)"
        )

    def _get_action(self) -> Any:
        raise NotImplementedError("solve_instance drives the planner directly")

    def _planner_for(self, env: Any, count: int | None) -> Packing3DPDDLStreamPlanner:
        if (
            not isinstance(env, VariableObjectCountEnv)
            or env.constant_object_env_path != PACKING3D_ENV_PATH
            or count is None
        ):
            raise NotImplementedError(
                "PDDLStreamPlanningApproach supports the variable-count Packing3D "
                f"env only, got {type(env).__name__} with count {count!r}"
            )
        return Packing3DPDDLStreamPlanner(num_parts=count)

    def solve_instance(
        self,
        *,
        env: Any,
        seed: int,
        budget_usd: float,
        output_subdir: Path,
        render: bool = False,
        count: int | None = None,
        max_steps: int | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> InstanceResult:
        """Plan once for this seed, then execute the plan open-loop.

        No plan within the shared eval timeout is scored as unsolved (not crashed).
        ``budget_usd`` is unused: the planner has no LLM cost, so ``cost_usd`` is
        always 0.0 and every seed is attempted. ``count`` pins the instance's part
        count so the planner faces the same instance as the generalized program.
        """
        del budget_usd
        if isinstance(env, (PR2PackedVariableCountEnv, PR2BlockedVariableCountEnv)):
            return self._solve_pr2_tamp(
                problem=(
                    "blocked"
                    if isinstance(env, PR2BlockedVariableCountEnv)
                    else "packed"
                ),
                env=env,
                seed=seed,
                render=render,
                count=count,
                max_steps=max_steps,
                progress_callback=progress_callback,
            )
        planner = self._planner_for(env, count)
        try:
            return self._solve_with_planner(
                planner=planner,
                env=env,
                seed=seed,
                output_subdir=output_subdir,
                render=render,
                count=count,
                max_steps=max_steps,
                progress_callback=progress_callback,
            )
        finally:
            planner.close()

    def _solve_pr2_tamp(
        self,
        *,
        problem: str,
        env: Any,
        seed: int,
        render: bool,
        count: int | None,
        max_steps: int | None,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> InstanceResult:
        """Plan one PR2 ``packed`` or ``blocked`` instance upstream and servo it back.

        Unlike Packing3D there is no twin to keep in step: the plan is computed from
        the instance's state in a separate process and comes back as joint-space
        waypoints, which the planner turns into actions by reading the evaluated
        environment's own observation. The environment is never mutated except
        through ``step``.
        """
        if count is None:
            raise NotImplementedError(
                "PDDLStreamPlanningApproach needs a pinned object count for the PR2 "
                "TAMP environments"
            )
        obs, _ = env.reset(seed=seed, options={"object_count": count})
        del obs

        frames: list[Any] = []

        def _capture() -> None:
            rendered = env.render()
            if isinstance(rendered, np.ndarray):
                frames.append(rendered)

        if render:
            _capture()
        if progress_callback is not None:
            progress_callback("planning", 0, 0)

        planner = PR2PackedPDDLStreamPlanner(env.current_backend, problem=problem)
        plan_start = time.perf_counter()
        try:
            steps = planner.plan(max_time=self._eval_timeout, seed=seed)
        except PlanningFailure as error:
            logger.info("PDDLStream found no plan for seed %d: %s", seed, error)
            return InstanceResult(
                solved=False,
                total_reward=None,
                num_steps=None,
                cost_usd=0.0,
                frames=frames if render else None,
                extras={
                    "planning_time": time.perf_counter() - plan_start,
                    "plan_found": False,
                    "plan_length": 0,
                    "object_count": count,
                },
            )
        planning_time = time.perf_counter() - plan_start

        episode_max_steps = env.max_steps_for_count(count)
        if max_steps is not None:
            episode_max_steps = min(episode_max_steps, max_steps)

        total_reward = 0.0
        num_steps = 0
        terminated = False
        exec_start = time.perf_counter()
        for action in planner.actions(steps):
            if num_steps >= episode_max_steps:
                break
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            num_steps += 1
            if progress_callback is not None:
                progress_callback("running episode", num_steps, episode_max_steps)
            if render:
                _capture()
            if terminated or truncated:
                break
        execution_time = time.perf_counter() - exec_start

        return InstanceResult(
            solved=bool(terminated),
            total_reward=total_reward,
            num_steps=num_steps,
            cost_usd=0.0,
            frames=frames if render else None,
            extras={
                "planning_time": planning_time,
                "execution_time": execution_time,
                "plan_found": True,
                "plan_length": num_steps,
                "waypoint_count": len(steps),
                "step_budget": episode_max_steps,
                "object_count": count,
            },
        )

    def _solve_with_planner(
        self,
        *,
        planner: Packing3DPDDLStreamPlanner,
        env: Any,
        seed: int,
        output_subdir: Path,
        render: bool,
        count: int | None,
        max_steps: int | None,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> InstanceResult:
        """Implement one attempt while :meth:`solve_instance` owns twin cleanup."""
        obs, _ = env.reset(seed=seed, options={"object_count": count})

        frames: list[Any] = []

        def _capture() -> None:
            rendered = env.render()
            if isinstance(rendered, np.ndarray):
                frames.append(rendered)

        # Capture before planning: a failed plan still shows the instance.
        if render:
            _capture()

        if progress_callback is not None:
            progress_callback("planning", 0, 0)
        plan_start = time.perf_counter()
        plan = planner.plan(
            obs,
            max_time=self._eval_timeout,
            seed=seed,
            output_dir=output_subdir,
        )
        planning_time = time.perf_counter() - plan_start
        count_extra = {"object_count": count}

        if plan is None:
            if progress_callback is not None:
                progress_callback("planning failed; saving initial state", 0, 0)
            return InstanceResult(
                solved=False,
                total_reward=None,
                num_steps=None,
                cost_usd=0.0,
                frames=frames if render else None,
                extras={
                    "planning_time": planning_time,
                    "plan_found": False,
                    "plan_length": 0,
                    **count_extra,
                },
            )

        episode_max_steps = env.max_steps_for_count(count)
        if max_steps is not None:
            episode_max_steps = min(episode_max_steps, max_steps)
        total_reward = 0.0
        num_steps = 0
        terminated = False
        env_step_time = 0.0
        twin_divergences = 0

        def _on_action(action: Any) -> None:
            nonlocal total_reward, num_steps, terminated, env_step_time
            nonlocal twin_divergences
            if num_steps >= episode_max_steps:
                raise StopExecution
            t0 = time.perf_counter()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            env_step_time += time.perf_counter() - t0
            total_reward += float(reward)
            num_steps += 1
            if progress_callback is not None:
                progress_callback("running episode", num_steps, episode_max_steps)
            if render:
                _capture()
            # Both states go through the evaluated env's Box view of this count.
            if not np.allclose(
                env.to_box(next_obs), env.to_box(planner.state()), atol=_TWIN_ATOL
            ):
                # The twin fell out of step with the evaluated env; put it back so the
                # rest of the plan is replayed from the real state.
                twin_divergences += 1
                planner.sync(next_obs)
            if terminated or truncated:
                raise StopExecution

        # Planning scratch-mutated the twin; replay from the instance's state.
        planner.sync(obs)
        exec_start = time.perf_counter()
        try:
            planner.execute(plan, _on_action)
        except StopExecution:
            pass
        execution_time = time.perf_counter() - exec_start - env_step_time
        if twin_divergences:
            logger.warning(
                "Twin sim diverged from the evaluated env %d time(s) on seed %d",
                twin_divergences,
                seed,
            )

        return InstanceResult(
            solved=bool(terminated),
            total_reward=total_reward,
            num_steps=num_steps,
            cost_usd=0.0,
            frames=frames if render else None,
            extras={
                "planning_time": planning_time,
                "execution_time": execution_time,
                "env_step_time": env_step_time,
                "plan_length": num_steps,
                "symbolic_plan_length": len(plan),
                "twin_divergences": twin_divergences,
                "plan_found": True,
                **count_extra,
            },
        )
