"""Per-instance bilevel planning (SeSamE) baseline.

For each eval seed the SeSamE planner is run once to produce an open-loop action
sequence, which is then executed without replanning. This is a reference baseline
whose planning cost grows with the object count `N`: it degrades (slow, then no
plan within the timeout) as instances get harder, in contrast to a frozen
generalized program. It runs entirely on the host (no LLM, no sandbox), so every
attempt is free (`cost_usd=0.0`) and per-instance time is bounded by the shared
`eval_timeout` (planning) and `max_steps` (execution).

The bilevel planning models (predicates, operators, parameterized skills,
samplers, and a transition simulator) come from `kinder_bilevel_planning`, built
from the env's `bilevel_env_name` / `bilevel_env_model_kwargs` mapping (see
`KinderGeom2DEnv`).
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
from kinder_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent

from robocode.approaches.base_approach import BaseApproach, InstanceResult
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.utils.bilevel import build_sesame_models


class BilevelPlanningApproach(BaseApproach[Any, Any]):
    """Solve each eval seed with a fresh SeSamE plan, executed open-loop."""

    per_instance = True

    def __init__(
        self,
        action_space: Any,
        observation_space: Any,
        seed: int,
        primitives: dict[str, Any],
        *,
        max_steps: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        eval_timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(action_space, observation_space, seed, primitives, **kwargs)
        self._max_steps = max_steps
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._eval_timeout = eval_timeout
        # Fixed-count env: SesameModels depend only on the env, so build once and reuse.
        # Variable-count env: the models bake in the object count, so they are rebuilt
        # per count and cached by count (see _get_models).
        self._models: Any | None = None
        self._models_by_count: dict[int, Any] = {}
        self._agent: BilevelPlanningAgent | None = None

    def train(self) -> None:
        # Per-instance approaches solve each seed via solve_instance; the runner
        # branches on approach.per_instance and never calls train().
        raise NotImplementedError(
            "BilevelPlanningApproach solves each seed via solve_instance; "
            "train() is not used (the runner branches on approach.per_instance)"
        )

    def _get_models(self, env: Any, count: int | None) -> Any:
        # Variable-count models bake in the count, so cache one bundle per count.
        if count is not None and isinstance(env, VariableObjectCountEnv):
            if count not in self._models_by_count:
                self._models_by_count[count] = env.models_for_count(count)
            return self._models_by_count[count]
        if self._models is None:
            self._models = build_sesame_models(env)
        return self._models

    @staticmethod
    def _planner_obs(env: Any, obs: Any) -> Any:
        # The SeSamE models consume a fixed-length Box. A variable-count env yields an
        # object-centric state, which it vectorizes through the current count's Box
        # space; a fixed-count env already yields the Box vector.
        if isinstance(env, VariableObjectCountEnv):
            return env.to_box(obs)
        return obs

    def _get_action(self) -> Any:
        # solve_instance drives the planner directly; this keeps the ABC honest.
        assert self._agent is not None, "solve_instance must run planning first"
        return self._agent.step()

    def solve_instance(
        self,
        *,
        env: Any,
        seed: int,
        budget_usd: float,
        output_subdir: Path,
        render: bool = False,
        count: int | None = None,
    ) -> InstanceResult:
        """Plan once for this seed, then execute the plan open-loop.

        A planning timeout or an unreachable goal surfaces as ``AgentFailure`` and
        is scored as an unsolved (not crashed) attempt. The dollar ``budget_usd``
        is unused: the planner has no LLM cost, so ``cost_usd`` is always 0.0 and
        every seed is attempted. ``count`` pins a variable-count env's object count so
        the planner faces the same instance as the generalized program; the SeSamE
        models are then rebuilt for that count and the object-centric observation is
        vectorized to that count's Box space for the planner.
        """
        del budget_usd, output_subdir
        models = self._get_models(env, count)
        agent: BilevelPlanningAgent[Any, Any, Any] = BilevelPlanningAgent(
            models,
            seed=seed,
            max_abstract_plans=self._max_abstract_plans,
            samples_per_step=self._samples_per_step,
            max_skill_horizon=self._max_skill_horizon,
            heuristic_name=self._heuristic_name,
            planning_timeout=self._eval_timeout,
        )
        self._agent = agent

        if count is not None:
            obs, info = env.reset(seed=seed, options={"object_count": count})
        else:
            obs, info = env.reset(seed=seed)
        object_count = count
        if object_count is None and isinstance(env, VariableObjectCountEnv):
            object_count = env.current_count

        plan_start = time.perf_counter()
        try:
            agent.reset(self._planner_obs(env, obs), info)
            plan_found = True
        except AgentFailure:
            plan_found = False
        planning_time = time.perf_counter() - plan_start

        if not plan_found:
            return InstanceResult(
                solved=False,
                total_reward=None,
                num_steps=None,
                cost_usd=0.0,
                extras={
                    "planning_time": planning_time,
                    "plan_found": False,
                    "plan_length": 0,
                    **(
                        {"object_count": object_count}
                        if object_count is not None
                        else {}
                    ),
                },
            )

        frames: list[Any] = []

        def _capture() -> None:
            rendered = env.render()
            # render() may return a single frame, a list, or None; keep only
            # numpy RGB frames (matches run_episode and what save_video expects).
            if isinstance(rendered, np.ndarray):
                frames.append(rendered)

        if render:
            _capture()

        total_reward = 0.0
        num_steps = 0
        terminated = False
        execution_time = 0.0
        env_step_time = 0.0
        # A variable-count instance gets a horizon that grows with its object count,
        # so a large instance is not cut off before its (longer) plan can execute.
        max_steps = (
            env.max_steps_for_count(count)
            if count is not None and isinstance(env, VariableObjectCountEnv)
            else self._max_steps
        )
        for _ in range(max_steps):
            t0 = time.perf_counter()
            try:
                action = agent.step()
            except AgentFailure:
                # Plan exhausted before the goal was reached: stop and score.
                execution_time += time.perf_counter() - t0
                break
            execution_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            env_step_time += time.perf_counter() - t0
            total_reward += float(reward)
            num_steps += 1
            t0 = time.perf_counter()
            agent.update(
                self._planner_obs(env, obs),
                float(reward),
                terminated or truncated,
                info,
            )
            execution_time += time.perf_counter() - t0
            if render:
                _capture()
            if terminated or truncated:
                break

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
                "plan_found": True,
                **({"object_count": object_count} if object_count is not None else {}),
            },
        )
