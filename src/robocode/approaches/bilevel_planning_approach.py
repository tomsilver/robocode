"""Per-instance bilevel planning (SeSamE) baseline.

For each eval seed the SeSamE planner is run once to produce an open-loop action
sequence, which is then executed without replanning. This is a reference baseline
whose planning cost grows with the object count `N`: it degrades (slow, then no
plan within the timeout) as instances get harder, in contrast to a frozen
generalized program. It runs entirely on the host (no LLM, no sandbox), so every
attempt is free (`cost_usd=0.0`) and per-instance time is bounded by
`planning_timeout` (planning) and `max_steps` (execution).

The bilevel planning models (predicates, operators, parameterized skills,
samplers, and a transition simulator) come from `kinder_bilevel_planning`, built
from the env's `bilevel_env_name` / `bilevel_env_model_kwargs` mapping (see
`KinderGeom2DEnv`).
"""

import time
from pathlib import Path
from typing import Any

from kinder_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from robocode.approaches.base_approach import BaseApproach, InstanceResult


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
        env: Any,
        max_steps: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        planning_timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(action_space, observation_space, seed, primitives, **kwargs)
        self._env = env
        self._max_steps = max_steps
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._planning_timeout = planning_timeout
        # SesameModels depend only on the (fixed) env, so build once and reuse.
        self._models: Any | None = None
        self._agent: BilevelPlanningAgent | None = None

    def train(self) -> None:
        # Per-instance approaches solve each seed via solve_instance; the runner
        # branches on approach.per_instance and never calls train().
        raise NotImplementedError(
            "BilevelPlanningApproach solves each seed via solve_instance; "
            "train() is not used (the runner branches on approach.per_instance)"
        )

    def _get_models(self) -> Any:
        if self._models is None:
            assert self._env.bilevel_env_name is not None, (
                "BilevelPlanningApproach needs bilevel_env_name on the environment; "
                "add bilevel_env_name and bilevel_env_model_kwargs to the env config."
            )
            self._models = create_bilevel_planning_models(
                self._env.bilevel_env_name,
                self._env.observation_space,
                self._env.action_space,
                **self._env.bilevel_env_model_kwargs,
            )
        return self._models

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
    ) -> InstanceResult:
        """Plan once for this seed, then execute the plan open-loop.

        A planning timeout or an unreachable goal surfaces as ``AgentFailure`` and
        is scored as an unsolved (not crashed) attempt. The dollar ``budget_usd``
        is unused: the planner has no LLM cost, so ``cost_usd`` is always 0.0 and
        every seed is attempted.
        """
        del budget_usd, output_subdir
        models = self._get_models()
        agent: BilevelPlanningAgent[Any, Any, Any] = BilevelPlanningAgent(
            models,
            seed=seed,
            max_abstract_plans=self._max_abstract_plans,
            samples_per_step=self._samples_per_step,
            max_skill_horizon=self._max_skill_horizon,
            heuristic_name=self._heuristic_name,
            planning_timeout=self._planning_timeout,
        )
        self._agent = agent

        obs, info = env.reset(seed=seed)

        plan_start = time.perf_counter()
        try:
            agent.reset(obs, info)
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
                },
            )

        frames: list[Any] = []

        def _capture() -> None:
            rendered = env.render()
            if rendered is not None:
                frames.append(rendered)

        if render:
            _capture()

        total_reward = 0.0
        num_steps = 0
        terminated = False
        execution_time = 0.0
        env_step_time = 0.0
        for _ in range(self._max_steps):
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
            agent.update(obs, float(reward), terminated or truncated, info)
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
            },
        )
