"""Per-instance agentic baseline: a fresh agent run per eval seed.

Unlike the generalized :class:`AgenticApproach` (one program for all seeds), this
baseline spends eval-time agent budget on each seed in turn: for every eval seed
the coding agent writes a program targeting *that* instance, which is then scored on
it -- ``env.reset(seed=S)``, or the pinned ``env.reset(seed=S, options={"object_count":
K})`` for a variable-count env. The runner (``run_per_instance_eval``) owns the global
budget that carries across seeds; this class solves one instance at a time.
"""

import logging
from pathlib import Path
from typing import Any

from robocode.approaches.agentic_base import GeneratedProgramApproach
from robocode.approaches.base_approach import InstanceResult
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.utils.episode import run_episode

logger = logging.getLogger(__name__)


class AgenticPerInstanceApproach(GeneratedProgramApproach):
    """Solve each eval seed with its own budget-bounded agent run."""

    per_instance = True

    def train(self) -> None:
        # Per-instance approaches do not train a single generalized policy; the
        # runner drives them seed by seed via solve_instance. Fail loudly if the
        # generalized lifecycle is invoked by mistake.
        raise NotImplementedError(
            "AgenticPerInstanceApproach solves each seed via solve_instance; "
            "train() is not used (the runner branches on approach.per_instance)"
        )

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
        """Run the agent on a single seed, then score the program it wrote.

        The agent is told its target is the exact instance it is scored on --
        ``env.reset(seed=S)``, or ``env.reset(seed=S, options={"object_count": K})``
        for a variable-count env -- and writes a program that only needs to solve that
        instance. A fresh sandbox is used per seed: no state from prior seeds is
        carried in.
        """
        sandbox_dir = output_subdir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # The local MCP render server reads env_config.json from the sandbox's
        # parent dir. Each per-instance sandbox needs its own copy (the runner
        # writes only one, at the top-level output dir, for the generalized
        # sandbox). self._env_cfg holds the exact same env config JSON.
        if self._mcp_tools and not self._blackbox:
            assert self._env_cfg is not None
            (output_subdir / "env_config.json").write_text(
                self._env_cfg, encoding="utf-8"
            )

        prompt, system_prompt, init_files = self._build_agentic_prompts(
            per_instance_seed=seed, per_instance_count=count
        )
        result = self._run_sandbox(
            sandbox_dir=sandbox_dir,
            prompt=prompt,
            system_prompt=system_prompt,
            max_budget_usd=budget_usd,
            init_files=init_files,
        )
        cost = result.total_cost_usd or 0.0

        if not (result.success and result.output_file is not None):
            logger.warning(
                "Agent produced no approach for seed %d (%s); scoring as failure",
                seed,
                result.error,
            )
            return InstanceResult(
                solved=False,
                total_reward=None,
                num_steps=None,
                cost_usd=cost,
                crashed=True,
            )
        if result.error:
            logger.info(
                "Agent stopped early for seed %d (%s) but committed an approach; "
                "scoring it.",
                seed,
                result.error,
            )

        assert self._max_steps is not None  # the runner always provides max_steps
        # Each seed gets a fresh program; clear any policy from a prior seed.
        self._generated = None
        episode_max_steps = (
            env.max_steps_for_count(count)
            if count is not None and isinstance(env, VariableObjectCountEnv)
            else self._max_steps
        )
        try:
            self._load_generated(result.output_file)
            metrics, frames, _ = run_episode(
                env, self, seed, episode_max_steps, render=render, count=count
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Isolate a single seed's bad program (load error or runtime crash):
            # score it as a failure rather than aborting the whole per-seed sweep,
            # mirroring the runner's documented eval-crash policy. Cost is still
            # charged against the global budget.
            logger.exception(
                "Per-instance attempt for seed %d crashed (%s); scoring as failure",
                seed,
                exc,
            )
            return InstanceResult(
                solved=False,
                total_reward=None,
                num_steps=None,
                cost_usd=cost,
                crashed=True,
            )

        return InstanceResult(
            solved=metrics["solved"],
            total_reward=metrics["total_reward"],
            num_steps=metrics["num_steps"],
            cost_usd=cost,
            crashed=False,
            frames=frames,
            extras=(
                {"object_count": metrics["object_count"]}
                if "object_count" in metrics
                else {}
            ),
        )
