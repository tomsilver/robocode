"""Best-of-K baseline: sample programs from the same prompt, keep the best.

The simplest LLM baseline, meant to isolate the *feedback* component of the
LLM-GenPlan loop. Under a fixed budget it draws independent ``GeneratedApproach``
candidates from the same prompt (no error feedback, no debug history), scores
each on the training tasks, and keeps the best one. Contrast with
:class:`~robocode.approaches.llm_genplan_approach.LLMGenPlanApproach`, which
re-prompts the model with structured error feedback between attempts.

It subclasses the GenPlan approach to reuse prompt construction, completion
clients, container execution, and policy delegation; only the generation loop
differs. ``chain_of_thought`` selects which GenPlan prompt to repeat: with it on,
each candidate runs GenPlan's summary -> strategy -> code flow independently;
off (the default), each candidate is a single prompt -> code sample.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from robocode.approaches.llm_genplan_approach import (
    LLMGenPlanApproach,
    _parse_python_code,
)
from robocode.utils.genplan_validate import TaskScore, score_tasks

logger = logging.getLogger(__name__)


class BestOfKApproach(LLMGenPlanApproach):
    """Sample independent policies from one prompt and keep the best-scoring one."""

    def __init__(
        self,
        *args: Any,
        chain_of_thought: bool = False,
        max_generation_steps: int | None = 5,
        **kwargs: Any,
    ) -> None:
        # GenPlan's CoT path is reused directly; Best-of-K has no debug attempts,
        # so its step cap is max_generation_steps (see _max_generations).
        kwargs["chain_of_thought"] = chain_of_thought
        kwargs["max_debug_attempts"] = None
        super().__init__(*args, **kwargs)
        self._max_generation_steps = max_generation_steps

    @property
    def _max_generations(self) -> int | None:
        """Number of candidates the step cap allows (None for no step cap)."""
        return self._max_generation_steps

    def _generate_local(
        self, approach_path: Path, sandbox_dir: Path, train_seeds: list[int]
    ) -> None:
        """Sample candidates from the same prompt; keep the best-scoring one."""
        context = self._build_context(train_seeds[: self._num_prompt_tasks])
        candidates_dir = sandbox_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        best_path: Path | None = None
        best_key: tuple[int, int, float] | None = None
        i = 0
        while self._within_budget(i):
            # Each candidate gets its own dir so its prompt/response artifacts
            # (CoT and implementation) are not overwritten by later candidates.
            # Fresh messages each candidate, with no feedback appended: a CoT run
            # resamples the summary/strategy independently, a non-CoT run resamples
            # the single prompt -> code completion.
            cand_dir = candidates_dir / f"cand{i}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            messages = self._build_initial_messages(context, cand_dir)
            response = self._complete(messages, cand_dir, "impl")
            cand_path = cand_dir / "approach.py"
            cand_path.write_text(_parse_python_code(response))
            if i == 0:
                self._assert_loop_bounded()

            score = self._score(cand_path, train_seeds)
            logger.info(
                "Candidate %d solved %d/%d (completed %d, mean reward %.3g)",
                i,
                score.num_solved,
                score.num_total,
                score.num_completed,
                score.mean_reward,
            )
            # Rank by solved, then completed (a runnable unsolved policy beats a
            # crashing one), then reward.
            key = (score.num_solved, score.num_completed, score.mean_reward)
            if best_key is None or key > best_key:
                best_key, best_path = key, cand_path
            i += 1
            if score.num_solved == score.num_total:
                logger.info(
                    "Candidate solved all %d training tasks; stopping", score.num_total
                )
                break

        assert best_path is not None, "no candidate was generated"
        self.num_generations = i  # number of candidates sampled
        approach_path.write_text(best_path.read_text())

    def _score(self, approach_path: Path, seeds: list[int]) -> TaskScore:
        """Score a candidate on the training tasks (see ``score_tasks``)."""
        assert self._env is not None
        return score_tasks(
            self._env,
            approach_path,
            self._action_space,
            self._state_space,
            self._primitives,
            seeds,
            self._max_steps,
            self._eval_timeout,
        )

    def _driver_config(self, completion: dict[str, Any]) -> dict[str, Any]:
        config = super()._driver_config(completion)
        config["approach"] = "bestofk"
        config["max_generation_steps"] = self._max_generation_steps
        return config
