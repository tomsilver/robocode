"""An approach that uses an LLM coding agent to generate approach code."""

import logging

from robocode.approaches.agentic_base import GeneratedProgramApproach

logger = logging.getLogger(__name__)


class AgenticApproach(GeneratedProgramApproach):
    """Generalized planning: one agent run writes one program for all seeds.

    ``train()`` runs the coding agent once (bounded by the dollar budget) and
    loads the resulting ``GeneratedApproach``; the runner then rolls that single
    frozen policy out over every eval seed at no further LLM cost. The reusable
    sandbox and prompt machinery lives in :class:`GeneratedProgramApproach`.
    """

    def train(self) -> None:
        if self._load_dir is not None:
            approach_file = self._load_dir / "sandbox" / "approach.py"
            if not approach_file.exists():
                raise FileNotFoundError(f"No approach file at {approach_file}")
            self._load_generated(approach_file)
            return

        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        prompt, system_prompt, init_files = self._build_agentic_prompts()
        result = self._run_sandbox(
            sandbox_dir=sandbox_dir,
            prompt=prompt,
            system_prompt=system_prompt,
            max_budget_usd=self._max_budget_usd,
            init_files=init_files,
        )

        self.total_cost_usd = result.total_cost_usd
        self.generation_metrics = result.generation_metrics

        if result.success and result.output_file is not None:
            if result.error:
                logger.info(
                    "Agent stopped early (%s) but committed an approach; "
                    "evaluating it.",
                    result.error,
                )
            self._load_generated(result.output_file)
        else:
            raise RuntimeError(f"Agent failed to generate an approach: {result.error}")
