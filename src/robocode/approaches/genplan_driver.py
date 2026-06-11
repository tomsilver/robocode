"""In-container entry point for the LLM-GenPlan baseline.

Runs the whole genplan loop inside the sandbox container (launched by
:func:`robocode.utils.docker_sandbox.run_genplan_in_docker`). Reads its config
from ``/sandbox/genplan_config.json`` and writes ``/sandbox/approach.py``. This
is the genplan analog of the agent CLI that the agentic approach runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from robocode.approaches.llm_genplan_approach import LLMGenPlanApproach
from robocode.primitives import build_primitives

_SANDBOX = Path("/sandbox")


def main() -> None:
    """Reconstruct the approach from the sandbox config and run its train loop."""
    cfg = json.loads((_SANDBOX / "genplan_config.json").read_text(encoding="utf-8"))
    env_cfg = OmegaConf.create(cfg["environment"])
    env = hydra.utils.instantiate(env_cfg)
    primitives = build_primitives(env, cfg["primitive_names"])

    desc_path: str | None = None
    if env.env_description is not None:
        (_SANDBOX / "env_description.md").write_text(env.env_description)
        desc_path = str(_SANDBOX / "env_description.md")

    approach = LLMGenPlanApproach(
        env.action_space,
        env.observation_space,
        cfg["seed"],
        primitives,
        completion=OmegaConf.create(cfg["completion"]),
        env=env,
        env_description_path=desc_path,
        output_dir="/",  # train() writes to output_dir/sandbox -> /sandbox
        max_steps=cfg["max_steps"],
        num_train_tasks=cfg["num_train_tasks"],
        num_prompt_tasks=cfg["num_prompt_tasks"],
        max_debug_attempts=cfg["max_debug_attempts"],
        skip_chain_of_thought=cfg["skip_chain_of_thought"],
        episode_timeout_s=cfg["episode_timeout_s"],
        container_backend="local",  # already isolated; run the loop in-process
    )
    approach.train()
    # The host approach reads this back; the container is the only place the
    # accumulated API cost exists.
    (_SANDBOX / "cost.json").write_text(
        json.dumps({"total_cost_usd": approach.total_cost_usd})
    )


if __name__ == "__main__":
    main()
