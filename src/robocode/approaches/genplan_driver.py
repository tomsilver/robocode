"""In-container entry point for the LLM-GenPlan and Best-of-K baselines.

Runs the whole generation loop inside the sandbox container (launched by
:func:`robocode.utils.docker_sandbox.run_genplan_in_docker`). Reads its config
from ``/sandbox/genplan_config.json`` and writes ``/sandbox/approach.py``. This
is the non-agentic analog of the agent CLI that the agentic approach runs. The
``approach`` field selects which baseline class to rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf

from robocode.approaches.best_of_k_approach import BestOfKApproach
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

    args = (env.action_space, env.observation_space, cfg["seed"], primitives)
    common: dict[str, Any] = {
        "completion": OmegaConf.create(cfg["completion"]),
        "env": env,
        "env_description_path": desc_path,
        "output_dir": "/",  # train() writes to output_dir/sandbox -> /sandbox
        "max_steps": cfg["max_steps"],
        "num_train_tasks": cfg["num_train_tasks"],
        "num_prompt_tasks": cfg["num_prompt_tasks"],
        "max_budget_usd": cfg["max_budget_usd"],
        "chain_of_thought": cfg["chain_of_thought"],
        "episode_timeout_s": cfg["episode_timeout_s"],
        "container_backend": "local",  # already isolated; run the loop in-process
    }
    approach: LLMGenPlanApproach
    if cfg.get("approach", "genplan") == "bestofk":
        approach = BestOfKApproach(
            *args, max_generation_steps=cfg["max_generation_steps"], **common
        )
    else:
        approach = LLMGenPlanApproach(
            *args, max_debug_attempts=cfg["max_debug_attempts"], **common
        )
    approach.train()
    # The host approach reads these back; the container is the only place the
    # accumulated API cost and generation count exist.
    (_SANDBOX / "cost.json").write_text(
        json.dumps(
            {
                "total_cost_usd": approach.total_cost_usd,
                "num_generations": approach.num_generations,
            }
        )
    )


if __name__ == "__main__":
    main()
