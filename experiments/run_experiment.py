"""Run an experiment with a given approach and environment.

Example usage:

    python experiments/run_experiment.py approach=agentic environment=motion2d_easy
    python experiments/run_experiment.py approach=agentic \
        approach.container_backend=docker 'primitives=[]' environment=motion2d_easy

The sandbox backend is selected with approach.container_backend=docker|apptainer|local
(default docker for agentic and llm_genplan; local runs unsandboxed in-process).

Parallel sweep with joblib launcher:

    python experiments/run_experiment.py -m \
        approach=agentic \
        approach.container_backend=docker \
        seed=42,24,424,444,222 \
        'primitives=[]' \
        environment=motion2d_easy,obstruction2d_easy,clutteredretrieval2d_easy \
        'hydra.sweep.dir=multirun/2026-02-23/no_primitives_5d_s42_24_424_444_222' \
        'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}' \
        hydra/launcher=joblib hydra.launcher.n_jobs=4
"""

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from robocode.primitives import build_primitives
from robocode.utils.approach_history import get_snapshots, record_episodes
from robocode.utils.episode import run_episode, save_video

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _main(cfg: DictConfig) -> float:
    """Run a single experiment."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = hydra.utils.instantiate(cfg.environment)

    # If the environment provides a description (e.g. kinder envs), write it
    # to a file so the agentic approach can read it in its sandbox. In blackbox
    # mode, use the variant that omits source-code pointers and direct-import
    # examples, since the agent has access to neither.
    blackbox = bool(cfg.approach.get("blackbox", False))
    description = env.env_description_blackbox if blackbox else env.env_description
    env_description_path: str | None = None
    if description is not None:
        desc_path = output_dir / "env_description.md"
        desc_path.write_text(description)
        env_description_path = str(desc_path)

    primitives = build_primitives(env, cfg.primitives)

    # Write env config for MCP server (if mcp_tools are configured).
    mcp_tools = tuple(cfg.get("mcp_tools", []))
    if mcp_tools:
        env_config_path = output_dir / "env_config.json"
        env_config_path.write_text(
            json.dumps(OmegaConf.to_container(cfg.environment, resolve=True))
        )

    # The Hydra environment choice name (e.g. "obstruction2d_medium") is used
    # to locate pre-written helper files in src/robocode/primitives/<env_name>/.
    env_name = HydraConfig.get().runtime.choices.get("environment")

    approach = hydra.utils.instantiate(
        cfg.approach,
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=cfg.seed,
        primitives=primitives,
        env_description_path=env_description_path,
        mcp_tools=mcp_tools,
        env_name=env_name,
        env=env,
        # JSON (not the DictConfig) so hydra.utils.instantiate doesn't recursively
        # instantiate it; the llm_genplan docker driver rebuilds the env from it.
        env_cfg=json.dumps(OmegaConf.to_container(cfg.environment, resolve=True)),
        max_steps=cfg.max_steps,
    )

    task_rng = np.random.default_rng(cfg.seed)
    num_eval = cfg.num_eval_tasks
    eval_seeds = [int(task_rng.integers(0, 2**63)) for _ in range(num_eval)]

    approach.train()

    # Record approach history: replay every sandbox snapshot.
    if cfg.record_approach_history:
        load_dir = cfg.approach.get("load_dir", None)
        sandbox_dir = Path(load_dir) / "sandbox" if load_dir else output_dir / "sandbox"
        snapshots = get_snapshots(sandbox_dir)
        record_episodes(
            snapshots,
            sandbox_dir,
            env,
            primitives,
            cfg.seed,
            max_steps=cfg.max_steps,
            output_dir=output_dir,
        )

    # Evaluate on held-out episodes.
    render = cfg.render_videos
    per_episode: list[dict[str, Any]] = []
    for i, s in enumerate(eval_seeds):
        try:
            episode_result, frames, _ = run_episode(
                env, approach, s, cfg.max_steps, render=render
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # A generated policy can raise on an unseen eval seed; count that as a
            # failed episode instead of aborting the whole evaluation. Record the
            # error so a crash is distinguishable from an ordinary unsolved episode.
            logger.exception("Eval episode (seed %d) crashed; counting as unsolved", s)
            per_episode.append(
                {
                    "total_reward": 0.0,
                    "num_steps": 0,
                    "solved": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        per_episode.append(episode_result)
        if frames:
            video_dir = output_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            save_video(frames, video_dir / f"episode_{i}.gif")

    mean_reward = float(np.mean([e["total_reward"] for e in per_episode]))
    mean_steps = float(np.mean([e["num_steps"] for e in per_episode]))
    solve_rate = float(np.mean([e["solved"] for e in per_episode]))

    results: dict[str, Any] = {
        "mean_eval_reward": mean_reward,
        "mean_eval_steps": mean_steps,
        "solve_rate": solve_rate,
        "num_eval_tasks": num_eval,
        "per_episode": per_episode,
    }
    agent_cost = getattr(approach, "total_cost_usd", None)
    if agent_cost is not None:
        results["agent_cost_usd"] = agent_cost
    num_generations = getattr(approach, "num_generations", None)
    if num_generations is not None:
        results["num_generations"] = num_generations
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    logger.info(
        "Mean reward: %.1f, Mean steps: %.1f, Solve rate: %.2f",
        mean_reward,
        mean_steps,
        solve_rate,
    )
    return mean_reward


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
