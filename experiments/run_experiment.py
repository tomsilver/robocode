"""Run an experiment with a given approach and environment."""

import json
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from robocode.approaches.base_approach import BaseApproach

logger = logging.getLogger(__name__)


def _run_episode(
    env: gym.Env,
    approach: BaseApproach,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Run a single evaluation episode and return metrics."""
    state, info = env.reset(seed=seed)
    approach.reset(state, info)

    total_reward = 0.0
    num_steps = 0
    terminated = False
    for _ in range(max_steps):
        action = approach.step()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        num_steps += 1
        approach.update(state, float(reward), terminated or truncated, info)
        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "num_steps": num_steps,
        "solved": terminated,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _main(cfg: DictConfig) -> float:
    """Run a single experiment."""
    env = hydra.utils.instantiate(cfg.environment)
    approach = hydra.utils.instantiate(
        cfg.approach,
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=cfg.seed,
    )

    task_rng = np.random.default_rng(cfg.seed)
    num_train = cfg.num_train_tasks
    num_eval = cfg.num_eval_tasks
    train_seeds = [int(task_rng.integers(0, 2**63)) for _ in range(num_train)]
    eval_seeds = [int(task_rng.integers(0, 2**63)) for _ in range(num_eval)]
    assert not set(train_seeds) & set(eval_seeds), "Train/eval seed collision"

    # Collect training initial states.
    train_states = []
    for s in train_seeds:
        state, info = env.reset(seed=s)
        train_states.append((state, info))
    approach.train(train_states)

    # Evaluate on held-out episodes.
    per_episode = []
    for s in eval_seeds:
        episode_result = _run_episode(env, approach, s, cfg.max_steps)
        per_episode.append(episode_result)

    mean_reward = float(np.mean([e["total_reward"] for e in per_episode]))
    mean_steps = float(np.mean([e["num_steps"] for e in per_episode]))
    solve_rate = float(np.mean([e["solved"] for e in per_episode]))

    results = {
        "mean_eval_reward": mean_reward,
        "mean_eval_steps": mean_steps,
        "solve_rate": solve_rate,
        "num_train_tasks": num_train,
        "num_eval_tasks": num_eval,
        "per_episode": per_episode,
    }
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
