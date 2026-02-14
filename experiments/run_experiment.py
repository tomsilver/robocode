"""Run an experiment with a given approach and environment."""

import json
import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from robocode.simulators.env_simulator import EnvSimulator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _main(cfg: DictConfig) -> float:
    """Run a single experiment."""
    env = hydra.utils.instantiate(cfg.environment)
    sim = EnvSimulator(hydra.utils.instantiate(cfg.environment))
    approach = hydra.utils.instantiate(
        cfg.approach,
        simulator=sim,
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=cfg.seed,
    )

    state, info = env.reset(seed=cfg.seed)
    approach.reset(state, info)

    total_reward = 0.0
    num_steps = 0
    terminated = False
    for _ in range(cfg.max_steps):
        action = approach.step()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        num_steps += 1
        approach.update(state, float(reward), terminated or truncated, info)
        if terminated or truncated:
            break

    results = {
        "total_reward": total_reward,
        "num_steps": num_steps,
        "solved": terminated,
    }
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    logger.info(
        "Total reward: %.1f, Steps: %d, Solved: %s",
        total_reward,
        num_steps,
        terminated,
    )
    return total_reward


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
