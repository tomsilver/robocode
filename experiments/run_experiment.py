"""Run an experiment with a given approach and environment."""

import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _main(cfg: DictConfig) -> float:
    """Run a single experiment."""
    env = hydra.utils.instantiate(cfg.environment)
    sim = hydra.utils.instantiate(cfg.environment)
    approach = hydra.utils.instantiate(cfg.approach, simulator=sim, seed=cfg.seed)

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

    logger.info(
        "Total reward: %.1f, Steps: %d, Solved: %s",
        total_reward,
        num_steps,
        terminated,
    )
    return total_reward


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
