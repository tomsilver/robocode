"""Run an experiment with a given approach and environment."""

import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import imageio.v3 as iio
import numpy as np
from hydra.core.hydra_config import HydraConfig
from numpy.typing import NDArray
from omegaconf import DictConfig

from robocode.approaches.base_approach import BaseApproach
from robocode.primitives.check_action_collision import check_action_collision
from robocode.primitives.motion_planning import BiRRT
from robocode.primitives.render_state import render_state

logger = logging.getLogger(__name__)


def _run_episode(
    env: gym.Env,
    approach: BaseApproach,
    seed: int,
    max_steps: int,
    render: bool = False,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]]]:
    """Run a single evaluation episode and return metrics + frames."""
    state, info = env.reset(seed=seed)
    approach.reset(state, info)

    frames: list[NDArray[np.uint8]] = []

    def _capture() -> None:
        rendered: Any = env.render()
        if isinstance(rendered, np.ndarray):
            frames.append(rendered)

    if render:
        _capture()

    total_reward = 0.0
    num_steps = 0
    terminated = False
    for _ in range(max_steps):
        action = approach.step()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        num_steps += 1
        approach.update(state, float(reward), terminated or truncated, info)
        if render:
            _capture()
        if terminated or truncated:
            break

    metrics = {
        "total_reward": total_reward,
        "num_steps": num_steps,
        "solved": bool(terminated),
    }
    return metrics, frames


def _save_video(frames: list[NDArray[np.uint8]], path: Path, fps: int = 10) -> None:
    """Save a list of RGB frames as a gif."""
    duration = 1000.0 / fps  # ms per frame
    iio.imwrite(str(path), frames, duration=duration, loop=0)
    logger.info("Saved video to %s", path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _main(cfg: DictConfig) -> float:
    """Run a single experiment."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = hydra.utils.instantiate(cfg.environment)

    # If the environment provides a description (e.g. kinder envs), write it
    # to a file so the agentic approach can read it in its sandbox.
    env_description_path: str | None = None
    if env.env_description is not None:
        desc_path = output_dir / "env_description.md"
        desc_path.write_text(env.env_description)
        env_description_path = str(desc_path)

    all_primitives = {
        "check_action_collision": partial(check_action_collision, env),
        "render_state": partial(render_state, env),
        "BiRRT": BiRRT,
    }
    primitives = {name: all_primitives[name] for name in cfg.primitives}

    approach = hydra.utils.instantiate(
        cfg.approach,
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=cfg.seed,
        primitives=primitives,
        env_description_path=env_description_path,
    )

    task_rng = np.random.default_rng(cfg.seed)
    num_eval = cfg.num_eval_tasks
    eval_seeds = [int(task_rng.integers(0, 2**63)) for _ in range(num_eval)]

    approach.train()

    # Evaluate on held-out episodes.
    render = cfg.render_videos
    per_episode = []
    for i, s in enumerate(eval_seeds):
        episode_result, frames = _run_episode(
            env, approach, s, cfg.max_steps, render=render
        )
        per_episode.append(episode_result)
        if frames:
            video_dir = output_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            _save_video(frames, video_dir / f"episode_{i}.gif")

    mean_reward = float(np.mean([e["total_reward"] for e in per_episode]))
    mean_steps = float(np.mean([e["num_steps"] for e in per_episode]))
    solve_rate = float(np.mean([e["solved"] for e in per_episode]))

    results = {
        "mean_eval_reward": mean_reward,
        "mean_eval_steps": mean_steps,
        "solve_rate": solve_rate,
        "num_eval_tasks": num_eval,
        "per_episode": per_episode,
    }
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
