"""Run an experiment with a given approach and environment.

Example usage:

    python experiments/run_experiment.py approach=agentic environment=motion2d_easy
    python experiments/run_experiment.py approach=agentic approach.use_docker=true \
        'primitives=[]' environment=motion2d_easy

Parallel sweep with joblib launcher:

    python experiments/run_experiment.py -m \
        approach=agentic \
        approach.use_docker=true \
        seed=42,24,424,444,222 \
        'primitives=[]' \
        environment=motion2d_easy,obstruction2d_easy,clutteredretrieval2d_easy \
        'hydra.sweep.dir=multirun/2026-02-23/no_primitives_5d_s42_24_424_444_222' \
        'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}' \
        hydra/launcher=joblib hydra.launcher.n_jobs=4
"""

import json
import logging
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import imageio.v3 as iio
import numpy as np
from gymnasium.wrappers import RecordVideo
from hydra.core.hydra_config import HydraConfig
from numpy.typing import NDArray
from omegaconf import DictConfig

from robocode.approaches.base_approach import BaseApproach
from robocode.primitives import csp as csp_module
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
    ics = getattr(env.unwrapped, "initial_constant_state", None)
    if ics is not None:
        info["initial_constant_state"] = ics
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


_SMALL_TEST_MAX_STEPS = 500


def _run_small_test_episode(
    env: gym.Env,
    approach: BaseApproach,
    init_state: Any,
    video_dir: Path,
    episode_id: str,
    max_steps: int = _SMALL_TEST_MAX_STEPS,
) -> dict[str, Any]:
    """Run one episode from a saved initial state, save video with tag."""
    tmp_video_dir = video_dir / f"_tmp_{episode_id}"
    rec_env = RecordVideo(
        env,
        str(tmp_video_dir),
        episode_trigger=lambda _: True,
        name_prefix=episode_id,
    )

    # Reset internals, then override with saved state.
    rec_env.reset(seed=0)
    env.unwrapped.set_state(init_state)
    state = init_state
    info: dict[str, Any] = {}
    ics = getattr(env.unwrapped, "initial_constant_state", None)
    if ics is not None:
        info["initial_constant_state"] = ics

    approach.reset(state, info)

    total_reward = 0.0
    solved = False
    num_steps = 0
    for num_steps in range(1, max_steps + 1):
        action = approach.step()
        state, reward, terminated, truncated, info = rec_env.step(action)
        total_reward += float(reward)
        approach.update(state, float(reward), terminated or truncated, info)
        if terminated:
            solved = True
            break
        if truncated:
            break

    rec_env.close()

    # Rename video with success/failed tag.
    tag = "success" if solved else "failed"
    for mp4 in Path(tmp_video_dir).glob("*.mp4"):
        dest = video_dir / f"{episode_id}_{tag}.mp4"
        shutil.move(str(mp4), str(dest))
    shutil.rmtree(tmp_video_dir, ignore_errors=True)

    return {
        "episode_id": episode_id,
        "total_reward": total_reward,
        "num_steps": num_steps,
        "solved": solved,
    }


def _run_small_test(
    approach: BaseApproach,
    output_dir: Path,
    failed_state_dir: str | None,
    success_state_dir: str | None,
) -> dict[str, Any]:
    """Evaluate the approach on failed + successful initial states."""
    from kinder.envs.kinematic2d.pushpullhook2d import (  # pylint: disable=import-outside-toplevel
        ObjectCentricPushPullHook2DEnv,
    )

    env = ObjectCentricPushPullHook2DEnv(
        render_mode="rgb_array", allow_state_access=True
    )
    video_dir = output_dir / "test_videos"
    video_dir.mkdir(exist_ok=True)

    episodes: list[tuple[str, Any]] = []
    for label, state_dir_str in [
        ("failed", failed_state_dir),
        ("success", success_state_dir),
    ]:
        if state_dir_str is None:
            continue
        state_dir = Path(state_dir_str)
        if not state_dir.is_dir():
            logger.warning("State dir not found, skipping: %s", state_dir)
            continue
        for npy_file in sorted(state_dir.glob("*.npy")):
            state = np.load(str(npy_file), allow_pickle=True).item()
            ep_id = f"{label}_{npy_file.stem}"
            episodes.append((ep_id, state))

    per_episode = []
    for ep_id, init_state in episodes:
        result = _run_small_test_episode(
            env, approach, init_state, video_dir, ep_id
        )
        per_episode.append(result)
        tag = "SOLVED" if result["solved"] else "FAILED"
        logger.info(
            "  %s: %s (steps=%d, reward=%.1f)",
            ep_id, tag, result["num_steps"], result["total_reward"],
        )

    # Compute per-group and overall solve rates.
    failed_eps = [e for e in per_episode if e["episode_id"].startswith("failed_")]
    success_eps = [e for e in per_episode if e["episode_id"].startswith("success_")]

    summary: dict[str, Any] = {"per_episode": per_episode}
    if failed_eps:
        rate = float(np.mean([e["solved"] for e in failed_eps]))
        summary["failed_solve_rate"] = rate
        logger.info("Failed states solve rate: %.0f%% (%d/%d)",
                     rate * 100, sum(e["solved"] for e in failed_eps), len(failed_eps))
    if success_eps:
        rate = float(np.mean([e["solved"] for e in success_eps]))
        summary["success_solve_rate"] = rate
        logger.info("Success states solve rate: %.0f%% (%d/%d)",
                     rate * 100, sum(e["solved"] for e in success_eps), len(success_eps))
    if per_episode:
        rate = float(np.mean([e["solved"] for e in per_episode]))
        summary["overall_solve_rate"] = rate
        summary["mean_reward"] = float(np.mean([e["total_reward"] for e in per_episode]))

    return summary


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
        "csp": csp_module,
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

    if cfg.small_test:
        # Quick evaluation on the failed/success state dirs used for training.
        failed_dir = getattr(cfg.approach, "failed_state_dir", None)
        success_dir = getattr(cfg.approach, "success_state_dir", None)
        results = _run_small_test(
            approach, output_dir, failed_dir, success_dir
        )
        mean_reward = results.get("mean_reward", 0.0)
    else:
        # Full evaluation on held-out episodes.
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

    agent_cost = getattr(approach, "total_cost_usd", None)
    if agent_cost is not None:
        results["agent_cost_usd"] = agent_cost
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    logger.info("Results saved to %s", results_path)
    return mean_reward


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
