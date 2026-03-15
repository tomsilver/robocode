"""Evaluate an approach on an environment over multiple seeds, saving videos."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic2d.pushpullhook2d import ObjectCentricPushPullHook2DEnv
from imageio.v2 import imwrite

from robocode.skills.pushpullhook2d.approach import GeneratedApproach
from tests.conftest import MAKE_VIDEOS

NUM_SEEDS = 10
MAX_STEPS = 500
VIDEO_DIR = "test_approach_videos"


def _run_approach_on_seed(
    seed: int,
    make_videos: bool = False,
) -> tuple[bool, int, float]:
    """Run the approach on one seed, return (solved, steps, total_reward)."""
    env = ObjectCentricPushPullHook2DEnv(render_mode="rgb_array")

    if make_videos:
        # Each seed gets its own temp folder; we rename after.
        seed_video_dir = os.path.join(VIDEO_DIR, f"_tmp_seed_{seed}")
        env = RecordVideo(
            env,
            seed_video_dir,
            episode_trigger=lambda _: True,
            name_prefix=f"seed_{seed:02d}",
        )

    state, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, 
                                 initial_constant_state=env.unwrapped.initial_constant_state)
    approach.reset(state, info)

    total_reward = 0.0
    solved = False
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        action = approach.get_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            solved = True
            break
        if truncated:
            break

    env.close()

    # Rename video file to include success/failure.
    if make_videos:
        tag = "success" if solved else "failed"
        src_dir = Path(seed_video_dir)
        for mp4 in src_dir.glob("*.mp4"):
            dest = Path(VIDEO_DIR) / f"episode_{seed:02d}_{tag}.mp4"
            shutil.move(str(mp4), str(dest))
        # Clean up temp dir.
        shutil.rmtree(seed_video_dir, ignore_errors=True)

    return solved, steps, total_reward


def test_pushpullhook2d_approach_10_seeds() -> None:
    """Run the PushPullHook2D approach on 10 seeds and report results."""
    if MAKE_VIDEOS:
        Path(VIDEO_DIR).mkdir(exist_ok=True)

    results: list[tuple[int, bool, int, float]] = []
    for seed in range(NUM_SEEDS):
        solved, steps, reward = _run_approach_on_seed(
            seed, make_videos=MAKE_VIDEOS
        )
        results.append((seed, solved, steps, reward))

    # Print summary table.
    num_solved = sum(1 for _, s, _, _ in results if s)
    print(f"\n{'Seed':>4} {'Result':>8} {'Steps':>6} {'Reward':>8}")
    print("-" * 32)
    for seed, solved, steps, reward in results:
        tag = "SOLVED" if solved else "FAILED"
        print(f"{seed:>4} {tag:>8} {steps:>6} {reward:>8.1f}")
    print("-" * 32)
    print(f"Solve rate: {num_solved}/{NUM_SEEDS}")

    # We don't assert a specific solve rate — this test is for evaluation
    # and video generation, not a pass/fail gate.
