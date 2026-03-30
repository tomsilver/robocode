"""Debug episode test for the ClutteredStorage2D-b3 oracle.

Runs a single episode, stores per-step debug logs, and saves a rollout video.
This test is intentionally lightweight and does not require the oracle to solve
the task yet.
"""

from __future__ import annotations

from pathlib import Path

import kinder
from gymnasium.wrappers import RecordVideo

from robocode.oracles.clutteredstorage2d_medium.approach import (
    ClutteredStorage2DOracleApproach,
)
from robocode.oracles.clutteredstorage2d_medium.obs_helpers import (
    BLOCK_NAMES,
    all_blocks_inside_shelf,
    extract_block,
    extract_robot,
    held_block_name,
    holding_any_block,
    outside_blocks,
)

ENV_ID = "kinder/ClutteredStorage2D-b3-v0"
SEED = 0
MAX_STEPS = 500
ARTIFACT_ROOT = Path("unit_test_artifacts/clutteredstorage2d_medium")
VIDEO_DIR = ARTIFACT_ROOT / "videos"
LOG_PATH = ARTIFACT_ROOT / "debug_seed0.log"


def _format_block_state(obs) -> str:
    entries: list[str] = []
    for name in BLOCK_NAMES:
        block = extract_block(obs, name)
        center_x, center_y = block.center
        entries.append(
            (
                f"{name}: pos=({block.x:.3f},{block.y:.3f}) "
                f"center=({center_x:.3f},{center_y:.3f}) "
                f"theta={block.theta:.3f}"
            )
        )
    return " | ".join(entries)


def test_debug_episode_writes_log_and_video():
    """Run one episode and persist debug artifacts for manual inspection."""
    kinder.register_all_environments()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = RecordVideo(
        kinder.make(ENV_ID, render_mode="rgb_array"),
        str(VIDEO_DIR),
        name_prefix="debug_seed0",
    )
    video_files: list[Path] = []
    try:
        obs, info = env.reset(seed=SEED)
        approach = ClutteredStorage2DOracleApproach(
            action_space=env.action_space,
            observation_space=env.observation_space,
        )
        approach.reset(obs, info)

        with LOG_PATH.open("w", encoding="utf-8") as log_file:
            log_file.write(f"env_id={ENV_ID}\n")
            log_file.write(f"seed={SEED}\n")
            log_file.write(f"max_steps={MAX_STEPS}\n\n")

            for step in range(MAX_STEPS):
                robot = extract_robot(obs)
                outside = outside_blocks(obs)
                holding = holding_any_block(obs)
                held_name = held_block_name(obs)
                action = approach.step()
                log_file.write(
                    f"step={step:03d} "
                    f"robot=({robot.x:.3f},{robot.y:.3f},{robot.theta:.3f},"
                    f"arm={robot.arm_joint:.3f},vac={robot.vacuum:.1f}) "
                    f"action=({action[0]:.3f},{action[1]:.3f},{action[2]:.3f},"
                    f"{action[3]:.3f},{action[4]:.1f}) "
                    f"outside={outside} holding={holding} held={held_name}\n"
                )
                log_file.write(f"    {_format_block_state(obs)}\n")

                obs, reward, terminated, truncated, info = env.step(action)
                approach.update(obs, float(reward), terminated or truncated, info)

                log_file.write(
                    f"    reward={float(reward):.1f} terminated={terminated} "
                    f"truncated={truncated} all_inside={all_blocks_inside_shelf(obs)}\n"
                )

                if terminated or truncated:
                    log_file.write(f"episode_end_step={step + 1}\n")
                    break
            else:
                log_file.write("episode_end_step=max_steps\n")
    finally:
        env.close()

    video_files = sorted(VIDEO_DIR.glob("debug_seed0-episode-*.mp4"))
    assert LOG_PATH.exists(), f"Expected debug log at {LOG_PATH}"
    assert video_files, f"No video file generated in {VIDEO_DIR}"
