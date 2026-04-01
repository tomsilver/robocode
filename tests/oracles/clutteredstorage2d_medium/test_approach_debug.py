"""Debug episode test for the ClutteredStorage2D-b3 oracle.

Runs a single episode, stores per-step debug logs, and saves a rollout video. This test
is intentionally lightweight and does not require the oracle to solve the task yet.
"""

from __future__ import annotations

from pathlib import Path

import kinder
import pytest
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
DEBUG_SEEDS = [7,11, 12, 13, 14,  17, 18]
DEBUG_SEEDS =[21,34,35,45,49,52,60,64,67,74,80,84]
DEBUG_SEEDS =[84] 
MAX_STEPS = 800
ARTIFACT_ROOT = Path("unit_test_artifacts/clutteredstorage2d_medium")
VIDEO_DIR = ARTIFACT_ROOT / "videos"


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


@pytest.mark.parametrize("seed", DEBUG_SEEDS)
def test_debug_episode_writes_log_and_video(seed: int):
    """Run one debug episode per seed and persist per-seed artifacts."""
    kinder.register_all_environments()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT_ROOT / f"debug_seed{seed}.log"

    env = RecordVideo(
        kinder.make(ENV_ID, render_mode="rgb_array"),
        str(VIDEO_DIR),
        name_prefix=f"debug_seed{seed}",
    )
    video_files: list[Path] = []
    try:
        obs, info = env.reset(seed=seed)
        approach = ClutteredStorage2DOracleApproach(
            action_space=env.action_space,
            observation_space=env.observation_space,
        )
        approach.reset(obs, info)

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"env_id={ENV_ID}\n")
            log_file.write(f"seed={seed}\n")
            log_file.write(f"max_steps={MAX_STEPS}\n\n")

            for step in range(MAX_STEPS):
                robot = extract_robot(obs)
                outside = outside_blocks(obs)
                holding = holding_any_block(obs)
                held_name = held_block_name(obs)
                debug = approach.debug_snapshot()
                action = approach.step()
                log_file.write(
                    f"step={step:03d} "
                    f"robot=({robot.x:.3f},{robot.y:.3f},{robot.theta:.3f},"
                    f"arm={robot.arm_joint:.3f},vac={robot.vacuum:.1f}) "
                    f"action=({action[0]:.3f},{action[1]:.3f},{action[2]:.3f},"
                    f"{action[3]:.3f},{action[4]:.1f}) "
                    f"outside={outside} holding={holding} held={held_name}\n"
                )
                log_file.write(
                    "    "
                    f"phase={debug.get('phase')} active={debug.get('active_block')} "
                    f"target={debug.get('target_center')} "
                    f"candidate={debug.get('chosen_pick_pose')} "
                    f"path_len={debug.get('path_len')} "
                    f"queued={debug.get('queued_actions')}\n"
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

    video_files = sorted(VIDEO_DIR.glob(f"debug_seed{seed}-episode-*.mp4"))
    assert log_path.exists(), f"Expected debug log at {log_path}"
    assert video_files, f"No video file generated in {VIDEO_DIR}"
