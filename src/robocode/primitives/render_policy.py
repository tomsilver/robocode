"""Render-policy primitive"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robocode.approaches.agentic_approach import AgenticApproach
from robocode.utils.episode import run_episode, save_frames


def render_policy(
    env: Any,
    primitives: dict[str, Any],
    approach_dir: str | Path,
    seed: int,
    output_dir: str | Path,
    max_steps: int = 1000,
    max_frames: int = 100,
) -> list[str]:
    """Run one episode of the approach and save every frame as a PNG."""
    
    approach_dir = Path(approach_dir)
    output_dir = Path(output_dir)

    approach = AgenticApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=seed,
        primitives=primitives,
        load_dir=str(approach_dir),
    )
    approach.train()

    saved_state = env.get_state()
    try:
        _, frames = run_episode(env, approach, seed, max_steps, render=True)
    finally:
        env.set_state(saved_state)

    return save_frames(frames, output_dir, max_frames=max_frames)
