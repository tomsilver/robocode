"""Render-policy helper: run an approach and save every frame as a PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robocode.utils.episode import load_generated_approach, save_frames


def render_policy(
    env: Any,
    primitives: dict[str, Any],
    approach_dir: str | Path,
    seed: int,
    output_dir: str | Path,
    max_steps: int = 1000,
    max_frames: int = 100,
    object_count: int | None = None,
) -> list[str]:
    """Run one episode of the sandbox's approach.py and save every frame.

    Loads the GeneratedApproach directly (reset/get_action interface) rather than
    wrapping it in AgenticApproach, so this module does not depend on the
    robocode.primitives package (stripped from the agent sandbox). ``object_count`` pins
    the object count for a variable-count env so the rollout matches the scored instance.
    The env's own state is saved and restored so rendering leaves no side effect.
    """
    approach_dir = Path(approach_dir)
    output_dir = Path(output_dir)
    path = approach_dir / "approach.py"
    if not path.exists():
        path = approach_dir / "sandbox" / "approach.py"
    approach = load_generated_approach(
        path, env.action_space, env.observation_space, primitives
    )
    options = {"object_count": object_count} if object_count is not None else None
    saved_state = env.get_state()
    frames: list[np.ndarray] = []
    try:
        obs, info = env.reset(seed=seed, options=options)
        approach.reset(obs, info)
        rendered = env.render()
        if isinstance(rendered, np.ndarray):
            frames.append(rendered)
        for _ in range(max_steps):
            obs, _, terminated, truncated, _ = env.step(approach.get_action(obs))
            rendered = env.render()
            if isinstance(rendered, np.ndarray):
                frames.append(rendered)
            if terminated or truncated:
                break
    finally:
        env.set_state(saved_state)
    return save_frames(frames, output_dir, max_frames=max_frames)
