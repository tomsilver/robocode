"""Evaluate a generated approach on PushPullHook2D, saving videos.

Usage examples:

    # Run 20 seeded episodes (default):
    python scripts/test_approach_generated.py path/to/approach.py

    # Run with custom seed count:
    python scripts/test_approach_generated.py path/to/approach.py --num_tests 50

    # Run on specific initial-state directories:
    python scripts/test_approach_generated.py path/to/approach.py \
        --state_dirs init_states/pushpullhook2d/pull init_states/pushpullhook2d/regrasp

Videos are saved to the approach's parent directory under a ``test_videos/``
subfolder, named ``episode_<id>_success.mp4`` or ``episode_<id>_failed.mp4``.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic2d.pushpullhook2d import ObjectCentricPushPullHook2DEnv
from tqdm import tqdm

MAX_STEPS = 500


# ------------------------------------------------------------------
# Loading helper
# ------------------------------------------------------------------


def _load_approach(approach_path: Path) -> type:
    """Load GeneratedApproach class from *approach_path*.

    The approach's parent directory is temporarily added to sys.path so
    that local skill imports (e.g. ``from pull_skill import PullController``)
    resolve naturally — no separate skill discovery needed.
    """
    skill_dir = str(approach_path.parent.resolve())
    added = skill_dir not in sys.path
    if added:
        sys.path.insert(0, skill_dir)
    try:
        source = approach_path.read_text(encoding="utf-8")
        namespace: dict[str, Any] = {}
        exec(compile(source, str(approach_path), "exec"), namespace)  # noqa: S102
        return namespace["GeneratedApproach"]
    finally:
        if added and skill_dir in sys.path:
            sys.path.remove(skill_dir)


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------


def _run_episode(
    approach_cls,
    video_dir: Path,
    episode_id: str,
    *,
    seed: int | None = None,
    init_state: Any | None = None,
) -> tuple[bool, int, float]:
    """Run one episode. Either *seed* or *init_state* must be provided."""
    env = ObjectCentricPushPullHook2DEnv(
        render_mode="rgb_array", allow_state_access=True
    )

    tmp_video_dir = video_dir / f"_tmp_{episode_id}"
    env = RecordVideo(
        env,
        str(tmp_video_dir),
        episode_trigger=lambda _: True,
        name_prefix=episode_id,
    )

    if init_state is not None:
        # Reset once to initialise internals, then override state.
        env.reset(seed=0)
        env.unwrapped.set_state(init_state)
        state = init_state
        info: dict[str, Any] = {}
    else:
        assert seed is not None
        state, info = env.reset(seed=seed)

    approach = approach_cls(
        env.action_space,
        env.observation_space,
        initial_constant_state=env.unwrapped.initial_constant_state,
    )
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

    # Rename video with success/failed tag.
    tag = "success" if solved else "failed"
    for mp4 in Path(tmp_video_dir).glob("*.mp4"):
        dest = video_dir / f"{episode_id}_{tag}.mp4"
        shutil.move(str(mp4), str(dest))
    shutil.rmtree(tmp_video_dir, ignore_errors=True)

    return solved, steps, total_reward


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a generated PushPullHook2D approach and save videos."
    )
    parser.add_argument(
        "approach_path",
        type=str,
        help="Path to approach.py (skills are loaded from the same directory).",
    )
    parser.add_argument(
        "--state_dirs",
        nargs="*",
        default=[],
        help="Directories containing initial state .npy files. "
        "If empty, run seeded episodes instead.",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=20,
        help="Number of seeded episodes when no state_dirs are given (default: 20).",
    )
    args = parser.parse_args()

    approach_path = Path(args.approach_path).resolve()
    if not approach_path.exists():
        print(f"ERROR: approach file not found: {approach_path}")
        sys.exit(1)

    skill_dir = approach_path.parent
    video_dir = skill_dir / "test_videos"
    video_dir.mkdir(exist_ok=True)

    approach_cls = _load_approach(approach_path)

    print(f"Loaded approach from {approach_path}")
    print(f"Videos will be saved to {video_dir}")

    # Build episode list.
    episodes: list[tuple[str, int | None, Any]] = []  # (id, seed, state)

    if args.state_dirs:
        for state_dir_path in args.state_dirs:
            state_dir = Path(state_dir_path).resolve()
            if not state_dir.is_dir():
                print(f"WARNING: state dir not found, skipping: {state_dir}")
                continue
            for npy_file in sorted(state_dir.glob("*.npy")):
                state = np.load(str(npy_file), allow_pickle=True).item()
                ep_id = f"{state_dir.name}_{npy_file.stem}"
                episodes.append((ep_id, None, state))
    else:
        for seed in range(args.num_tests):
            episodes.append((f"seed_{seed:03d}", seed, None))

    if not episodes:
        print("No episodes to run.")
        sys.exit(0)

    print(f"\nRunning {len(episodes)} episodes...\n")

    results: list[tuple[str, bool, int, float]] = []
    for ep_id, seed, init_state in tqdm(episodes):
        solved, steps, reward = _run_episode(
            approach_cls,
            video_dir,
            ep_id,
            seed=seed,
            init_state=init_state,
        )
        results.append((ep_id, solved, steps, reward))

    # Print summary.
    num_solved = sum(1 for _, s, _, _ in results if s)
    print(f"\n{'Episode':<40} {'Result':>8} {'Steps':>6} {'Reward':>8}")
    print("-" * 66)
    for ep_id, solved, steps, reward in results:
        tag = "SOLVED" if solved else "FAILED"
        print(f"{ep_id:<40} {tag:>8} {steps:>6} {reward:>8.1f}")
    print("-" * 66)
    print(f"Solve rate: {num_solved}/{len(results)}")


if __name__ == "__main__":
    main()
