"""Tests for GraspRotate behavior on PushPullHook2D."""

import kinder
from gymnasium.wrappers import RecordVideo

from robocode.oracles.pushpullhook2d.behaviors import GraspRotate
from robocode.oracles.pushpullhook2d.obs_helpers import (
    extract_hook,
    extract_robot,
    hook_center,
    holding_hook,
    hook_is_horizontal,
    hook_at_center,
)
from tests.conftest import MAKE_VIDEOS

ENV_ID = "kinder/PushPullHook2D-v0"
MAX_STEPS = 1500


def _run_grasp_rotate(seed: int) -> tuple[bool, int]:
    """Run GraspRotate on a single seed. Return (success, steps)."""
    kinder.register_all_environments()
    render_mode = "rgb_array" if MAKE_VIDEOS else None
    env = kinder.make(ENV_ID, render_mode=render_mode)
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=seed)
    behavior = GraspRotate()

    robot = extract_robot(obs)
    hook = extract_hook(obs)
    cx, cy = hook_center(hook)
    print(
        f"seed={seed}  robot=({robot.x:.3f}, {robot.y:.3f}, {robot.theta:.3f})  "
        f"hook=({hook.x:.3f}, {hook.y:.3f}, {hook.theta:.3f})  "
        f"hook_center=({cx:.3f}, {cy:.3f})"
    )

    if not behavior.initializable(obs):
        print("  Precondition NOT satisfied (hook already in target state).")
        env.close()
        return True, 0

    behavior.reset(obs)
    for s in range(MAX_STEPS):
        action = behavior.step(obs)
        obs, _, _, _, _ = env.step(action)
        if behavior.terminated(obs):
            print(f"  Subgoal achieved in {s + 1} steps.")
            env.close()
            return True, s + 1

    # Debug: print final state
    robot = extract_robot(obs)
    hook = extract_hook(obs)
    cx, cy = hook_center(hook)
    print(
        f"  FAILED after {MAX_STEPS} steps.  "
        f"holding={holding_hook(obs)}  horiz={hook_is_horizontal(obs)}  "
        f"centered={hook_at_center(obs)}  "
        f"hook_theta={hook.theta:.3f}  hook_center=({cx:.3f}, {cy:.3f})"
    )
    env.close()
    return False, MAX_STEPS


def test_grasp_rotate_single_seed():
    """GraspRotate should succeed on a known seed."""
    success, steps = _run_grasp_rotate(seed=42)
    assert success, f"GraspRotate failed on seed 42 after {steps} steps."


def test_grasp_rotate_multiple_seeds():
    """GraspRotate should succeed on several random seeds."""
    seeds = [42, 123, 636, 7, 2025]
    results = [_run_grasp_rotate(seed=s) for s in seeds]
    successes = sum(ok for ok, _ in results)
    mean_steps = sum(st for _, st in results) / len(results)
    print(f"\nSolve rate: {successes}/{len(seeds)}  mean steps: {mean_steps:.0f}")
    assert successes == len(seeds), (
        f"GraspRotate failed on {len(seeds) - successes}/{len(seeds)} seeds."
    )
