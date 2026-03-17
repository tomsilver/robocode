"""Tests for PickTargetBlock behavior on Obstruction2D-o2."""

import kinder

from robocode.oracles.obstruction2d_medium.behaviors import PickTargetBlock
from imageio.v2 import imwrite

ENV_ID = "kinder/Obstruction2D-o2-v0"
MAX_STEPS = 500

def test_holding_target_after_pick():
    """After PickTargetBlock, robot should hold the block."""
    kinder.register_all_environments()
    env = kinder.make(ENV_ID)
    obs_init, _ = env.reset(seed=0)
    behavior = PickTargetBlock()

    assert not behavior.initializable(obs_init), "Precondition should not be satisfied at the start."

    obs_init[29] += 0.5
    obs_init[39] += 0.5
    assert behavior.initializable(obs_init), "Precondition should be satisfied after moving obstructions."

    obs, _ = env.reset(options={"init_state": obs_init})
    assert not behavior.terminated(obs), "Subgoal should not be satisfied at the start."

    behavior.reset(obs)
    for s in range(MAX_STEPS):
        action = behavior.step(obs)
        obs, _, _, _ = env.step(action)
        if behavior.terminated(obs):
            print(f"Subgoal achieved in {s+1} steps.")
            break
    