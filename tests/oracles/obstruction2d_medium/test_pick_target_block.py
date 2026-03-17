"""Tests for PickTargetBlock behavior on Obstruction2D-o2."""

import pytest
import kinder

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.oracles.obstruction2d_medium.behaviors import PickTargetBlock

ENV_ID = "kinder/Obstruction2D-o2-v0"
MAX_STEPS = 500

def test_holding_target_after_pick():
    """After PickTargetBlock, robot should hold the block."""
    kinder.register_all_environments()
    env = kinder.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    behavior = PickTargetBlock()

    assert not behavior.initializable(obs), "Precondition should not be satisfied at the start."
    