"""Lightweight internal tests for the ClutteredStorage2D oracle behaviors."""

# pylint: disable=protected-access

from __future__ import annotations

from typing import cast

import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.oracles.clutteredstorage2d_medium.approach import (
    ClutteredStorage2DOracleApproach,
)
from robocode.oracles.clutteredstorage2d_medium.behaviors import (
    ClearCompactBlocker,
    CompactShelfBlocks,
    StoreOutsideBlock,
)
from robocode.oracles.clutteredstorage2d_medium.obs_helpers import outside_blocks

ENV_ID = "kinder/ClutteredStorage2D-b3-v0"


@pytest.fixture(name="clutteredstorage_env")
def _clutteredstorage_env() -> KinderGeom2DEnv:
    """Create the ClutteredStorage2D-b3 test environment."""
    return KinderGeom2DEnv(ENV_ID)


def test_compact_behavior_reset_builds_initial_plan(
    clutteredstorage_env: KinderGeom2DEnv,
) -> None:
    """Compact behavior should expose an initial execution plan after reset."""
    state, _ = clutteredstorage_env.reset(seed=0)
    behavior = CompactShelfBlocks(seed=0)

    assert behavior.initializable(state)

    behavior.reset(state)
    snapshot = behavior.debug_snapshot()
    queued_actions = cast(int, snapshot["queued_actions"])

    assert behavior.result() == "running"
    assert snapshot["phase"] == "compact"
    assert snapshot["active_block"] is not None
    assert snapshot["chosen_pick_pose"] is not None
    assert queued_actions > 0


def test_store_behavior_reset_builds_initial_plan(
    clutteredstorage_env: KinderGeom2DEnv,
) -> None:
    """Store behavior should select one outside block and queue actions."""
    state, _ = clutteredstorage_env.reset(seed=0)
    behavior = StoreOutsideBlock(seed=0)

    assert behavior.initializable(state)

    behavior.reset(state)
    snapshot = behavior.debug_snapshot()
    queued_actions = cast(int, snapshot["queued_actions"])

    assert behavior.result() == "running"
    assert snapshot["phase"] == "store"
    assert snapshot["active_block"] in outside_blocks(state)
    assert snapshot["target_center"] is not None
    assert queued_actions > 0


def test_clear_behavior_with_store_selection_targets_staging(
    clutteredstorage_env: KinderGeom2DEnv,
) -> None:
    """Clear behavior should build a staging plan when using store selection."""
    state, _ = clutteredstorage_env.reset(seed=0)
    behavior = ClearCompactBlocker(seed=0, use_store_selection=True)

    assert behavior.initializable(state)

    behavior.reset(state)
    snapshot = behavior.debug_snapshot()
    queued_actions = cast(int, snapshot["queued_actions"])

    assert behavior.result() == "running"
    assert snapshot["phase"] == "clear_compact"
    assert snapshot["active_block"] in outside_blocks(state)
    assert snapshot["target_center"] is not None
    assert queued_actions > 0


def test_approach_starts_in_compact_and_disables_it_after_store(
    clutteredstorage_env: KinderGeom2DEnv,
) -> None:
    """The approach should only use compact during the startup window."""
    state, info = clutteredstorage_env.reset(seed=0)
    approach = ClutteredStorage2DOracleApproach(
        action_space=clutteredstorage_env.action_space,
        observation_space=clutteredstorage_env.observation_space,
    )

    approach.reset(state, info)

    assert approach.debug_snapshot().get("phase") == "compact"

    approach._allow_startup_compact = False
    selected = approach._select_behavior(state, previous_result=None)

    assert isinstance(selected, StoreOutsideBlock)
