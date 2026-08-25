"""Tests for the oracle approach dispatcher."""

import numpy as np
import pytest
from gymnasium.spaces import Box

from robocode.approaches.base_approach import BaseApproach
from robocode.approaches.oracle_approach import (
    ORACLE_TARGETS,
    build_oracle_approach,
    resolve_oracle_class,
)
from robocode.environments.pr2_tamp_env import PR2PackedEnv
from robocode.oracles.pr2packed.approach import PR2PackedOracleApproach


@pytest.mark.parametrize("env_name", sorted(ORACLE_TARGETS))
def test_every_registered_target_resolves(env_name: str) -> None:
    """A registry entry that no longer imports would fail only at experiment time."""
    assert issubclass(resolve_oracle_class(env_name), BaseApproach)


def test_unknown_environment_lists_the_known_ones() -> None:
    """The error names the environments that do have an oracle."""
    with pytest.raises(ValueError, match="No oracle is registered"):
        resolve_oracle_class("motion2d_easy")


def test_missing_env_name_is_rejected() -> None:
    """env_name comes from the runner; without it there is nothing to dispatch on."""
    with pytest.raises(ValueError, match="needs the environment choice name"):
        resolve_oracle_class(None)


def test_builds_pr2_oracle_with_the_env() -> None:
    """The PR2 oracle needs the env to plan, so it must be forwarded."""
    env = PR2PackedEnv(num_blocks=1)
    try:
        approach = build_oracle_approach(
            action_space=env.action_space,
            observation_space=env.observation_space,
            seed=0,
            env=env,
            env_name="pr2packed_easy",
            max_steps=100,
            mcp_tools=(),
        )
        assert isinstance(approach, PR2PackedOracleApproach)
    finally:
        env.close()


def test_filters_arguments_the_oracle_does_not_accept() -> None:
    """The runner passes every approach the same wide kwargs; older oracles take few.

    The kinder oracles predate ``env``/``max_steps`` and declare neither, so forwarding
    the runner's kwargs blindly would raise TypeError.
    """
    space = Box(-1.0, 1.0, (4,), dtype=np.float32)
    approach = build_oracle_approach(
        action_space=space,
        observation_space=space,
        seed=0,
        env_name="obstruction2d_medium",
        env=object(),
        max_steps=500,
        eval_timeout=60,
        mcp_tools=(),
        env_cfg="{}",
    )
    assert isinstance(approach, BaseApproach)
