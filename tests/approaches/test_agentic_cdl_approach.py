"""Tests for agentic_cdl_approach.py."""

from robocode.approaches.agentic_cdl_approach import AgenticCDLApproach
from robocode.environments.maze_env import MazeEnv
from robocode.utils.backends import DEFAULT_BACKEND_CFG


def test_accepts_run_experiment_kwargs():
    """The extra kwargs run_experiment passes to every approach are absorbed."""
    env = MazeEnv(5, 8, 5, 8)
    approach = AgenticCDLApproach(
        action_space=env.action_space,
        observation_space=env.observation_space,
        seed=123,
        primitives={},
        backend=DEFAULT_BACKEND_CFG,
        env=env,
        env_cfg="{}",
        max_steps=100,
    )
    assert approach.total_cost_usd is None
