"""Tests for EnvSimulator."""

from robocode.environments.maze_env import MazeEnv
from robocode.simulators.env_simulator import EnvSimulator


def test_env_simulator_matches_env_step():
    """EnvSimulator produces the same next state as stepping the env."""
    env = MazeEnv(5, 8, 5, 8)
    state, _ = env.reset(seed=123)
    env.action_space.seed(seed=123)

    simulator = EnvSimulator(env)
    action = env.action_space.sample()

    sim_next = simulator.sample_next_state(state, action, env.np_random)

    env.set_state(state)
    env_next, _, _, _, _ = env.step(action)

    assert sim_next == env_next
