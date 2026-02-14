"""Tests for maze_benchmark.py."""

import numpy as np

from robocode.benchmarks.maze_benchmark import MazeBenchmark


def test_maze_benchmark():
    """Tests for MazeBenchmark()."""
    env = MazeBenchmark(5, 8, 5, 8)
    rng = np.random.default_rng(123)
    tasks = env.generate_tasks(3, "train", rng)
    assert len(tasks) == 3
    task = tasks[0]
    state = task.init
    goal = task.goal
    assert not env.check_goal(state, goal)
    for action in env.get_actions():
        next_state = env.get_next_state(state, action)
        assert env.get_cost(state, action, next_state) == 1.0
