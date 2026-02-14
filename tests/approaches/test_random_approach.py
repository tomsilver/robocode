"""Tests for random_approach.py."""

import numpy as np

from robocode.approaches.random_approach import RandomApproach
from robocode.benchmarks.maze_benchmark import MazeBenchmark


def test_random_approach():
    """Tests for RandomApproach()."""
    benchmark = MazeBenchmark(5, 8, 5, 8)
    approach = RandomApproach(
        benchmark.get_actions(),
        benchmark.get_next_state,
        benchmark.get_cost,
        benchmark.check_goal,
    )
    rng = np.random.default_rng(123)
    task = benchmark.generate_tasks(1, "train", rng)[0]
    plan = approach.generate_plan(task, "test", 1e-1, rng)
    assert len(plan) > 0
