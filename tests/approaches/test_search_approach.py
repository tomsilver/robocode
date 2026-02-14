"""Tests for search_approach.py."""

import numpy as np

from robocode.approaches.search_approach import SearchApproach
from robocode.benchmarks.maze_benchmark import MazeBenchmark
from robocode.utils import plan_is_valid


def test_search_approach():
    """Tests for SearchApproach()."""
    benchmark = MazeBenchmark(5, 8, 5, 8)
    approach = SearchApproach(
        benchmark.get_actions(),
        benchmark.get_next_state,
        benchmark.get_cost,
        benchmark.check_goal,
    )
    rng = np.random.default_rng(123)
    task = benchmark.generate_tasks(1, "train", rng)[0]
    plan = approach.generate_plan(task, "test", 1.0, rng)
    assert plan_is_valid(plan, task, benchmark)
