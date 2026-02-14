"""Utility functions."""

from robocode.benchmarks.base_benchmark import Benchmark
from robocode.structs import Action, Task


def plan_is_valid(plan: list[Action], task: Task, benchmark: Benchmark) -> bool:
    """Checks if the plan solves the task."""
    state = task.init
    for action in plan:
        state = benchmark.get_next_state(state, action)
    return benchmark.check_goal(state, task.goal)


def get_plan_cost(plan: list[Action], task: Task, benchmark: Benchmark) -> float:
    """Get the total plan cost."""
    cost = 0.0
    state = task.init
    for action in plan:
        next_state = benchmark.get_next_state(state, action)
        cost += benchmark.get_cost(state, action, next_state)
        state = next_state
    return cost
