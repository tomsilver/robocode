"""End-to-end eval-loop coverage for a variable-object-count env.

Exercises count pinning through ``run_episode`` and by-count aggregation over real
rollouts (design plus held-out counts), complementing the synthetic-dict
``summarize_by_count`` unit tests.
"""

from __future__ import annotations

from robocode.approaches.random_approach import RandomApproach
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.utils.episode import run_episode, summarize_by_count

_OBSTRUCTION2D = "kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv"


def _env() -> VariableObjectCountEnv:
    # Design range is [0, 1]; 2 is held out (reached only via an explicit pin).
    return VariableObjectCountEnv(
        constant_object_env_path=_OBSTRUCTION2D,
        count_kwarg="num_obstructions",
        count_object_prefix="obstruction",
        design_counts=[0, 1],
        eval_counts=[0, 1, 2],
        bilevel_env_name="obstruction2d",
    )


def _random(env: VariableObjectCountEnv) -> RandomApproach:
    return RandomApproach(env.action_space, env.observation_space, 0, {})


def test_run_episode_pins_held_out_count() -> None:
    """run_episode(count=k) pins the object count and reports it, held-out count
    included."""
    env = _env()
    try:
        metrics, _, _ = run_episode(env, _random(env), seed=0, max_steps=3, count=2)
        assert metrics["object_count"] == 2  # held out: design range never samples it
    finally:
        env.close()


def test_by_count_aggregates_over_real_rollouts() -> None:
    """A scheduled sweep over design + held-out counts yields honest per-count
    denominators from real episodes."""
    env = _env()
    try:
        approach = _random(env)
        scheduled = [0, 1, 2, 2]
        per_episode = [
            run_episode(env, approach, seed=s, max_steps=3, count=c)[0]
            for s, c in enumerate(scheduled)
        ]

        by_count, largest_all, largest_any = summarize_by_count(scheduled, per_episode)

        assert set(by_count) == {0, 1, 2}
        assert by_count[2]["n"] == 2  # both count-2 episodes pool into one bucket
        assert sum(b["n"] for b in by_count.values()) == len(scheduled)
        # A random policy solves nothing in 3 steps, so every scheduled episode counts
        # as unsolved against the full denominator (no count qualifies as solved).
        assert all(b["solve_rate"] == 0.0 for b in by_count.values())
        assert largest_all is None and largest_any is None
    finally:
        env.close()
