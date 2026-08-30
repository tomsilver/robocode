"""Tests for episode utilities."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import math
import multiprocessing as mp
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import _thread
import imageio.v3 as iio
import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box

from robocode.approaches.base_approach import BaseApproach, InstanceResult
from robocode.utils.episode import (
    _EPISODE_FORK_SAFE,
    load_generated_approach,
    open_video_writer,
    run_episode,
    run_episode_with_timeout,
    run_in_forked_worker,
    run_per_instance_eval,
    save_frames,
    save_video,
    summarize_by_count,
    summarize_count_regimes,
    summarize_eval_episodes,
)
from robocode.utils.scored_env_guard import (
    ScoredEnvMutationError,
    readonly_view,
)


def test_summarize_eval_episodes_counts_a_crash_as_unsolved() -> None:
    """A crashed episode is a failure in solve_rate, not an episode that never ran.

    Excluding it instead rescales the headline to whatever survived, so a policy that
    crashes on the instances it cannot handle scores higher the more of them it kills.
    """
    per_episode: list[dict[str, Any]] = [
        {"solved": True, "total_reward": -10.0, "num_steps": 10},
        {"solved": False, "crashed": True, "total_reward": None, "num_steps": None},
        {"solved": False, "crashed": True, "total_reward": None, "num_steps": None},
        {"solved": False, "crashed": True, "total_reward": None, "num_steps": None},
    ]
    summary = summarize_eval_episodes(per_episode)
    assert summary["solve_rate"] == 0.25  # 1 of 4 scheduled, not 1 of 1 survivor
    assert summary["num_eval_tasks"] == 4
    assert summary["num_evaluated_episodes"] == 1
    assert summary["num_crashed_episodes"] == 3
    assert summary["eval_complete"] is False
    # Reward and steps keep the scored denominator: there is no worst-case return to
    # charge a crash with, so they describe the one episode that produced a number.
    assert summary["mean_eval_reward"] == -10.0
    assert summary["mean_eval_steps"] == 10.0


def test_summarize_eval_episodes_agrees_with_the_by_count_curve() -> None:
    """The headline solve rate matches the scaling curve it is supposed to summarize.

    Both use the full scheduled denominator. Before that was true they diverged exactly
    when episodes crashed, i.e. when a reader most needs them to agree.
    """
    scheduled = [2, 2, 5, 5]
    per_episode: list[dict[str, Any]] = [
        {"solved": True, "total_reward": -5.0, "num_steps": 5},
        {"solved": False, "total_reward": -9.0, "num_steps": 9},
        {"solved": False, "crashed": True, "total_reward": None, "num_steps": None},
        {"solved": False, "crashed": True, "total_reward": None, "num_steps": None},
    ]
    summary = summarize_eval_episodes(per_episode)
    by_count, _largest_all, _largest_any = summarize_by_count(scheduled, per_episode)
    pooled = sum(e["n_solved"] for e in by_count.values()) / sum(
        e["n"] for e in by_count.values()
    )
    assert summary["solve_rate"] == pooled == 0.25


def test_summarize_eval_episodes_flags_an_unattempted_suite_incomplete() -> None:
    """Budget exhaustion leaves seeds unattempted; that is a partial eval too."""
    per_episode: list[dict[str, Any]] = [
        {"solved": True, "attempted": True, "total_reward": -5.0, "num_steps": 5},
        {"solved": False, "attempted": False, "total_reward": None, "num_steps": None},
    ]
    summary = summarize_eval_episodes(per_episode)
    assert summary["eval_complete"] is False
    assert summary["num_crashed_episodes"] == 0  # unattempted is not a crash
    assert summary["num_evaluated_episodes"] == 1
    assert summary["solve_rate"] == 0.5


def test_summarize_eval_episodes_marks_a_clean_suite_complete() -> None:
    """Nothing crashed and nothing was skipped, so the means cover the whole suite."""
    per_episode: list[dict[str, Any]] = [
        {"solved": True, "total_reward": -5.0, "num_steps": 5},
        {"solved": False, "total_reward": -9.0, "num_steps": 9},
    ]
    summary = summarize_eval_episodes(per_episode)
    assert summary["eval_complete"] is True
    assert summary["solve_rate"] == 0.5
    assert summary["num_evaluated_episodes"] == 2


def test_summarize_eval_episodes_reports_nan_when_everything_crashed() -> None:
    """With no scored episode there is no mean, but solve_rate is 0, not nan.

    This is the shape a macOS run produced before the eval fork fix: every episode
    crashed, and a nan solve rate reads as "no data" when the honest reading is
    "nothing was solved".
    """
    per_episode: list[dict[str, Any]] = [
        {"solved": False, "crashed": True, "total_reward": None, "num_steps": None},
        {"solved": False, "crashed": True, "total_reward": None, "num_steps": None},
    ]
    summary = summarize_eval_episodes(per_episode)
    assert summary["solve_rate"] == 0.0
    assert math.isnan(summary["mean_eval_reward"])
    assert math.isnan(summary["mean_eval_steps"])
    assert summary["eval_complete"] is False


def test_summarize_eval_episodes_handles_an_empty_suite() -> None:
    """No scheduled episodes yields nan rather than dividing by zero."""
    summary = summarize_eval_episodes([])
    assert math.isnan(summary["solve_rate"])
    assert summary["num_eval_tasks"] == 0
    assert summary["eval_complete"] is True


def test_summarize_by_count_uses_full_scheduled_denominator() -> None:
    """Per-count solve rate counts every scheduled episode; crashes/unattempted fail."""
    scheduled = [2, 2, 2, 5, 5]
    per_episode: list[dict[str, Any]] = [
        {"solved": True, "num_steps": 10},  # count 2
        {"solved": False, "num_steps": 30},  # count 2
        {"solved": False, "crashed": True},  # count 2, crash -> failure in denominator
        {"solved": False, "attempted": False},  # count 5, unattempted -> failure
        {"solved": True, "num_steps": 40, "planning_time": 3.0},  # count 5
    ]
    by_count, largest_all, largest_any = summarize_by_count(scheduled, per_episode)
    assert by_count[2]["n"] == 3 and by_count[2]["solve_rate"] == 1 / 3
    assert by_count[5]["n"] == 2 and by_count[5]["solve_rate"] == 1 / 2
    # numeric extras averaged per count over whoever has them.
    assert by_count[5]["mean_planning_time"] == 3.0
    # largest_count_all_solved needs solve_rate == 1.0 (no count qualifies here).
    assert largest_all is None
    assert largest_any == 5  # count 5 has a solve


def test_summarize_by_count_largest_all_solved() -> None:
    """largest_count_all_solved is the biggest count solved on every episode."""
    scheduled = [1, 1, 3]
    per_episode: list[dict[str, Any]] = [
        {"solved": True, "num_steps": 5},
        {"solved": True, "num_steps": 6},
        {"solved": False, "num_steps": 9},
    ]
    _by_count, largest_all, largest_any = summarize_by_count(scheduled, per_episode)
    assert largest_all == 1  # count 1 fully solved, count 3 not
    assert largest_any == 1


def test_summarize_by_count_rejects_length_mismatch() -> None:
    """Scheduled counts and episode entries must be parallel."""
    with pytest.raises(ValueError, match="scheduled_counts and per_episode"):
        summarize_by_count([1, 2], [{"solved": True}])


def test_summarize_count_regimes_reports_design_and_held_out_separately() -> None:
    """Count-regime rates pool scheduled episodes within each protocol regime."""
    by_count = {
        1: {"n": 2, "n_solved": 2, "solve_rate": 1.0},
        3: {"n": 2, "n_solved": 1, "solve_rate": 0.5},
        5: {"n": 3, "n_solved": 1, "solve_rate": 1 / 3},
    }
    regimes = summarize_count_regimes(by_count, design_counts=[1, 3])
    assert regimes["design"] == {
        "counts": [1, 3],
        "n": 4,
        "n_solved": 3,
        "solve_rate": 0.75,
    }
    assert regimes["held_out"] == {
        "counts": [5],
        "n": 3,
        "n_solved": 1,
        "solve_rate": 1 / 3,
    }


class _ScriptedPerInstanceApproach(BaseApproach[Any, Any]):
    """A per-instance approach whose solve_instance returns canned results."""

    per_instance = True

    def __init__(self, results: list[InstanceResult]) -> None:
        space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        super().__init__(space, space, 0, {})
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def _get_action(self) -> Any:
        raise NotImplementedError

    def solve_instance(
        self,
        *,
        env: Any,
        seed: int,
        budget_usd: float,
        output_subdir: Path,
        render: bool = False,
        count: int | None = None,
        max_steps: int | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> InstanceResult:
        del env, output_subdir, max_steps, progress_callback
        self.calls.append(
            {"seed": seed, "budget_usd": budget_usd, "render": render, "count": count}
        )
        return self._results.pop(0)


def test_per_instance_eval_stops_when_budget_exhausted(tmp_path: Path) -> None:
    """Once the global budget is spent, remaining seeds are left unattempted."""
    results = [
        InstanceResult(solved=True, total_reward=1.0, num_steps=3, cost_usd=1.0)
        for _ in range(3)
    ]
    approach = _ScriptedPerInstanceApproach(results)
    out = run_per_instance_eval(
        None,
        approach,
        [10, 11, 12, 13, 14],
        max_budget_usd=3.0,
        output_dir=tmp_path,
    )
    assert len(approach.calls) == 3  # only 3 attempts fit in a $3 budget
    assert out["num_attempted"] == 3
    assert out["num_solved"] == 3
    # solve_rate is over ALL 5 seeds; the 2 unreached count as failures.
    assert out["solve_rate"] == pytest.approx(3 / 5)
    assert out["per_episode"][3]["attempted"] is False
    assert out["per_episode"][4]["attempted"] is False
    assert out["total_cost_usd"] == pytest.approx(3.0)


def test_per_instance_eval_respects_per_instance_cap(tmp_path: Path) -> None:
    """A per-instance cap bounds each attempt to min(cap, remaining)."""
    results = [
        InstanceResult(solved=False, total_reward=0.0, num_steps=10, cost_usd=0.5)
        for _ in range(4)
    ]
    approach = _ScriptedPerInstanceApproach(results)
    out = run_per_instance_eval(
        None,
        approach,
        [1, 2, 3, 4],
        max_budget_usd=10.0,
        output_dir=tmp_path,
        max_budget_per_instance_usd=2.0,
    )
    assert all(c["budget_usd"] == pytest.approx(2.0) for c in approach.calls)
    assert out["num_attempted"] == 4
    assert out["solve_rate"] == 0.0


def test_per_instance_eval_charges_crashed_attempts(tmp_path: Path) -> None:
    """Crashed attempts still charge cost and count as solve failures, but are excluded
    from the reward/step means."""
    results = [
        InstanceResult(solved=True, total_reward=5.0, num_steps=4, cost_usd=1.0),
        InstanceResult(
            solved=False,
            total_reward=None,
            num_steps=None,
            cost_usd=2.0,
            crashed=True,
        ),
    ]
    approach = _ScriptedPerInstanceApproach(results)
    out = run_per_instance_eval(
        None, approach, [7, 8], max_budget_usd=10.0, output_dir=tmp_path
    )
    assert out["total_cost_usd"] == pytest.approx(3.0)  # crash cost charged
    assert out["num_crashed_episodes"] == 1
    assert out["num_evaluated_episodes"] == 1  # only the non-crashed is scored
    assert out["mean_eval_reward"] == pytest.approx(5.0)
    assert out["mean_eval_steps"] == pytest.approx(4.0)
    assert out["solve_rate"] == pytest.approx(0.5)  # 1 of 2 seeds solved
    assert out["per_episode"][1]["crashed"] is True


def test_per_instance_eval_tags_every_entry_with_scheduled_count(
    tmp_path: Path,
) -> None:
    """Crashed and budget-exhausted entries keep their scheduled object_count, so the
    by-count denominator covers every scheduled episode (nothing silently dropped)."""
    results = [
        InstanceResult(solved=True, total_reward=1.0, num_steps=3, cost_usd=1.0),
        InstanceResult(
            solved=False,
            total_reward=None,
            num_steps=None,
            cost_usd=2.0,
            crashed=True,
        ),
    ]
    approach = _ScriptedPerInstanceApproach(results)
    out = run_per_instance_eval(
        None,
        approach,
        [10, 11, 12, 13],
        max_budget_usd=3.0,  # fits solved ($1) + crashed ($2); seeds 12,13 unattempted
        output_dir=tmp_path,
        eval_counts=[2, 4, 6, 8],
    )
    per = out["per_episode"]
    assert [e.get("object_count") for e in per] == [2, 4, 6, 8]
    assert per[1]["crashed"] is True and per[1]["object_count"] == 4
    assert per[2]["attempted"] is False and per[2]["object_count"] == 6
    # by-count covers all scheduled episodes; the unreached count-8 episode is a failure.
    assert out["by_count"][8]["n"] == 1
    assert out["by_count"][8]["n_solved"] == 0


def test_per_instance_eval_rejects_mismatched_eval_counts(tmp_path: Path) -> None:
    """Eval counts must be parallel to eval seeds."""
    approach = _ScriptedPerInstanceApproach([])
    with pytest.raises(ValueError, match="eval_counts and eval_seeds"):
        run_per_instance_eval(
            None,
            approach,
            [10, 11],
            max_budget_usd=1.0,
            output_dir=tmp_path,
            eval_counts=[2],
        )


def test_per_instance_eval_aggregates_extras(tmp_path: Path) -> None:
    """Per-instance extras are merged into per_episode and averaged as mean_<key>.

    Extras keys may differ across instances (a failed attempt reports fewer): the
    aggregation averages each numeric key over whichever scored episodes have it,
    ignores bools (e.g. a flag), and never lets extras clobber the fixed keys.
    """
    results = [
        InstanceResult(
            solved=True,
            total_reward=1.0,
            num_steps=5,
            cost_usd=0.0,
            extras={
                "planning_time": 2.0,
                "execution_time": 4.0,
                "plan_found": True,
                "seed": 999,  # collides with a fixed key; must be ignored
            },
        ),
        InstanceResult(
            solved=False,
            total_reward=None,
            num_steps=None,
            cost_usd=0.0,
            # A failed plan reports planning_time but no execution_time.
            extras={"planning_time": 6.0, "plan_found": False},
        ),
    ]
    approach = _ScriptedPerInstanceApproach(results)
    out = run_per_instance_eval(
        None, approach, [3, 4], max_budget_usd=1.0, output_dir=tmp_path
    )
    # planning_time is present on both scored episodes -> mean of 2.0 and 6.0.
    assert out["mean_planning_time"] == pytest.approx(4.0)
    # execution_time only on the first -> mean over the one that has it.
    assert out["mean_execution_time"] == pytest.approx(4.0)
    # bool extras are not averaged.
    assert "mean_plan_found" not in out
    # extras are exposed per-episode; the fixed "seed" key is not overwritten.
    assert out["per_episode"][0]["planning_time"] == pytest.approx(2.0)
    assert out["per_episode"][0]["seed"] == 3
    assert out["per_episode"][0]["plan_found"] is True


def test_per_instance_eval_threads_render_and_saves_video(tmp_path: Path) -> None:
    """Render is passed to solve_instance and returned frames are saved as a gif."""
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (8, 8, 3), dtype=np.uint8) for _ in range(3)]
    results = [
        InstanceResult(
            solved=True, total_reward=1.0, num_steps=3, cost_usd=0.5, frames=frames
        )
    ]
    approach = _ScriptedPerInstanceApproach(results)
    run_per_instance_eval(
        None, approach, [5], max_budget_usd=2.0, output_dir=tmp_path, render=True
    )
    assert approach.calls[0]["render"] is True
    assert (tmp_path / "videos" / "episode_0.gif").exists()


@pytest.fixture()
def dummy_approach_file(tmp_path: Path) -> Path:
    """Write a minimal GeneratedApproach to a temp file."""
    approach_py = tmp_path / "approach.py"
    approach_py.write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._action_space = action_space\n"
        "        self._primitives = primitives\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        return self._action_space.sample()\n"
    )
    return approach_py


@pytest.fixture()
def sample_frames() -> list[np.ndarray]:
    """Create a list of small random RGB frames."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (8, 8, 3), dtype=np.uint8) for _ in range(5)]


def test_load_generated_approach(dummy_approach_file: Path) -> None:
    """load_generated_approach returns an instance with expected methods."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    approach = load_generated_approach(dummy_approach_file, action_space, obs_space, {})
    assert hasattr(approach, "get_action")
    assert hasattr(approach, "reset")


def test_load_generated_approach_receives_primitives(
    dummy_approach_file: Path,
) -> None:
    """Primitives dict is passed through to the loaded approach."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    prims = {"my_prim": lambda: None}
    approach = load_generated_approach(
        dummy_approach_file, action_space, obs_space, prims
    )
    assert approach._primitives is prims  # pylint: disable=protected-access


def test_load_cleans_sys_path(dummy_approach_file: Path) -> None:
    """sys.path is cleaned up after loading."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    sandbox_dir = str(dummy_approach_file.parent.resolve())

    load_generated_approach(dummy_approach_file, action_space, obs_space, {})
    assert sandbox_dir not in sys.path


_APPROACH_TEMPLATE = (
    "{extra}\n"
    "class GeneratedApproach:\n"
    "    def __init__(self, action_space, observation_space, primitives):\n"
    "        self._action_space = action_space\n"
    "    def reset(self, state, info):\n"
    "        pass\n"
    "    def get_action(self, state):\n"
    "        return self._action_space.sample()\n"
)


def _write_approach(tmp_path: Path, extra: str = "") -> Path:
    approach_py = tmp_path / "approach.py"
    approach_py.write_text(_APPROACH_TEMPLATE.format(extra=extra))
    return approach_py


def test_anti_cheat_rejects_planner_refs_with_bilevel_models(tmp_path: Path) -> None:
    """With bilevel_models, referencing the SeSamE planner is rejected at load."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    path = _write_approach(
        tmp_path, extra="from bilevel_planning.sesame import run_sesame"
    )
    with pytest.raises(ValueError, match="bilevel planner"):
        load_generated_approach(
            path, action_space, obs_space, {"bilevel_models": object()}
        )


def test_anti_cheat_follows_planner_refs_into_a_sibling_module(tmp_path: Path) -> None:
    """Moving the planner call one file over does not evade the check.

    The loader puts the sandbox directory on ``sys.path`` so ``approach.py`` can
    import siblings, so a scan of ``approach.py`` alone is bypassed by writing the
    call in ``planner.py`` instead. The check walks the import graph for that reason.
    """
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    (tmp_path / "planner.py").write_text(
        "from bilevel_planning.sesame import run_sesame\n"
    )
    path = _write_approach(tmp_path, extra="import planner")
    with pytest.raises(ValueError, match="planner.py"):
        load_generated_approach(
            path, action_space, obs_space, {"bilevel_models": object()}
        )


def test_anti_cheat_follows_planner_refs_through_a_sibling_package(
    tmp_path: Path,
) -> None:
    """A package the agent wrote is walked too, relative imports included."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    pkg = tmp_path / "solver"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .search import plan\n")
    (pkg / "search.py").write_text(
        "def plan():\n    from bilevel_planning.sesame import run_sesame\n"
    )
    path = _write_approach(tmp_path, extra="from solver import plan")
    with pytest.raises(ValueError, match="search.py"):
        load_generated_approach(
            path, action_space, obs_space, {"bilevel_models": object()}
        )


def test_anti_cheat_ignores_sandbox_files_the_approach_never_imports(
    tmp_path: Path,
) -> None:
    """Only the modules actually reachable from approach.py are scanned.

    A scratch file the agent left in the sandbox is dead code at evaluation time, so
    rejecting the run because of it would be another false positive.
    """
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    (tmp_path / "scratch.py").write_text(
        "from bilevel_planning.sesame import run_sesame\n"
    )
    path = _write_approach(tmp_path)
    approach = load_generated_approach(
        path, action_space, obs_space, {"bilevel_models": object()}
    )
    assert hasattr(approach, "get_action")


def test_anti_cheat_allows_clean_program_with_bilevel_models(tmp_path: Path) -> None:
    """A program that does not touch the planner loads normally."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    path = _write_approach(tmp_path, extra="# uses primitives['bilevel_models'] only")
    approach = load_generated_approach(
        path, action_space, obs_space, {"bilevel_models": object()}
    )
    assert hasattr(approach, "get_action")


def test_anti_cheat_not_enforced_without_bilevel_models(tmp_path: Path) -> None:
    """The check only applies when the bilevel_models primitive is present."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    path = _write_approach(
        tmp_path, extra="run_sesame = None  # not the primitive setting"
    )
    approach = load_generated_approach(path, action_space, obs_space, {})
    assert hasattr(approach, "get_action")


@pytest.mark.parametrize("mutator", ["set_state", "sample_next_state"])
def test_mentioning_a_mutator_no_longer_blocks_loading(
    tmp_path: Path, mutator: str
) -> None:
    """Naming a mutator in source is not itself cheating.

    The scan this replaced rejected any source containing ``.set_state`` -- including,
    as here, a string literal, and including a program planning in an environment it
    built itself, which cannot affect scoring. What matters is whether the *scored*
    environment can be mutated, and that is enforced on the object (see
    ``tests/utils/test_scored_env_guard.py``) rather than by reading source.
    """
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    path = _write_approach(tmp_path, extra=f"_ = 'env.{mutator}(x)'")
    assert load_generated_approach(path, action_space, obs_space, {}) is not None


def test_anti_cheat_allows_state_reads(tmp_path: Path) -> None:
    """get_state, reset_state, and longer set_state* names do not false-positive."""
    action_space = Box(low=-1, high=1, shape=(2,))
    obs_space = Box(low=0, high=1, shape=(4,))
    path = _write_approach(
        tmp_path,
        extra=(
            "_ = 'env.get_state(); self.reset_state();"
            " env.set_stateful(); env.set_state_from_bytes()'"
        ),
    )
    approach = load_generated_approach(path, action_space, obs_space, {})
    assert hasattr(approach, "get_action")


class _StatefulGoalEnv(Env):
    """Toy env with get_state/set_state; step terminates once position reaches 5.0.

    Honest play needs ~50 steps of +0.1; a teleport to the goal state solves in one.
    """

    def __init__(self) -> None:
        self.observation_space = Box(0.0, 10.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._pos = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = 0.0
        return np.array([self._pos], dtype=np.float32), {}

    def step(self, action):
        del action
        self._pos = min(10.0, self._pos + 0.1)
        obs = np.array([self._pos], dtype=np.float32)
        return obs, 0.0, self._pos >= 5.0, False, {}

    def get_state(self):
        """Snapshot the current position."""
        return np.array([self._pos], dtype=np.float32)

    def set_state(self, state):
        """Restore the position from a snapshot."""
        self._pos = float(state[0])

    def goal_state(self):
        """A state sitting at the goal position (5.0)."""
        return np.array([5.0], dtype=np.float32)

    def render(self):
        return None


def test_set_state_teleport_fabricates_a_solve() -> None:
    """Red-team: teleporting into the goal state fakes a one-step solve."""
    env: Any = _StatefulGoalEnv()
    env.reset(seed=0)
    env.set_state(env.goal_state())
    _, _, terminated, _, _ = env.step(env.action_space.sample())
    assert terminated  # a policy that could set_state would score solved doing nothing


def test_generated_policy_cannot_teleport_via_primitive_closure(tmp_path: Path) -> None:
    """Red-team: an env-bound primitive exposes the env; the guard blocks the write.

    ``build_primitives`` binds primitives to a read-only view for exactly this reason;
    the wrapping is mirrored here because this test uses a toy env rather than a
    configured one.
    """
    env = _StatefulGoalEnv()
    # Mirrors build_primitives' partial(check_action_collision, readonly_view(env)):
    # the bound env is reachable from a generated policy as primitives["peek"].args[0].
    prims = {"peek": partial(lambda e, s: None, readonly_view(env))}
    assert prims["peek"].args[0] is not env  # reachable, but not the live object
    teleport = tmp_path / "approach.py"
    teleport.write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives):\n"
        "        self._action_space = action_space\n"
        "        self._primitives = primitives\n"
        "    def reset(self, state, info):\n"
        "        pass\n"
        "    def get_action(self, state):\n"
        "        env = self._primitives['peek'].args[0]\n"
        "        env.set_state(env.goal_state())  # teleport into the goal\n"
        "        return self._action_space.sample()\n"
    )
    approach = load_generated_approach(
        teleport, env.action_space, env.observation_space, prims
    )
    start = np.array([0.0], dtype=np.float32)
    approach.reset(start, {})
    with pytest.raises(ScoredEnvMutationError):
        approach.get_action(start)
    # And the scored env is untouched: the teleport never landed.
    final = np.asarray(env.get_state())  # type: ignore[no-untyped-call]
    assert final.item() == pytest.approx(0.0)


class _CountEnv(Env):
    """A tiny env that terminates after three steps (goal at pos 3.0)."""

    def __init__(self) -> None:
        self.observation_space = Box(0.0, 10.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._pos = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = 0.0
        return np.array([self._pos], dtype=np.float32), {}

    def step(self, action):
        self._pos += 1.0
        obs = np.array([self._pos], dtype=np.float32)
        return obs, 0.0, self._pos >= 3.0, False, {}

    def render(self):
        return None


class _NoopApproach(BaseApproach[Any, Any]):
    """Always returns the zero action; the env drives termination."""

    def _get_action(self) -> Any:
        return np.zeros(1, dtype=np.float32)


class _SlowApproach(BaseApproach[Any, Any]):
    """Sleeps far past any test timeout, so the rollout must be killed."""

    def _get_action(self) -> Any:
        time.sleep(30)
        return np.zeros(1, dtype=np.float32)


class _CrashApproach(BaseApproach[Any, Any]):
    """Raises on the first action, to exercise worker-crash propagation."""

    def _get_action(self) -> Any:
        raise ValueError("boom")


def _worker_writes_result(result: Any) -> None:
    result["value"] = 42


def _worker_sleeps(result: Any) -> None:
    del result
    time.sleep(30)


def test_run_in_forked_worker_finishes() -> None:
    """A worker that returns writes its result and reports 'finished'."""
    ctx = mp.get_context("fork")
    with ctx.Manager() as manager:
        result = manager.dict()
        outcome, exitcode = run_in_forked_worker(
            ctx, _worker_writes_result, (result,), timeout=10
        )
        assert outcome == "finished"
        assert result["value"] == 42
        assert exitcode == 0


def test_run_in_forked_worker_times_out() -> None:
    """A worker that overruns the timeout is killed and reports 'timeout'."""
    ctx = mp.get_context("fork")
    with ctx.Manager() as manager:
        result = manager.dict()
        outcome, _ = run_in_forked_worker(ctx, _worker_sleeps, (result,), timeout=0.3)
        assert outcome == "timeout"
        assert "value" not in result


def test_run_episode_returns_final_state() -> None:
    """run_episode returns the observation the episode ended on."""
    env = _CountEnv()
    approach = _NoopApproach(env.action_space, env.observation_space, 0, {})
    metrics, _, final_state = run_episode(env, approach, seed=0, max_steps=10)
    assert metrics["solved"]
    assert final_state == np.array([3.0], dtype=np.float32)


def test_run_episode_reports_step_progress() -> None:
    """Replay callers can expose actual rollout progress after every step."""
    env = _CountEnv()
    approach = _NoopApproach(env.action_space, env.observation_space, 0, {})
    updates: list[tuple[int, int]] = []

    metrics, _, _ = run_episode(
        env,
        approach,
        seed=0,
        max_steps=10,
        progress_callback=lambda current, total: updates.append((current, total)),
    )

    assert updates == [(1, 10), (2, 10), (3, 10)]
    assert metrics["num_steps"] == updates[-1][0]


def test_run_episode_with_timeout_solves_within_budget() -> None:
    """A fast policy finishes inside the forked worker and is scored normally."""
    env = _CountEnv()
    approach = _NoopApproach(env.action_space, env.observation_space, 0, {})
    metrics, _, _ = run_episode_with_timeout(
        env, approach, seed=0, max_steps=10, timeout=30
    )
    assert metrics["solved"]
    assert not metrics.get("timed_out")


def test_run_episode_with_timeout_kills_slow_policy() -> None:
    """A policy that overruns the budget is killed and scored unsolved."""
    env = _CountEnv()
    approach = _SlowApproach(env.action_space, env.observation_space, 0, {})
    metrics, frames, final_state = run_episode_with_timeout(
        env, approach, seed=0, max_steps=10, timeout=0.5
    )
    assert metrics["solved"] is False
    assert metrics["timed_out"] is True
    assert metrics["num_steps"] == 0
    assert not frames
    assert final_state is None


def test_run_episode_with_timeout_reraises_worker_crash() -> None:
    """A crash in the policy is carried back and re-raised for the caller to score."""
    env = _CountEnv()
    approach = _CrashApproach(env.action_space, env.observation_space, 0, {})
    with pytest.raises(RuntimeError, match="boom"):
        run_episode_with_timeout(env, approach, seed=0, max_steps=10, timeout=30)


class _SlowPerStepApproach(BaseApproach[Any, Any]):
    """Slow across many steps rather than stuck in one, so only a deadline sees it."""

    def _get_action(self) -> Any:
        time.sleep(0.2)
        return np.zeros(1, dtype=np.float32)


def test_fork_is_avoided_on_macos() -> None:
    """MacOS must take the in-process path; a forked child cannot touch MuJoCo there."""
    assert _EPISODE_FORK_SAFE == (sys.platform != "darwin")


def test_in_process_episode_scores_like_the_forked_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-process path scores a fast policy exactly as the forked one does."""
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", False)
    env = _CountEnv()
    approach = _NoopApproach(env.action_space, env.observation_space, 0, {})
    metrics, _, _ = run_episode_with_timeout(
        env, approach, seed=0, max_steps=10, timeout=30
    )
    assert metrics["solved"]
    assert not metrics.get("timed_out")


def test_in_process_episode_stops_a_policy_hung_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single call that never returns is cut off by the alarm, not left to run.

    Without the alarm a deadline checked between steps would never be reached, so this
    is the case that would hang the whole campaign.
    """
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", False)
    env = _CountEnv()
    approach = _SlowApproach(env.action_space, env.observation_space, 0, {})
    metrics, frames, final_state = run_episode_with_timeout(
        env, approach, seed=0, max_steps=10, timeout=0.5
    )
    assert metrics["solved"] is False
    assert metrics["timed_out"] is True
    assert metrics["num_steps"] == 0
    assert not frames
    assert final_state is None


def test_in_process_episode_stops_a_policy_slow_across_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy that is merely slow is stopped by the per-step deadline."""
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", False)
    env = _CountEnv()
    approach = _SlowPerStepApproach(env.action_space, env.observation_space, 0, {})
    metrics, _, _ = run_episode_with_timeout(
        env, approach, seed=0, max_steps=100, timeout=0.5
    )
    assert metrics["timed_out"] is True
    assert metrics["solved"] is False


def test_in_process_episode_reraises_crash_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy crash surfaces as RuntimeError, matching the forked path's shape."""
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", False)
    env = _CountEnv()
    approach = _CrashApproach(env.action_space, env.observation_space, 0, {})
    with pytest.raises(RuntimeError, match="boom"):
        run_episode_with_timeout(env, approach, seed=0, max_steps=10, timeout=30)


def test_in_process_episode_restores_the_previous_alarm_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not leak its handler, or the next episode inherits the alarm."""
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", False)
    before = signal.getsignal(signal.SIGALRM)
    env = _CountEnv()
    approach = _NoopApproach(env.action_space, env.observation_space, 0, {})
    run_episode_with_timeout(env, approach, seed=0, max_steps=10, timeout=30)
    assert signal.getsignal(signal.SIGALRM) is before
    # A leaked handler and a leaked alarm are separate failures, and the alarm is
    # the one that would fire partway through the next episode.
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


class _SwallowingApproach(BaseApproach[Any, Any]):
    """Retries forever behind ``except Exception``, a common generated-code shape."""

    def _get_action(self) -> Any:
        while True:
            try:
                time.sleep(0.05)
            except Exception:  # pylint: disable=broad-exception-caught
                continue


def test_in_process_episode_stops_a_policy_that_catches_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy swallowing Exception must not escape its budget.

    EpisodeTimeout is a BaseException precisely so ``except Exception`` cannot catch
    it. When it subclassed Exception this policy ran unbounded -- 45s and counting on
    a 3s budget -- while the forked path stopped it on time, so the in-process path
    was quietly weaker rather than equivalent.
    """
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", False)
    env = _CountEnv()
    approach = _SwallowingApproach(env.action_space, env.observation_space, 0, {})
    # Regressing this makes the policy unbounded, so without a backstop the test
    # would hang CI instead of failing it. KeyboardInterrupt is the right lever:
    # the policy under test catches Exception, which does not cover it.
    rescue = threading.Timer(15.0, _thread.interrupt_main)
    rescue.start()
    started = time.monotonic()
    try:
        metrics, _, _ = run_episode_with_timeout(
            env, approach, seed=0, max_steps=1000, timeout=1.0
        )
    except KeyboardInterrupt:
        pytest.fail("policy escaped its 1s budget; the timeout was swallowed")
    finally:
        rescue.cancel()
    assert metrics["timed_out"] is True
    assert time.monotonic() - started < 10


def test_in_process_episode_returns_rendered_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render=True is served by the in-process path, not only the forked one."""
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", False)

    class _RenderingCountEnv(_CountEnv):
        def render(self) -> Any:
            return np.zeros((4, 4, 3), dtype=np.uint8)

    env = _RenderingCountEnv()
    approach = _NoopApproach(env.action_space, env.observation_space, 0, {})
    metrics, frames, _ = run_episode_with_timeout(
        env, approach, seed=0, max_steps=10, timeout=30, render=True
    )
    # One frame for the initial state plus one per step.
    assert len(frames) == metrics["num_steps"] + 1


def test_in_process_timed_out_metrics_keep_the_scheduled_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out episode still reports the count it was scheduled at.

    The scaling curve buckets by object_count, so losing it on the in-process path would
    drop timed-out episodes out of their bucket instead of scoring them as failures
    there.
    """
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", False)
    env = _CountEnv()
    approach = _SlowApproach(env.action_space, env.observation_space, 0, {})
    metrics, _, _ = run_episode_with_timeout(
        env, approach, seed=0, max_steps=10, timeout=0.5, count=7
    )
    assert metrics["timed_out"] is True
    assert metrics["object_count"] == 7


def test_save_frames_creates_pngs(
    tmp_path: Path, sample_frames: list[np.ndarray]
) -> None:
    """save_frames writes PNG files and returns their names."""
    out = tmp_path / "frames"
    filenames = save_frames(sample_frames, out)
    assert len(filenames) == 5
    for f in filenames:
        assert f.endswith(".png")
        assert (out / f).exists()


def test_save_frames_max_frames(
    tmp_path: Path, sample_frames: list[np.ndarray]
) -> None:
    """save_frames respects the max_frames limit."""
    out = tmp_path / "frames"
    filenames = save_frames(sample_frames, out, max_frames=2)
    assert len(filenames) == 2


def test_save_frames_content_readable(
    tmp_path: Path, sample_frames: list[np.ndarray]
) -> None:
    """Saved frames can be read back as valid images."""
    out = tmp_path / "frames"
    filenames = save_frames(sample_frames, out)
    img = iio.imread(str(out / filenames[0]))
    assert img.shape[:2] == (8, 8)


def test_save_video_creates_gif(
    tmp_path: Path, sample_frames: list[np.ndarray]
) -> None:
    """save_video writes a GIF file."""
    gif_path = tmp_path / "test.gif"
    save_video(sample_frames, gif_path)
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_open_video_writer_streams_frames(
    tmp_path: Path, sample_frames: list[np.ndarray]
) -> None:
    """Frames appended one at a time produce the same GIF as save_video."""
    streamed = tmp_path / "streamed.gif"
    with open_video_writer(streamed) as append_frame:
        for frame in sample_frames:
            append_frame(frame)
    batch = tmp_path / "batch.gif"
    save_video(sample_frames, batch)
    assert streamed.read_bytes() == batch.read_bytes()


def test_run_episode_frame_sink_receives_frames() -> None:
    """With a frame_sink, frames go to the sink and the returned list is empty."""

    class _RenderingCountEnv(_CountEnv):
        def render(self):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    env = _RenderingCountEnv()
    approach = _NoopApproach(env.action_space, env.observation_space, 0, {})
    sunk: list[np.ndarray] = []
    metrics, frames, _ = run_episode(
        env, approach, seed=0, max_steps=10, render=True, frame_sink=sunk.append
    )
    assert not frames
    # One frame for the initial state plus one per step.
    assert len(sunk) == metrics["num_steps"] + 1
