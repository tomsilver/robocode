"""Tests for episode utilities."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box

from robocode.approaches.base_approach import BaseApproach, InstanceResult
from robocode.utils.episode import (
    load_generated_approach,
    run_episode,
    run_per_instance_eval,
    save_frames,
    save_video,
    summarize_by_count,
)


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
    ) -> InstanceResult:
        del env, output_subdir
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


def test_run_episode_returns_final_state() -> None:
    """run_episode returns the observation the episode ended on."""

    class _CountEnv(Env):
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
        def _get_action(self) -> Any:
            return np.zeros(1, dtype=np.float32)

    env = _CountEnv()
    approach = _NoopApproach(env.action_space, env.observation_space, 0, {})
    metrics, _, final_state = run_episode(env, approach, seed=0, max_steps=10)
    assert metrics["solved"]
    assert final_state == np.array([3.0], dtype=np.float32)


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
