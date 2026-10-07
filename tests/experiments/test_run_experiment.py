"""Tests for the experiment runner's seed protocol.

``experiments/`` is a scripts directory rather than an installed package, so
the runner module is loaded from its path.
"""

import importlib.util
import math
import multiprocessing
import sys
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box
from omegaconf import OmegaConf

_MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments" / "run_experiment.py"
_SPEC = importlib.util.spec_from_file_location("run_experiment", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run_experiment: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_experiment)


class _IncrementEnv(Env):  # type: ignore[type-arg]
    """A tiny env whose step count is set by the approach's action magnitude.

    ``step`` advances ``_pos`` by the action and terminates once ``_pos`` reaches
    1.0, so the episode length equals ``ceil(1.0 / action)``. A deterministic
    approach (fixed action) therefore always takes the same number of steps,
    while one that changes its action between resets produces replays that
    disagree.
    """

    def __init__(self) -> None:
        self.observation_space = Box(0.0, 10.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self._pos = 0.0

    def reset(self, *, seed: Any = None, options: Any = None) -> Any:
        super().reset(seed=seed)
        self._pos = 0.0
        return np.array([self._pos], dtype=np.float32), {}

    def step(self, action: Any) -> Any:
        self._pos += float(np.asarray(action).item())
        obs = np.array([self._pos], dtype=np.float32)
        return obs, 0.0, self._pos >= 1.0, False, {}

    def render(self) -> None:
        return None


class _FixedActionApproach:
    """Always acts with the same magnitude, so every replay takes the same steps."""

    def __init__(self, action: float) -> None:
        self._action = action

    def reset(self, state: Any, info: Any) -> None:
        """Start an episode; the action is fixed so length never varies."""
        del state, info

    def step(self) -> Any:
        """Return the fixed action that always ends the episode in two steps."""
        return np.array([self._action], dtype=np.float32)

    def update(self, state: Any, reward: float, done: bool, info: Any) -> None:
        """Record the outcome, matching the BaseApproach interface."""
        del state, reward, done, info


class _AlternatingActionApproach:
    """Alternates its action magnitude across resets, so replays disagree.

    The first reset in a fresh sequence uses a large action (short episode) and the
    second a small one (long episode), which is the same shape as a policy whose per-
    plan-call budget is decided by wall clock.
    """

    def __init__(self) -> None:
        self._reset_count = 0

    def reset(self, state: Any, info: Any) -> None:
        """Start an episode, flipping the action for the next replay."""
        del state, info
        self._reset_count += 1

    def step(self) -> Any:
        """Return the action whose magnitude sets the episode length."""
        action = 0.5 if self._reset_count % 2 else 0.25
        return np.array([action], dtype=np.float32)

    def update(self, state: Any, reward: float, done: bool, info: Any) -> None:
        """Record the outcome, matching the BaseApproach interface."""
        del state, reward, done, info


class _CrashOnSecondResetApproach:
    """Plays one normal replay, then raises on every step of the next."""

    def __init__(self) -> None:
        self._reset_count = 0

    def reset(self, state: Any, info: Any) -> None:
        """Count resets so only the second replay crashes."""
        del state, info
        self._reset_count += 1

    def step(self) -> Any:
        """Solve normally on the first replay, crash on the second."""
        if self._reset_count > 1:
            raise RuntimeError("policy exploded on the replayed seed")
        return np.array([0.5], dtype=np.float32)

    def update(self, state: Any, reward: float, done: bool, info: Any) -> None:
        """Record the outcome, matching the BaseApproach interface."""
        del state, reward, done, info


class _SharedCounterApproach:
    """Length varies across resets through a fork-shared multiprocessing counter.

    Each forked worker inherits the counter's current value, and the worker-side
    reset bumps it, so the parent-side counter advance between replays makes the
    two replays take deliberately different step counts without randomness.
    """

    def __init__(self, counter: "Synchronized[int]") -> None:
        self._counter = counter

    def reset(self, state: Any, info: Any) -> None:
        """Bump the shared counter so the episode length flips each replay."""
        del state, info
        with self._counter.get_lock():
            self._counter.value += 1

    def step(self) -> Any:
        """Return the action whose magnitude sets the episode length."""
        action = 0.5 if self._counter.value % 2 else 0.25
        return np.array([action], dtype=np.float32)

    def update(self, state: Any, reward: float, done: bool, info: Any) -> None:
        """Record the outcome, matching the BaseApproach interface."""
        del state, reward, done, info


@pytest.fixture(autouse=True)
def _in_process_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run replays in-process so the test is not platform-dependent.

    The forked path copies the approach into a child per replay, which would hide the
    alternating approach's cross-reset state. The in-process path keeps the same
    instance, so the two replays observe the state change.
    """
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", False)


def test_determinism_check_reports_full_agreement() -> None:
    """A fixed-action policy replays to the same step count every time."""
    env = _IncrementEnv()
    approach = _FixedActionApproach(0.5)  # 2 steps every episode
    result = run_experiment.run_determinism_check(
        env,
        approach,
        [10, 11, 12],
        num_episodes=3,
        max_steps=100,
        timeout=30,
    )
    assert result["determinism_check_num_episodes"] == 3
    assert result["determinism_check_agreement_rate"] == 1.0
    assert result["determinism_check"]["num_episodes"] == 3
    assert all(e["agrees"] for e in result["determinism_check"]["episodes"])


def test_determinism_check_flags_a_nondeterministic_policy() -> None:
    """An approach whose step count varies across replays drops the agreement rate."""
    env = _IncrementEnv()
    approach = _AlternatingActionApproach()
    result = run_experiment.run_determinism_check(
        env,
        approach,
        [10, 11],
        num_episodes=2,
        max_steps=100,
        timeout=30,
    )
    assert result["determinism_check_num_episodes"] == 2
    assert result["determinism_check_agreement_rate"] == 0.0
    # The two replays of the first seed disagree: one is short, one is long.
    first = result["determinism_check"]["episodes"][0]
    assert first["agrees"] is False
    assert first["num_steps"] != first["replay2_num_steps"]


def test_determinism_check_clamps_to_available_seeds() -> None:
    """Requesting more replays than eval seeds replays every seed once."""
    env = _IncrementEnv()
    approach = _FixedActionApproach(0.5)
    result = run_experiment.run_determinism_check(
        env,
        approach,
        [10, 11],
        num_episodes=10,
        max_steps=100,
        timeout=30,
    )
    assert result["determinism_check_num_episodes"] == 2
    assert result["determinism_check"]["num_episodes"] == 2


def test_determinism_check_disabled_returns_empty() -> None:
    """num_episodes=0 records no replays and no agreement."""
    env = _IncrementEnv()
    approach = _FixedActionApproach(0.5)
    result = run_experiment.run_determinism_check(
        env,
        approach,
        [10, 11],
        num_episodes=0,
        max_steps=100,
        timeout=30,
    )
    assert result["determinism_check_num_episodes"] == 0
    assert result["determinism_check"]["num_episodes"] == 0
    assert result["determinism_check"]["episodes"] == []
    assert math.isnan(result["determinism_check_agreement_rate"])


def test_repository_protocol_requires_explicit_eval_seed() -> None:
    """The public config does not contain the private evaluation seed."""
    cfg = OmegaConf.load(_MODULE_PATH.parent / "conf" / "config.yaml")
    assert cfg.eval_timeout == 60
    with pytest.raises(ValueError, match="must be set explicitly"):
        run_experiment.resolve_eval_seed(cfg)


def test_eval_seed_is_required() -> None:
    """Evaluation must never silently follow the replicate seed."""
    cfg = OmegaConf.create({"replicate_seed": 42})
    with pytest.raises(ValueError, match="must be set explicitly"):
        run_experiment.resolve_eval_seed(cfg)


def test_eval_seed_cannot_be_null() -> None:
    """An explicit null is rejected rather than falling back."""
    cfg = OmegaConf.create({"replicate_seed": 42, "eval_seed": None})
    with pytest.raises(ValueError, match="must be set explicitly"):
        run_experiment.resolve_eval_seed(cfg)


@pytest.mark.parametrize("eval_seed", [-1, 42.5])
def test_invalid_eval_seed_is_rejected(eval_seed: Any) -> None:
    """Invalid evaluation seeds fail before synthesis work can begin."""
    with pytest.raises(ValueError, match="eval_seed must be a nonnegative integer"):
        run_experiment.resolve_eval_seed(OmegaConf.create({"eval_seed": eval_seed}))


def test_pinned_eval_seed_yields_identical_suite_across_replicates() -> None:
    """Replicate changes do not change the ordered evaluation episodes."""
    cfg_a = OmegaConf.create({"replicate_seed": 24, "eval_seed": 918273645})
    cfg_b = OmegaConf.create({"replicate_seed": 424, "eval_seed": 918273645})
    suite_a = run_experiment.generate_eval_seeds(
        run_experiment.resolve_eval_seed(cfg_a), 100
    )
    suite_b = run_experiment.generate_eval_seeds(
        run_experiment.resolve_eval_seed(cfg_b), 100
    )
    assert suite_a == suite_b
    assert len(suite_a) == len(set(suite_a)) == 100


def test_eval_suite_changes_only_when_protocol_seed_changes() -> None:
    """Changing the master seed produces a different episode suite."""
    assert run_experiment.generate_eval_seeds(918273645, 5) != (
        run_experiment.generate_eval_seeds(918273646, 5)
    )


def test_eval_suite_requires_positive_size() -> None:
    """An empty evaluation suite is a configuration error."""
    with pytest.raises(ValueError, match="positive"):
        run_experiment.generate_eval_seeds(918273645, 0)


def test_local_generated_code_backend_is_rejected() -> None:
    """The host-readable local sandbox cannot protect experimenter config."""
    cfg = OmegaConf.create({"approach": {"container_backend": "local"}})
    with pytest.raises(ValueError, match="cannot isolate eval_seed"):
        run_experiment.validate_eval_seed_isolation(cfg)


@pytest.mark.parametrize("backend", ["docker", "apptainer"])
def test_container_backends_satisfy_seed_isolation_boundary(backend: str) -> None:
    """Both supported experiment backends provide a filesystem boundary."""
    cfg = OmegaConf.create({"approach": {"container_backend": backend}})
    run_experiment.validate_eval_seed_isolation(cfg)


def test_non_generated_approach_needs_no_container_backend() -> None:
    """Ordinary baselines do not launch an untrusted synthesis process."""
    cfg = OmegaConf.create({"approach": {"_target_": "example.RandomApproach"}})
    run_experiment.validate_eval_seed_isolation(cfg)


def test_tracker_experiment_id_is_accepted() -> None:
    """Generated identifiers are available to output metadata."""
    condition_id = "motion2d_easy__agentic__none__whitebox__abc12345"
    cfg = OmegaConf.create({"experiment_id": condition_id})
    assert run_experiment.resolve_experiment_id(cfg) == condition_id


def test_missing_experiment_id_is_allowed_for_manual_runs() -> None:
    """Existing ad-hoc commands remain valid without tracker metadata."""
    assert run_experiment.resolve_experiment_id(OmegaConf.create({})) is None


@pytest.mark.parametrize(
    "condition_id",
    ["renamed by hand", "../../other-result", "condition__not-a-hash"],
)
def test_invalid_experiment_id_is_rejected(condition_id: str) -> None:
    """Unsafe or hand-edited identifiers fail before an experiment starts."""
    cfg = OmegaConf.create({"experiment_id": condition_id})
    with pytest.raises(ValueError, match="tracker-generated"):
        run_experiment.resolve_experiment_id(cfg)


def test_determinism_check_survives_a_crashing_replay() -> None:
    """A policy crash on a replayed seed is recorded, not fatal to the run."""
    env = _IncrementEnv()
    approach = _CrashOnSecondResetApproach()
    result = run_experiment.run_determinism_check(
        env,
        approach,
        [10],
        num_episodes=1,
        max_steps=100,
        timeout=30,
    )
    assert result["determinism_check_num_episodes"] == 1
    first = result["determinism_check"]["episodes"][0]
    assert first["num_steps"] is not None
    assert first["replay2_num_steps"] is None
    assert first["replay2_solved"] is False
    assert first["agrees"] is False
    assert result["determinism_check_agreement_rate"] == 0.0


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="forked workers are disabled on darwin",
)
def test_determinism_check_covers_forked_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checker reports disagreement through real forked workers.

    A shared multiprocessing counter makes the two replays take deliberately
    different lengths: each forked child inherits the counter, its reset bumps
    it, and the step action reads the bumped value, so replay one is short and
    replay two is long without relying on random outcomes or wall clock.
    """
    monkeypatch.setattr("robocode.utils.episode._EPISODE_FORK_SAFE", True)
    counter = multiprocessing.Value("i", 0)
    env = _IncrementEnv()
    approach = _SharedCounterApproach(counter)
    result = run_experiment.run_determinism_check(
        env,
        approach,
        [10],
        num_episodes=1,
        max_steps=100,
        timeout=30,
    )
    first = result["determinism_check"]["episodes"][0]
    assert first["num_steps"] != first["replay2_num_steps"]
    assert first["agrees"] is False
    assert result["determinism_check_agreement_rate"] == 0.0
