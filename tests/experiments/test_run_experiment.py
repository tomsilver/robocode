"""Tests for the experiment runner's seed protocol.

``experiments/`` is a scripts directory rather than an installed package, so
the runner module is loaded from its path.
"""

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

_MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments" / "run_experiment.py"
_SPEC = importlib.util.spec_from_file_location("run_experiment", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run_experiment: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_experiment)


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
