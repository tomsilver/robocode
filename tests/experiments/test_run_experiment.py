"""Tests for run_experiment helpers.

experiments/ is a scripts directory, not an installed package, so the module is
loaded from its path. Covers eval-suite seed resolution: `eval_seed` pins the
suite independently of `seed`; when unset, it follows `seed`.
"""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

_MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments" / "run_experiment.py"
_SPEC = importlib.util.spec_from_file_location("run_experiment", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run_experiment: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_experiment)


def _suite(seed: int, num_eval: int = 5) -> list[int]:
    """Derive the eval seed list the way _main does."""
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 2**63)) for _ in range(num_eval)]


def test_eval_seed_defaults_to_seed() -> None:
    """When unset, eval_seed follows seed."""
    cfg = OmegaConf.create({"seed": 42, "eval_seed": None})
    assert run_experiment.resolve_eval_seed(cfg) == 42


def test_eval_seed_missing_key_defaults_to_seed() -> None:
    """A config without the key behaves like eval_seed: null."""
    cfg = OmegaConf.create({"seed": 42})
    assert run_experiment.resolve_eval_seed(cfg) == 42


def test_eval_seed_overrides_seed() -> None:
    """A set eval_seed wins over seed."""
    cfg = OmegaConf.create({"seed": 42, "eval_seed": 1000})
    assert run_experiment.resolve_eval_seed(cfg) == 1000


def test_pinned_eval_seed_yields_identical_suite_across_training_seeds() -> None:
    """Pinning eval_seed gives differently-seeded runs the same eval suite."""
    cfg_a = OmegaConf.create({"seed": 24, "eval_seed": 1000})
    cfg_b = OmegaConf.create({"seed": 424, "eval_seed": 1000})
    suite_a = _suite(run_experiment.resolve_eval_seed(cfg_a))
    suite_b = _suite(run_experiment.resolve_eval_seed(cfg_b))
    assert suite_a == suite_b


def test_non_integer_eval_seed_rejected() -> None:
    """A fractional seed raises instead of silently truncating."""
    cfg = OmegaConf.create({"seed": 42, "eval_seed": 42.5})
    with pytest.raises(ValueError, match="integer"):
        run_experiment.resolve_eval_seed(cfg)


def test_unpinned_eval_seed_tracks_training_seed() -> None:
    """Without a pin, the eval seed is the training seed, so runs differ."""
    cfg_a = OmegaConf.create({"seed": 24, "eval_seed": None})
    cfg_b = OmegaConf.create({"seed": 424, "eval_seed": None})
    assert run_experiment.resolve_eval_seed(cfg_a) == 24
    assert run_experiment.resolve_eval_seed(cfg_b) == 424
