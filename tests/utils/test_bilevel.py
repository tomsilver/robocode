"""Tests for the shared bilevel model builder."""

from __future__ import annotations

import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.utils.bilevel import build_sesame_models


def test_build_sesame_models_returns_full_bundle() -> None:
    """A mapped env yields a SesameModels bundle with the expected pieces."""
    env = KinderGeom2DEnv(
        "kinder/Obstruction2D-o0-v0",
        bilevel_env_name="obstruction2d",
        bilevel_env_model_kwargs={"num_obstructions": 0},
    )
    models = build_sesame_models(env)
    for attr in (
        "skills",
        "predicates",
        "types",
        "observation_to_state",
        "state_abstractor",
        "goal_deriver",
        "transition_fn",
    ):
        assert hasattr(models, attr)
    assert len(models.skills) > 0


def test_build_sesame_models_requires_mapping() -> None:
    """An env without the bilevel mapping fails loudly rather than silently."""
    env = KinderGeom2DEnv("kinder/Obstruction2D-o0-v0")  # no bilevel_env_name
    with pytest.raises(AssertionError, match="bilevel_env_name"):
        build_sesame_models(env)
