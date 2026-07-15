"""Tests for the shared bilevel model builder."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.utils.bilevel import build_sesame_models, infer_bilevel_mapping


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
    """build_sesame_models fails loudly for an env with no bilevel model."""
    env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")  # infers to (None, {})
    assert env.bilevel_env_name is None
    with pytest.raises(AssertionError, match="bilevel_env_name"):
        build_sesame_models(env)


def test_infer_bilevel_mapping_agrees_with_env_configs() -> None:
    """Inference from env_id matches every explicit env-config mapping (so the fallback
    and the configs can never silently diverge)."""
    conf_dir = Path("experiments/conf/environment")
    checked = 0
    for yaml_file in sorted(conf_dir.glob("*.yaml")):
        cfg = OmegaConf.to_container(OmegaConf.load(yaml_file), resolve=True)
        assert isinstance(cfg, dict)
        name = cfg.get("bilevel_env_name")
        if name is None:
            continue
        # env_id-based inference only applies to the fixed-count KinderGeom2DEnv
        # configs; variable-count (VariableObjectCountEnv) configs carry no env_id and
        # resolve their per-count model kwargs differently.
        if "env_id" not in cfg:
            continue
        inferred_name, inferred_kwargs = infer_bilevel_mapping(str(cfg["env_id"]))
        assert inferred_name == name, yaml_file.name
        assert inferred_kwargs == dict(cfg["bilevel_env_model_kwargs"]), yaml_file.name
        checked += 1
    assert checked >= 12  # all mapped 2D families x easy/medium/hard


def test_infer_bilevel_mapping_none_for_unsupported() -> None:
    """Envs with no bilevel model (pushpullhook, 3D) infer to (None, {})."""
    for env_id in (
        "kinder/PushPullHook2D-v0",
        "kinder/Motion3D-v0",
        "kinder/Obstruction3D-o2-v0",
    ):
        assert infer_bilevel_mapping(env_id) == (None, {})


def test_kinder_env_infers_mapping_when_not_given() -> None:
    """A plain KinderGeom2DEnv (no explicit mapping) infers it from env_id, so the
    bilevel_models primitive works on an env the agent builds by hand to test."""
    env = KinderGeom2DEnv("kinder/StickButton2D-b3-v0")
    assert env.bilevel_env_name == "stickbutton2d"
    assert env.bilevel_env_model_kwargs == {"num_buttons": 3}


def test_explicit_mapping_overrides_inference() -> None:
    """An explicitly passed mapping is used as-is (configs still win)."""
    env = KinderGeom2DEnv(
        "kinder/Obstruction2D-o2-v0",
        bilevel_env_name="obstruction2d",
        bilevel_env_model_kwargs={"num_obstructions": 2},
    )
    assert env.bilevel_env_name == "obstruction2d"
    assert env.bilevel_env_model_kwargs == {"num_obstructions": 2}
