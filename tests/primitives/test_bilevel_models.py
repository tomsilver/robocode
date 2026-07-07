"""Tests for the bilevel_models primitive."""

from __future__ import annotations

import pytest
from bilevel_planning.structs import SesameModels

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.primitives import build_primitives


def _obstruction_env() -> KinderGeom2DEnv:
    return KinderGeom2DEnv(
        "kinder/Obstruction2D-o2-v0",
        bilevel_env_name="obstruction2d",
        bilevel_env_model_kwargs={"num_obstructions": 2},
    )


def test_registered_as_raw_models() -> None:
    """build_primitives exposes bilevel_models as the raw SesameModels bundle."""
    env = _obstruction_env()
    prims = build_primitives(env, ["bilevel_models"])
    assert isinstance(prims["bilevel_models"], SesameModels)


def test_only_requested_primitives_are_built() -> None:
    """build_primitives builds only the requested primitives, so an env with no bilevel
    mapping can still use other primitives without a models build."""
    env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")  # infers to no mapping
    prims = build_primitives(env, ["csp"])  # must not raise
    assert "bilevel_models" not in prims


def test_use_on_unmapped_env_fails_loudly() -> None:
    """Requesting bilevel_models on an env with no bilevel model fails loudly."""
    env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
    with pytest.raises(AssertionError, match="bilevel_env_name"):
        build_primitives(env, ["bilevel_models"])


def test_symbolic_layer() -> None:
    """Abstract state, goal, operators, and objects report the domain correctly."""
    env = _obstruction_env()
    models = build_primitives(env, ["bilevel_models"])["bilevel_models"]
    obs, _ = env.reset(seed=0)
    state = models.observation_to_state(obs)

    atoms = {str(a) for a in models.state_abstractor(state).atoms}
    assert "(HandEmpty robot)" in atoms
    assert "(OnTable target_block)" in atoms

    goal = {str(a) for a in models.goal_deriver(state).atoms}
    assert goal == {"(OnTarget target_block)"}

    skill_names = {s.operator.name for s in models.skills}
    assert skill_names == {
        "PickFromTable",
        "PickFromTarget",
        "PlaceOnTable",
        "PlaceOnTarget",
    }
    # operators are inspectable (name + parameters).
    op = next(o for o in models.operators if o.name == "PickFromTable")
    assert len(op.parameters) == 2

    all_names = {o.name for o in state}
    assert {"robot", "target_block", "target_surface"} <= all_names
    rectangles = [o for o in state if o.type.name == "rectangle"]
    assert rectangles


def test_skills_usable_for_control() -> None:
    """Each skill exposes an operator and a controller that grounds to objects, and the
    bundle carries the transition simulator."""
    env = _obstruction_env()
    models = build_primitives(env, ["bilevel_models"])["bilevel_models"]
    obs, _ = env.reset(seed=0)
    state = models.observation_to_state(obs)
    objs = {o.name: o for o in state}

    skill = next(s for s in models.skills if s.operator.name == "PickFromTable")
    controller = skill.controller.ground((objs["robot"], objs["target_block"]))
    assert hasattr(controller, "sample_parameters")
    assert hasattr(controller, "step")
    assert callable(models.transition_fn)
