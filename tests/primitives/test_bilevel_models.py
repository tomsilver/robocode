"""Tests for the bilevel_models primitive."""

from __future__ import annotations

import pytest
from bilevel_planning.structs import SesameModels

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.primitives import build_primitives
from robocode.utils.bilevel import VariableCountBilevelModels


def _obstruction_env() -> KinderGeom2DEnv:
    return KinderGeom2DEnv(
        "kinder/Obstruction2D-o2-v0",
        bilevel_env_name="obstruction2d",
        bilevel_env_model_kwargs={"num_obstructions": 2},
    )


def _variable_obstruction_env() -> VariableObjectCountEnv:
    return VariableObjectCountEnv(
        constant_object_env_path=(
            "kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv"
        ),
        count_kwarg="num_obstructions",
        count_object_prefix="obstruction",
        design_counts=[1, 2],
        eval_counts=[1, 2],
        bilevel_env_name="obstruction2d",
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


def test_variable_count_exposes_accessor() -> None:
    """A variable-count env yields a count-dispatching accessor, not one bundle."""
    env = _variable_obstruction_env()
    prims = build_primitives(env, ["bilevel_models"])
    assert isinstance(prims["bilevel_models"], VariableCountBilevelModels)


def test_models_for_state_resolves_and_caches_by_count() -> None:
    """models_for_state builds the bundle for the state's count and caches per count."""
    env = _variable_obstruction_env()
    accessor = build_primitives(env, ["bilevel_models"])["bilevel_models"]
    state, _ = env.reset(seed=0, options={"object_count": 2})

    models = accessor.models_for_state(state)
    assert isinstance(models, SesameModels)
    assert accessor.models_for_count(2) is models  # cached: same count, same bundle
    assert accessor.models_for_count(1) is not models  # a different count is distinct


def test_variable_count_symbolic_layer() -> None:
    """The per-count bundle abstracts the object-centric state and lists the
    operators."""
    env = _variable_obstruction_env()
    accessor = build_primitives(env, ["bilevel_models"])["bilevel_models"]
    state, _ = env.reset(seed=0, options={"object_count": 2})
    models = accessor.models_for_state(state)

    atoms = {str(a) for a in models.state_abstractor(state).atoms}
    assert "(HandEmpty robot)" in atoms

    skill_names = {s.operator.name for s in models.skills}
    assert skill_names == {
        "PickFromTable",
        "PickFromTarget",
        "PlaceOnTable",
        "PlaceOnTarget",
    }
