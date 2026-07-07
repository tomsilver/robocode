"""Tests for the bilevel_models primitive."""

from __future__ import annotations

import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.primitives import build_primitives
from robocode.primitives.bilevel_models import BilevelModels


def _obstruction_env() -> KinderGeom2DEnv:
    return KinderGeom2DEnv(
        "kinder/Obstruction2D-o2-v0",
        bilevel_env_name="obstruction2d",
        bilevel_env_model_kwargs={"num_obstructions": 2},
    )


def test_registered_and_built_env_bound() -> None:
    """build_primitives exposes bilevel_models as an env-bound BilevelModels."""
    env = _obstruction_env()
    prims = build_primitives(env, ["bilevel_models"])
    assert isinstance(prims["bilevel_models"], BilevelModels)


def test_construction_is_lazy() -> None:
    """Constructing the primitive must not build models eagerly (the factory builds it
    for every env, so it must be cheap until actually used)."""
    env = _obstruction_env()
    bm = BilevelModels(env)
    assert bm._models is None  # pylint: disable=protected-access


def test_use_on_unmapped_env_fails_loudly() -> None:
    """Using the primitive on an env with no bilevel model fails loudly."""
    env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")  # infers to no mapping
    bm = BilevelModels(env)  # must not raise
    with pytest.raises(AssertionError, match="bilevel_env_name"):
        _ = bm.models


def test_symbolic_layer() -> None:
    """Abstract state, goal, operators, and objects report the domain correctly."""
    env = _obstruction_env()
    bm = build_primitives(env, ["bilevel_models"])["bilevel_models"]
    obs, _ = env.reset(seed=0)

    atoms = {str(a) for a in bm.get_abstract_state(obs)}
    assert "(HandEmpty robot)" in atoms
    assert "(OnTable target_block)" in atoms

    goal = {str(a) for a in bm.get_goal_atoms(obs)}
    assert goal == {"(OnTarget target_block)"}

    assert set(bm.skill_names) == {
        "PickFromTable",
        "PickFromTarget",
        "PlaceOnTable",
        "PlaceOnTarget",
    }
    # operators are inspectable (name + parameters).
    op = next(o for o in bm.operators if o.name == "PickFromTable")
    assert len(op.parameters) == 2

    all_names = {o.name for o in bm.get_objects(obs)}
    assert {"robot", "target_block", "target_surface"} <= all_names
    rectangles = bm.get_objects(obs, type_name="rectangle")
    assert rectangles and all(o.type.name == "rectangle" for o in rectangles)


def test_raw_models_usable_for_skills() -> None:
    """The `.models` bundle exposes the skills (with controllers) and the transition
    simulator."""
    env = _obstruction_env()
    bm = build_primitives(env, ["bilevel_models"])["bilevel_models"]
    obs, _ = env.reset(seed=0)
    state = bm.models.observation_to_state(obs)
    objs = {o.name: o for o in state}

    skill = next(s for s in bm.models.skills if s.operator.name == "PickFromTable")
    controller = skill.ground((objs["robot"], objs["target_block"])).controller
    assert hasattr(controller, "sample_parameters")
    assert hasattr(controller, "step")
    assert callable(bm.models.transition_fn)
