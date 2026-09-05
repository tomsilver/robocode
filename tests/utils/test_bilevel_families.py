"""Family mapping and model construction for the dynamic 2D and 3D kinder families."""

from __future__ import annotations

from typing import Any

import pytest

from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.utils.bilevel import (
    bilevel_count_kwarg,
    build_sesame_models,
    infer_bilevel_env_name_from_path,
    infer_bilevel_mapping,
)

_ID_CASES = [
    ("kinder/DynObstruction2D-o2-v0", "dynobstruction2d", {"num_obstructions": 2}),
    ("kinder/DynPushPullHook2D-o5-v0", "dynpushpullhook2d", {"num_obstructions": 5}),
    ("kinder/Transport3D-o2-v0", "transport3d", {"num_objects": 2}),
    ("kinder/KinematicShelf3D-o3-v0", "shelf3d", {"num_objects": 3}),
    ("kinder/Tossing3D-o1-v0", "tidybot3d_tossing3D", {"num_objects": 1}),
]

_PATH_CASES = [
    ("kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv", "dynobstruction2d"),
    (
        "kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
        "dynpushpullhook2d",
    ),
    ("kinder.envs.kinematic3d.transport3d:Transport3DEnv", "transport3d"),
    ("kinder.envs.kinematic3d.shelf3d:Shelf3DEnv", "shelf3d"),
    ("kinder.envs.dynamic3d.task_families:Tossing3DEnv", "tidybot3d_tossing3D"),
    ("kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv", "obstruction2d"),
    # No models: Obstruction3D has skills only; the MuJoCo Shelf3D is a different env.
    ("kinder.envs.kinematic3d.obstruction3d:Obstruction3DEnv", None),
    ("kinder.envs.dynamic3d.task_families:Shelf3DEnv", None),
    ("robocode.environments.dyn_scoop_sort2d:DynScoopSort2DEnv", None),
]


@pytest.mark.parametrize("env_id,name,kwargs", _ID_CASES)
def test_infer_mapping_for_new_families(
    env_id: str, name: str, kwargs: dict[str, int]
) -> None:
    """Dynamic 2D and 3D ids map to their (case-sensitive) model module and kwarg."""
    assert infer_bilevel_mapping(env_id) == (name, kwargs)


@pytest.mark.parametrize("env_path,name", _PATH_CASES)
def test_infer_env_name_from_path(env_path: str, name: str | None) -> None:
    """A variable-count env's class path maps to its family, or None when unmapped."""
    assert infer_bilevel_env_name_from_path(env_path) == name


def test_count_kwarg_differs_from_env_kwarg_for_kinematic3d() -> None:
    """Transport3D/Shelf3D envs take num_cubes but their models take num_objects."""
    assert bilevel_count_kwarg("transport3d") == "num_objects"
    assert bilevel_count_kwarg("shelf3d") == "num_objects"
    assert bilevel_count_kwarg("obstruction2d") == "num_obstructions"
    with pytest.raises(AssertionError, match="not a known bilevel env family"):
        bilevel_count_kwarg("packing3d")


def test_kinder_geom3d_env_infers_mapping_and_builds_models() -> None:
    """A fixed-count 3D env infers its family from env_id and yields a full bundle."""
    env = KinderGeom3DEnv("kinder/Transport3D-o1-v0")
    try:
        assert env.bilevel_env_name == "transport3d"
        assert env.bilevel_env_model_kwargs == {"num_objects": 1}
        models = build_sesame_models(env)
        assert len(models.skills) > 0
        obs, _ = env.reset(seed=0)
        state = models.observation_to_state(obs)
        assert models.state_abstractor(state) is not None
    finally:
        env.close()


def test_kinder_geom3d_env_without_models_has_no_mapping() -> None:
    """Obstruction3D has no bilevel models, so the loud config check fires."""
    env = KinderGeom3DEnv("kinder/Obstruction3D-o0-v0")
    try:
        assert env.bilevel_env_name is None
        with pytest.raises(AssertionError, match="bilevel_env_name"):
            build_sesame_models(env)
    finally:
        env.close()


_VARIABLE_COUNT_CASES: list[Any] = [
    pytest.param(
        {
            "constant_object_env_path": (
                "kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv"
            ),
            "count_kwarg": "num_obstructions",
            "count_object_prefix": "obstruction",
            "design_counts": [0, 1],
            "eval_counts": [0, 1, 2],
        },
        "dynobstruction2d",
        1,
        id="dynobstruction2d",
    ),
    pytest.param(
        {
            "constant_object_env_path": (
                "kinder.envs.kinematic3d.transport3d:Transport3DEnv"
            ),
            "count_kwarg": "num_cubes",
            "count_object_prefix": "cube",
            "design_counts": [1],
            "eval_counts": [1, 2],
            "constant_object_env_kwargs": {"num_boxes": 1},
        },
        "transport3d",
        2,
        id="transport3d",
    ),
]


@pytest.mark.parametrize("env_kwargs,name,count", _VARIABLE_COUNT_CASES)
def test_variable_count_models_round_trip_state(
    env_kwargs: dict[str, Any], name: str, count: int
) -> None:
    """Inferred family; per-count models accept the env's Box view of a state, and the
    model's own sim names the same objects (guards the count kwarg translation and the
    fixed-kwarg defaults such as Transport3D's box count)."""
    env = VariableObjectCountEnv(**env_kwargs)
    try:
        assert env.bilevel_env_name == name
        state, _ = env.reset(seed=0, options={"object_count": count})
        models = env.models_for_count(count)
        round_trip = models.observation_to_state(env.to_box(state))
        assert set(round_trip.get_object_names()) == set(state.get_object_names())
        assert models.state_abstractor(round_trip) is not None
    finally:
        env.close()


def test_inferred_family_is_not_advertised_in_description() -> None:
    """Only an explicitly configured family shows up in the env card's example, so
    inferring one for planner use does not change the agent-facing prompt."""
    inferred = VariableObjectCountEnv(
        constant_object_env_path=(
            "kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv"
        ),
        count_kwarg="num_obstructions",
        count_object_prefix="obstruction",
        design_counts=[0],
        eval_counts=[0, 1],
    )
    try:
        assert inferred.bilevel_env_name == "dynobstruction2d"
        assert "bilevel_env_name" not in inferred.env_description
    finally:
        inferred.close()
    explicit = VariableObjectCountEnv(
        constant_object_env_path=(
            "kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv"
        ),
        count_kwarg="num_obstructions",
        count_object_prefix="obstruction",
        design_counts=[0],
        eval_counts=[0, 1],
        bilevel_env_name="obstruction2d",
    )
    try:
        assert 'bilevel_env_name="obstruction2d"' in explicit.env_description
    finally:
        explicit.close()


def test_tossing3d_models_round_trip_state() -> None:
    """Tossing3D's models (MuJoCo + PyBullet) accept the env's count-1 Box view."""
    try:
        import mujoco  # pylint: disable=import-outside-toplevel

        mujoco.GLContext(max_width=16, max_height=16).free()
    except Exception as e:  # pylint: disable=broad-except
        pytest.skip(f"mujoco GL runtime unavailable: {e}")
    env = VariableObjectCountEnv(
        constant_object_env_path="kinder.envs.dynamic3d.task_families:Tossing3DEnv",
        count_kwarg="num_objects",
        count_object_prefix="cube_",
        design_counts=[1],
        eval_counts=[1, 2],
        constant_object_env_kwargs={"scene_bg": False},
    )
    try:
        assert env.bilevel_env_name == "tidybot3d_tossing3D"
        state, _ = env.reset(seed=0, options={"object_count": 1})
        models = env.models_for_count(1)
        round_trip = models.observation_to_state(env.to_box(state))
        assert set(round_trip.get_object_names()) == set(state.get_object_names())
        # The upstream operators name a single cube, so count 2 has no models.
        with pytest.raises(NotImplementedError):
            env.models_for_count(2)
    finally:
        env.close()
