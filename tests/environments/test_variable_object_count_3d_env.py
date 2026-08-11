"""Kinematic3D coverage for the shared variable-object-count environment."""

from __future__ import annotations

from typing import Any

import pybullet as p
import pytest
from relational_structs import ObjectCentricState

from robocode.environments.constrained_cupboard3d import ConstrainedCupboard3DEnv
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.utils.object_centric_codec import (
    decode_object_centric_state,
    encode_object_centric_state,
)

_FAMILY_CASES = [
    pytest.param(
        "kinder.envs.kinematic3d.table3d:Table3DEnv",
        "num_cubes",
        "cube",
        {},
        True,
        id="table3d-cubes",
    ),
    pytest.param(
        "kinder.envs.kinematic3d.transport3d:Transport3DEnv",
        "num_cubes",
        "cube",
        {"num_boxes": 1},
        True,
        id="transport3d-cubes",
    ),
    pytest.param(
        "kinder.envs.kinematic3d.shelf3d:Shelf3DEnv",
        "num_cubes",
        "cube",
        {},
        True,
        id="shelf3d-cubes",
    ),
    pytest.param(
        "kinder.envs.kinematic3d.obstruction3d:Obstruction3DEnv",
        "num_obstructions",
        "obstruction",
        {},
        True,
        id="obstruction3d-obstructions",
    ),
    pytest.param(
        "kinder.envs.kinematic3d.packing3d:Packing3DEnv",
        "num_parts",
        "part",
        {},
        False,
        id="packing3d-parts",
    ),
]


def _num_prefixed(state: ObjectCentricState, prefix: str) -> int:
    return sum(name.startswith(prefix) for name in state.get_object_names())


@pytest.mark.parametrize(
    "path,count_kwarg,prefix,fixed_kwargs,exact_roundtrip", _FAMILY_CASES
)
def test_kinematic3d_family_supports_variable_count_roundtrip(
    path: str,
    count_kwarg: str,
    prefix: str,
    fixed_kwargs: dict[str, Any],
    exact_roundtrip: bool,
) -> None:
    """Every 3D count axis supports pinned reset, wire-style restore, and step."""
    env = VariableObjectCountEnv(
        constant_object_env_path=path,
        count_kwarg=count_kwarg,
        count_object_prefix=prefix,
        design_counts=[1],
        eval_counts=[1, 2],
        constant_object_env_kwargs={"realistic_bg": False, **fixed_kwargs},
    )
    physics_client_ids: list[int] = []
    try:
        state, info = env.reset(seed=2, options={"object_count": 2})
        assert info["object_count"] == 2
        assert env.current_count == 2
        assert _num_prefixed(state, prefix) == 2
        assert state.__class__ is not ObjectCentricState
        assert env.observation_space.contains(state)

        # The black-box codec intentionally reconstructs the portable base class.
        # set_state() must rehydrate the backend's specialized 3D state class before
        # its robot/grasp helpers are used.
        decoded = decode_object_centric_state(encode_object_centric_state(state))
        assert decoded.__class__ is ObjectCentricState
        env.set_state(decoded)
        restored = env.get_state()
        assert restored.__class__ is state.__class__
        if exact_roundtrip:
            assert restored.allclose(state)
        else:
            # The pinned Kinder revision does not preserve Packing3D's randomized
            # part geometry during restoration, so this checks only shared routing.
            assert restored.get_object_names() == state.get_object_names()

        next_state, _, _, _, step_info = env.step(env.action_space.sample())
        assert next_state.__class__ is state.__class__
        assert step_info["object_count"] == 2
        assert env.to_box(next_state).ndim == 1

        # The init_state reset route needs the same rehydration as set_state().
        reset_state, reset_info = env.reset(options={"init_state": decoded})
        assert reset_state.__class__ is state.__class__
        assert reset_info["object_count"] == 2

        card = env.env_description
        assert "VARIABLE number of objects" in card
        assert prefix in card
        assert "constant_object_env_kwargs=" in card
        assert "state.get(obj, 'x')" not in card  # invalid for most 3D types
    finally:
        physics_client_ids.extend(
            getattr(getattr(backend, "_object_centric_env"), "physics_client_id")
            for backend in env._backends.values()  # pylint: disable=protected-access
        )
        env.close()

    # The wrapper owns all per-count PyBullet clients and must release them.
    assert all(not p.isConnected(client_id) for client_id in physics_client_ids)


def test_transport3d_varies_cubes_with_one_box() -> None:
    """Transport3D keeps one box while varying its cube count."""
    env = VariableObjectCountEnv(
        constant_object_env_path=("kinder.envs.kinematic3d.transport3d:Transport3DEnv"),
        count_kwarg="num_cubes",
        count_object_prefix="cube",
        design_counts=[1],
        eval_counts=[1, 4],
        constant_object_env_kwargs={
            "num_boxes": 1,
            "realistic_bg": False,
        },
    )
    try:
        design_state, _ = env.reset(seed=0, options={"object_count": 1})
        assert _num_prefixed(design_state, "cube") == 1
        assert _num_prefixed(design_state, "box") == 1

        eval_state, info = env.reset(seed=0, options={"object_count": 4})
        assert info["object_count"] == 4
        assert _num_prefixed(eval_state, "cube") == 4
        assert _num_prefixed(eval_state, "box") == 1

        decoded = decode_object_centric_state(encode_object_centric_state(eval_state))
        env.set_state(decoded)
        assert env.current_count == 4
        assert env.get_state().allclose(eval_state)

    finally:
        env.close()


def test_constrainedcupboard3d_uses_registered_object_counts() -> None:
    """ConstrainedCupboard3D selects its existing one, two, and six-rod tasks."""
    env = VariableObjectCountEnv(
        constant_object_env_path=(
            "robocode.environments.constrained_cupboard3d:ConstrainedCupboard3DEnv"
        ),
        count_kwarg="num_objects",
        count_object_prefix="cuboid_",
        design_counts=[1, 2],
        eval_counts=[1, 2, 6],
        constant_object_env_kwargs={"scene_bg": False},
    )
    try:
        for count in env.eval_counts:
            state, info = env.reset(seed=0, options={"object_count": count})
            assert info["object_count"] == count
            assert _num_prefixed(state, "cuboid_") == count

        assert "`cuboid_0`, `cuboid_1`, ..." in env.env_description

        decoded = decode_object_centric_state(encode_object_centric_state(state))
        env.set_state(decoded)
        assert env.current_count == 6
        assert env.get_state().allclose(state)
    finally:
        env.close()


def test_constrainedcupboard3d_adapter_kwargs() -> None:
    """The adapter owns its task path while allowing a camera override."""
    with pytest.raises(ValueError, match="selects task_config_path"):
        ConstrainedCupboard3DEnv(
            num_objects=1,
            task_config_path="ignored.json",
        )

    env = ConstrainedCupboard3DEnv(
        num_objects=1,
        scene_bg=False,
        scene_render_camera="task_view",
    )
    try:
        env.reset(seed=0)
    finally:
        env.close()
        getattr(env, "_object_centric_env").close()
