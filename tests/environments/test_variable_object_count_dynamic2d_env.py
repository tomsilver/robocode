"""Dynamic2D coverage for the variable-object-count environment."""

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from relational_structs import ObjectCentricState

from robocode.environments.variable_object_count_env import VariableObjectCountEnv

_SCOOPPOUR_CONFIG: dict[str, Any] = {
    "constant_object_env_path": (
        "robocode.environments.generalized_dynamic2d:"
        "CountParameterizedDynScoopPour2DEnv"
    ),
    "count_kwarg": "num_small_objects",
    "count_object_prefix": "small_",
    "design_counts": [10, 20, 30],
    "eval_counts": [10, 20, 30, 50],
}

_FAMILY_CASES = [
    pytest.param(
        {
            "constant_object_env_path": (
                "kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv"
            ),
            "count_kwarg": "num_obstructions",
            "count_object_prefix": "obstruction",
            "design_counts": [0, 1, 2],
            "eval_counts": [0, 1, 2, 3],
        },
        3,
        id="dynobstruction2d",
    ),
    pytest.param(
        {
            "constant_object_env_path": (
                "kinder.envs.dynamic2d.dyn_pushpullhook2d:" "DynPushPullHook2DEnv"
            ),
            "count_kwarg": "num_obstructions",
            "count_object_prefix": "obstruction",
            "design_counts": [0, 1, 2, 3],
            "eval_counts": [0, 1, 2, 3, 5],
        },
        5,
        id="dynpushpullhook2d",
    ),
    pytest.param(
        _SCOOPPOUR_CONFIG,
        50,
        id="dynscooppour2d",
    ),
]


def _num_prefixed(state: ObjectCentricState, prefix: str) -> int:
    return sum(name.startswith(prefix) for name in state.get_object_names())


@pytest.mark.parametrize("config,eval_count", _FAMILY_CASES)
def test_dynamic2d_family_supports_variable_counts(
    config: dict[str, Any], eval_count: int
) -> None:
    """Each dynamic family supports pinned reset, state routing, step, and render."""
    env = VariableObjectCountEnv(**config)
    try:
        state, info = env.reset(seed=123, options={"object_count": eval_count})
        assert info["object_count"] == eval_count
        assert env.current_count == eval_count
        assert _num_prefixed(state, config["count_object_prefix"]) == eval_count
        assert env.observation_space.contains(state)

        env.set_state(state)
        assert env.current_count == eval_count
        assert env.get_state().get_object_names() == state.get_object_names()

        assert env.action_space.shape is not None
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_state, _, _, _, step_info = env.step(action)
        assert env.observation_space.contains(next_state)
        assert step_info["object_count"] == eval_count

        frame = cast(NDArray[Any], env.render())
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
    finally:
        env.close()


def test_scooppour_balances_circle_and_square_counts() -> None:
    """The single total count preserves ScoopPour's mixed-shape distribution."""
    count = 50
    env = VariableObjectCountEnv(**_SCOOPPOUR_CONFIG)
    try:
        state, _ = env.reset(seed=0, options={"object_count": count})
        names = state.get_object_names()
        assert sum(name.startswith("small_circle") for name in names) == count // 2
        assert sum(name.startswith("small_square") for name in names) == count // 2
    finally:
        env.close()
