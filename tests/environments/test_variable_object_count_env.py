"""Tests for variable_object_count_env.py."""

from typing import Any

import numpy as np
import pytest

from robocode.environments.variable_object_count_env import VariableObjectCountEnv

OBSTRUCTION2D: dict[str, Any] = {
    "constant_object_env_path": "kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",
    "count_kwarg": "num_obstructions",
    "count_object_prefix": "obstruction",
    "design_counts": [0, 1, 2],
    "eval_counts": [0, 1, 2, 3, 4],
    "bilevel_env_name": "obstruction2d",
}


def _num_prefixed(state, prefix: str) -> int:
    return sum(1 for name in state.get_object_names() if name.startswith(prefix))


def test_unpinned_reset_stays_in_design_range() -> None:
    """An unpinned reset never samples a held-out count (OOD hygiene)."""
    env = VariableObjectCountEnv(**OBSTRUCTION2D)
    seen = {env.reset(seed=s)[1]["object_count"] for s in range(50)}
    assert seen <= set(env.design_counts)
    # It should exercise more than one design count over many seeds.
    assert len(seen) > 1
    env.close()


def test_determinism_for_fixed_seed_and_count() -> None:
    """A fixed (seed, count) reproduces the same instance for both approaches."""
    env = VariableObjectCountEnv(**OBSTRUCTION2D)
    a, _ = env.reset(seed=7, options={"object_count": 3})
    b, _ = env.reset(seed=7, options={"object_count": 3})
    assert a.allclose(b)
    env.close()


def test_get_set_state_roundtrip_and_count_inference() -> None:
    """set_state restores the state and infers its count by object-name prefix."""
    env = VariableObjectCountEnv(**OBSTRUCTION2D)
    env.reset(seed=1, options={"object_count": 2})
    saved = env.get_state()
    env.step(env.action_space.sample())
    env.set_state(saved)
    assert env.current_count == 2
    assert saved.allclose(env.get_state())
    # Stepping from a restored state is reproducible.
    action = env.action_space.sample()
    env.set_state(saved)
    s1, _, _, _, _ = env.step(action)
    env.set_state(saved)
    s2, _, _, _, _ = env.step(action)
    assert s1.allclose(s2)
    env.close()


def test_spaces_are_count_invariant_but_box_view_is_not() -> None:
    """The object-centric obs/action spaces are shared; the Box view grows with k."""
    env = VariableObjectCountEnv(**OBSTRUCTION2D)
    obs_space, act_space = env.observation_space, env.action_space
    box_dims = {}
    for k in [0, 2, 4]:
        env.reset(seed=0, options={"object_count": k})
        assert env.observation_space is obs_space
        assert env.action_space is act_space
        box_dims[k] = env.current_box_obs().shape[0]
    # Each object adds features, so the per-count Box view is strictly larger.
    assert box_dims[0] < box_dims[2] < box_dims[4]
    env.close()


def test_sample_next_state_with_object_centric_state() -> None:
    """sample_next_state (used by primitives) works with object-centric states."""
    env = VariableObjectCountEnv(**OBSTRUCTION2D)
    state, _ = env.reset(seed=5, options={"object_count": 2})
    nxt = env.sample_next_state(
        state, env.action_space.sample(), np.random.default_rng(0)
    )
    assert env.observation_space.contains(nxt)
    env.close()


def test_infeasible_count_raises_clearly() -> None:
    """A configured count the scene cannot fit fails with an informative error."""
    with pytest.raises(ValueError, match="feasible object count|could not build"):
        VariableObjectCountEnv(
            **{**OBSTRUCTION2D, "design_counts": [0], "eval_counts": [0, 8]}
        )


def test_description_is_object_centric() -> None:
    """The env card describes an object-centric, variable-count observation."""
    env = VariableObjectCountEnv(**OBSTRUCTION2D)
    full = env.env_description
    blackbox = env.env_description_blackbox
    env.close()

    assert "## Generalization" in full
    assert "## Observation" in full
    assert "ObjectCentricState" in full
    assert "VARIABLE number of objects" in full
    # No fixed-vector language for this setting.
    assert "obs.shape" not in full
    assert "devectorize" not in full
    # Blackbox omits the direct-import example.
    assert "## Example Usage" in full
    assert "## Example Usage" not in blackbox
    assert full.startswith(blackbox)


# Every 2D family. ``one_to_one`` marks families whose count-defining objects are named
# ``<prefix>0``..``<prefix>(k-1)`` exactly; Motion2D makes ~2 obstacle objects per
# passage, so its prefixed-object count is not the count parameter (inference recovers
# it from the prefix->count map regardless).
_FAMILY_CASES = [
    pytest.param(OBSTRUCTION2D, True, id="obstruction2d"),
    pytest.param(
        {
            "constant_object_env_path": (
                "kinder.envs.kinematic2d.clutteredretrieval2d:ClutteredRetrieval2DEnv"
            ),
            "count_kwarg": "num_obstructions",
            "count_object_prefix": "obstruction",
            "design_counts": [1, 3],
            "eval_counts": [1, 3, 5],
            "bilevel_env_name": "clutteredretrieval2d",
        },
        True,
        id="clutteredretrieval2d",
    ),
    pytest.param(
        {
            "constant_object_env_path": (
                "kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv"
            ),
            "count_kwarg": "num_blocks",
            "count_object_prefix": "block",
            "design_counts": [1, 3],
            "eval_counts": [1, 3, 7],  # odd only (family requires it)
            "bilevel_env_name": "clutteredstorage2d",
        },
        True,
        id="clutteredstorage2d",
    ),
    pytest.param(
        {
            "constant_object_env_path": (
                "kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv"
            ),
            "count_kwarg": "num_buttons",
            "count_object_prefix": "button",
            "design_counts": [1, 2],
            "eval_counts": [1, 2, 3],
            "bilevel_env_name": "stickbutton2d",
        },
        True,
        id="stickbutton2d",
    ),
    pytest.param(
        {
            "constant_object_env_path": "kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            "count_kwarg": "num_passages",
            "count_object_prefix": "obstacle",
            "design_counts": [1, 2],
            "eval_counts": [1, 2, 3],
            "bilevel_env_name": "motion2d",
        },
        False,
        id="motion2d",
    ),
]


@pytest.mark.parametrize("cfg, one_to_one", _FAMILY_CASES)
def test_family_pinned_count_and_prefix_inference(
    cfg: dict[str, Any], one_to_one: bool
) -> None:
    """For every 2D family: a pinned count is reported and set_state infers it back from
    the object-name prefix. One-to-one families expose exactly that many prefixed
    objects; Motion2D makes more (inference uses the prefix->count map either way)."""
    env = VariableObjectCountEnv(**cfg)
    prefix = cfg["count_object_prefix"]
    for k in cfg["eval_counts"]:  # design AND held-out counts
        state, info = env.reset(seed=k, options={"object_count": k})
        assert env.observation_space.contains(state)
        assert info["object_count"] == k
        assert env.current_count == k
        n_prefixed = _num_prefixed(state, prefix)
        assert n_prefixed == k if one_to_one else n_prefixed > k
    # Inference from a bare state recovers its count, not the current instance's.
    biggest = cfg["eval_counts"][-1]
    state, _ = env.reset(seed=0, options={"object_count": biggest})
    env.reset(seed=1, options={"object_count": cfg["design_counts"][0]})
    env.set_state(state)
    assert env.current_count == biggest
    env.close()


def test_clutteredstorage_requires_odd_count() -> None:
    """ClutteredStorage2D splits its blocks across a shelf, so an even count is rejected
    at construction (the backend is built eagerly per configured count)."""
    with pytest.raises(AssertionError, match="must be odd"):
        VariableObjectCountEnv(
            constant_object_env_path=(
                "kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv"
            ),
            count_kwarg="num_blocks",
            count_object_prefix="block",
            design_counts=[2],
            eval_counts=[2],
            bilevel_env_name="clutteredstorage2d",
        )
