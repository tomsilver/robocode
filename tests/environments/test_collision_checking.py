"""Tests for check_action_collision across environments."""

import numpy as np
from gymnasium.spaces import Box

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.environments.maze_env import MazeEnv, _MazeState
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.primitives.check_action_collision import check_action_collision

# Maze action constants (public in MazeEnv but prefixed with _).
_UP, _DOWN, _LEFT, _RIGHT = 0, 1, 2, 3


def test_maze_obstacle_collision():
    """Moving into an obstacle is a collision."""
    state = _MazeState(
        agent=(1, 1),
        obstacles=frozenset([(0, 1)]),
        height=3,
        width=3,
        goal=(2, 2),
    )
    env = MazeEnv(3, 3, 3, 3)
    assert check_action_collision(env, state, _UP)


def test_maze_boundary_collision():
    """Moving out of bounds is a collision."""
    state = _MazeState(
        agent=(0, 0),
        obstacles=frozenset(),
        height=3,
        width=3,
        goal=(2, 2),
    )
    env = MazeEnv(3, 3, 3, 3)
    assert check_action_collision(env, state, _UP)
    assert check_action_collision(env, state, _LEFT)


def test_maze_free_move():
    """Moving into an empty cell is not a collision."""
    state = _MazeState(
        agent=(1, 1),
        obstacles=frozenset(),
        height=3,
        width=3,
        goal=(2, 2),
    )
    env = MazeEnv(3, 3, 3, 3)
    for action in [_UP, _DOWN, _LEFT, _RIGHT]:
        assert not check_action_collision(env, state, action)


def test_maze_state_preservation():
    """check_action_collision does not mutate env state."""
    env = MazeEnv(5, 8, 5, 8)
    state, _ = env.reset(seed=42)
    saved = env.get_state()
    check_action_collision(env, state, _UP)
    assert env.get_state() == saved


def _find_kinder_collision(
    env: KinderGeom2DEnv,
) -> tuple[np.ndarray, np.ndarray]:
    """Step the agent repeatedly until a collision occurs.

    Returns (state, action) where the action collides.
    """
    assert isinstance(env.action_space, Box)
    action = env.action_space.high.copy()
    for _ in range(500):
        state = env.get_state()
        if check_action_collision(env, state, action):
            return state, action
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.reset()
    raise RuntimeError("Could not find a collision")


def test_kinder_collision():
    """Stepping into a wall should collide."""
    env = KinderGeom2DEnv("kinder/Motion2D-p1-v0")
    env.reset(seed=0)
    state, action = _find_kinder_collision(env)
    assert check_action_collision(env, state, action)
    env.close()


def test_kinder_free_move():
    """A zero action should not collide."""
    env = KinderGeom2DEnv("kinder/Motion2D-p1-v0")
    env.reset(seed=0)
    state = env.get_state()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    assert not check_action_collision(env, state, action)
    env.close()


def test_kinder_state_preservation():
    """check_action_collision restores env state."""
    env = KinderGeom2DEnv("kinder/Motion2D-p1-v0")
    env.reset(seed=0)
    saved = env.get_state()
    state, action = _find_kinder_collision(env)
    env.set_state(saved)
    check_action_collision(env, state, action)
    np.testing.assert_array_equal(env.get_state(), saved)
    env.close()


def _make_variable_count_env() -> VariableObjectCountEnv:
    """A one-passage Motion2D wrapped as a variable-count env (walls to hit)."""
    return VariableObjectCountEnv(
        constant_object_env_path="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
        count_kwarg="num_passages",
        count_object_prefix="obstacle",
        design_counts=[1],
        eval_counts=[1],
        bilevel_env_name="motion2d",
    )


def _find_variable_count_collision(
    env: VariableObjectCountEnv,
) -> tuple[object, np.ndarray]:
    """Drive with the max action until a collision occurs; return (state, action)."""
    assert isinstance(env.action_space, Box)
    action = env.action_space.high.copy()
    for _ in range(500):
        state = env.get_state()
        if check_action_collision(env, state, action):
            return state, action
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.reset(seed=0, options={"object_count": 1})
    raise RuntimeError("Could not find a collision")


def test_variable_count_collision():
    """Stepping into a wall collides through the variable-count dispatch too."""
    env = _make_variable_count_env()
    try:
        env.reset(seed=0, options={"object_count": 1})
        state, action = _find_variable_count_collision(env)
        assert check_action_collision(env, state, action)
    finally:
        env.close()


def test_variable_count_free_move():
    """A zero action does not collide."""
    env = _make_variable_count_env()
    try:
        state, _ = env.reset(seed=0, options={"object_count": 1})
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        assert not check_action_collision(env, state, action)
    finally:
        env.close()


def test_variable_count_state_preservation():
    """check_action_collision restores the object-centric env state."""
    env = _make_variable_count_env()
    try:
        saved, _ = env.reset(seed=0, options={"object_count": 1})
        state, action = _find_variable_count_collision(env)
        env.set_state(saved)
        check_action_collision(env, state, action)
        assert env.get_state().allclose(saved)
    finally:
        env.close()


def _step_identity_collision(env, backend_attr, state, action):
    """Ground-truth collision: a full kinder step leaves _current_state unreplaced.

    This is the exact semantics the primitive must reproduce: kinder's step only
    reassigns the current-state object when the action is collision-free, so identity
    equality before vs after the step is a collision. Restores env state afterward.
    """
    # pylint: disable=protected-access
    saved = env.get_state()
    env.set_state(state)
    inner = getattr(env, backend_attr)._object_centric_env
    before = inner._current_state
    env.step(np.asarray(action, dtype=np.float32))
    after = inner._current_state
    env.set_state(saved)
    return after is before


def _candidate_actions(action_space, rng):
    """Max/min/zero and random actions covering collisions, free moves, and grasps."""
    assert isinstance(action_space, Box)
    high = action_space.high.astype(np.float32)
    low = action_space.low.astype(np.float32)
    randoms = [rng.uniform(low, high).astype(np.float32) for _ in range(4)]
    return [high.copy(), low.copy(), np.zeros_like(high), *randoms]


def _assert_equivalent_over_episode(env, backend_attr, seed, num_steps=80):
    """Drive an episode and assert the direct check matches the step-identity oracle."""
    rng = np.random.default_rng(seed)
    for _ in range(num_steps):
        state = env.get_state()
        for action in _candidate_actions(env.action_space, rng):
            assert check_action_collision(
                env, state, action
            ) == _step_identity_collision(env, backend_attr, state, action)
        _, _, terminated, truncated, _ = env.step(
            np.asarray(env.action_space.sample(), dtype=np.float32)
        )
        if terminated or truncated:
            env.reset(seed=seed)


def test_equivalence_kinder_motion2d():
    """Direct check matches a full step over a Motion2D episode (numpy path, walls)."""
    env = KinderGeom2DEnv("kinder/Motion2D-p1-v0")
    env.reset(seed=0)
    try:
        _assert_equivalent_over_episode(env, "_kinder_env", seed=0)
    finally:
        env.close()


def test_equivalence_kinder_grasping():
    """Direct check matches a full step on StickButton2D (suction and held objects)."""
    env = KinderGeom2DEnv("kinder/StickButton2D-b3-v0")
    env.reset(seed=0)
    try:
        _assert_equivalent_over_episode(env, "_kinder_env", seed=1)
    finally:
        env.close()


def test_equivalence_variable_count():
    """Direct check matches a full step through the variable-count dispatch (ocs
    path)."""
    env = _make_variable_count_env()
    env.reset(seed=0, options={"object_count": 1})
    try:
        _assert_equivalent_over_episode(env, "_current_backend", seed=2)
    finally:
        env.close()
