"""Tests for check_action_collision across environments."""

import numpy as np
from gymnasium.spaces import Box
from kinder.envs.kinematic2d.object_types import CRVRobotType
from kinder.envs.kinematic2d.utils import (
    get_suctioned_objects,
    snap_suctioned_objects,
)
from prpl_utils.utils import wrap_angle

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.environments.maze_env import MazeEnv, _MazeState
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.oracles.pushpullhook2d.behaviors import (
    GraspRotate,
    PrePushPull,
    Push,
    Sweep,
)
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


def _drive_behavior(env, obs, behavior, max_steps, name):
    """Advance a pushpull oracle behavior to termination to reach a later phase."""
    assert behavior.initializable(obs), f"{name} precondition not met."
    behavior.reset(obs)
    for _ in range(max_steps):
        obs, _, _, _, _ = env.step(np.asarray(behavior.step(obs), dtype=np.float32))
        if behavior.terminated(obs):
            return obs
    raise AssertionError(f"{name} did not finish in {max_steps} steps.")


def _grasp_and_contact(env, ocs, action):
    """Whether *action* has an object grasped, and whether it pushes another object.

    The second flag detects PushPullHook2D's contact propagation firing, i.e. the
    grasped hook moving the movable button through get_objects_to_move.
    """
    # pylint: disable=protected-access
    inner = env._kinder_env._object_centric_env
    robot = next(o for o in ocs if o.is_instance(CRVRobotType))
    suctioned = get_suctioned_objects(ocs, robot)
    dx, dy, dtheta, darm, vac = np.asarray(action, dtype=np.float32)
    state = ocs.copy()
    state.set(robot, "x", state.get(robot, "x") + float(dx))
    state.set(robot, "y", state.get(robot, "y") + float(dy))
    state.set(robot, "theta", wrap_angle(state.get(robot, "theta") + float(dtheta)))
    lo, hi = state.get(robot, "base_radius"), state.get(robot, "arm_length")
    state.set(
        robot,
        "arm_joint",
        float(np.clip(state.get(robot, "arm_joint") + float(darm), lo, hi)),
    )
    state.set(robot, "vacuum", float(vac))
    snap_suctioned_objects(state, robot, suctioned)
    _, moved = inner.get_objects_to_move(state, suctioned)
    return len(suctioned) > 0, len(moved) > 0


def test_equivalence_pushpull_grasp_and_push():
    """Direct check matches a full step while the grasped hook pushes the button.

    Drives the PushPullHook2D oracle phases to reach Push, where the grasped hook
    contacts and moves the movable button. This is the only path exercising
    PushPullHook2D's overridden get_objects_to_move (contact propagation), so it guards
    that branch. Asserts the hook is grasped and contact propagation fires.
    """
    # pylint: disable=protected-access
    env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
    try:
        obs, _ = env.reset(seed=0)
        obs = _drive_behavior(env, obs, GraspRotate(), 500, "GraspRotate")
        obs = _drive_behavior(env, obs, Sweep(), 1000, "Sweep")
        obs = _drive_behavior(env, obs, PrePushPull(), 2000, "PrePushPull")

        push = Push()
        assert push.initializable(obs), "Push precondition not met."
        push.reset(obs)
        box = env._kinder_env.observation_space
        grasped = contact = 0
        for _ in range(2000):
            state = env.get_state()
            ocs = box.devectorize(np.asarray(state, dtype=np.float32))
            action = push.step(obs)
            for cand in (action, np.zeros_like(action), env.action_space.high):
                fast = check_action_collision(env, state, cand)
                slow = _step_identity_collision(env, "_kinder_env", state, cand)
                assert fast == slow, f"MISMATCH fast={fast} slow={slow}"
                g, m = _grasp_and_contact(env, ocs, cand)
                grasped += int(g)
                contact += int(m)
            obs, _, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
            if push.terminated(obs):
                break
        assert push.terminated(obs), "Push did not complete."
        assert grasped > 0, "Hook was never grasped during Push."
        assert contact > 0, "Contact propagation never fired."
    finally:
        env.close()
