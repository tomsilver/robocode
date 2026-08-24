"""Tests for pr2_tamp_env.py."""

import threading
from typing import Any

import numpy as np
import pytest

from robocode.environments import ss_pybullet as sp
from robocode.environments.pr2_tamp_env import PR2PackedEnv

_NOOP = np.zeros(11, dtype=np.float32)
_CLOSE = np.array([0] * 10 + [-1], dtype=np.float32)
_OPEN = np.array([0] * 10 + [1], dtype=np.float32)


@pytest.fixture(name="env")
def _env():
    env = PR2PackedEnv(num_blocks=2)
    yield env
    env.close()


def test_pr2_packed_basic(env: PR2PackedEnv) -> None:
    """Basic functionality: reset, step, get/set state."""
    env.action_space.seed(123)
    state, _ = env.reset(seed=123)
    assert env.observation_space.contains(state)

    action = env.action_space.sample()
    assert env.action_space.contains(action)
    next_state, reward, terminated, truncated, _ = env.step(action)
    assert env.observation_space.contains(next_state)
    assert reward == -1.0
    assert not terminated
    assert not truncated
    assert np.array_equal(env.get_state(), next_state)


def test_pr2_packed_reset_is_seeded(env: PR2PackedEnv) -> None:
    """The same seed gives the same instance, different seeds different ones."""
    first, _ = env.reset(seed=7)
    repeat, _ = env.reset(seed=7)
    other, _ = env.reset(seed=8)
    assert np.array_equal(first, repeat)
    assert not np.array_equal(first, other)


def test_pr2_packed_set_state_restores(env: PR2PackedEnv) -> None:
    """set_state undoes the effect of stepping."""
    env.action_space.seed(0)
    saved, _ = env.reset(seed=1)
    for _ in range(20):
        env.step(env.action_space.sample())
    assert not np.allclose(env.get_state(), saved)
    env.set_state(saved)
    assert np.allclose(env.get_state(), saved, atol=1e-5)


def test_pr2_packed_render(env: PR2PackedEnv) -> None:
    """Rendering returns an RGB frame."""
    env.reset(seed=0)
    # render() is typed against gymnasium's unbound RenderFrame TypeVar.
    frame: Any = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_pr2_packed_grasp_carry_place(env: PR2PackedEnv) -> None:
    """A block can be grasped, carried by the base, released, and reach the goal.

    The arm is not driven there by inverse kinematics; the block is teleported to the
    tool frame so that the grasp/carry/release machinery is exercised on its own.
    """
    # pylint: disable=protected-access
    obs, _ = env.reset(seed=5)
    sp.set_client(env._client)
    assert obs[11] == 0.0  # nothing held

    tool_point = sp.get_link_pose(env._robot, env._tool_link)[0]
    block = env._blocks[0]
    sp.set_point(block, sp.Point(*tool_point))
    obs, _, terminated, _, _ = env.step(_CLOSE)
    assert obs[11] == 1.0
    assert not terminated

    # The held block travels with the base.
    before = sp.get_pose(block)[0]
    drive = _NOOP.copy()
    drive[0] = 0.15
    env.step(drive)
    after = sp.get_pose(block)[0]
    assert np.isclose(after[0] - before[0], 0.15, atol=1e-3)

    # Releasing over the plate places it there, and the goal needs every block.
    # Cells are a block-width apart plus clearance, and squared up with the plate:
    # a block still carrying its random table yaw has a wider axis-aligned footprint,
    # and two of those at this pitch interpenetrate, which the drop now refuses.
    for i, held in enumerate(env._blocks):
        sp.set_pose(held, ((0.085 * i - 0.0425, 0.0, 1.1), (0.0, 0.0, 0.0, 1.0)))
        env._attachment = sp.create_attachment(env._robot, env._tool_link, held)
        obs, _, terminated, _, _ = env.step(_OPEN)
        assert obs[11] == 0.0
        assert sp.is_placement(held, env._plate)
    assert terminated


def test_pr2_packed_collision_blocks_motion(env: PR2PackedEnv) -> None:
    """Driving the base into the table is rejected rather than allowed to pass."""
    # pylint: disable=protected-access
    obs, _ = env.reset(seed=2)
    assert obs[0] == pytest.approx(-1.0)
    drive = _NOOP.copy()
    drive[0] = 0.2
    for _ in range(40):
        obs, _, _, _, _ = env.step(drive)
    # The base advances until the table stops it, well short of both the table at
    # the origin and the +5 base joint limit it would otherwise have reached.
    assert -1.0 < obs[0] < -0.5
    sp.set_client(env._client)
    assert not sp.pairwise_collision(env._robot, env._table)


def test_pr2_packed_concurrent_envs_do_not_interfere() -> None:
    """Envs stepped concurrently on threads match the same envs stepped serially.

    pybullet_tools keeps the live physics client in a module global, and env_server
    hands each agent connection its own env on its own ThreadingTCPServer thread, so a
    missing lock here would silently let one env drive another's simulation.
    """

    def drive(env: PR2PackedEnv, seed: int) -> np.ndarray:
        env.action_space.seed(seed)
        env.reset(seed=seed)
        obs = np.zeros(env.observation_space.shape or (0,), dtype=np.float32)
        for _ in range(30):
            obs, _, _, _, _ = env.step(env.action_space.sample())
        return obs

    seeds = [100, 101, 102]
    envs = [PR2PackedEnv(num_blocks=2) for _ in seeds]
    try:
        concurrent: dict[int, np.ndarray] = {}
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                concurrent[index] = drive(envs[index], seeds[index])
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(len(seeds))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(concurrent) == len(seeds)
        for index, env in enumerate(envs):
            assert np.allclose(concurrent[index], drive(env, seeds[index]), atol=1e-6)
    finally:
        for env in envs:
            env.close()


def test_pr2_packed_refuses_to_stack_blocks(env: PR2PackedEnv) -> None:
    """Releasing a block into one already on the plate is refused, not accepted.

    ``is_placement`` is a per-block test, so without this a policy could drop every
    block over one point and satisfy it N times over with N interpenetrating blocks,
    defeating the packing the benchmark is about.
    """
    # pylint: disable=protected-access
    env.reset(seed=0)
    sp.set_client(env._client)
    for index, block in enumerate(env._blocks):
        tool = sp.get_link_pose(env._robot, env._tool_link)
        sp.set_pose(block, sp.multiply(tool, ((0.05, 0.0, 0.0), (0, 0, 0, 1))))
        obs, _, _, _, _ = env.step(_CLOSE)
        assert obs[11] == 1.0, "the gripper should have closed around the block"
        # Aim every block at the same point on the plate.
        sp.set_point(block, sp.Point(x=0.0, y=0.0, z=1.1))
        env._attachment = sp.create_attachment(env._robot, env._tool_link, block)
        obs, _, terminated, _, _ = env.step(_OPEN)
        if index == 0:
            assert obs[11] == 0.0, "the first block has a clear plate to land on"
        else:
            assert obs[11] == 1.0, "a drop onto an occupied pose must be refused"
        assert not terminated


def test_pr2_packed_grasp_needs_the_gripper_around_the_block(
    env: PR2PackedEnv,
) -> None:
    """Hovering above a block is not a grasp, however close the tool frame is."""
    # pylint: disable=protected-access
    env.reset(seed=0)
    sp.set_client(env._client)
    block = env._blocks[0]
    tool = sp.get_link_pose(env._robot, env._tool_link)
    # Well inside the grasp radius, but the fingers cannot reach past the block top.
    hovering = 0.074
    sp.set_pose(block, sp.multiply(tool, ((hovering, 0.0, 0.0), (0, 0, 0, 1))))
    assert hovering < env._grasp_radius
    obs, _, _, _, _ = env.step(_CLOSE)
    assert obs[11] == 0.0, "a hover should not grasp"
    # Brought properly between the fingers, the same block is grasped.
    sp.set_pose(block, sp.multiply(tool, ((0.05, 0.0, 0.0), (0, 0, 0, 1))))
    obs, _, _, _, _ = env.step(_CLOSE)
    assert obs[11] == 1.0


def test_pr2_packed_does_not_disturb_the_global_numpy_rng() -> None:
    """Placement sampling draws from the global legacy RNG but must restore it.

    That RNG is shared with approaches and anything else in the process, so an
    environment reseeding it as a side effect of construction or reset would make
    unrelated code's sampling depend on how many episodes had been run.
    """

    def keys() -> list[int]:
        """The first few words of the global legacy RNG's key."""
        # legacy=True returns the MT19937 tuple; the annotation says dict.
        state: Any = np.random.get_state()
        return list(state[1][:8])

    env = PR2PackedEnv(num_blocks=2)
    try:
        np.random.seed(1234)
        expected = keys()
        env.reset(seed=7)
        env.reset(seed=8)
        assert keys() == expected

        np.random.seed(99)
        expected = keys()
        other = PR2PackedEnv(num_blocks=2)
        other.close()
        assert keys() == expected
    finally:
        env.close()


def test_pr2_packed_instances_stay_reproducible(env: PR2PackedEnv) -> None:
    """Restoring the global RNG must not cost per-seed reproducibility."""
    first, _ = env.reset(seed=5)
    again, _ = env.reset(seed=5)
    other, _ = env.reset(seed=6)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_pr2_packed_client_context_is_reentrant(env: PR2PackedEnv) -> None:
    """The public client guard is the lock the env's own methods take."""
    env.reset(seed=0)
    with env.client() as client:
        with env.client() as inner:
            assert inner == client
        # Env methods are callable from inside it, which a non-reentrant lock
        # would deadlock on.
        env.step(_NOOP)
        assert sp.get_pose(env.blocks[0]) is not None
