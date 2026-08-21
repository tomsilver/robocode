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
    for i, held in enumerate(env._blocks):
        sp.set_point(held, sp.Point(x=0.06 * i - 0.03, y=0.0, z=1.1))
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
