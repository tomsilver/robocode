"""Tests for pr2_tamp_blocked_env.py.

The cases that matter here are the ones that distinguish ``blocked`` from ``packed``:
the goal is existential and the red block does not satisfy it, the grasp is a side
grasp, the pen really does obstruct the target, and the fixed layout is restored on
reset rather than resampled.
"""

from typing import Any

import numpy as np
import pytest

from robocode.environments import ss_pybullet as sp
from robocode.environments.pr2_tamp_blocked_env import PR2BlockedEnv
from robocode.environments.pr2_tamp_scenes import (
    BLOCKED_BLOCK_HEIGHT,
    BLOCKED_BLOCK_WIDTH,
)

_NOOP = np.zeros(11, dtype=np.float32)


@pytest.fixture(name="env")
def _env():
    env = PR2BlockedEnv(num_spares=1)
    yield env
    env.close()


def _place_on_plate(env: PR2BlockedEnv, body: int) -> None:
    """Drop *body* flush onto the middle of the plate."""
    with env.client():
        point = sp.get_pose(env.plate)[0]
        sp.set_point(
            body,
            sp.Point(x=point[0], y=point[1], z=sp.stable_z(body, env.plate) + 1e-4),
        )


def test_blocked_basic(env: PR2BlockedEnv) -> None:
    """Reset, step and the observation layout agree with the declared space."""
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    nxt, reward, terminated, truncated, _ = env.step(_NOOP)
    assert nxt.shape == obs.shape
    assert reward == -1.0
    assert not truncated
    # A fresh instance is not already solved.
    assert not terminated


def test_scene_has_a_penned_block_a_blocker_and_three_walls(
    env: PR2BlockedEnv,
) -> None:
    """The handles describe the benchmark's layout."""
    env.reset(seed=0)
    assert env.penned == env.greens[0]
    assert env.spares == env.greens[1:]
    assert len(env.walls) == 3
    assert env.blocker not in env.greens
    # The blocker is movable -- relocating it is the task -- but not a goal object.
    assert env.blocker in env.movables


def test_any_green_block_on_the_plate_ends_the_episode(env: PR2BlockedEnv) -> None:
    """The goal is existential: one green block is enough, whichever it is."""
    env.reset(seed=0)
    _place_on_plate(env, env.spares[0])
    _, _, terminated, _, _ = env.step(_NOOP)
    assert terminated


def test_the_red_blocker_on_the_plate_does_not_end_the_episode(
    env: PR2BlockedEnv,
) -> None:
    """Parking the blocker on the goal surface proves nothing."""
    env.reset(seed=0)
    _place_on_plate(env, env.blocker)
    with env.client():
        assert sp.is_placement(env.blocker, env.plate), "blocker is not on the plate"
    _, _, terminated, _, _ = env.step(_NOOP)
    assert not terminated


def test_the_blocker_obstructs_the_penned_block(env: PR2BlockedEnv) -> None:
    """The pen plus the blocker leave no side grasp of the target reachable.

    This is the premise of the whole task, so it is asserted rather than assumed: with
    the blocker where the scene puts it no side grasp is collision-free, and with it
    moved away some are. Without this the environment would be a plain pick-and-place
    wearing ``blocked``'s name.
    """
    env.reset(seed=0)
    obstacles = [env.near_table, env.far_table, *env.walls]

    def reachable(with_blocker: bool) -> int:
        rng = np.random.default_rng(0)
        found = 0
        bodies = obstacles + ([env.blocker] if with_blocker else [])
        block_pose = sp.get_pose(env.penned)
        grasps = list(sp.get_side_grasps(env.penned, grasp_length=0.0))
        for _ in range(120):
            radius, theta = rng.uniform(0.45, 0.85), rng.uniform(-np.pi, np.pi)
            bx = block_pose[0][0] + radius * np.cos(theta)
            by = block_pose[0][1] + radius * np.sin(theta)
            facing = np.arctan2(block_pose[0][1] - by, block_pose[0][0] - bx)
            sp.set_joint_positions(env.robot, env.base_joints, [bx, by, facing])
            sp.set_joint_positions(env.robot, env.arm_joints, env.initial_arm_conf)
            if any(sp.pairwise_collision(env.robot, b) for b in bodies):
                continue
            for grasp in grasps:
                target = sp.multiply(block_pose, sp.invert(grasp))
                if (
                    sp.sub_inverse_kinematics(
                        env.robot, env.arm_joints[0], env.tool_link, target
                    )
                    is None
                ):
                    continue
                if not any(sp.pairwise_collision(env.robot, b) for b in bodies):
                    found += 1
        return found

    with env.client(), sp.WorldSaver():
        blocked = reachable(with_blocker=True)
        sp.set_point(env.blocker, sp.Point(x=0.0, y=5.0, z=0.8))
        unblocked = reachable(with_blocker=False)
    assert blocked == 0, "a side grasp reached the target through the blocker"
    assert unblocked > 0, "the target is unreachable even with the blocker gone"


def test_reset_restores_the_fixed_layout(env: PR2BlockedEnv) -> None:
    """A solved episode must not leave the next one already solved.

    Nothing about the pen is resampled, so unlike ``packed`` the fixed bodies have to
    be put back explicitly; without that a green block left on the plate carries over.
    """
    env.reset(seed=0)
    with env.client():
        before = np.array(sp.get_pose(env.penned)[0])
    _place_on_plate(env, env.penned)
    assert env.step(_NOOP)[2], "the episode should be solved before the reset"
    env.reset(seed=1)
    with env.client():
        after = np.array(sp.get_pose(env.penned)[0])
    assert np.allclose(before, after)
    assert not env.step(_NOOP)[2]


def test_spares_are_resampled_and_reproducible() -> None:
    """Spares move between episodes, and the same seed reproduces them."""
    env = PR2BlockedEnv(num_spares=2)
    try:
        first, _ = env.reset(seed=7)
        other, _ = env.reset(seed=8)
        again, _ = env.reset(seed=7)
        assert not np.allclose(first, other), "spares did not vary across seeds"
        assert np.allclose(first, again), "the same seed gave a different instance"
    finally:
        env.close()


def test_zero_spares_is_a_valid_instance() -> None:
    """The count may be zero: the penned block is then the only green one."""
    env = PR2BlockedEnv(num_spares=0)
    try:
        obs, _ = env.reset(seed=0)
        assert not env.spares
        assert env.greens == [env.penned]
        assert obs.shape == env.observation_space.shape
    finally:
        env.close()


def test_set_state_restores(env: PR2BlockedEnv) -> None:
    """A saved state round-trips exactly, blocker and spares included."""
    env.reset(seed=0)
    saved = env.get_state()
    for _ in range(3):
        env.step(env.action_space.sample())
    env.set_state(saved)
    assert np.allclose(env.get_state(), saved)


def test_render_returns_a_frame(env: PR2BlockedEnv) -> None:
    """Rendering produces an RGB frame of the documented size."""
    env.reset(seed=0)
    frame: Any = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_grasp_needs_the_fingers_around_the_block_from_the_side(
    env: PR2BlockedEnv,
) -> None:
    """The grasp window is half a block *width* along the approach, not half a height.

    ``blocked``'s blocks are twice as tall as they are wide, so a top-grasp window
    would admit a gripper a full half-height away -- which for a side grasp is nowhere
    near the block.
    """
    env.reset(seed=0)
    assert env._max_grasp_approach == pytest.approx(  # pylint: disable=protected-access
        BLOCKED_BLOCK_WIDTH / 2 + 0.01
    )
    lateral = env._grasp_lateral_limit  # pylint: disable=protected-access
    assert lateral == (BLOCKED_BLOCK_WIDTH / 2, BLOCKED_BLOCK_HEIGHT / 2)


def test_card_describes_the_side_grasp_and_the_existential_goal(
    env: PR2BlockedEnv,
) -> None:
    """The agent-facing card states what actually decides the episode."""
    card = env.describe(include_access=True)
    lowered = card.lower()
    assert "side grasp" in lowered
    assert "any" in lowered and "green" in lowered
    # The red block's status is the easiest thing to get wrong from the card alone.
    assert "does not finish the episode" in lowered or "not a green block" in lowered
    blackbox: Any = env.env_description_blackbox
    assert "Source Code" not in blackbox
