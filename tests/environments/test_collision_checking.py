"""Tests for check_action_collision across environments."""

import numpy as np
from gymnasium.spaces import Box

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.environments.maze_env import MazeEnv, _MazeState

# Maze action constants (public in MazeEnv but prefixed with _).
_UP, _DOWN, _LEFT, _RIGHT = 0, 1, 2, 3


class TestMazeCollision:
    """Collision checking for MazeEnv."""

    def test_obstacle_collision(self):
        """Moving into an obstacle is a collision."""
        state = _MazeState(
            agent=(1, 1),
            obstacles=frozenset([(0, 1)]),
            height=3,
            width=3,
            goal=(2, 2),
        )
        env = MazeEnv(3, 3, 3, 3)
        assert env.check_action_collision(state, _UP)

    def test_boundary_collision(self):
        """Moving out of bounds is a collision."""
        state = _MazeState(
            agent=(0, 0),
            obstacles=frozenset(),
            height=3,
            width=3,
            goal=(2, 2),
        )
        env = MazeEnv(3, 3, 3, 3)
        assert env.check_action_collision(state, _UP)
        assert env.check_action_collision(state, _LEFT)

    def test_free_move(self):
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
            assert not env.check_action_collision(state, action)

    def test_state_preservation(self):
        """check_action_collision does not mutate env state."""
        env = MazeEnv(5, 8, 5, 8)
        state, _ = env.reset(seed=42)
        saved = env.get_state()
        env.check_action_collision(state, _UP)
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
        if env.check_action_collision(state, action):
            return state, action
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.reset()
    raise RuntimeError("Could not find a collision")


class TestKinderCollision:
    """Collision checking for KinderGeom2DEnv."""

    def test_collision(self):
        """Stepping into a wall should collide."""
        env = KinderGeom2DEnv("kinder/Motion2D-p1-v0")
        env.reset(seed=0)
        state, action = _find_kinder_collision(env)
        assert env.check_action_collision(state, action)
        env.close()

    def test_free_move(self):
        """A zero action should not collide."""
        env = KinderGeom2DEnv("kinder/Motion2D-p1-v0")
        env.reset(seed=0)
        state = env.get_state()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        assert not env.check_action_collision(state, action)
        env.close()

    def test_state_preservation(self):
        """check_action_collision restores env state."""
        env = KinderGeom2DEnv("kinder/Motion2D-p1-v0")
        env.reset(seed=0)
        saved = env.get_state()
        state, action = _find_kinder_collision(env)
        env.set_state(saved)
        env.check_action_collision(state, action)
        np.testing.assert_array_equal(env.get_state(), saved)
        env.close()
