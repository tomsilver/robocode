"""Tests for the object-centric StickButton2D oracle.

Each test passes an explicit config, so the stick spawn range under test does not
depend on the values the checked-out kinder source happens to carry.
"""

from dataclasses import replace

import numpy as np
import pytest
from kinder.envs.kinematic2d.stickbutton2d import (
    ObjectCentricStickButton2DEnv,
    StickButton2DEnvConfig,
)
from kinder.envs.kinematic2d.structs import SE2Pose

from robocode.oracles.stickbutton2d_generalized.approach import (
    StickButton2DGeneralizedOracleApproach,
)
from robocode.oracles.stickbutton2d_generalized.obs_helpers import (
    all_buttons_pressed,
    button_names,
    extract_robot,
    has_space_stick_bottom,
)

BASE = StickButton2DEnvConfig()
_TABLE_Y = BASE.table_pose.y
_STICK_H = BASE.stick_shape[1]

# Full-width stick spawn, and a spawn range pinned against the right wall where
# no robot placement can reach under the stick to grasp it from below.
FULL_WIDTH_BOUNDS = (
    SE2Pose(BASE.world_min_x, _TABLE_Y - _STICK_H / 2, 0),
    SE2Pose(BASE.world_max_x - BASE.stick_shape[0], _TABLE_Y - _STICK_H / 10, 0),
)
AT_WALL_BOUNDS = (
    SE2Pose(3.435, _TABLE_Y - _STICK_H / 2, 0),
    SE2Pose(BASE.world_max_x - BASE.stick_shape[0], _TABLE_Y - _STICK_H / 10, 0),
)

MAX_STEPS = 1500


def _solve(bounds, num_buttons, seed):
    env = ObjectCentricStickButton2DEnv(
        num_buttons=num_buttons,
        config=replace(BASE, stick_init_pose_bounds=bounds),
    )
    state, info = env.reset(seed=seed)
    approach = StickButton2DGeneralizedOracleApproach(
        env.action_space, env.observation_space, seed=0
    )
    approach.reset(state, info)
    for _ in range(MAX_STEPS):
        state, reward, terminated, truncated, info = env.step(approach.step())
        approach.update(state, float(reward), terminated or truncated, info)
        if terminated:
            return True, state
    return False, state


@pytest.mark.parametrize("num_buttons", [1, 3, 5])
def test_solves_full_width_spawn(num_buttons):
    """The oracle presses every button when the stick spawns anywhere."""
    solved, state = _solve(FULL_WIDTH_BOUNDS, num_buttons, seed=0)
    assert solved
    assert all_buttons_pressed(state)
    assert len(button_names(state)) == num_buttons


@pytest.mark.parametrize("num_buttons", [1, 3])
def test_solves_stick_at_wall(num_buttons):
    """A stick pinned at the wall is solved by repositioning it first."""
    env = ObjectCentricStickButton2DEnv(
        num_buttons=num_buttons,
        config=replace(BASE, stick_init_pose_bounds=AT_WALL_BOUNDS),
    )
    state, _ = env.reset(seed=0)
    assert not has_space_stick_bottom(state)

    solved, state = _solve(AT_WALL_BOUNDS, num_buttons, seed=0)
    assert solved
    assert all_buttons_pressed(state)


def test_button_names_ordered_numerically():
    """Buttons are ordered by index, not lexicographically."""
    env = ObjectCentricStickButton2DEnv(num_buttons=10, config=BASE)
    state, _ = env.reset(seed=0)
    assert button_names(state) == [f"button{i}" for i in range(10)]


def test_extract_robot_matches_state():
    """Robot features are read by name, not by a fixed index layout."""
    env = ObjectCentricStickButton2DEnv(num_buttons=2, config=BASE)
    state, _ = env.reset(seed=1)
    robot = extract_robot(state)
    obj = state.get_object_from_name("robot")
    assert np.isclose(robot.x, state.get(obj, "x"))
    assert np.isclose(robot.base_radius, state.get(obj, "base_radius"))
