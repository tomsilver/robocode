"""Tests for rovers_env.py.

The cases that matter are the ones that make this the ``rovers`` benchmark rather than
a driving task: the operators enforce their preconditions, the arena really is split in
two so neither rover can finish alone, sight crosses the divider even though driving
cannot, and the goal needs every clause.
"""

from typing import Any

import numpy as np
import pytest

from robocode.environments import ss_pybullet as sp
from robocode.environments.rovers_env import (
    CALIBRATE,
    DROP,
    IMAGE,
    NOOP,
    SAMPLE,
    SEND,
    RoversEnv,
    operator_action,
)


@pytest.fixture(name="env")
def _env():
    env = RoversEnv(num_objectives=2)
    yield env
    env.close()


def _action(*ops: int) -> np.ndarray:
    """An action applying one operator per rover and no motion."""
    a = np.zeros(8, dtype=np.float32)
    for i, op in enumerate(ops):
        a[4 * i + 3] = operator_action(op)
    return a


def _park(
    env: RoversEnv, index: int, xy: Any, heading: float = 0.0
) -> None:  # noqa: E501
    with env.client():
        conf: Any = [float(xy[0]), float(xy[1]), float(heading)]
        sp.set_joint_positions(env.rovers[index], env.base_joint_groups[index], conf)


def test_rovers_basic(env: RoversEnv) -> None:
    """Reset, step and the observation layout agree with the declared spaces."""
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    nxt, reward, terminated, truncated, _ = env.step(np.zeros(8, dtype=np.float32))
    assert nxt.shape == obs.shape
    assert reward == -1.0
    assert not terminated and not truncated


def test_an_all_zero_action_does_nothing(env: RoversEnv) -> None:
    """The neutral action must not fire an operator.

    The selector is a continuous number split into bands, so which operator sits on zero
    is a choice; noop has to, because zero is what a policy emits when it has nothing to
    do. An earlier layout put *image* there, and idle rovers spent every step
    photographing.
    """
    env.reset(seed=0)
    # pylint: disable=protected-access
    assert env._operator(0.0) == NOOP
    for operator in (NOOP, SAMPLE, CALIBRATE, IMAGE, SEND, DROP):
        assert env._operator(operator_action(operator)) == operator


def test_sampling_from_nowhere_is_refused(env: RoversEnv) -> None:
    """sample_rock's precondition: the base has to be over a sample."""
    env.reset(seed=1)
    _park(env, 0, (0.5, 0.5))
    with env.client():
        assert env.sample_under(0) is None
    env.step(_action(SAMPLE, NOOP))
    # pylint: disable=protected-access
    assert not env._store_full[0], "sampled from nowhere near a sample"


def test_sampling_over_a_rock_fills_the_store(env: RoversEnv) -> None:
    """Standing on a sample fills the store and records the analysis; drop frees it."""
    env.reset(seed=1)
    with env.client():
        point: Any = sp.get_point(env.rocks[0])
    _park(env, 0, point[:2])
    env.step(_action(SAMPLE, NOOP))
    # pylint: disable=protected-access
    assert env._store_full[0]
    assert env.rocks[0] in env._analyzed[0]
    env.step(_action(DROP, NOOP))
    assert not env._store_full[0]


def test_imaging_needs_calibration_and_spends_it(env: RoversEnv) -> None:
    """take_image needs a calibrated camera and a visible objective, one shot each."""
    env.reset(seed=1)
    with env.client():
        point = sp.get_point(env.objectives[0])
    _park(env, 0, (point[0], point[1] - 1.0), np.pi / 2)
    with env.client():
        assert env.image_visible(0, env.objectives[0])
    # pylint: disable=protected-access
    env.step(_action(IMAGE, NOOP))
    assert not env._have_image[0], "imaged without calibrating"
    env.step(_action(CALIBRATE, NOOP))
    assert env._calibrated[0]
    env.step(_action(IMAGE, NOOP))
    assert len(env._have_image[0]) == 1, "one calibration should buy one photograph"
    assert not env._calibrated[0]


def test_only_the_holder_can_send(env: RoversEnv) -> None:
    """Send transmits what *this* rover holds, and needs the lander in view."""
    env.reset(seed=1)
    with env.client():
        rock, lander = sp.get_point(env.rocks[0]), sp.get_point(env.lander)
    _park(env, 0, rock[:2])
    env.step(_action(SAMPLE, NOOP))
    # pylint: disable=protected-access
    assert env._analyzed[0] and not env._analyzed[1]
    # Rover 1 has nothing to send, so sending changes nothing.
    env.step(_action(NOOP, SEND))
    assert not env._received_analysis
    _park(env, 0, (lander[0] + 1.0, lander[1] + 0.6))
    with env.client():
        assert env.com_visible(0)
    env.step(_action(SEND, NOOP))
    assert env.rocks[0] in env._received_analysis


def test_the_arena_is_split_but_sight_is_not(env: RoversEnv) -> None:
    """The wall divides the floor into two halves, and the cameras see over it.

    Both halves of that are the benchmark's own scene and together are what make this a
    two-robot task: neither rover can drive to the other side, so the errands there are
    not its to run, yet either can still radio the lander wherever it stands.
    """
    env.reset(seed=0)
    with env.client():
        starts = [env.rover_conf(i)[0] for i in (0, 1)]
        assert starts[0] > 0 > starts[1], "rovers should start on opposite sides"
        # Driving across is blocked: the rover's body is far taller than the wall.
        _park(env, 0, (0.6, -1.75))
        before = env.rover_conf(0)[0]
    for _ in range(20):
        env.step(np.array([-0.2, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    with env.client():
        assert env.rover_conf(0)[0] > 0, "a rover drove through the dividing wall"
        assert env.rover_conf(0)[0] < before + 1e-6
        # Sight is not blocked: the lander sits on the far side and stays visible.
        assert sp.get_point(env.lander)[0] < 0
        assert env.com_visible(0), "the radio should reach over an ankle-high wall"


def test_goal_needs_every_clause(env: RoversEnv) -> None:
    """Samples of both types, every image, both rovers home, both stores empty."""
    env.reset(seed=3)
    with env.client():
        lander = np.array(sp.get_point(env.lander))[:2]
    # pylint: disable=protected-access
    for body in (env.rocks[0], env.soils[0]):
        with env.client():
            point = np.array(sp.get_point(body))[:2]
        _park(env, 0, point)
        env.step(_action(SAMPLE, NOOP))
        _park(env, 0, lander + np.array([1.0, 0.6]))
        env.step(_action(SEND, NOOP))
        env.step(_action(DROP, NOOP))
    assert len(env._received_analysis) == 2
    for objective in env.objectives:
        with env.client():
            point = np.array(sp.get_point(objective))[:2]
        _park(env, 0, point - np.array([0.0, 1.0]), np.pi / 2)
        env.step(_action(CALIBRATE, NOOP))
        env.step(_action(IMAGE, NOOP))
        _park(env, 0, lander + np.array([1.0, 0.6]))
        env.step(_action(SEND, NOOP))
    assert len(env._received_image) == len(env.objectives)
    # Everything done but the rovers are not home, so the episode is not over.
    assert not env.step(_action(NOOP, NOOP))[2]
    for index in (0, 1):
        home = env.home_confs[index]
        _park(env, index, home[:2], float(home[2]))
    assert env.step(_action(NOOP, NOOP))[2]


def test_a_red_herring_sample_does_not_substitute(env: RoversEnv) -> None:
    """Two stones are not a stone and a soil.

    The goal names the types, so an agent that grabs whatever is nearest twice has not
    finished.
    """
    env.reset(seed=3)
    with env.client():
        lander = np.array(sp.get_point(env.lander))[:2]
    for body in (env.rocks[0], env.rocks[1]):
        with env.client():
            point = np.array(sp.get_point(body))[:2]
        _park(env, 0, point)
        env.step(_action(SAMPLE, NOOP))
        _park(env, 0, lander + np.array([1.0, 0.6]))
        env.step(_action(SEND, NOOP))
        env.step(_action(DROP, NOOP))
    # pylint: disable=protected-access
    assert len(env._received_analysis) == 2
    assert not env._goal_reached()


def test_reset_is_reproducible_and_varies(env: RoversEnv) -> None:
    """The same seed gives the same layout; different seeds do not."""
    first, _ = env.reset(seed=5)
    other, _ = env.reset(seed=6)
    again, _ = env.reset(seed=5)
    assert np.allclose(first, again)
    assert not np.allclose(first, other)


def test_state_roundtrip(env: RoversEnv) -> None:
    """A saved state restores exactly, task fluents included."""
    env.reset(seed=2)
    with env.client():
        point = sp.get_point(env.rocks[0])
    _park(env, 0, point[:2])
    env.step(_action(SAMPLE, NOOP))
    saved = env.get_state()
    for _ in range(3):
        env.step(env.action_space.sample())
    env.set_state(saved)
    assert np.allclose(env.get_state(), saved)
    # pylint: disable=protected-access
    assert env._store_full[0]


def test_render_returns_a_frame(env: RoversEnv) -> None:
    """Rendering produces an RGB frame of the documented size."""
    env.reset(seed=0)
    frame: Any = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)


def test_card_describes_the_operators_and_the_goal(env: RoversEnv) -> None:
    """The agent-facing card states what actually decides the episode."""
    card = env.describe(include_access=True).lower()
    for word in ("sample", "calibrate", "image", "send", "drop", "lander"):
        assert word in card
    assert "noop is the band containing zero" in card
    assert "Source Code" not in env.env_description_blackbox
