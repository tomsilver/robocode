"""Tests for render_state primitive."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import numpy as np
import pytest

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from robocode.primitives.render_state import render_state


@pytest.fixture()
def env() -> KinderGeom2DEnv:
    """Create a KinderGeom2DEnv for testing."""
    e = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
    e.reset(seed=0)
    return e


def test_no_callback_returns_valid_image(env: KinderGeom2DEnv) -> None:
    """No-callback render returns a valid RGB image."""
    state = env.get_state()
    img = render_state(env, state)
    assert img.ndim == 3
    assert img.shape[2] in (3, 4)
    assert img.dtype == np.uint8


def test_callback_render_differs_from_plain(env: KinderGeom2DEnv) -> None:
    """Callback render returns correct shape/dtype and differs from plain."""
    state = env.get_state()
    plain = render_state(env, state)
    with_cb = render_state(
        env, state, ax_callback=lambda ax: ax.plot(1.0, 1.0, "ro", markersize=10)
    )
    assert with_cb.shape == plain.shape
    assert with_cb.dtype == np.uint8
    assert not np.array_equal(plain, with_cb)


def test_state_preserved_without_callback(env: KinderGeom2DEnv) -> None:
    """Env state is preserved after plain render."""
    state = env.get_state()
    render_state(env, state)
    np.testing.assert_array_equal(env.get_state(), state)


def test_state_preserved_with_callback(env: KinderGeom2DEnv) -> None:
    """Env state is preserved after callback render."""
    state = env.get_state()
    render_state(env, state, ax_callback=lambda ax: ax.plot(1.0, 1.0, "ro"))
    np.testing.assert_array_equal(env.get_state(), state)


def test_not_implemented_for_unsupported_env() -> None:
    """NotImplementedError raised for unsupported env types with ax_callback."""

    class _FakeEnv:
        """Stub env for testing."""

    with pytest.raises(NotImplementedError):
        render_state(_FakeEnv(), None, ax_callback=lambda ax: None)


def test_output_dimensions_match(env: KinderGeom2DEnv) -> None:
    """Output dimensions match between callback and plain renders."""
    state = env.get_state()
    plain = render_state(env, state)
    with_cb = render_state(env, state, ax_callback=lambda ax: ax.plot(1.0, 1.0, "ro"))
    assert plain.shape == with_cb.shape
