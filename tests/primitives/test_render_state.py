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


def test_no_labels_returns_valid_image(env: KinderGeom2DEnv) -> None:
    """No-labels render returns a valid RGB image."""
    state = env.get_state()
    img = render_state(env, state)
    assert img.ndim == 3
    assert img.shape[2] in (3, 4)
    assert img.dtype == np.uint8


def test_labeled_render_differs_from_unlabeled(env: KinderGeom2DEnv) -> None:
    """Labeled render returns correct shape/dtype and differs from unlabeled."""
    state = env.get_state()
    plain = render_state(env, state)
    labeled = render_state(env, state, labels=[(1.0, 1.0, "A")])
    assert labeled.shape == plain.shape
    assert labeled.dtype == np.uint8
    assert not np.array_equal(plain, labeled)


def test_multiple_labels(env: KinderGeom2DEnv) -> None:
    """Multiple labels work."""
    state = env.get_state()
    labels = [(0.5, 0.5, "P1"), (1.5, 1.5, "P2"), (2.0, 0.5, "P3")]
    img = render_state(env, state, labels=labels)
    assert img.ndim == 3
    assert img.dtype == np.uint8


def test_empty_labels_same_as_no_labels(env: KinderGeom2DEnv) -> None:
    """Empty labels behaves like no labels."""
    state = env.get_state()
    plain = render_state(env, state)
    empty = render_state(env, state, labels=[])
    assert plain.shape == empty.shape
    np.testing.assert_array_equal(plain, empty)


def test_state_preserved_after_unlabeled(env: KinderGeom2DEnv) -> None:
    """Env state is preserved after unlabeled render."""
    state = env.get_state()
    render_state(env, state)
    np.testing.assert_array_equal(env.get_state(), state)


def test_state_preserved_after_labeled(env: KinderGeom2DEnv) -> None:
    """Env state is preserved after labeled render."""
    state = env.get_state()
    render_state(env, state, labels=[(1.0, 1.0, "X")])
    np.testing.assert_array_equal(env.get_state(), state)


def test_not_implemented_for_unsupported_env() -> None:
    """NotImplementedError raised for unsupported env types with labels."""

    class _FakeEnv:
        """Stub env for testing."""

    with pytest.raises(NotImplementedError):
        render_state(_FakeEnv(), None, labels=[(0.0, 0.0, "X")])


def test_output_dimensions_match(env: KinderGeom2DEnv) -> None:
    """Output dimensions match between labeled and unlabeled renders."""
    state = env.get_state()
    plain = render_state(env, state)
    labeled = render_state(env, state, labels=[(1.0, 1.0, "A")])
    assert plain.shape == labeled.shape
