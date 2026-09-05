"""Tests for the MuJoCo GL helpers."""

from robocode.environments.mujoco_gl import uses_mujoco


def test_uses_mujoco_detects_kinder_dynamic3d_hierarchy() -> None:
    """Any class descending from kinder.envs.dynamic3d counts as MuJoCo-backed."""
    base = type("TidyBot3DEnv", (), {})
    base.__module__ = "kinder.envs.dynamic3d.envs"
    derived = type("Family", (base,), {})
    derived.__module__ = "kinder.envs.dynamic3d.task_families"
    assert uses_mujoco(base)
    assert uses_mujoco(derived)


def test_uses_mujoco_is_false_for_other_environments() -> None:
    """Kinematic and dynamic2d families keep the forked evaluation worker."""
    other = type("Obstruction2DEnv", (), {})
    other.__module__ = "kinder.envs.kinematic2d.obstruction2d"
    assert not uses_mujoco(other)
    assert not uses_mujoco(object)
