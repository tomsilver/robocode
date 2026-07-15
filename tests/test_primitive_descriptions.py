"""Tests for format_primitives_description (black-box CRV notes across obs models)."""

from robocode.primitive_descriptions import format_primitives_description


def test_blackbox_crv_note_vector_uses_devectorize():
    """The default black-box CRV note builds the planner state via devectorize."""
    desc = format_primitives_description(["crv_motion_planning"], blackbox=True)
    assert "observation_space.devectorize(obs)" in desc


def test_blackbox_crv_note_object_centric_passes_state_directly():
    """A variable-count black-box CRV note passes the ObjectCentricState straight in --
    the object-centric client space has no devectorize."""
    desc = format_primitives_description(
        ["crv_motion_planning", "crv_motion_planning_grasp"],
        blackbox=True,
        object_centric=True,
    )
    assert "devectorize" not in desc
    assert "ObjectCentricState" in desc
    assert "plan_crv_actions(state," in desc


def test_object_centric_crv_note_is_a_blackbox_addendum_only():
    """Object-centric only swaps the black-box note; clearbox descriptions are
    unaffected (no black-box note appears)."""
    desc = format_primitives_description(
        ["crv_motion_planning"], blackbox=False, object_centric=True
    )
    assert "Black-box note" not in desc
