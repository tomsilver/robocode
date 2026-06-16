"""Tests for render_paths.py."""

from pathlib import Path

from robocode.utils.render_paths import safe_label, unique_path


def test_safe_label_keeps_allowed_characters() -> None:
    """Alphanumerics, dashes, and underscores pass through unchanged."""
    assert safe_label("my_label-1") == "my_label-1"


def test_safe_label_sanitizes_path_separators() -> None:
    """Path separators and ``..`` collapse to underscores (no escaping)."""
    assert safe_label("../../escape") == "_escape"
    assert safe_label("a/b\\c") == "a_b_c"


def test_safe_label_collapses_runs_of_disallowed_characters() -> None:
    """A run of disallowed characters becomes a single underscore."""
    assert safe_label("a   b!!c") == "a_b_c"


def test_unique_path_returns_plain_path_when_free(tmp_path: Path) -> None:
    """With nothing on disk, the unsuffixed ``stem.ext`` is returned."""
    assert unique_path(tmp_path, "frame") == tmp_path / "frame.png"


def test_unique_path_appends_counter_when_taken(tmp_path: Path) -> None:
    """Existing files are skipped by appending _1, _2, ..."""
    (tmp_path / "frame.png").touch()
    assert unique_path(tmp_path, "frame") == tmp_path / "frame_1.png"
    (tmp_path / "frame_1.png").touch()
    assert unique_path(tmp_path, "frame") == tmp_path / "frame_2.png"


def test_unique_path_honors_custom_extension(tmp_path: Path) -> None:
    """The extension argument overrides the ``.png`` default."""
    assert unique_path(tmp_path, "clip", ".gif") == tmp_path / "clip.gif"
