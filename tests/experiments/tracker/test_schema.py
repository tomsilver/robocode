"""Tests for tracker column ownership and display order."""

import importlib
import sys
from pathlib import Path
from typing import Any

_TRACKER_DIR = Path(__file__).resolve().parents[3] / "experiments" / "tracker"
sys.path.insert(0, str(_TRACKER_DIR))
schema: Any = importlib.import_module("schema")
sys.path.pop(0)


def test_priority_is_immediately_after_seed_protocol() -> None:
    """The main scheduling signal stays visible before the long command field."""
    assert schema.ALL_COLUMNS[:2] == ("Status", "Owner")
    assert schema.ALL_COLUMNS.index("Priority") == (
        schema.ALL_COLUMNS.index("Evaluation Seed") + 1
    )
    assert "Priority" in schema.HUMAN_COLUMNS


def test_column_ownership_is_a_complete_partition() -> None:
    """Every Sheet column has exactly one writer."""
    assert set(schema.GENERATED_COLUMNS).isdisjoint(schema.HUMAN_COLUMNS)
    assert set(schema.ALL_COLUMNS) == set(schema.GENERATED_COLUMNS) | set(
        schema.HUMAN_COLUMNS
    )
    assert set(schema.COLUMN_WIDTHS) == set(schema.ALL_COLUMNS)
