"""Tests for preservation-safe Google Sheet tracker upserts."""

import importlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

_TRACKER_DIR = Path(__file__).resolve().parents[3] / "experiments" / "tracker"
sys.path.insert(0, str(_TRACKER_DIR))
schema: Any = importlib.import_module("schema")
sync: Any = importlib.import_module("sync_google_sheet")
sys.path.pop(0)

_TEST_EVAL_SEED = 918273645


def _generated_row(
    condition_id: str, campaign: str = "campaign_a", command: str = "old command"
) -> dict[str, str]:
    row = {column: "" for column in schema.ALL_COLUMNS}
    row.update(
        {
            "Experiment ID": condition_id,
            "Campaign": campaign,
            "Environment": "motion2d_easy",
            "Method": "agentic",
            "Primitive Level": "none",
            "Access": "whitebox",
            "Model / Backend": "claude_opus5",
            "Replicate Seeds": "[42, 24]",
            "Evaluation Seed": str(_TEST_EVAL_SEED),
            "Command": command,
            "Active": "TRUE",
            "Status": "Todo",
            "Progress": "0/2",
        }
    )
    return row


def test_load_generated_rows_fills_truncated_cells_with_blanks(tmp_path: Path) -> None:
    """Short CSV rows never emit null values that Sheets silently skips."""
    csv_path = tmp_path / "short.csv"
    csv_path.write_text(
        ",".join(schema.GENERATED_COLUMNS) + "\ncondition-1,campaign_a,motion2d_easy\n",
        encoding="utf-8",
    )
    [row] = sync.load_generated_rows([csv_path])
    assert row["Experiment ID"] == "condition-1"
    assert row["Command"] == ""
    assert all(value is not None for value in row.values())


def test_upsert_updates_generated_cells_and_preserves_human_cells() -> None:
    """Existing human fields are never included in the planned write ranges."""
    old = _generated_row("condition-1")
    existing_row = [old[column] for column in schema.ALL_COLUMNS]
    existing_row[schema.ALL_COLUMNS.index("Owner")] = "Ada"
    existing_row[schema.ALL_COLUMNS.index("Status")] = "Running"
    existing_row[schema.ALL_COLUMNS.index("Notes")] = "Do not overwrite"
    new = _generated_row("condition-1", command="new command")
    plan = sync.plan_upsert([list(schema.ALL_COLUMNS), existing_row], [new])
    assert plan.updated_experiments == 1
    assert not plan.new_rows
    command_column = schema.ALL_COLUMNS.index("Command") + 1
    assert sync.CellUpdate(2, command_column, "new command") in plan.updates
    human_columns = {
        schema.ALL_COLUMNS.index(column) + 1 for column in schema.HUMAN_COLUMNS
    }
    assert not any(update.column in human_columns for update in plan.updates)


def test_upsert_rejects_seed_changes_for_the_same_experiment_id() -> None:
    """A malformed old-style ID cannot relabel an existing row's protocol."""
    old = _generated_row("condition-1")
    existing = [
        list(schema.ALL_COLUMNS),
        [old[column] for column in schema.ALL_COLUMNS],
    ]
    changed = _generated_row("condition-1")
    changed["Replicate Seeds"] = "[42, 24, 424]"
    with pytest.raises(ValueError, match="changes its run protocol"):
        sync.plan_upsert(existing, [changed])


def test_changed_protocol_appends_and_preserves_completed_results() -> None:
    """A new canonical ID leaves an earlier completed row untouched."""
    old = _generated_row("old-protocol")
    new = _generated_row("new-protocol")
    existing_row = [old[column] for column in schema.ALL_COLUMNS]
    existing_row[schema.ALL_COLUMNS.index("Status")] = "Done"
    existing_row[schema.ALL_COLUMNS.index("Progress")] = "2/2"
    existing_row[schema.ALL_COLUMNS.index("Results")] = "result-link"
    existing_row[schema.ALL_COLUMNS.index("Git SHA")] = "abc123"
    plan = sync.plan_upsert([list(schema.ALL_COLUMNS), existing_row], [new])
    assert len(plan.new_rows) == 1
    protected = {
        schema.ALL_COLUMNS.index(column) + 1
        for column in ("Status", "Progress", "Results", "Git SHA")
    }
    assert not any(update.column in protected for update in plan.updates)


def test_upsert_inactivates_only_campaigns_in_the_input() -> None:
    """Syncing one campaign does not retire rows owned by other campaigns."""
    old_a = _generated_row("old-a", campaign="campaign_a")
    old_b = _generated_row("old-b", campaign="campaign_b")
    new_a = _generated_row("new-a", campaign="campaign_a")
    existing = [
        list(schema.ALL_COLUMNS),
        [old_a[column] for column in schema.ALL_COLUMNS],
        [old_b[column] for column in schema.ALL_COLUMNS],
    ]
    plan = sync.plan_upsert(existing, [new_a])
    active_column = schema.ALL_COLUMNS.index("Active") + 1
    assert sync.CellUpdate(2, active_column, "FALSE") in plan.updates
    assert not any(update.row == 3 for update in plan.updates)
    assert plan.inactivated_experiments == 1


def test_new_rows_receive_initial_human_defaults() -> None:
    """Human defaults are applied only when an experiment is first appended."""
    plan = sync.plan_upsert([], [_generated_row("new")])
    appended = dict(zip(plan.header, plan.new_rows[0], strict=True))
    assert plan.initialize_header
    assert appended["Status"] == "Todo"
    assert appended["Progress"] == "0/2"
    assert appended["Owner"] == ""


def test_apply_upsert_coalesces_adjacent_generated_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One changed row uses contiguous ranges instead of one range per cell."""
    gspread_utils = Mock()
    gspread_utils.rowcol_to_a1.side_effect = lambda row, column: (
        f"{chr(64 + column)}{row}"
    )
    monkeypatch.setattr(
        sync.importlib, "import_module", Mock(return_value=gspread_utils)
    )
    worksheet = Mock()
    plan = sync.UpsertPlan(
        tuple(schema.ALL_COLUMNS),
        False,
        (
            sync.CellUpdate(2, 3, "a"),
            sync.CellUpdate(2, 4, "b"),
            sync.CellUpdate(2, 6, "c"),
        ),
        (),
        1,
        0,
    )
    sync.apply_upsert(worksheet, plan)
    assert worksheet.batch_update.call_args.args[0] == [
        {"range": "C2:D2", "values": [["a", "b"]]},
        {"range": "F2", "values": [["c"]]},
    ]


def test_project_upsert_materializes_updates_and_appends() -> None:
    """The in-memory table state matches the write plan."""
    old = _generated_row("condition-1")
    existing = [
        list(schema.ALL_COLUMNS),
        [old[column] for column in schema.ALL_COLUMNS],
    ]
    changed = _generated_row("condition-1", command="new command")
    added = _generated_row("condition-2")
    plan = sync.plan_upsert(existing, [changed, added])
    projected = sync.project_upsert(existing, plan)
    command_index = schema.ALL_COLUMNS.index("Command")
    assert projected[1][command_index] == "new command"
    assert projected[2][schema.ALL_COLUMNS.index("Experiment ID")] == "condition-2"


def test_categorical_table_columns_use_live_dropdown_options() -> None:
    """Categorical chips include every value already present in the table."""
    first = _generated_row("first", campaign="older_campaign")
    second = _generated_row("second", campaign="new_campaign")
    second["Primitive Level"] = "bilevel"
    table_values = [
        list(schema.ALL_COLUMNS),
        [first[column] for column in schema.ALL_COLUMNS],
        [second[column] for column in schema.ALL_COLUMNS],
    ]
    campaign = sync.build_table_column("Campaign", 1, table_values)
    primitive_level = sync.build_table_column("Primitive Level", 4, table_values)
    active = sync.build_table_column("Active", 9, table_values)
    assert campaign["columnType"] == "DROPDOWN"
    assert campaign["dataValidationRule"]["condition"]["values"] == [
        {"userEnteredValue": "older_campaign"},
        {"userEnteredValue": "new_campaign"},
    ]
    assert primitive_level["dataValidationRule"]["condition"]["values"] == [
        {"userEnteredValue": "none"},
        {"userEnteredValue": "bilevel"},
    ]
    assert active["dataValidationRule"]["condition"]["values"] == [
        {"userEnteredValue": "TRUE"},
        {"userEnteredValue": "FALSE"},
    ]


def test_blank_categorical_column_remains_text() -> None:
    """A category with no current choices does not create an empty dropdown."""
    row = _generated_row("first")
    row["Model / Backend"] = ""
    table_values = [
        list(schema.ALL_COLUMNS),
        [row[column] for column in schema.ALL_COLUMNS],
    ]
    assert (
        sync.build_table_column("Model / Backend", 6, table_values)["columnType"]
        == "TEXT"
    )


def test_table_column_signature_ignores_omitted_zero_index() -> None:
    """An unchanged schema does not rewrite native dropdown chip styling."""
    row = _generated_row("first")
    table_values = [
        list(schema.ALL_COLUMNS),
        [row[column] for column in schema.ALL_COLUMNS],
    ]
    desired = [
        sync.build_table_column(column, index, table_values)
        for index, column in enumerate(schema.ALL_COLUMNS)
    ]
    current = [dict(column) for column in desired]
    current[0].pop("columnIndex")
    # pylint: disable-next=protected-access
    current_signature = sync._table_column_signature(current)
    # pylint: disable-next=protected-access
    desired_signature = sync._table_column_signature(desired)
    assert current_signature == desired_signature


def test_missing_title_does_not_reuse_a_nonempty_worksheet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Content outside A1 prevents the default sheet from being overwritten."""

    class WorksheetNotFound(Exception):
        """Test substitute for gspread's missing-worksheet exception."""

    gspread = Mock(WorksheetNotFound=WorksheetNotFound)
    monkeypatch.setattr(sync.importlib, "import_module", Mock(return_value=gspread))
    default = Mock()
    default.get_all_values.return_value = [["", "existing note"]]
    spreadsheet = Mock()
    spreadsheet.worksheet.side_effect = WorksheetNotFound
    spreadsheet.worksheets.return_value = [default]
    # pylint: disable-next=protected-access
    created = sync._get_worksheet(spreadsheet, "Tracker")
    assert created == spreadsheet.add_worksheet.return_value
    default.update_title.assert_not_called()


def test_dry_run_does_not_create_a_missing_worksheet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preview of a new tracker remains read-only at spreadsheet level."""

    class WorksheetNotFound(Exception):
        """Test substitute for gspread's missing-worksheet exception."""

    gspread = Mock(WorksheetNotFound=WorksheetNotFound)
    spreadsheet = Mock()
    spreadsheet.worksheet.side_effect = WorksheetNotFound
    client = Mock()
    client.open_by_key.return_value = spreadsheet
    args = Mock(
        csv_files=[Path("generated.csv")],
        sheet_id="sheet-id",
        worksheet="Tracker",
        dry_run=True,
    )
    monkeypatch.setattr(
        sync, "importlib", Mock(import_module=Mock(return_value=gspread))
    )
    monkeypatch.setattr(sync, "_parse_args", Mock(return_value=args))
    monkeypatch.setattr(
        sync, "load_generated_rows", Mock(return_value=[_generated_row("new")])
    )
    monkeypatch.setattr(sync, "_authorize", Mock(return_value=client))
    sync.main()
    spreadsheet.worksheet.assert_called_once_with("Tracker")
    spreadsheet.worksheets.assert_not_called()
    spreadsheet.add_worksheet.assert_not_called()


def test_sync_fetches_existing_values_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Table refresh uses the projected state instead of re-downloading the Sheet."""
    row = _generated_row("condition-1")
    existing = [
        list(schema.ALL_COLUMNS),
        [row[column] for column in schema.ALL_COLUMNS],
    ]
    worksheet = Mock()
    worksheet.get_all_values.return_value = existing
    spreadsheet = Mock()
    client = Mock()
    client.open_by_key.return_value = spreadsheet
    args = Mock(
        csv_files=[Path("generated.csv")],
        sheet_id="sheet-id",
        worksheet="Tracker",
        dry_run=False,
    )
    ensure = Mock()
    monkeypatch.setattr(sync, "_parse_args", Mock(return_value=args))
    monkeypatch.setattr(sync, "load_generated_rows", Mock(return_value=[row]))
    monkeypatch.setattr(sync, "_authorize", Mock(return_value=client))
    monkeypatch.setattr(sync, "_get_worksheet", Mock(return_value=worksheet))
    monkeypatch.setattr(sync, "apply_upsert", Mock())
    monkeypatch.setattr(sync, "ensure_tracker_table", ensure)
    sync.main()
    worksheet.get_all_values.assert_called_once_with()
    assert ensure.call_args.args[2] == existing
