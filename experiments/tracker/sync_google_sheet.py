"""Safely upsert generated experiment rows into a Google Sheet."""

import argparse
import csv
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schema import (
    ALL_COLUMNS,
    CATEGORICAL_COLUMNS,
    GENERATED_COLUMNS,
    HUMAN_COLUMNS,
    PRIORITY_OPTIONS,
    SEMANTIC_COLORS,
    STATUS_OPTIONS,
)


@dataclass(frozen=True)
class CellUpdate:
    """One generated cell that differs from its current displayed value."""

    row: int
    column: int
    value: str


@dataclass(frozen=True)
class UpsertPlan:
    """A preservation-safe set of changes for one tracker worksheet."""

    header: tuple[str, ...]
    initialize_header: bool
    updates: tuple[CellUpdate, ...]
    new_rows: tuple[tuple[str, ...], ...]
    updated_experiments: int
    inactivated_experiments: int


def load_generated_rows(paths: list[Path]) -> list[dict[str, str]]:
    """Read generated CSVs and require their generated schema."""
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            header = reader.fieldnames or []
            missing = set(GENERATED_COLUMNS) - set(header)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            rows.extend(dict(row) for row in reader)
    if not rows:
        raise ValueError("At least one generated experiment row is required")
    ids = [row["Experiment ID"] for row in rows]
    if any(not condition_id for condition_id in ids):
        raise ValueError("Every generated row must have an Experiment ID")
    if len(ids) != len(set(ids)):
        raise ValueError("Generated CSVs contain duplicate Experiment IDs")
    return rows


def _padded(row: list[str], width: int) -> list[str]:
    return [*row, *("" for _ in range(width - len(row)))]


def plan_upsert(
    existing_values: list[list[str]], generated_rows: list[dict[str, str]]
) -> UpsertPlan:
    """Plan an upsert that never targets a human-owned column."""
    initialize_header = not existing_values or not any(existing_values[0])
    header = tuple(ALL_COLUMNS if initialize_header else existing_values[0])
    if len(header) != len(set(header)):
        raise ValueError("Tracker header contains duplicate column names")
    missing = set(ALL_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"Tracker is missing columns {sorted(missing)}")
    column_by_name = {name: index + 1 for index, name in enumerate(header)}

    existing_by_id: dict[str, tuple[int, list[str]]] = {}
    if not initialize_header:
        id_index = column_by_name["Experiment ID"] - 1
        for row_number, raw_row in enumerate(existing_values[1:], start=2):
            row = _padded(raw_row, len(header))
            condition_id = row[id_index]
            if not condition_id:
                continue
            if condition_id in existing_by_id:
                raise ValueError(f"Duplicate Experiment ID in Sheet: {condition_id}")
            existing_by_id[condition_id] = (row_number, row)

    updates: list[CellUpdate] = []
    new_rows: list[tuple[str, ...]] = []
    updated_experiments = 0
    incoming_ids = {row["Experiment ID"] for row in generated_rows}
    incoming_campaigns = {row["Campaign"] for row in generated_rows}

    for generated in generated_rows:
        condition_id = generated["Experiment ID"]
        if condition_id not in existing_by_id:
            new_rows.append(tuple(generated.get(column, "") for column in header))
            continue
        row_number, current = existing_by_id[condition_id]
        changed = False
        for column in GENERATED_COLUMNS:
            column_number = column_by_name[column]
            value = generated[column]
            if current[column_number - 1] != value:
                updates.append(CellUpdate(row_number, column_number, value))
                changed = True
        updated_experiments += int(changed)

    campaign_index = column_by_name["Campaign"] - 1
    active_column = column_by_name["Active"]
    inactivated_experiments = 0
    for condition_id, (row_number, current) in existing_by_id.items():
        if condition_id in incoming_ids:
            continue
        if current[campaign_index] not in incoming_campaigns:
            continue
        if current[active_column - 1] != "FALSE":
            updates.append(CellUpdate(row_number, active_column, "FALSE"))
            inactivated_experiments += 1

    human_column_numbers = {column_by_name[column] for column in HUMAN_COLUMNS}
    assert all(update.column not in human_column_numbers for update in updates)
    return UpsertPlan(
        header,
        initialize_header,
        tuple(updates),
        tuple(new_rows),
        updated_experiments,
        inactivated_experiments,
    )


def _column_letter(column: int) -> str:
    result = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        result = chr(65 + remainder) + result
    return result


def apply_upsert(worksheet: Any, plan: UpsertPlan) -> None:
    """Apply a plan using generated-cell ranges and whole new rows."""
    if plan.initialize_header:
        worksheet.update([list(plan.header)], "A1", value_input_option="RAW")
    if plan.updates:
        worksheet.batch_update(
            [
                {
                    "range": f"{_column_letter(update.column)}{update.row}",
                    "values": [[update.value]],
                }
                for update in plan.updates
            ],
            value_input_option="RAW",
        )
    if plan.new_rows:
        worksheet.append_rows(
            [list(row) for row in plan.new_rows], value_input_option="RAW"
        )


def _dropdown_options(column: str, table_values: list[list[str]]) -> tuple[str, ...]:
    if column == "Status":
        return STATUS_OPTIONS
    if column == "Priority":
        return PRIORITY_OPTIONS
    if column == "Active":
        return ("TRUE", "FALSE")
    header = table_values[0]
    column_index = header.index(column)
    options: list[str] = []
    for raw_row in table_values[1:]:
        row = _padded(raw_row, len(header))
        value = row[column_index]
        if value and value not in options:
            options.append(value)
    return tuple(options)


def build_table_column(
    column: str, index: int, table_values: list[list[str]]
) -> dict[str, Any]:
    """Describe one native table column and its allowed values."""
    result: dict[str, Any] = {
        "columnIndex": index,
        "columnName": column,
        "columnType": "TEXT",
    }
    if column == "Owner":
        result["columnType"] = "PEOPLE_CHIP"
    elif column == "Results":
        result["columnType"] = "FILES_CHIP"
    elif column in CATEGORICAL_COLUMNS:
        options = _dropdown_options(column, table_values)
        if options:
            result.update(
                {
                    "columnType": "DROPDOWN",
                    "dataValidationRule": {
                        "condition": {
                            "type": "ONE_OF_LIST",
                            "values": [
                                {"userEnteredValue": option} for option in options
                            ],
                        }
                    },
                }
            )
    return result


def _rgb_color(hex_color: str) -> dict[str, float]:
    """Convert a six-digit hex color to the Sheets API RGB shape."""
    return {
        channel: int(hex_color[offset : offset + 2], 16) / 255
        for channel, offset in (("red", 0), ("green", 2), ("blue", 4))
    }


def _conditional_rule_key(rule: dict[str, Any]) -> tuple[int, str] | None:
    """Return the column/value key for a simple text-equality format rule."""
    ranges = rule.get("ranges", [])
    condition = rule.get("booleanRule", {}).get("condition", {})
    values = condition.get("values", [])
    if (
        len(ranges) != 1
        or condition.get("type") != "TEXT_EQ"
        or len(values) != 1
        or "userEnteredValue" not in values[0]
    ):
        return None
    grid_range = ranges[0]
    start = grid_range.get("startColumnIndex")
    end = grid_range.get("endColumnIndex")
    if not isinstance(start, int) or end != start + 1:
        return None
    return start, values[0]["userEnteredValue"]


def build_semantic_format_requests(
    sheet_id: int,
    header: tuple[str, ...],
    row_count: int,
    existing_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add missing semantic colors while preserving user-created rules."""
    existing_keys = {
        key for rule in existing_rules if (key := _conditional_rule_key(rule))
    }
    requests: list[dict[str, Any]] = []
    next_index = len(existing_rules)
    for column, value_colors in SEMANTIC_COLORS.items():
        column_index = header.index(column)
        for value, hex_color in value_colors.items():
            if (column_index, value) in existing_keys:
                continue
            requests.append(
                {
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [
                                {
                                    "sheetId": sheet_id,
                                    "startRowIndex": 1,
                                    "endRowIndex": row_count,
                                    "startColumnIndex": column_index,
                                    "endColumnIndex": column_index + 1,
                                }
                            ],
                            "booleanRule": {
                                "condition": {
                                    "type": "TEXT_EQ",
                                    "values": [{"userEnteredValue": value}],
                                },
                                "format": {
                                    "backgroundColorStyle": {
                                        "rgbColor": _rgb_color(hex_color)
                                    }
                                },
                            },
                        },
                        "index": next_index,
                    }
                }
            )
            next_index += 1
    return requests


def ensure_tracker_table(
    spreadsheet: Any, worksheet: Any, table_values: list[list[str]]
) -> None:
    """Create or refresh the native table and its categorical columns."""
    header = tuple(table_values[0])
    used_rows = len(table_values)
    metadata = spreadsheet.fetch_sheet_metadata()
    sheet = next(
        item
        for item in metadata["sheets"]
        if item["properties"]["sheetId"] == worksheet.id
    )
    matching_tables = [
        table
        for table in sheet.get("tables", [])
        if table["range"].get("startRowIndex", 0) == 0
        and table["range"].get("startColumnIndex", 0) == 0
    ]
    table_range = {
        "sheetId": worksheet.id,
        "startRowIndex": 0,
        "endRowIndex": used_rows,
        "startColumnIndex": 0,
        "endColumnIndex": len(header),
    }
    requests: list[dict[str, Any]] = []
    rows_properties = {
        "headerColorStyle": {"rgbColor": {"red": 0.12, "green": 0.31, "blue": 0.55}},
        "firstBandColorStyle": {"rgbColor": {"red": 1, "green": 1, "blue": 1}},
        "secondBandColorStyle": {
            "rgbColor": {"red": 0.97, "green": 0.97, "blue": 0.97}
        },
    }
    column_properties = [
        build_table_column(column, index, table_values)
        for index, column in enumerate(header)
    ]
    if matching_tables:
        if len(matching_tables) != 1:
            raise ValueError("Tracker worksheet has multiple tables starting at A1")
        requests.append(
            {
                "updateTable": {
                    "table": {
                        "tableId": matching_tables[0]["tableId"],
                        "range": table_range,
                        "rowsProperties": rows_properties,
                        "columnProperties": column_properties,
                    },
                    "fields": "range,rowsProperties,columnProperties",
                }
            }
        )
    else:
        requests.extend(
            [
                {
                    "addTable": {
                        "table": {
                            "name": f"RobocodeExperimentTracker_{worksheet.id}",
                            "range": table_range,
                            "rowsProperties": rows_properties,
                            "columnProperties": column_properties,
                        }
                    }
                },
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": worksheet.id,
                            "gridProperties": {"frozenRowCount": 1},
                        },
                        "fields": "gridProperties.frozenRowCount",
                    }
                },
            ]
        )
        widths = {
            "Experiment ID": 300,
            "Campaign": 180,
            "Environment": 180,
            "Method": 120,
            "Primitive Level": 140,
            "Access": 110,
            "Model / Backend": 180,
            "Seeds": 140,
            "Command": 420,
            "Owner": 180,
            "Status": 120,
            "Progress": 90,
            "Priority": 100,
            "Notes": 320,
            "Results": 220,
            "Git SHA": 120,
        }
        requests.extend(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet.id,
                        "dimension": "COLUMNS",
                        "startIndex": index,
                        "endIndex": index + 1,
                    },
                    "properties": {"pixelSize": widths.get(column, 110)},
                    "fields": "pixelSize",
                }
            }
            for index, column in enumerate(header)
        )
        for column in ("Experiment ID", "Command", "Notes", "Results"):
            index = header.index(column)
            requests.append(
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": worksheet.id,
                            "startRowIndex": 1,
                            "endRowIndex": used_rows,
                            "startColumnIndex": index,
                            "endColumnIndex": index + 1,
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "wrapStrategy": "WRAP",
                                "verticalAlignment": "TOP",
                            }
                        },
                        "fields": ("userEnteredFormat(wrapStrategy,verticalAlignment)"),
                    }
                }
            )
        requests.append(
            {
                "autoResizeDimensions": {
                    "dimensions": {
                        "sheetId": worksheet.id,
                        "dimension": "ROWS",
                        "startIndex": 0,
                        "endIndex": used_rows,
                    }
                }
            }
        )
    requests.append(
        {
            "repeatCell": {
                "range": {
                    "sheetId": worksheet.id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": len(header),
                },
                "cell": {
                    "userEnteredFormat": {
                        "textFormat": {
                            "bold": True,
                            "foregroundColorStyle": {
                                "rgbColor": {"red": 1, "green": 1, "blue": 1}
                            },
                        }
                    }
                },
                "fields": "userEnteredFormat.textFormat",
            }
        }
    )
    row_count = sheet["properties"]["gridProperties"]["rowCount"]
    requests.extend(
        build_semantic_format_requests(
            worksheet.id,
            header,
            row_count,
            sheet.get("conditionalFormats", []),
        )
    )
    spreadsheet.batch_update({"requests": requests})


def _authorize(args: argparse.Namespace) -> Any:
    gspread = importlib.import_module("gspread")

    kwargs: dict[str, str] = {}
    credentials = args.credentials or os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
    authorized_user = args.authorized_user or os.environ.get("GOOGLE_AUTHORIZED_USER")
    if credentials:
        kwargs["credentials_filename"] = credentials
    if authorized_user:
        kwargs["authorized_user_filename"] = authorized_user
    return gspread.oauth(**kwargs)


def _get_worksheet(spreadsheet: Any, title: str) -> Any:
    gspread = importlib.import_module("gspread")

    try:
        return spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        worksheets = spreadsheet.worksheets()
        if len(worksheets) == 1 and not worksheets[0].acell("A1").value:
            worksheets[0].update_title(title)
            return worksheets[0]
        return spreadsheet.add_worksheet(title=title, rows=100, cols=len(ALL_COLUMNS))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", type=Path, nargs="+")
    parser.add_argument("--sheet-id", required=True)
    parser.add_argument("--worksheet", default="Tracker")
    parser.add_argument("--credentials", type=str)
    parser.add_argument("--authorized-user", type=str)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the Google Sheet upsert CLI."""
    args = _parse_args()
    generated_rows = load_generated_rows(args.csv_files)
    client = _authorize(args)
    spreadsheet = client.open_by_key(args.sheet_id)
    worksheet = _get_worksheet(spreadsheet, args.worksheet)
    existing = worksheet.get_all_values()
    plan = plan_upsert(existing, generated_rows)
    print(f"{len(plan.new_rows)} new experiments")
    print(f"{plan.updated_experiments} existing experiments updated")
    print(f"{plan.inactivated_experiments} experiments marked inactive")
    if args.dry_run:
        return
    apply_upsert(worksheet, plan)
    table_values = worksheet.get_all_values()
    ensure_tracker_table(spreadsheet, worksheet, table_values)
    print(f"Synchronized https://docs.google.com/spreadsheets/d/{args.sheet_id}")


if __name__ == "__main__":
    main()
