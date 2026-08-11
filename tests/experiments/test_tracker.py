"""Tests for campaign expansion and preservation-safe tracker upserts."""

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKER_DIR = _REPO_ROOT / "experiments" / "tracker"
sys.path.insert(0, str(_TRACKER_DIR))
generate: Any = importlib.import_module("generate")
schema: Any = importlib.import_module("schema")
sync: Any = importlib.import_module("sync_google_sheet")
sys.path.pop(0)


def test_priority_is_immediately_after_seeds() -> None:
    """The main scheduling signal stays visible before the long command field."""
    assert schema.ALL_COLUMNS.index("Priority") == schema.ALL_COLUMNS.index("Seeds") + 1
    assert "Priority" in schema.HUMAN_COLUMNS


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
            "Seeds": "[42, 24]",
            "Command": command,
            "Active": "TRUE",
            "Status": "Todo",
            "Progress": "0/2",
        }
    )
    return row


def test_sample_campaign_expansion_and_constraint() -> None:
    """The smoke campaign yields five rows and excludes one blackbox bilevel cell."""
    campaign = _REPO_ROOT / "experiments" / "campaigns" / "tracker_smoke_test.yaml"
    rows, excluded = generate.generate_rows([campaign])
    assert len(rows) == 5
    assert len(excluded) == 1
    assert excluded[0][1] == "bilevel_models unavailable under blackbox"
    assert all(row["Seeds"] == "[42, 24]" for row in rows)
    assert all("seed=42,24" in row["Command"] for row in rows)


def test_experiment_id_is_independent_of_seeds_and_mapping_order() -> None:
    """Replicate order and YAML mapping order do not change a condition ID."""
    values = {
        "environment": "motion2d_easy",
        "approach": "agentic",
        "primitive_level": "low_level",
        "approach.blackbox": False,
        "approach/backend": "claude_opus5",
    }
    first = generate.ExperimentConfig("campaign_a", values, (42, 24))
    second = generate.ExperimentConfig(
        "campaign_b", dict(reversed(tuple(values.items()))), (424,)
    )
    assert generate.experiment_id(first) == generate.experiment_id(second)


def test_missing_hydra_choice_fails_generation(tmp_path: Path) -> None:
    """Campaign typos fail at Hydra composition instead of reaching the Sheet."""
    campaign = tmp_path / "bad.yaml"
    campaign.write_text("""name: typo
matrix:
  environment: [not_an_environment]
  approach: [agentic]
  primitive_level: [none]
seeds: [42]
""")
    with pytest.raises(ValueError, match="Hydra could not compose"):
        generate.generate_rows([campaign])


@pytest.mark.parametrize(
    ("primitive_level", "expected"),
    [
        ("none", []),
        ("low_level", ["check_action_collision", "BiRRT"]),
        ("bilevel", ["bilevel_models"]),
    ],
)
def test_primitive_level_composes_to_implementation_list(
    primitive_level: str, expected: list[str]
) -> None:
    """The semantic Hydra group remains the single source for primitive lists."""
    config = generate.ExperimentConfig(
        "test",
        {
            "environment": "motion2d_easy",
            "approach": "agentic",
            "primitive_level": primitive_level,
        },
        (42,),
    )
    with initialize_config_dir(
        config_dir=str(_REPO_ROOT / "experiments" / "conf"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=generate.hydra_overrides(config))
    assert cfg.primitive_level == primitive_level
    assert list(cfg.primitives) == expected


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
    assert len(plan.new_rows) == 0
    command_column = schema.ALL_COLUMNS.index("Command") + 1
    assert sync.CellUpdate(2, command_column, "new command") in plan.updates
    human_columns = {
        schema.ALL_COLUMNS.index(column) + 1 for column in schema.HUMAN_COLUMNS
    }
    assert not any(update.column in human_columns for update in plan.updates)


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
    row = _generated_row("new")
    plan = sync.plan_upsert([], [row])
    appended = dict(zip(plan.header, plan.new_rows[0], strict=True))
    assert plan.initialize_header
    assert appended["Status"] == "Todo"
    assert appended["Progress"] == "0/2"
    assert appended["Owner"] == ""


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
    campaign_values = campaign["dataValidationRule"]["condition"]["values"]
    assert campaign_values == [
        {"userEnteredValue": "older_campaign"},
        {"userEnteredValue": "new_campaign"},
    ]
    primitive_values = primitive_level["dataValidationRule"]["condition"]["values"]
    assert primitive_values == [
        {"userEnteredValue": "none"},
        {"userEnteredValue": "bilevel"},
    ]
    active_values = active["dataValidationRule"]["condition"]["values"]
    assert active_values == [
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
    model = sync.build_table_column("Model / Backend", 6, table_values)
    assert model["columnType"] == "TEXT"


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

    assert sync._table_column_signature(current) == sync._table_column_signature(
        desired
    )
