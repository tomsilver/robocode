"""Shared columns for generated experiment files and the Google Sheet."""

GENERATED_COLUMNS = (
    "Experiment ID",
    "Campaign",
    "Environment",
    "Method",
    "Primitive Level",
    "Access",
    "Model / Backend",
    "Replicate Seeds",
    "Evaluation Seed",
    "Command",
    "Active",
)

HUMAN_COLUMNS = (
    "Owner",
    "Status",
    "Progress",
    "Priority",
    "Notes",
    "Results",
    "Git SHA",
)

# Display order is intentionally independent from column ownership. Priority is a
# human-owned field, but belongs next to the seed protocol so it is visible before
# the long generated Command column.
ALL_COLUMNS = (
    "Status",
    "Owner",
    "Experiment ID",
    "Campaign",
    "Environment",
    "Method",
    "Primitive Level",
    "Access",
    "Model / Backend",
    "Replicate Seeds",
    "Evaluation Seed",
    "Priority",
    "Progress",
    "Command",
    "Active",
    "Notes",
    "Results",
    "Git SHA",
)

STATUS_OPTIONS = (
    "Todo",
    "Ready",
    "Running",
    "Analysis",
    "Done",
    "Blocked",
    "Needs rerun",
)

PRIORITY_OPTIONS = ("Low", "Medium", "High")

CATEGORICAL_COLUMNS = (
    "Campaign",
    "Environment",
    "Method",
    "Primitive Level",
    "Access",
    "Model / Backend",
    "Active",
    "Status",
    "Priority",
)

COLUMN_WIDTHS = {
    "Status": 120,
    "Owner": 180,
    "Experiment ID": 300,
    "Campaign": 180,
    "Environment": 180,
    "Method": 120,
    "Primitive Level": 140,
    "Access": 110,
    "Model / Backend": 180,
    "Replicate Seeds": 160,
    "Evaluation Seed": 140,
    "Priority": 100,
    "Progress": 90,
    "Command": 420,
    "Active": 110,
    "Notes": 320,
    "Results": 220,
    "Git SHA": 120,
}

WRAPPED_COLUMNS = ("Experiment ID", "Command", "Notes", "Results")

assert len(ALL_COLUMNS) == len(set(ALL_COLUMNS))
assert set(GENERATED_COLUMNS).isdisjoint(HUMAN_COLUMNS)
assert set(ALL_COLUMNS) == set(GENERATED_COLUMNS) | set(HUMAN_COLUMNS)
assert set(COLUMN_WIDTHS) == set(ALL_COLUMNS)
assert set(WRAPPED_COLUMNS) <= set(ALL_COLUMNS)
