"""Shared columns for generated experiment files and the Google Sheet."""

GENERATED_COLUMNS = (
    "Experiment ID",
    "Campaign",
    "Environment",
    "Method",
    "Primitive Level",
    "Access",
    "Model / Backend",
    "Seeds",
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
# human-owned field, but belongs next to Seeds so it is visible before the long
# generated Command column.
ALL_COLUMNS = (
    "Experiment ID",
    "Campaign",
    "Environment",
    "Method",
    "Primitive Level",
    "Access",
    "Model / Backend",
    "Seeds",
    "Priority",
    "Command",
    "Active",
    "Owner",
    "Status",
    "Progress",
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
