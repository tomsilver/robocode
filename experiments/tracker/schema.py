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

ALL_COLUMNS = GENERATED_COLUMNS + HUMAN_COLUMNS

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
