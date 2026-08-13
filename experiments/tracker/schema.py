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
