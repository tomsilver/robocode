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

# Light semantic fills keep dropdown cells scannable without competing with the
# table header. Google Sheets does not expose dropdown-option chip colors through
# its API, so synchronization applies these as conditional cell formats.
SEMANTIC_COLORS = {
    "Method": {
        "agentic": "D9EAF7",
        "agentic_cdl": "D0E0E3",
        "agentic_per_instance": "D9EAD3",
        "best_of_k": "FFF2CC",
        "bilevel_planning": "E4D7F5",
        "llm_genplan": "FCE5CD",
        "random": "E7E7E7",
    },
    "Primitive Level": {
        "none": "E7E7E7",
        "low_level": "CFE2F3",
        "bilevel": "E4D7F5",
    },
    "Access": {
        "whitebox": "D9EAD3",
        "blackbox": "FCE5CD",
    },
    "Model / Backend": {
        "claude_opus48": "E4D7F5",
        "claude_opus5": "E4D7F5",
        "claude_sonnet46": "CFE2F3",
        "claude_sonnet5": "CFE2F3",
        "claude_haiku45": "D9EAD3",
        "claude_ollama_qwen": "D0E0E3",
        "opencode_gpt4omini": "FCE5CD",
        "opencode_gpt54": "F9CB9C",
        "opencode_gpt5nano": "FFF2CC",
        "opencode_qwen": "D0E0E3",
    },
    "Priority": {
        "Low": "D9EAD3",
        "Medium": "FFF2CC",
        "High": "F4CCCC",
    },
    "Active": {
        "TRUE": "D9EAD3",
        "FALSE": "E7E7E7",
    },
    "Status": {
        "Todo": "E7E7E7",
        "Ready": "CFE2F3",
        "Running": "FFF2CC",
        "Analysis": "E4D7F5",
        "Done": "D9EAD3",
        "Blocked": "F4CCCC",
        "Needs rerun": "FCE5CD",
    },
}
