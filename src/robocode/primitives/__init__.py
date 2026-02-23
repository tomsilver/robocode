"""Robocode primitives — canonical registry of primitive name → file mappings."""

# Mapping from primitive name (as used in the primitives dict) to the source
# file basename (without .py) under ``src/robocode/primitives/``.
PRIMITIVE_NAME_TO_FILE: dict[str, str] = {
    "check_action_collision": "check_action_collision",
    "render_state": "render_state",
    "csp": "csp",
    "BiRRT": "motion_planning",
}
