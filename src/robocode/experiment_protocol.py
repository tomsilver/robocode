"""Shared validation for experiment-tracker protocol fields."""

import re
from typing import Any

EXPERIMENT_ID_SEPARATOR = "__"
EXPERIMENT_ID_DIGEST_LENGTH = 8
EXPERIMENT_ID_PATTERN = re.compile(
    rf"[a-z0-9][a-z0-9_]*{EXPERIMENT_ID_SEPARATOR}"
    rf"[0-9a-f]{{{EXPERIMENT_ID_DIGEST_LENGTH}}}"
)


def validate_eval_seed(value: Any) -> int:
    """Return a nonnegative integer evaluation seed."""
    if value is None:
        raise ValueError(
            "eval_seed must be set explicitly; it must not follow replicate_seed"
        )
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"eval_seed must be a nonnegative integer, got {value!r}")
    return value
