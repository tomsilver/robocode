"""Explicit validity constraints for campaign experiment conditions."""

from dataclasses import dataclass
from typing import TypeAlias

Scalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class ExperimentConfig:
    """One condition selected by a campaign before seed expansion."""

    campaign: str
    values: dict[str, Scalar]
    replicate_seeds: tuple[int, ...]


def is_valid_experiment(config: ExperimentConfig) -> tuple[bool, str | None]:
    """Return whether a Hydra-canonicalized condition is runnable."""
    blackbox = config.values.get("approach.blackbox") is True
    bilevel = config.values.get("primitive_level") == "bilevel"
    if blackbox and bilevel:
        return False, "bilevel_models unavailable under blackbox"
    return True, None
