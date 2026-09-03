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


# Primitive levels that grant a primitive bound to the live environment (see
# ENV_DEPENDENT_PRIMITIVES: bilevel grants bilevel_models, low_level grants
# check_action_collision). Such a primitive closes over the env at eval time, so a
# black-box program granted one could read the environment out of its closure --
# build_primitives refuses the combination, and campaigns must not schedule it.
_ENV_BOUND_PRIMITIVE_LEVELS = frozenset({"bilevel", "low_level"})


def is_valid_experiment(config: ExperimentConfig) -> tuple[bool, str | None]:
    """Return whether a Hydra-canonicalized condition is runnable."""
    blackbox = config.values.get("approach.blackbox") is True
    strict = config.values.get("approach.blackbox_strict") is True
    level = config.values.get("primitive_level")
    if strict and not blackbox:
        return False, "strict blackbox runtime requires blackbox access"
    if strict and level != "none":
        return False, "strict blackbox runtime requires primitive_level=none"
    if blackbox and level == "bilevel":
        return False, "bilevel_models unavailable under blackbox"
    if blackbox and level in _ENV_BOUND_PRIMITIVE_LEVELS:
        return False, "env-bound primitives unavailable under blackbox"
    return True, None
