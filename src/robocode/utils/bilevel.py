"""Shared helper for building bilevel planning models from a robocode env.

Both the SeSamE baseline (`BilevelPlanningApproach`) and the `bilevel_models`
primitive need the same `SesameModels` bundle for an environment, built from the
`bilevel_env_name` / `bilevel_env_model_kwargs` mapping carried on the env (see
`KinderGeom2DEnv`). This module owns that one construction so the two callers do
not duplicate the mapping read.
"""

from __future__ import annotations

from typing import Any

from kinder_bilevel_planning.env_models import create_bilevel_planning_models


def build_sesame_models(env: Any) -> Any:
    """Build the `SesameModels` (predicates, operators, skills, transition sim).

    Reads the bilevel env-family name and object-count kwargs off *env*. Fails loudly if
    the env config is missing the mapping rather than planning silently.
    """
    assert env.bilevel_env_name is not None, (
        "bilevel_env_name is not set on the environment; add bilevel_env_name and "
        "bilevel_env_model_kwargs to the env config to use bilevel planning models."
    )
    return create_bilevel_planning_models(
        env.bilevel_env_name,
        env.observation_space,
        env.action_space,
        **env.bilevel_env_model_kwargs,
    )
