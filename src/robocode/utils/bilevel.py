"""Shared helper for building bilevel planning models from a robocode env.

Both the SeSamE baseline (`BilevelPlanningApproach`) and the `bilevel_models`
primitive need the same `SesameModels` bundle for an environment, built from the
`bilevel_env_name` / `bilevel_env_model_kwargs` mapping carried on the env (see
`KinderGeom2DEnv`). This module owns that one construction so the two callers do
not duplicate the mapping read.
"""

from __future__ import annotations

import re
from typing import Any

# Family (bilevel_env_name) -> the object-count kwarg its model builder expects.
_BILEVEL_KWARG_BY_FAMILY: dict[str, str] = {
    "obstruction2d": "num_obstructions",
    "stickbutton2d": "num_buttons",
    "clutteredstorage2d": "num_blocks",
    "clutteredretrieval2d": "num_obstructions",
    "motion2d": "num_passages",
}

# e.g. "kinder/Obstruction2D-o2-v0" -> name="Obstruction2D", count=2.
_ENV_ID_RE = re.compile(r"kinder/([A-Za-z0-9]+2D)-[a-z](\d+)-v\d+")


def infer_bilevel_mapping(env_id: str) -> tuple[str | None, dict[str, int]]:
    """Infer ``(bilevel_env_name, model_kwargs)`` from a kinder 2D env id.

    e.g. ``"kinder/Obstruction2D-o2-v0" -> ("obstruction2d", {"num_obstructions": 2})``.
    Returns ``(None, {})`` for env ids that have no bilevel planning model (3D
    envs, pushpullhook, mazes). Used as a fallback so a plain ``KinderGeom2DEnv``
    (e.g. one the agent builds to test) can use the ``bilevel_models`` primitive
    without the explicit mapping; the env configs still set it explicitly, and a
    test checks the two agree.
    """
    match = _ENV_ID_RE.fullmatch(env_id)
    if match is None:
        return None, {}
    family = match.group(1).lower()
    kwarg = _BILEVEL_KWARG_BY_FAMILY.get(family)
    if kwarg is None:
        return None, {}
    return family, {kwarg: int(match.group(2))}


def build_sesame_models(env: Any) -> Any:
    """Build the `SesameModels` (predicates, operators, skills, transition sim).

    Reads the bilevel env-family name and object-count kwargs off *env*. Fails loudly if
    the env config is missing the mapping rather than planning silently.

    `kinder_bilevel_planning` is imported lazily (it is an optional `bilevel` extra),
    so `import robocode.primitives` works even where the extra is not installed -- e.g.
    a "models OFF" sandbox. Only actually using the bilevel models requires it.
    """
    # pylint: disable=import-outside-toplevel
    from kinder_bilevel_planning.env_models import create_bilevel_planning_models

    # We never run under python -O, so this assert fires as a loud config check.
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
