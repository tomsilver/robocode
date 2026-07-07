"""Robocode primitives: source-importing factory with metadata re-exported.

The source-free primitive metadata (names, file mapping, env-dependence flags,
black-box manifest builder) lives in ``robocode.primitive_specs`` so it stays
importable where this package's source is stripped (the agentic sandbox). The
human-facing descriptions (and ``format_primitives_description``) live in the
host-only ``robocode.primitive_descriptions`` module, which is NOT shipped to the
sandbox. This module adds the heavy factory (``build_primitives``) and re-exports
both sets of names so every existing ``from robocode.primitives import ...`` keeps
working. It is never imported inside the agent sandbox (this package's source is
stripped there); the genplan container keeps ``primitive_descriptions.py`` so the
re-export resolves.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from robocode.primitive_descriptions import (
    PRIMITIVE_DESCRIPTIONS,
    format_primitives_description,
)
from robocode.primitive_specs import (
    ENV_DEPENDENT_PRIMITIVES,
    GENERIC_PRIMITIVE_ATTR,
    PRIMITIVE_NAME_TO_FILE,
    REMOTE_MODULE_PRIMITIVES,
    blackbox_primitive_manifest,
)
from robocode.primitives import crv_motion_planning as crv_motion_planning_module
from robocode.primitives import crv_motion_planning_grasp as crv_grasp_module
from robocode.primitives import csp as csp_module
from robocode.primitives.check_action_collision import check_action_collision
from robocode.primitives.motion_planning import BiRRT
from robocode.utils.bilevel import build_sesame_models

__all__ = [
    "ENV_DEPENDENT_PRIMITIVES",
    "GENERIC_PRIMITIVE_ATTR",
    "PRIMITIVE_DESCRIPTIONS",
    "PRIMITIVE_NAME_TO_FILE",
    "REMOTE_MODULE_PRIMITIVES",
    "blackbox_primitive_manifest",
    "build_primitives",
    "format_primitives_description",
]


# Per-name builders bound to the live env. build_primitives invokes only the
# requested ones, so bilevel_models builds its SeSamE models only when granted
# (they require an env with a bilevel mapping); the others stay cheap.
_PRIMITIVE_BUILDERS: dict[str, Callable[[Any], Any]] = {
    "check_action_collision": lambda env: partial(check_action_collision, env),
    "csp": lambda env: csp_module,
    "crv_motion_planning": lambda env: crv_motion_planning_module,
    "crv_motion_planning_grasp": lambda env: crv_grasp_module,
    "BiRRT": lambda env: BiRRT,
    "bilevel_models": build_sesame_models,
}


def build_primitives(env: Any, names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Build a primitives dict containing only the requested *names*."""
    return {name: _PRIMITIVE_BUILDERS[name](env) for name in names}
