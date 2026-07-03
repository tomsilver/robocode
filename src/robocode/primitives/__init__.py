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
from robocode.primitives.bilevel_models import BilevelModels
from robocode.primitives.check_action_collision import check_action_collision
from robocode.primitives.motion_planning import BiRRT

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


def _all_primitives(env: Any) -> dict[str, Any]:
    """Return the full primitives dict for a given environment."""
    return {
        "check_action_collision": partial(check_action_collision, env),
        "csp": csp_module,
        "crv_motion_planning": crv_motion_planning_module,
        "crv_motion_planning_grasp": crv_grasp_module,
        "BiRRT": BiRRT,
        "bilevel_models": BilevelModels(env),
    }


def build_primitives(env: Any, names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Build a primitives dict containing only the requested *names*."""
    all_prims = _all_primitives(env)
    return {name: all_prims[name] for name in names}
