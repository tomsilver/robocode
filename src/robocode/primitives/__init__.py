"""Robocode primitives — source-importing factory; metadata re-exported.

The primitive metadata (names, file mapping, descriptions, manifest/description
builders) lives in the source-free ``robocode.primitive_specs`` module so it
stays importable where this package's source is stripped (the agentic sandbox).
This module adds the heavy factory (``build_primitives``), which imports the
actual primitive implementations, and re-exports the metadata names so every
existing ``from robocode.primitives import ...`` keeps working.
"""

from __future__ import annotations

from functools import partial
from typing import Any

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
    "PRIMITIVE_NAME_TO_FILE",
    "REMOTE_MODULE_PRIMITIVES",
    "blackbox_primitive_manifest",
    "build_primitives",
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
