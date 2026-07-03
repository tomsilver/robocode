"""Source-free primitive registry (sandbox-safe metadata only).

Holds the names, file mapping, env-dependence flags, and the black-box manifest
builder for the robocode primitives, but imports NO primitive implementation
(and no kinder or environment code). It depends only on the standard library and
typing, so it stays importable in the sandbox (the in-container render server
needs this metadata to rebuild the primitives dict).

The human-facing *descriptions* live in ``robocode.primitive_descriptions``
instead, used only host-side to build prompts and NOT shipped into the sandbox,
so an agent cannot read the description of a primitive it was not granted. The
heavy factory (``build_primitives``) lives in ``robocode.primitives``.
"""

from __future__ import annotations

from typing import Any

# Mapping from primitive name (as used in the primitives dict) to the source
# file basename (without .py) under ``src/robocode/primitives/``.
PRIMITIVE_NAME_TO_FILE: dict[str, str] = {
    "check_action_collision": "check_action_collision",
    "csp": "csp",
    "crv_motion_planning": "crv_motion_planning",
    "crv_motion_planning_grasp": "crv_motion_planning_grasp",
    "BiRRT": "motion_planning",
    "bilevel_models": "bilevel_models",
}

# Primitives whose construction needs the live environment. In black-box mode
# the sandbox has no env source, so these run on the host via the env server
# (see env_client.BlackboxEnv) and their source is NOT copied into the sandbox:
# it would not import (it imports the hidden env) and it would leak the env
# structure. Every other primitive is generic and imported directly.
# ``bilevel_models`` is env-bound too: it builds the SeSamE models for the live
# env. Full black-box exposure (proxying its object methods over the handle
# protocol) is deferred; for now it runs clearbox/local like check_action_collision.
ENV_DEPENDENT_PRIMITIVES: frozenset[str] = frozenset(
    {"check_action_collision", "bilevel_models"}
)

# Primitives whose whole module runs on the host in black-box mode. Their
# source imports the stripped kinder.envs.* (kinematic2d, utils), so just like
# ENV_DEPENDENT_PRIMITIVES they are NOT copied into the sandbox; but unlike the
# per-callable host proxies, the agent reaches the ENTIRE module via a
# remote-module proxy (env_client._RemoteModule). The sandbox calls e.g.
# primitives['crv_motion_planning'].plan_crv_actions(ocs, cfg, ...) and the call,
# its ObjectCentricState/CRVConfig arguments, and its result all travel over the
# generic remote-handle protocol so the host executes the planner against the
# real env source.
REMOTE_MODULE_PRIMITIVES: frozenset[str] = frozenset(
    {"crv_motion_planning", "crv_motion_planning_grasp"}
)

# For generic primitives, the attribute to pull from the source module named in
# PRIMITIVE_NAME_TO_FILE. None means the primitive IS the module object.
GENERIC_PRIMITIVE_ATTR: dict[str, str | None] = {
    "csp": None,
    "crv_motion_planning": None,
    "crv_motion_planning_grasp": None,
    "BiRRT": "BiRRT",
}


def blackbox_primitive_manifest(
    names: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Describe how a black-box sandbox should build each requested primitive.

    Returns a JSON-serializable spec list for ``env_spaces.json`` that
    ``env_client.BlackboxEnv.make_primitives`` consumes to reconstruct the same
    dict ``build_primitives`` produces at eval time. Env-dependent primitives
    become host proxies (run on the host via the env server); remote-module
    primitives become whole-module proxies (the agent calls into them over the
    wire); generic ones name the source module (and attribute) the sandbox
    imports from its copy.
    """
    manifest: list[dict[str, Any]] = []
    for name in names:
        if name in ENV_DEPENDENT_PRIMITIVES:
            manifest.append({"name": name, "kind": "host_proxy"})
        elif name in REMOTE_MODULE_PRIMITIVES:
            manifest.append({"name": name, "kind": "remote_module", "module": name})
        else:
            manifest.append(
                {
                    "name": name,
                    "kind": "generic",
                    "module": PRIMITIVE_NAME_TO_FILE[name],
                    "attr": GENERIC_PRIMITIVE_ATTR[name],
                }
            )
    return manifest
