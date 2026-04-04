"""Robocode primitives — canonical registry, descriptions, and factory."""

from __future__ import annotations

from functools import partial
from typing import Any

from robocode.primitives import crv_motion_planning as crv_motion_planning_module
from robocode.primitives import crv_motion_planning_grasp as crv_grasp_module
from robocode.primitives import csp as csp_module
from robocode.primitives.check_action_collision import check_action_collision
from robocode.primitives.motion_planning import BiRRT

# Mapping from primitive name (as used in the primitives dict) to the source
# file basename (without .py) under ``src/robocode/primitives/``.
PRIMITIVE_NAME_TO_FILE: dict[str, str] = {
    "check_action_collision": "check_action_collision",
    "csp": "csp",
    "crv_motion_planning": "crv_motion_planning",
    "crv_motion_planning_grasp": "crv_motion_planning_grasp",
    "BiRRT": "motion_planning",
}

# Descriptions shown to the Claude agent so it knows how to call each
# primitive. Keyed by the same names as PRIMITIVE_NAME_TO_FILE.
PRIMITIVE_DESCRIPTIONS: dict[str, str] = {
    "check_action_collision": (
        "`check_action_collision(state, action) -> bool` returns True when "
        "taking `action` in `state` would cause a collision (i.e. the agent "
        "stays in place). Use it to avoid wasted steps \u2014 e.g. in search or "
        "planning algorithms, skip actions that collide."
    ),
    "csp": (
        "`csp` is a module providing a constraint satisfaction problem (CSP) "
        "solver. Use it to sample configurations (e.g. placements, grasps) "
        "that satisfy constraints (e.g. collision-free). Key classes:\n"
        "  - `csp.CSPVariable(name, domain)` \u2014 a variable with a "
        "`gymnasium.spaces.Space` domain.\n"
        "  - `csp.FunctionalCSPConstraint(name, variables, fn)` \u2014 a "
        "constraint where `fn(*vals) -> bool`.\n"
        "  - `csp.CSP(variables, constraints, cost=None)` \u2014 the problem.\n"
        "  - `csp.FunctionalCSPSampler(fn, csp, sampled_vars)` \u2014 a "
        "sampler where `fn(current_vals, rng) -> dict | None`.\n"
        "  - `csp.RandomWalkCSPSolver(seed)` \u2014 solver; call "
        "`.solve(csp, initialization, samplers)` to get a satisfying "
        "assignment or None.\n"
        "  - `csp.CSPCost(name, variables, cost_fn)` \u2014 optional cost to "
        "minimize.\n"
        "  - `csp.LogProbCSPConstraint(name, variables, logprob_fn, "
        "threshold)` \u2014 constraint from log probabilities.\n"
        "  Access via `primitives['csp']`, e.g. "
        "`primitives['csp'].CSPVariable(...)`."
    ),
    "crv_motion_planning": (
        "`crv_motion_planning` is a module with generic CRV robot motion "
        "planners. Use `plan_crv_actions(...)` with object-centric state and a "
        "target `CRVConfig` to get collision-free action sequences, and set "
        "`carrying=True` for holding-aware planning. Compatibility wrappers "
        "`plan_crv_base_actions(...)` and `plan_crv_holding_actions(...)` are "
        "also available. The module exports `CRVConfig`, `CRVActionLimits`, and "
        "helpers to convert between pose plans and action plans."
    ),
    "crv_motion_planning_grasp": (
        "`crv_motion_planning_grasp` is a module that plans one CRV grasp "
        "maneuver from an object-centric state, a target object, a relative "
        "grasp pose, and a grasp arm length. Use `plan_crv_grasp(...)` to get "
        "collision-free grasp waypoints, and handle the explicit suction "
        "failure errors."
    ),
    "BiRRT": (
        "`BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng, "
        "num_attempts, num_iters, smooth_amt)` \u2014 Bidirectional RRT motion "
        "planner. Construct one, then call `birrt.query(start, goal)` to get "
        "a collision-free path (list of states) or None. "
        "`sample_fn(state) -> state` samples a random state, "
        "`extend_fn(s1, s2) -> Iterable[state]` interpolates between states, "
        "`collision_fn(state) -> bool` returns True if state is in collision, "
        "`distance_fn(s1, s2) -> float` returns distance between states, "
        "`rng` is a `np.random.Generator`."
    ),
}


def _all_primitives(env: Any) -> dict[str, Any]:
    """Return the full primitives dict for a given environment."""
    return {
        "check_action_collision": partial(check_action_collision, env),
        "csp": csp_module,
        "crv_motion_planning": crv_motion_planning_module,
        "crv_motion_planning_grasp": crv_grasp_module,
        "BiRRT": BiRRT,
    }


def build_primitives(env: Any, names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Build a primitives dict containing only the requested *names*."""
    all_prims = _all_primitives(env)
    return {name: all_prims[name] for name in names}
