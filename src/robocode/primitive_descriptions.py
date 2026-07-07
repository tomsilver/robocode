"""Human-facing primitive descriptions used to build agent prompts.

Kept SEPARATE from ``robocode.primitive_specs`` (the sandbox-safe name/attr
metadata that the in-container render server needs) so these descriptions are
NOT shipped into the agent sandbox. They are used only host-side, when
constructing the prompt, so an agent cannot read the description of a primitive
it was not granted (e.g. the ``bilevel_models`` symbolic structure). The sandbox
mount strips this module (see ``docker_sandbox._copy_src``).
"""

from __future__ import annotations

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
    "bilevel_models": (
        "`primitives['bilevel_models']` gives you the bilevel planning MODELS for "
        "this environment: the symbolic predicates and operators and the "
        "parameterized skills the SeSamE planner is built from, without the planner "
        "itself (you must not call `run_sesame` or any planner/search). These "
        "obs-based queries take the raw observation `obs`:\n"
        "  - `get_abstract_state(obs) -> set[GroundAtom]`: the atoms true now "
        "(e.g. `Holding(robot, block)`, `OnTarget(block)`).\n"
        "  - `get_goal_atoms(obs) -> set[GroundAtom]`: the atoms the goal "
        "requires.\n"
        "  - `operators`: the lifted operators, one per skill, each with `.name` "
        "(the skill name), `.parameters`, `.preconditions`, `.add_effects`, and "
        "`.delete_effects`.\n"
        "  - `skill_names`: the available skill names.\n"
        "  - `get_objects(obs, type_name=None) -> list[Object]`: the objects, "
        "optionally filtered by type name (e.g. `'circle'`).\n"
        "The raw `SesameModels` bundle is `primitives['bilevel_models'].models`: "
        "the skills and their controllers (`.ground(objects).controller`), the "
        "`transition_fn(state, action)` simulator, `state_abstractor`, "
        "`goal_deriver`, and the observation/state converters. The model source is "
        "available; read it for the details."
    ),
}


# Extra guidance appended in black-box mode only. In black-box the planner
# state is not an obs vector but the object-centric view from
# observation_space.devectorize(obs), and the CRV module runs on the host via a
# remote-module proxy; spell that out so the agent calls them correctly. Normal
# mode descriptions stay unchanged.
_BLACKBOX_PRIMITIVE_NOTES: dict[str, str] = {
    "crv_motion_planning": (
        "  Black-box note: build the planner state with "
        "`ocs = observation_space.devectorize(obs)`, then call "
        "`primitives['crv_motion_planning'].plan_crv_actions(ocs, "
        "primitives['crv_motion_planning'].CRVConfig(x, y, theta), "
        "carrying=..., seed=...)`. The module runs on the host; pass the "
        "ObjectCentricState and CRVConfig straight to it."
    ),
    "crv_motion_planning_grasp": (
        "  Black-box note: build the planner state with "
        "`ocs = observation_space.devectorize(obs)` and pass it (plus the "
        "target object name and a "
        "`primitives['crv_motion_planning_grasp'].RelativeGraspPose(...)`) to "
        "`primitives['crv_motion_planning_grasp'].plan_crv_grasp(...)`. The "
        "module runs on the host."
    ),
}


def format_primitives_description(names: list[str], blackbox: bool = False) -> str:
    """Markdown describing the ``primitives`` dict passed to GeneratedApproach.

    Shared by the agentic and llm_genplan prompts so both describe primitives the same
    way. With *blackbox*, appends per-primitive notes (e.g. for the CRV planners, how to
    build the planner state via observation_space.devectorize) without changing the
    normal-mode descriptions.
    """
    if not names:
        return "`primitives` is an empty dict."
    lines = ["`primitives` is a dict with these callables:\n"]
    for name in sorted(names):
        lines.append(f"- {PRIMITIVE_DESCRIPTIONS.get(name, f'`{name}`')}")
        if blackbox and name in _BLACKBOX_PRIMITIVE_NOTES:
            lines.append(_BLACKBOX_PRIMITIVE_NOTES[name])
    listed = ", ".join(f"`{n}`" for n in sorted(names))
    lines.append(
        f"\nIMPORTANT: Your approach MUST use the following primitives: {listed}. "
        "These are essential for solving this environment. Read their descriptions "
        "above and integrate them into your solution."
    )
    return "\n".join(lines)
