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
        "stays in place)."
    ),
    "csp": (
        "`csp` is a module providing a constraint satisfaction problem (CSP) "
        "solver. Key classes:\n"
        "  - `csp.CSPVariable(name, domain)`: a variable with a "
        "`gymnasium.spaces.Space` domain.\n"
        "  - `csp.FunctionalCSPConstraint(name, variables, fn)`: a "
        "constraint where `fn(*vals) -> bool`.\n"
        "  - `csp.CSP(variables, constraints, cost=None)`: the problem.\n"
        "  - `csp.FunctionalCSPSampler(fn, csp, sampled_vars)`: a "
        "sampler where `fn(current_vals, rng) -> dict | None`.\n"
        "  - `csp.RandomWalkCSPSolver(seed)`: a solver whose "
        "`.solve(csp, initialization, samplers)` returns a satisfying "
        "assignment or None.\n"
        "  - `csp.CSPCost(name, variables, cost_fn)`: an optional cost to "
        "minimize.\n"
        "  - `csp.LogProbCSPConstraint(name, variables, logprob_fn, "
        "threshold)`: a constraint from log probabilities.\n"
        "  Accessed via `primitives['csp']`, e.g. "
        "`primitives['csp'].CSPVariable(...)`."
    ),
    "crv_motion_planning": (
        "`crv_motion_planning` is a module with generic CRV robot motion "
        "planners. `plan_crv_actions(state, target_config, carrying=..., ...)` "
        "takes an object-centric state and a target `CRVConfig` and returns a "
        "collision-free action sequence, or None if no plan is found; "
        "`carrying=True` makes planning holding-aware. Wrappers "
        "`plan_crv_base_actions(...)` and `plan_crv_holding_actions(...)` are "
        "also available. The module exports `CRVConfig`, `CRVActionLimits`, and "
        "helpers to convert between pose plans and action plans."
    ),
    "crv_motion_planning_grasp": (
        "`crv_motion_planning_grasp` is a module that plans one CRV grasp "
        "maneuver. `plan_crv_grasp(state, target_object, relative_grasp_pose, "
        "grasping_arm_length, ...)` returns collision-free grasp waypoints, or "
        "raises `SuctionFailedEmptySpaceError` or "
        "`SuctionFailedNoCollisionFreePathError`."
    ),
    "BiRRT": (
        "`BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng, "
        "num_attempts, num_iters, smooth_amt)` is a bidirectional RRT motion "
        "planner; `birrt.query(start, goal)` returns a collision-free path "
        "(list of states) or None. "
        "`sample_fn(state) -> state` samples a random state, "
        "`extend_fn(s1, s2) -> Iterable[state]` interpolates between states, "
        "`collision_fn(state) -> bool` returns True if state is in collision, "
        "`distance_fn(s1, s2) -> float` returns distance between states, "
        "`rng` is a `np.random.Generator`."
    ),
    "bilevel_models": (
        "`primitives['bilevel_models']` gives you the `SesameModels` bundle for this "
        "environment: the symbolic predicates and operators and the parameterized "
        "skills the SeSamE planner is built from, without the planner itself "
        "(`run_sesame` and any planner/search are rejected at load time). For a "
        "fixed-count environment it IS the bundle; for a VARIABLE object count the "
        "models depend on the count, so it is instead a count-dispatching accessor: "
        "`primitives['bilevel_models'].models_for_state(state)` returns the bundle "
        "for the current instance's count. The bundle's fields:\n"
        "  - `predicates`, `types`: the domain's predicates (classifiers) and "
        "object types.\n"
        "  - `skills`: a set of `LiftedSkill`, one per operator, each pairing a "
        "symbolic operator with an executable controller. `.operator` is the "
        "action's symbolic template (`.name`, typed `.parameters`, "
        "`.preconditions` that must hold to apply it, and "
        "`.add_effects`/`.delete_effects` describing how the atoms change). "
        "`operators` lists these directly. `.controller` is the policy that "
        "executes the "
        "skill: `.ground(objects)` binds it to specific objects, and a grounded "
        "controller takes a continuous parameter (its `params_space`, e.g. a "
        "grasp offset) and produces the low-level actions.\n"
        "  - `state_abstractor(state)`: its `.atoms` are the ground atoms true in "
        "`state` (e.g. `Holding(robot, block)`, `OnTarget(block)`).\n"
        "  - `goal_deriver(state)`: its `.atoms` are the atoms the goal requires.\n"
        "  - `transition_fn(state, action) -> state`: a simulator of the "
        "environment dynamics.\n"
        "  - `observation_to_state(obs) -> state`: converts a raw observation "
        "vector into the object-centric `state` the fields above take; iterating a "
        "`state` yields its objects, each with a `.type` (whose `.name` is e.g. "
        "`'circle'`).\n"
        "The full model source is readable. Each controller's policy is in the "
        "`kinder_models` package (an env's controllers are in "
        "`.../envs/<name>/parameterized_skills.py`), reachable via "
        "`inspect.getsource(skill.controller)`."
    ),
}


# Extra guidance appended in black-box mode only. In black-box with a flat-vector
# observation the planner state is the object-centric view from
# observation_space.devectorize(obs); the CRV module runs on the host via a
# remote-module proxy. Normal-mode descriptions stay unchanged.
_BLACKBOX_PRIMITIVE_NOTES: dict[str, str] = {
    "crv_motion_planning": (
        "  Black-box note: the planner state is the object-centric view "
        "`observation_space.devectorize(obs)`. "
        "`primitives['crv_motion_planning'].plan_crv_actions(ocs, "
        "primitives['crv_motion_planning'].CRVConfig(x, y, theta), "
        "carrying=..., seed=...)` runs on the host and takes the "
        "ObjectCentricState and CRVConfig directly."
    ),
    "crv_motion_planning_grasp": (
        "  Black-box note: the planner state is the object-centric view "
        "`observation_space.devectorize(obs)`. "
        "`primitives['crv_motion_planning_grasp'].plan_crv_grasp(...)` runs on "
        "the host and takes that state, the target object name, and a "
        "`primitives['crv_motion_planning_grasp'].RelativeGraspPose(...)`."
    ),
}

# Black-box notes for a variable-count env: the observation and get_state() are
# already ObjectCentricState objects (there is no devectorize), so the state is
# passed straight to the host-side CRV planners.
_BLACKBOX_PRIMITIVE_NOTES_OBJECT_CENTRIC: dict[str, str] = {
    "crv_motion_planning": (
        "  Black-box note: the observation and `env.get_state()` are already "
        "`ObjectCentricState` objects. "
        "`primitives['crv_motion_planning'].plan_crv_actions(state, "
        "primitives['crv_motion_planning'].CRVConfig(x, y, theta), "
        "carrying=..., seed=...)` runs on the host and takes the state directly."
    ),
    "crv_motion_planning_grasp": (
        "  Black-box note: the observation and `env.get_state()` are already "
        "`ObjectCentricState` objects. "
        "`primitives['crv_motion_planning_grasp'].plan_crv_grasp(...)` runs on "
        "the host and takes that state, the target object name, and a "
        "`primitives['crv_motion_planning_grasp'].RelativeGraspPose(...)`."
    ),
}


def format_primitives_description(
    names: list[str], blackbox: bool = False, object_centric: bool = False
) -> str:
    """Markdown describing the ``primitives`` dict passed to GeneratedApproach.

    Shared by the agentic and llm_genplan prompts so both describe primitives the same
    way. With *blackbox*, appends per-primitive CRV-planner notes; *object_centric*
    selects the variant that passes the state directly (no ``devectorize``) for a
    variable-count env. Normal-mode descriptions are unchanged.
    """
    if not names:
        return "`primitives` is an empty dict."
    blackbox_notes = (
        _BLACKBOX_PRIMITIVE_NOTES_OBJECT_CENTRIC
        if object_centric
        else _BLACKBOX_PRIMITIVE_NOTES
    )
    lines = ["`primitives` is a dict with these callables:\n"]
    for name in sorted(names):
        lines.append(f"- {PRIMITIVE_DESCRIPTIONS.get(name, f'`{name}`')}")
        if blackbox and name in blackbox_notes:
            lines.append(blackbox_notes[name])
    return "\n".join(lines)
