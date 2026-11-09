"""Independent configuration, intervention, and environment correspondence checks."""

# Bool and integer intervention IDs must remain distinct.
# pylint: disable=unidiomatic-typecheck

from dataclasses import asdict, replace

from .checkpoints import Edit, Program, Trace
from .engine import Action, Configuration, Transition, label, state_data, step
from .environment import check_environment
from .evidence import computation_features
from .explanation import CostMapping, Explanation
from .heuristics import CostConfiguration
from .tasks import TaskCase, TaskSearchSpec, task_environment

ERRORS = (ValueError, KeyError, TypeError, IndexError, RuntimeError)


def intervene(
    state: Configuration, operation: str, value: int = 0, node: int | None = None
) -> Configuration:
    """The abstract edit, independently defined from the concrete edit adapter."""
    frontier, parents = list(state.frontier), dict(state.parents)
    if state.phase != "ready":
        raise ValueError("Interventions require a live search configuration")
    if operation in ("costs", "heuristic"):
        if (
            not isinstance(state, CostConfiguration)
            or type(value) is not int
            or value < 0
        ):
            raise ValueError("Priority edits require nonnegative represented values")
        values = dict(getattr(state, operation))
        if node not in values:
            raise ValueError("Priority intervention node is not represented")
        values[node] = value
        return (
            replace(state, costs=values)
            if operation == "costs"
            else replace(state, heuristic=values)
        )
    if operation == "clear":
        frontier = []
    elif operation == "reverse":
        frontier = list(reversed(frontier))
    elif operation == "drop":
        if type(value) is not int or not 0 <= value < len(frontier):
            raise ValueError("Drop index outside frontier")
        frontier = frontier[:value] + frontier[value + 1 :]
    elif operation == "set":
        if type(value) is not int or value not in (-1, 0, 1):
            raise ValueError("Invalid primitive action")
        if len(frontier) != 1 or not isinstance(frontier[0], Action):
            raise ValueError("Action replacement requires one terminal child")
        old = frontier[0]
        child = Action(value)
        del parents[old]
        parents[child] = state.root
        frontier = [child]
    else:
        raise ValueError("Unknown abstract intervention")
    return replace(state, frontier=tuple(frontier), parents=parents)


def predict(explanation: Explanation, position: int | TaskCase, patch=None):
    """Predict the entire abstract run WITHOUT access to a concrete trace."""
    problem = explanation.problem(position)
    states = [
        problem.initialize(
            problem.abstract(
                position.start if isinstance(position, TaskCase) else position
            )
        )
    ]
    transitions: list[Transition] = []
    before = None
    for index in range(problem.step_budget):
        if patch and index == patch[0]:
            before = states[-1]
            states[-1] = intervene(
                before, patch[1], patch[2], patch[3] if len(patch) > 3 else None
            )
        if states[-1].phase == "done":
            if patch and before is None:
                raise ValueError("Abstract intervention checkpoint was not reached")
            return states, transitions, before
        transition = step(problem, states[-1])
        transitions.append(transition)
        states.append(transition.state)
    raise RuntimeError("Abstract step budget exhausted")


def compare(
    program: Program, explanation: Explanation, trace: Trace, patch=None
) -> dict:
    """Compare every aligned configuration, output, and termination event."""
    states, transitions, before = predict(explanation, trace.input, patch)
    records: list[dict] = []
    result = {
        "position": trace.position,
        "status": "supported",
        "predicted_output": states[-1].output,
        "observed_output": trace.output,
        "alignments": records,
    }
    if trace.case:
        result["case"] = trace.case.name
    try:
        aligned = explanation.aligned(trace)
        for i in range(max(len(states), len(aligned))):
            if i >= len(states) or i >= len(aligned):
                result.update(
                    status="rejected",
                    reason="step_count",
                    counterexample={
                        "step": i,
                        "predicted_checkpoints": len(states),
                        "observed_checkpoints": len(aligned),
                    },
                )
                return result
            snapshot = aligned[i]
            observed = explanation.decode(snapshot, program.argument)
            operation = "initialize" if i == 0 else transitions[i - 1].operation
            record = {
                "step": i,
                "checkpoint": snapshot.site,
                "occurrence": snapshot.occurrence,
                "concrete_index": snapshot.index,
                "concrete_interval": [
                    0 if i == 0 else aligned[i - 1].index,
                    snapshot.index,
                ],
                "operation": operation,
                "predicted": state_data(states[i]),
                "observed": state_data(observed),
                "passed": observed == states[i],
            }
            if i:
                transition = transitions[i - 1]
                record.update(
                    selected=label(transition.selected),
                    expanded=transition.expanded,
                    generated=[label(n) for n in transition.generated],
                    added=[label(n) for n in transition.added],
                    **transition.extra_data(),
                )
            if patch and i == patch[0]:
                if snapshot.before is None:
                    raise ValueError("Concrete intervention was not recorded")
                decoded_before = explanation.decode(
                    replace(snapshot, values=snapshot.before), program.argument
                )
                record["edit_correspondence"] = (
                    decoded_before == before and observed == states[i]
                )
                record["before_edit"] = state_data(decoded_before)
                record["passed"] &= record["edit_correspondence"]
            records.append(record)
            if not record["passed"]:
                result.update(
                    status="rejected",
                    reason="configuration_mismatch",
                    counterexample=record,
                )
                return result
        if states[-1].output != trace.output:
            result.update(status="rejected", reason="output_mismatch")
    except ERRORS as exc:
        result.update(
            status="rejected",
            reason="decode_or_alignment",
            error=f"{type(exc).__name__}: {exc}",
        )
    if not patch and result["status"] == "supported":
        result.update(computation_features(transitions))
    return result


def probes(explanation: Explanation, trace: Trace, argument: str):
    """Enumerate edits from ordinary checkpoints; no patched runs get weights."""
    for i, snapshot in enumerate(explanation.aligned(trace)):
        if snapshot.kind == "exit":
            continue
        state = explanation.decode(snapshot, argument)
        if isinstance(state, CostConfiguration):
            mapping = explanation.mapping
            assert isinstance(mapping, CostMapping)
            for node in state.frontier:
                assert isinstance(node, int)
                for role in ("heuristic", "costs"):
                    values = getattr(state, role)
                    replacements: tuple[int, ...]
                    if role == "heuristic":
                        if len(state.frontier) < 2:
                            continue
                        replacements = (
                            0,
                            max(state.heuristic.values())
                            + max(state.costs.values())
                            + 10,
                        )
                    else:
                        # Raising an unexpanded leaf preserves acyclic parent paths.
                        # Arbitrary cost edits can make a parent cycle and are outside
                        # this bounded intervention family.
                        if node == state.root or node in state.parents.values():
                            continue
                        replacements = (values[node] + 2,)
                    for value in replacements:
                        if value != values[node]:
                            selector = (
                                f"{getattr(mapping, role)}[{node + mapping.offset}]"
                            )
                            yield i, Edit(
                                snapshot.site,
                                snapshot.occurrence,
                                selector,
                                "set",
                                value,
                                role,
                                node,
                            )
        if explanation.spec.mode == "reactive":
            if snapshot.kind != "entry":
                child = state.frontier[0]
                assert isinstance(child, Action)
                old = child.value
                for value in (-1, 0, 1):
                    if value != old:
                        yield i, Edit(
                            snapshot.site,
                            snapshot.occurrence,
                            explanation.mapping.action,
                            "set",
                            value,
                        )
        elif state.frontier:
            variable = explanation.mapping.frontier
            yield i, Edit(snapshot.site, snapshot.occurrence, variable, "clear")
            if len(state.frontier) > 1:
                yield i, Edit(snapshot.site, snapshot.occurrence, variable, "reverse")
                for index in range(len(state.frontier)):
                    yield i, Edit(
                        snapshot.site, snapshot.occurrence, variable, "drop", index
                    )


def unchanged_prefix(ordinary: Trace, patched: Trace, index: int) -> bool:
    """Verify deterministic replay to the intervention, including its pre-edit state."""
    if patched.edits != 1 or len(patched.snapshots) <= index:
        return False
    for i in range(index + 1):
        old, new = ordinary.snapshots[i], patched.snapshots[i]
        if (old.site, old.occurrence, old.kind) != (new.site, new.occurrence, new.kind):
            return False
        if old.values != (new.before if i == index else new.values):
            return False
    return True


def control_path(states: list[dict]) -> list[dict]:
    """Inspect downstream control, excluding the directly edited cost/h values."""
    return [
        {k: s[k] for k in ("root", "frontier", "parents", "phase", "output")}
        for s in states
    ]


def _verify_computation(
    program: Program,
    explanation: Explanation,
    positions,
    traces=None,
    interventions: bool = True,
) -> dict:
    """Verify a supplied explanation; never enumerate or repair its mappings."""
    result: dict = {
        "id": explanation.identifier,
        "status": "supported",
        "ordinary": [],
        "interventions": [],
        "observation_checks": 0,
    }
    try:
        explanation.admissible(program.sites)
    except ValueError as exc:
        return {
            **result,
            "status": "rejected",
            "reason": "admissibility",
            "error": str(exc),
        }
    if not positions:
        return {**result, "status": "unresolved", "reason": "no_inputs"}
    runs = []
    for position in positions:
        try:
            trace = program.observe(position) if traces is None else traces[position]
        except (AssertionError, *ERRORS) as exc:
            return {
                **result,
                "status": "unresolved",
                "reason": "observation_or_execution",
                "position": (
                    position.start if isinstance(position, TaskCase) else position
                ),
                "error": str(exc),
            }
        result["observation_checks"] += 1
        ordinary = compare(program, explanation, trace)
        result["ordinary"].append(ordinary)
        if ordinary["status"] != "supported":
            return {
                **result,
                "status": ordinary["status"],
                "reason": "ordinary",
                "counterexample": ordinary,
            }
        runs.append((trace, ordinary))
    if not interventions:
        return {**result, "status": "unresolved", "reason": "ordinary_only"}
    for trace, ordinary in runs:
        aligned = explanation.aligned(trace)
        informative = 0
        priority_effects = {"costs": 0, "heuristic": 0}
        for i, edit in probes(explanation, trace, program.argument):
            patch = (i, edit.role or edit.operation, edit.value, edit.node)
            predicted, _, _ = predict(explanation, trace.input, patch)
            record = {
                "position": trace.position,
                "step": i,
                "edit": asdict(edit),
                "baseline_output": trace.output,
                "predicted_output": predicted[-1].output,
                "informative": predicted[-1].output != trace.output,
                "predicted_states": [state_data(s) for s in predicted],
            }
            if trace.case:
                record["case"] = trace.case.name
            try:
                patched = program.run(trace.input, edit)
                prefix = unchanged_prefix(trace, patched, aligned[i].index)
                checked = compare(program, explanation, patched, patch)
                record.update(
                    observed_output=patched.output,
                    prefix_preserved=prefix,
                    edit_count=patched.edits,
                    comparison=checked,
                    passed=prefix and checked["status"] == "supported",
                )
                baseline = control_path(
                    [r["observed"] for r in ordinary["alignments"]][i + 1 :]
                )
                record["predicted_control_effect"] = (
                    control_path([state_data(s) for s in predicted][i + 1 :])
                    != baseline
                )
                record["observed_control_effect"] = (
                    control_path(
                        [
                            r["observed"]
                            for r in checked["alignments"]
                            if "observed" in r
                        ][i + 1 :]
                    )
                    != baseline
                )
            except RuntimeError as exc:
                record.update(passed=False, error=str(exc), status="unresolved")
            except ERRORS as exc:
                record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            result["interventions"].append(record)
            if not record["passed"]:
                return {
                    **result,
                    "status": record.get("status", "rejected"),
                    "reason": "intervention",
                    "counterexample": record,
                }
            informative += record["informative"]
            if edit.role:
                priority_effects[edit.role] += record["predicted_control_effect"]
        ordinary["decision_effects"] = informative
        ordinary["priority_control_effects"] = priority_effects
        ordinary["intervention_count"] = sum(
            r.get("case", r["position"])
            == (trace.case.name if trace.case else trace.position)
            for r in result["interventions"]
        )
        ordinary["mechanism_finding"] = (
            "reactive_baseline"
            if explanation.spec.mode == "reactive"
            else (
                "search_supported"
                if ordinary["search_expansions"] and informative
                else (
                    "no_search_expansion_observed"
                    if ordinary["N"] == 0
                    else "unresolved"
                )
            )
        )
        if ordinary["N"] > 0 and not informative:
            return {
                **result,
                "status": "unresolved",
                "reason": "no_decision_effect",
                "position": trace.position,
            }
    result["informative_interventions"] = sum(
        r["informative"] for r in result["interventions"]
    )
    return result


def verify(
    program: Program,
    explanation: Explanation,
    positions,
    traces=None,
    witness_limit: int = 3,
    interventions: bool = True,
) -> dict:
    """Check computational causality and task grounding as separate evidence axes."""
    positions = tuple(positions)
    if isinstance(explanation.spec, TaskSearchSpec):
        if not all(isinstance(case, TaskCase) for case in positions):
            raise ValueError("Task explanations require task inputs")
        checks = [task_environment(explanation.spec, case) for case in positions]
        environment = {
            "status": (
                "not_applicable"
                if explanation.spec.mode == "reactive"
                else (
                    "supported"
                    if all(c["status"] == "supported" for c in checks)
                    else "rejected"
                )
            ),
            "cases": checks,
            "attributed_policy_work": 0,
        }
    else:
        environment = check_environment(explanation.spec, witness_limit)
        if environment["status"] != "supported":
            return {
                "id": explanation.identifier,
                "status": environment["status"],
                "mode": explanation.spec.mode,
                "mechanism_status": "not_tested",
                "ordinary": [],
                "interventions": [],
                "environment": environment,
                "reason": "environment",
                "counterexample": environment,
            }
    result = _verify_computation(program, explanation, positions, traces, interventions)
    result.update(
        mode=explanation.spec.mode,
        mechanism_status=result["status"],
        environment=environment,
    )
    if result["status"] == "supported" and environment["status"] not in (
        "supported",
        "not_applicable",
    ):
        result.update(
            status=environment["status"],
            reason="environment",
            counterexample=environment,
        )
    return result
