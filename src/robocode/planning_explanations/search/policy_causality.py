"""Bounded search -> stored plan -> later action experiments.

A readout template (cache attribute, point callable, final-point kind) is
proposed on development runs only and frozen with the bank. The final point is
absent, follows the chain's last cell, or stands in for it: some policies drop
the endpoint cell and append an exact target inside it instead. On each plan
version a template binds three values from the ordinary stored plan: the
endpoint cell, whether the root cell is absent (zero or one leading cell, e.g.
consumed in the same call), and the final point. Those stay fixed when
predicting the stored plan of an edited search.

The readout says nothing about what a policy stores once an edit loses that
endpoint, or once the policy picks another destination; such probes are
unresolved, never counterexamples. Only a wrong prediction for an endpoint the
edited search still reaches refutes the readout. Counterfactual policies receive
the ordinary episode's observations, so these are controlled causal probes, not
alternative closed-loop task-success trials.
"""

import copy
from dataclasses import asdict, dataclass

import numpy as np

from .policy_runner import SearchEdit, Session, action_tuple, plan_attributes
from .policy_verify import ERRORS, abstract_edit, bind, compare_run

CHAIN_HORIZON = 32
DETOUR_CANDIDATES = 3
# Final point: none, after the chain's last cell, or in place of it.
NO_TAIL, TAIL_AFTER, TAIL_REPLACES = 0, 1, 2
TAILS = (NO_TAIL, TAIL_AFTER, TAIL_REPLACES)
CHAIN_RULES = {
    "readout": "complete-parent-chain-v3",
    "bound_per_plan": ["endpoint", "absent leading root cell (0 or 1)", "final point"],
    "final_point": {
        "0": "none",
        "1": "after the chain's last cell",
        "2": (
            "in place of the chain's last cell: the tree cell whose point is "
            "uniquely nearest the final point, and whose parent is the last "
            "stored cell"
        ),
    },
    "producer_probe": (
        "per stored chain, the first of its first three interior cells whose "
        "removal keeps the endpoint per the explanation: drop every frontier "
        "entry of it at the visit just before its first ordinary expansion"
    ),
    "out_of_domain": "endpoint lost or plan rebound under an edit: unresolved",
    "future_calls": CHAIN_HORIZON,
    "observations": "ordinary episode observations held fixed",
    "mediation": (
        "changed-plan-only predicts action; restoring ordinary plan restores action"
    ),
}


@dataclass(frozen=True)
class PlanRule:
    """A small readout grammar, independent of variable meaning."""

    attribute: str
    function: str
    tail: int

    def __post_init__(self):
        if self.tail not in TAILS:
            raise ValueError("Unknown final-point kind")


@dataclass(frozen=True)
class Binding:
    """What a readout binds from one ordinary stored plan, held fixed under edits."""

    skip: int
    endpoint: tuple
    tail: tuple
    replaced: bool = False


def _path(parents, root, endpoint) -> list:
    if endpoint not in parents:
        return []
    path, seen = [], set()
    cell = endpoint
    while cell is not None:
        if cell in seen or cell not in parents:
            raise ValueError("Invalid parent chain")
        seen.add(cell)
        path.append(cell)
        cell = parents[cell]
    path.reverse()
    if path[0] != root:
        raise ValueError("Parent chain does not start at root")
    return path


def read_plan(engine, point, binding: Binding) -> tuple:
    """The stored waypoints a bound readout predicts from a final parent map."""
    cells = _path(engine.parents, engine.root, binding.endpoint)
    if binding.replaced:
        cells = cells[:-1]
    cells = cells[binding.skip :]
    return tuple(tuple(float(v) for v in point(c)) for c in cells) + binding.tail


def _cell_holding(inverse: dict, target) -> tuple:
    """The tree cell whose point is uniquely nearest `target`."""
    ranked = sorted(
        (sum((a - b) ** 2 for a, b in zip(value, target)), cell)
        for value, cell in inverse.items()
    )
    if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        raise ValueError("No unique cell holds the final point")
    return ranked[0][1]


def bind_plan(rule: PlanRule, points, engine, point) -> Binding:
    """Read a stored plan as a *whole* parent chain, optionally without the root."""
    points = tuple(points or ())
    mapped = points[:-1] if rule.tail else points
    if not mapped:
        raise ValueError("No represented path")
    inverse = {}
    for cell in engine.parents:
        value = tuple(float(v) for v in point(cell))
        if value in inverse:
            raise ValueError("Non-injective point readout")
        inverse[value] = cell
    endpoint = inverse[tuple(mapped[-1])]
    tail = points[-1:] if rule.tail else ()
    replaced = rule.tail == TAIL_REPLACES
    if replaced:
        # The final point stands in for one more cell, which the policy dropped.
        last, endpoint = endpoint, _cell_holding(inverse, tail[0])
        if engine.parents.get(endpoint) != last:
            raise ValueError("The final point's cell does not extend the chain")
    for skip in (0, 1):
        binding = Binding(skip, endpoint, tail, replaced)
        if read_plan(engine, point, binding) == points:
            return binding
    raise ValueError("Stored plan is not a complete parent chain")


def propose_rules(attribute, points, engine, functions, names) -> list[dict]:
    """At most three templates per point callable; DEVELOPMENT ONLY."""
    rules = []
    for name in names:
        if name not in functions:
            continue
        for tail in TAILS:
            rule = PlanRule(attribute, name, tail)
            try:
                bind_plan(rule, points, engine, functions[name])
            except ERRORS:
                continue
            rules.append(asdict(rule))
    return rules


def candidate_templates(plans: dict, names) -> list[dict]:
    """Every template the development grammar could propose for these plans."""
    return [
        asdict(PlanRule(attribute, name, tail))
        for attribute in sorted(plans)
        for name in names
        for tail in TAILS
    ]


def _detour(explanation, run, engine, cell, endpoint) -> SearchEdit | None:
    """Drop `cell` before its first expansion if, per the model, the endpoint stays.

    The choice uses the ordinary run and the explanation alone; the live policy's
    response to the edit is unknown when it is made.
    """
    _, values = next(run.capture.frames())
    model = bind(explanation, values, run.capture.callables)
    visit, found = 0, False
    while model.phase == "ready" and not found:
        found = model.step().expanded == cell
        visit += not found
    if not found:
        return None
    edit = SearchEdit(
        run.site,
        run.ordinal,
        visit,
        "drop_cell",
        explanation.mapping.frontier,
        cell=cell,
        heap=not explanation.spec.ordered,
    )
    model = bind(explanation, values, run.capture.callables)
    for _ in range(visit):
        model.step()
    abstract_edit(model, edit)
    while model.phase == "ready":
        model.step()
    kept = endpoint in model.parents and model.reached_goal == engine.reached_goal
    return edit if kept else None


def detour_edits(explanation, run, engine, plans, templates, functions) -> list:
    """Predeclared producer probes, chosen from the ordinary run alone.

    For every stored plan a template reads as a complete chain of this run's tree,
    take the first of its first DETOUR_CANDIDATES interior cells (starting at the
    root's child) whose removal keeps the endpoint according to the explanation,
    and drop all of its frontier entries just before its first expansion. The
    route changes near its start, so a consumer acts on the change within the
    replay horizon.
    """
    chains: list = []
    for template in templates:
        rule = PlanRule(**template)
        points = plans.get(rule.attribute)
        if not points or rule.function not in functions:
            continue
        try:
            binding = bind_plan(rule, points, engine, functions[rule.function])
            path = _path(engine.parents, engine.root, binding.endpoint)
        except ERRORS:
            continue
        if path not in chains:
            chains.append(path)
    edits: list = []
    for path in chains:
        for cell in path[1:-1][:DETOUR_CANDIDATES]:
            try:
                edit = _detour(explanation, run, engine, cell, path[-1])
            except (*ERRORS, StopIteration):
                continue
            if edit is not None:
                if edit not in edits:
                    edits.append(edit)
                break
    return edits


def judge_probe(rule, binding, points, engine, patched, point, observed) -> dict:
    """One edited stored plan against the bound readout's prediction.

    `status` is "replay" (changed exactly as predicted: test later actions),
    "unresolved" or "rejected". Only a kept endpoint can yield a counterexample.
    """
    if (
        binding.endpoint not in patched.parents
        or patched.reached_goal != engine.reached_goal
    ):
        return {"status": "unresolved", "reason": "endpoint_lost"}
    predicted = read_plan(patched, point, binding)
    fields = {
        "predicted_plan": predicted,
        "prediction_matches": predicted == observed,
        "changed": predicted != points,
    }
    if predicted == observed:
        if predicted == points:
            return {**fields, "status": "unresolved", "reason": "plan_unchanged"}
        return {**fields, "status": "replay"}
    try:
        other = bind_plan(rule, observed, patched, point)
    except ERRORS:
        return {
            **fields,
            "status": "rejected",
            "reason": "stored_plan_prediction_mismatch",
        }
    # Still a complete chain of the edited tree, read with other bound values:
    # the policy chose another destination, which this readout does not model.
    moved = (other.endpoint, other.tail) != (binding.endpoint, binding.tail)
    return {
        **fields,
        "status": "unresolved",
        "reason": "destination_changed" if moved else "root_offset_changed",
        "rebound": {"endpoint": other.endpoint, "root_absent": other.skip},
    }


def _install(policy, attribute, points) -> None:
    """Replace the actual cache, preserving list aliases and point types."""
    original = getattr(policy, attribute)
    sample = original[0] if len(original) else ()
    values: list
    if isinstance(sample, np.ndarray):
        values = [np.array(p, dtype=sample.dtype) for p in points]
    elif isinstance(sample, list):
        values = [list(p) for p in points]
    else:
        values = list(points)
    if isinstance(original, list):
        original[:] = values
    else:
        setattr(policy, attribute, tuple(values))


def _same(left, right) -> bool:
    return len(left) == len(right) and all(
        abs(a - b) <= 1e-9 for a, b in zip(left, right)
    )


def _consumed(old, new) -> bool:
    """Unchanged or a suffix (including an exhausted cache)."""
    return len(new) <= len(old) and (not new or old[-len(new) :] == new)


class ChainReplay:
    """Carry one search intervention into future calls, with mediation controls."""

    def __init__(
        self, evidence, attribute, ordinary, patched, predicted, baseline
    ) -> None:
        self.evidence, self.attribute = evidence, attribute
        self.policies = {
            "baseline": copy.deepcopy(ordinary),
            "search_edit": copy.deepcopy(patched),
            "plan_only": copy.deepcopy(ordinary),
            "restored": copy.deepcopy(patched),
        }
        _install(self.policies["plan_only"], attribute, predicted)
        _install(self.policies["restored"], attribute, baseline)
        evidence.update(status="pending", future=[])

    def step(self, program, domain, call, state, extra, search_sites) -> None:
        """Advance all four branches on the *same* ordinary observation."""
        record = self.evidence
        if record["status"] != "pending" or call.method != "get_action":
            return
        actions = {}
        try:
            for name, policy in self.policies.items():
                before = plan_attributes(policy).get(self.attribute, ())
                session = Session(capture=False)
                output = program.call(
                    policy, call.method, (domain.copy_state(state), *extra), session
                )
                after = plan_attributes(policy).get(self.attribute, ())
                if any(r.site in search_sites for r in session.runs):
                    record.update(
                        status="unresolved", reason="replanned_before_witness"
                    )
                    return
                if not _consumed(before, after):
                    record.update(
                        status="unresolved", reason="cache_replaced_before_witness"
                    )
                    return
                actions[name] = action_tuple(output)
        except ERRORS as exc:
            record.update(status="unresolved", reason="replay_error", error=str(exc))
            return
        witness = {
            "call": call.index,
            "ordinary_action": call.action,
            "observed_action": actions["search_edit"],
            "predicted_action": actions["plan_only"],
            "restored_action": actions["restored"],
            "baseline_replay_matches": _same(actions["baseline"], call.action),
            "changed": not _same(actions["search_edit"], call.action),
            "prediction_matches": _same(actions["plan_only"], actions["search_edit"]),
            "restoration_matches": _same(actions["restored"], call.action),
        }
        record["future"].append(witness)
        if not witness["baseline_replay_matches"]:
            record.update(status="unresolved", reason="baseline_replay_mismatch")
        elif not witness["prediction_matches"] or not witness["restoration_matches"]:
            record.update(status="rejected", reason="plan_does_not_mediate_action")
        elif witness["changed"]:
            record.update(status="supported", reason="search_plan_action_witness")
        elif len(record["future"]) >= CHAIN_HORIZON:
            record.update(status="unresolved", reason="no_action_effect_within_horizon")


def check_plan_chain(
    rule, points, engine, functions, explanation, executions, ordinary
) -> tuple[dict, list]:
    """Predict each intervention's complete cache, then schedule future replays."""
    rule = PlanRule(**rule)
    evidence: dict = {"rule": asdict(rule), "status": "unresolved", "probes": []}
    replays: list[ChainReplay] = []
    try:
        point = functions[rule.function]
        binding = bind_plan(rule, points, engine, point)
    except ERRORS as exc:
        # This template does not read the ordinary plan: no claim, no evidence.
        evidence.update(
            status="not_applicable", reason="ordinary_readout", error=str(exc)
        )
        return evidence, replays
    evidence.update(
        endpoint=binding.endpoint,
        root_absent=binding.skip,
        fixed_tail=binding.tail,
        endpoint_replaced_by_final_point=binding.replaced,
        ordinary_plan=points,
    )
    for edit, executed, search_check in executions:
        probe: dict = {"edit": search_check["edit"], "status": "unresolved"}
        evidence["probes"].append(probe)
        if not search_check["passed"] or executed.get("error"):
            probe.update(reason="search_edit_failed", error=executed.get("error"))
            continue
        _, patched = compare_run(explanation, executed["target"], edit)
        if patched is None:
            probe.update(reason="search_prediction_missing")
            continue
        observed = executed["plans"].get(rule.attribute)
        probe["observed_plan"] = observed
        try:
            outcome = judge_probe(
                rule, binding, points, engine, patched, point, observed
            )
        except ERRORS as exc:
            probe.update(reason="readout_error", error=str(exc))
            continue
        if outcome.pop("status") == "replay":
            probe.update(outcome)
            replays.append(
                ChainReplay(
                    probe,
                    rule.attribute,
                    ordinary,
                    executed["policy"],
                    outcome["predicted_plan"],
                    points,
                )
            )
            continue
        probe.update(outcome)
        if outcome["reason"] == "stored_plan_prediction_mismatch":
            probe["status"] = "rejected"
    return evidence, replays


def finish_chains(versions) -> None:
    """Chain and plan-version verdicts once every replay has ended.

    A failed prediction refutes a readout even if another probe had a witness.
    A version is "unmatched" when no frozen readout reads it at all.
    """
    for version in versions:
        applicable = []
        for chain in version["chains"]:
            if chain["status"] == "not_applicable":
                continue
            applicable.append(chain)
            for probe in chain["probes"]:
                if probe["status"] == "pending":
                    probe.update(
                        status="unresolved", reason="episode_ended_before_witness"
                    )
            statuses = {p["status"] for p in chain["probes"]}
            if "rejected" in statuses:
                chain.update(status="rejected", reason="counterexample")
            elif "supported" in statuses:
                chain.update(status="supported", reason="search_plan_action_witness")
            elif not chain["probes"]:
                chain.update(status="unresolved", reason="producer_not_probed")
            else:
                chain.update(status="unresolved", reason="no_complete_chain_witness")
        statuses = {c["status"] for c in applicable}
        if "supported" in statuses:
            version.update(
                causal_status="supported", causal_reason="complete_chain_witness"
            )
        elif statuses == {"rejected"}:
            version.update(
                causal_status="rejected", causal_reason="readout_counterexample"
            )
        elif applicable:
            version.update(
                causal_status="unresolved", causal_reason="no_complete_chain_witness"
            )
        else:
            version.update(
                causal_status="unmatched",
                causal_reason=(
                    "no_complete_chain_readout"
                    if version["chains"]
                    else "no_frozen_readout"
                ),
            )
