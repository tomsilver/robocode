"""Bounded search -> stored plan -> later action experiments.

The readout is conditional on the ordinary endpoint and optional final target.
It reconstructs an existing parent chain; it does not discover a new route or
explain target selection. Templates are proposed on development runs only.
Counterfactual policies receive the ordinary episode's observations, so these
are controlled causal probes, not alternative closed-loop task-success trials.
"""

import copy
from dataclasses import asdict, dataclass

import numpy as np

from .policy_runner import Session, action_tuple, plan_attributes
from .policy_verify import ERRORS, compare_run

CHAIN_HORIZON = 32
CHAIN_RULES = {
    "readout": "fixed-terminal-parent-chain-v1",
    "skip_root": [0, 1],
    "fixed_tail_points": [0, 1],
    "extra_search_probe": "clear frontier at visit zero",
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
    skip_root: int
    tail: int


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


def read_plan(rule, engine, point, endpoint, tail) -> tuple:
    """Predict actual stored waypoints from the abstract final parent map."""
    cells = _path(engine.parents, engine.root, endpoint)[rule.skip_root :]
    return tuple(tuple(float(v) for v in point(c)) for c in cells) + tail


def context(rule, points, engine, point) -> tuple:
    """Bind the ordinary endpoint, requiring a *whole* root-to-endpoint path."""
    mapped = points[:-1] if rule.tail else points
    if not mapped:
        raise ValueError("No represented path")
    inverse = {}
    for cell in engine.parents:
        value = tuple(float(v) for v in point(cell))
        if value in inverse:
            raise ValueError("Non-injective point readout")
        inverse[value] = cell
    endpoint = inverse[mapped[-1]]
    tail = points[-1:] if rule.tail else ()
    if read_plan(rule, engine, point, endpoint, tail) != points:
        raise ValueError("Stored plan is not the declared complete parent chain")
    return endpoint, tail


def propose_rules(attribute, points, engine, functions, names) -> list[dict]:
    """At most four templates per point callable; DEVELOPMENT ONLY."""
    rules = []
    for name in names:
        if name not in functions:
            continue
        for skip in (0, 1):
            for tail in (0, 1):
                rule = PlanRule(attribute, name, skip, tail)
                try:
                    context(rule, points, engine, functions[name])
                except ERRORS:
                    continue
                rules.append(asdict(rule))
    return rules


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
        endpoint, tail = context(rule, points, engine, point)
    except ERRORS as exc:
        evidence.update(status="rejected", reason="ordinary_readout", error=str(exc))
        return evidence, replays
    evidence.update(endpoint=endpoint, fixed_tail=tail, ordinary_plan=points)
    for edit, executed, search_check in executions:
        probe = {"edit": search_check["edit"], "status": "unresolved"}
        evidence["probes"].append(probe)
        if not search_check["passed"] or executed.get("error"):
            probe.update(reason="search_edit_failed", error=executed.get("error"))
            continue
        _, patched = compare_run(explanation, executed["target"], edit)
        if patched is None:
            probe.update(reason="search_prediction_missing")
            continue
        predicted = read_plan(rule, patched, point, endpoint, tail)
        observed = executed["plans"].get(rule.attribute)
        probe.update(
            predicted_plan=predicted,
            observed_plan=observed,
            prediction_matches=predicted == observed,
            changed=predicted != points,
        )
        if predicted != observed:
            probe.update(status="rejected", reason="stored_plan_prediction_mismatch")
        elif predicted == points:
            probe.update(reason="plan_unchanged")
        else:
            replays.append(
                ChainReplay(
                    probe,
                    rule.attribute,
                    ordinary,
                    executed["policy"],
                    predicted,
                    points,
                )
            )
    return evidence, replays


def finish_chains(versions) -> None:
    """A failed prediction refutes a readout even if another probe had a witness."""
    for version in versions:
        for chain in version["chains"]:
            if chain["status"] == "rejected":
                continue
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
            else:
                chain.update(status="unresolved", reason="no_complete_chain_witness")
        statuses = {c["status"] for c in version["chains"]}
        if "supported" in statuses:
            version.update(
                causal_status="supported", causal_reason="complete_chain_witness"
            )
        elif statuses == {"rejected"}:
            version.update(
                causal_status="rejected", causal_reason="readout_counterexample"
            )
        else:
            version.update(
                causal_status="unresolved",
                causal_reason=(
                    "no_complete_chain_witness" if statuses else "no_frozen_readout"
                ),
            )
