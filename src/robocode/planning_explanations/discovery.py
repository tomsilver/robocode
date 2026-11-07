"""Bounded hypothesis/mapping enumeration and paired causal experiments.

Candidate mappings only select trace values. Task-model computations live here,
outside measured execution. No source names or fixture identities are consulted.
"""

# Exclude booleans and arbitrary subclasses from numeric mapping coordinates.
# pylint: disable=unidiomatic-typecheck

import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite

from .instrument import Patch, Selector
from .task import backup, choose, rollout


@dataclass(frozen=True)
class Limits:
    """Explicit search and evidence budgets, recorded in the report."""

    series: int = 256
    permutations: int = 4096
    candidates: int = 128
    intervention_cases: int = 2

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Search and evidence budgets must be positive integers")


@dataclass(frozen=True)
class Explanation:
    """A frozen computation and selection-only alignment for one route."""

    route: str
    horizon: int
    mapping: tuple

    @property
    def identifier(self):
        """A stable identity for the entire model and mapping."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def selectors(self):
        """View the immutable mapping by role."""
        return dict(self.mapping)

    def read(self, trace):
        """Select and rearrange existing values, with no model computation."""
        return {
            role: [s.read(trace) for s in selectors] for role, selectors in self.mapping
        }


@dataclass(frozen=True)
class FrozenBank:
    """Discovery output; validation never adds or repairs an explanation."""

    program_digest: str
    discovery_inputs: tuple
    explanations: tuple
    audit: dict

    @property
    def digest(self):
        """Freeze the source and complete explanation library together."""
        payload = {
            "source": self.program_digest,
            "explanations": [asdict(e) for e in self.explanations],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def leaves(value, path=()):
    """Inventory numeric scalar locations, without interpreting identifiers."""
    if type(value) in (int, float):
        yield path
    elif type(value) in (list, dict):
        keys = range(len(value)) if type(value) is list else value.keys()
        for key in keys:
            yield from leaves(value[key], path + (key,))


def inventory(trace, limit):
    """Group scalar occurrences at one site/path or leaves of one container.

    This intentionally does not enumerate arbitrary subsets of all checkpoints.
    Values assembled across unrelated sites are outside this first search space.
    """
    grouped = defaultdict(list)
    vectors = []
    for event in trace.events:
        locations = [
            Selector(event.site, event.occurrence, path) for path in leaves(event.value)
        ]
        if len(locations) in (2, 4):
            vectors.append(tuple(locations))
        for selector in locations:
            grouped[(selector.site, selector.path)].append(selector)
    vectors.extend(tuple(v) for v in grouped.values() if len(v) in (2, 4))
    vectors = list(dict.fromkeys(vectors))
    return vectors[:limit], len(vectors) > limit, len(vectors)


def ordinary(explanation, case, trace):
    """Check task semantics, temporal order and output on an ordinary execution."""
    expected = rollout(
        case.state, case.transitions, case.utilities, explanation.horizon
    )
    try:
        observed = explanation.read(trace)
        if observed != expected:
            return {
                "reason": "role_values",
                "case": case.label,
                "predicted": expected,
                "observed": observed,
            }
        mapping = explanation.selectors()
        times = {
            r: [trace.event(s).index for s in selectors]
            for r, selectors in mapping.items()
        }
        horizon = explanation.horizon
        for action in (0, 1):
            indices = [action] if horizon == 1 else [2 * action, 2 * action + 1]
            for i in indices:
                if horizon == 2 and times["states_1"][action] >= times["states_2"][i]:
                    return {"reason": "state_order", "case": case.label}
                if times[f"states_{horizon}"][i] >= times["evaluations"][i]:
                    return {"reason": "evaluation_order", "case": case.label}
                if times["evaluations"][i] > times["scores"][action]:
                    return {"reason": "backup_order", "case": case.label}
        if trace.output != choose(expected["scores"]):
            return {
                "reason": "ordinary_output",
                "case": case.label,
                "predicted": choose(expected["scores"]),
                "observed": trace.output,
            }
    except (KeyError, IndexError, TypeError) as exc:
        return {"reason": "missing_location", "case": case.label, "detail": str(exc)}
    return None


def role_options(series, role, horizon, runs, limits, audit):
    """Enumerate rearrangements and reject semantically inconsistent mappings."""
    expected = [
        rollout(c.state, c.transitions, c.utilities, horizon)[role] for c, _ in runs
    ]
    found = []
    tried = 0
    for vector in series:
        if len(vector) != len(expected[0]):
            continue
        for selectors in itertools.permutations(vector):
            if tried == limits.permutations:
                audit["truncated"] = True
                return found
            tried += 1
            mismatch = None
            for (case, trace), predicted in zip(runs, expected):
                try:
                    observed = [s.read(trace) for s in selectors]
                except (KeyError, IndexError, TypeError):
                    observed = None
                if observed != predicted:
                    mismatch = {
                        "case": case.label,
                        "predicted": predicted,
                        "observed": observed,
                    }
                    break
            audit["role_mappings_tested"] += 1
            if mismatch:
                audit["role_mappings_rejected"] += 1
                # A diagnostic sample, not every failed permutation.
                if len(audit["role_rejection_examples"]) < 16:
                    audit["role_rejection_examples"].append(
                        {
                            "role": role,
                            "horizon": horizon,
                            "mapping": [asdict(s) for s in selectors],
                            **mismatch,
                        }
                    )
            else:
                found.append(selectors)
    return list(dict.fromkeys(found))


def predict_patch(case, horizon, role, index, value):
    """Intervene in the candidate computation and propagate its effects."""
    predicted = rollout(case.state, case.transitions, case.utilities, horizon)
    predicted[role][index] = value
    if role.startswith("states_"):
        if horizon == 2 and role == "states_1":
            predicted["states_2"][2 * index : 2 * index + 2] = case.transitions[value]
        predicted["evaluations"] = [
            case.utilities[s] for s in predicted[f"states_{horizon}"]
        ]
    if role != "scores":
        predicted["scores"] = backup(predicted["evaluations"], horizon)
    return predicted


def interventions(explanation, case, trace):
    """State substitutions and low/high value substitutions for every role."""
    low, high = min(case.utilities) - 100, max(case.utilities) + 100
    for role, selectors in explanation.mapping:
        for i, selector in enumerate(selectors):
            current = selector.read(trace)
            values = (
                ((int(current) + 1) % 6, (int(current) + 3) % 6)
                if role.startswith("states_")
                else (low, high)
            )
            for value in values:
                yield role, i, Patch(selector, value)


def paired_test(program, explanation, case, trace):
    """Compare model predictions to actual downstream checkpoints and output."""
    records = []
    for role, i, patch in interventions(explanation, case, trace):
        predicted = predict_patch(case, explanation.horizon, role, i, patch.value)
        record = {
            "case": case.label,
            "role": role,
            "coordinate": i,
            "target": asdict(patch.selector),
            "old_value": patch.selector.read(trace),
            "replacement": patch.value,
            "baseline_output": trace.output,
            "predicted_output": choose(predicted["scores"]),
            "predicted_roles": predicted,
        }
        try:
            patched = program.run(case, patch)
            observed = explanation.read(patched)
            # Earlier snapshots are historical facts, not overwritten retroactively.
            target_time = trace.event(patch.selector).index
            checks = []
            for name, selectors in explanation.mapping:
                for coordinate, selector in enumerate(selectors):
                    if (
                        trace.event(selector).index > target_time
                        or selector == patch.selector
                    ):
                        checks.append(
                            {
                                "role": name,
                                "coordinate": coordinate,
                                "predicted": predicted[name][coordinate],
                                "observed": observed[name][coordinate],
                            }
                        )
            record.update(
                {
                    "observed_output": patched.output,
                    "observed_roles": observed,
                    "downstream_checks": checks,
                    "patch_count": patched.patch_count,
                    "output_changed": patched.output != trace.output,
                    "passed": patched.patch_count == 1
                    and patched.output == record["predicted_output"]
                    and all(c["predicted"] == c["observed"] for c in checks),
                }
            )
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            record.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
        records.append(record)
        if not record["passed"]:
            return records, {
                "reason": "intervention_mismatch",
                "counterexample": record,
            }
    if not any(r["output_changed"] for r in records):
        return records, {"reason": "no_decision_effect", "case": case.label}
    return records, None


def discover(program, inputs, limits=Limits()):
    """Discover on this split only, then return an immutable explanation bank."""
    if not inputs:
        raise ValueError("Discovery inputs cannot be empty")
    runs = [(c, program.check_observation(c)) for c in inputs]
    routes = defaultdict(list)
    for case, trace in runs:
        routes[trace.route].append((case, trace))
    audit = {"limits": asdict(limits), "routes": {}, "observation_checks": len(runs)}
    accepted = []
    for route, route_runs in routes.items():
        series, truncated, available = inventory(route_runs[0][1], limits.series)
        details = {
            "branches": route_runs[0][1].branches,
            "cases": [c.label for c, _ in route_runs],
            "inventory": [[asdict(s) for s in v] for v in series],
            "available_series": available,
            "truncated": truncated,
            "role_mappings_tested": 0,
            "role_mappings_rejected": 0,
            "role_rejection_examples": [],
            "candidates_tested": 0,
            "rejected_candidates": [],
            "supported_candidates": [],
        }
        audit["routes"][route] = details
        exhausted = False
        for horizon in (1, 2):
            roles = ["states_1"] + (["states_2"] if horizon == 2 else [])
            roles += ["evaluations", "scores"]
            options = [
                role_options(series, role, horizon, route_runs, limits, details)
                for role in roles
            ]
            for combination in itertools.product(*options):
                if details["candidates_tested"] == limits.candidates:
                    details["truncated"] = True
                    exhausted = True
                    break
                explanation = Explanation(
                    route, horizon, tuple(zip(roles, combination))
                )
                details["candidates_tested"] += 1
                rejection = None
                evidence = []
                for case, trace in route_runs:
                    rejection = ordinary(explanation, case, trace)
                    if rejection:
                        break
                if not rejection:
                    for case, trace in route_runs[: limits.intervention_cases]:
                        records, rejection = paired_test(
                            program, explanation, case, trace
                        )
                        evidence.extend(records)
                        if rejection:
                            break
                entry = {
                    "id": explanation.identifier,
                    "explanation": asdict(explanation),
                }
                if rejection:
                    details["rejected_candidates"].append({**entry, **rejection})
                else:
                    accepted.append(explanation)
                    details["supported_candidates"].append(
                        {
                            **entry,
                            "ordinary_cases": len(route_runs),
                            "interventions": evidence,
                        }
                    )
            if exhausted:
                break
    return FrozenBank(program.digest, tuple(inputs), tuple(accepted), audit)


def validate(program, bank, inputs, state_weights=None, intervention_cases=2):
    """Validate frozen mappings and describe ordinary execution features.

    No enumeration or mapping repair happens here. Unknown routes remain unknown.
    State weights define the ordinary population, never the patched experiments.
    """
    if program.digest != bank.program_digest:
        raise ValueError("Frozen explanations belong to a different program")
    if type(intervention_cases) is not int or intervention_cases < 1:
        raise ValueError("At least one validation intervention case is required")
    discovery_fingerprints = {
        (c.state, c.transitions, c.utilities, c.cached_actions)
        for c in bank.discovery_inputs
    }
    if any(
        (c.state, c.transitions, c.utilities, c.cached_actions)
        in discovery_fingerprints
        for c in inputs
    ):
        raise ValueError(
            "Validation inputs must be fresh, not relabeled discovery inputs"
        )
    if not inputs:
        raise ValueError("Validation inputs cannot be empty")
    weights = {s: 1 / 6 for s in range(6)} if state_weights is None else state_weights
    if (
        set(weights) != set(range(6))
        or any(not isfinite(w) or w < 0 for w in weights.values())
        or abs(sum(weights.values()) - 1) > 1e-9
    ):
        raise ValueError("Supply six nonnegative state weights summing to one")
    counts = defaultdict(int)
    runs = [(c, program.check_observation(c)) for c in inputs]
    for case, _ in runs:
        counts[case.state] += 1
    if any(weights[s] > 0 and not counts[s] for s in weights):
        raise ValueError("Missing validation inputs for a positive-mass state")
    validations = []
    supported = defaultdict(list)
    for explanation in bank.explanations:
        matching = [(c, t) for c, t in runs if t.route == explanation.route]
        failures = [
            failure
            for c, t in matching
            if (failure := ordinary(explanation, c, t)) is not None
        ]
        records = []
        if not failures:
            for case, trace in matching[:intervention_cases]:
                evidence, failure = paired_test(program, explanation, case, trace)
                records.extend(evidence)
                if failure:
                    failures.append(failure)
                    break
        status = "supported" if matching and not failures and records else "unresolved"
        if failures:
            status = "rejected"
        if status == "supported":
            supported[explanation.route].append(explanation)
        validations.append(
            {
                "id": explanation.identifier,
                "status": status,
                "ordinary_cases": len(matching),
                "interventions": records,
                "counterexamples": failures,
            }
        )
    population = []
    for case, trace in runs:
        explanations = supported[trace.route]
        population.append(
            {
                "case": case.label,
                "state": case.state,
                "weight": weights[case.state] / counts[case.state],
                "route": trace.route,
                "branches": trace.branches,
                "output": trace.output,
                "original_checkpoints": len(trace.events),
                "status": "supported_planning" if explanations else "unresolved",
                "supported_explanations": [e.identifier for e in explanations],
                "features": {
                    "supported_horizons": sorted({e.horizon for e in explanations}),
                    "evaluated_alternatives": sorted(
                        {2**e.horizon for e in explanations}
                    ),
                    "root_actions": [2] if explanations else [],
                    "amount_if_unresolved": None,
                },
                "ambiguity": len(explanations) > 1,
            }
        )
    return {
        "frozen_digest": bank.digest,
        "state_distribution": weights,
        "utility_distribution": (
            "Uniform over the listed fresh contexts conditional on state"
        ),
        "observation_checks": len(runs),
        "explanations": validations,
        "ordinary_population": population,
        "evidence_coverage": {
            "supported_mass": sum(
                r["weight"] for r in population if r["status"] == "supported_planning"
            ),
            "unresolved_mass": sum(
                r["weight"] for r in population if r["status"] == "unresolved"
            ),
            "meaning": "Coverage by tested explanations; NOT a planningness score",
        },
        "planningness": None,
    }
