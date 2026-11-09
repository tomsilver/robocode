"""Frozen task-level discovery/validation and independent closed-loop assessment."""

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .checkpoints import Program
from .discover import (
    Limits,
    binding_candidates,
    digest,
    exhaust_candidates,
    truncation_audit,
)
from .evidence import FEATURES, summarize_findings
from .explanation import Explanation, from_data
from .tasks import TASK_DOMAIN, TaskCase, TaskSearchSpec, case_from_data
from .verify import ERRORS, verify

GRAMMAR = "task-search-v1"
SELECTION_RULE = (
    "Prefer causally supported graph search; mean development M; "
    "fewer fixed task sources; identifier"
)


def check_partition(development, validation) -> None:
    """Fresh means distinct task contents, not cosmetic names or case objects."""
    cases = (*development, *validation)
    if (
        not development
        or not validation
        or not all(isinstance(c, TaskCase) for c in cases)
    ):
        raise ValueError("Provide nonempty development and validation TaskCases")
    if len({c.name for c in cases}) != len(cases) or len(
        {c.signature for c in cases}
    ) != len(cases):
        raise ValueError(
            "Case names and full input contents must be distinct across partitions"
        )

    def task_identity(case: TaskCase) -> tuple:
        return case.transitions, case.goal, getattr(case, "heuristic", None)

    development_tasks = {task_identity(case) for case in development}
    if any(task_identity(case) in development_tasks for case in validation):
        raise ValueError(
            "Task validation must change transitions, goal or supplied heuristic, "
            "not only start"
        )


@dataclass(frozen=True)
class FrozenTaskBank:
    """Fix proposals, source, full task splits and distribution before validation."""

    source_digest: str
    discovery_inputs: tuple[TaskCase, ...]
    validation_inputs: tuple[TaskCase, ...]
    explanations: tuple[Explanation, ...]
    selected: str | None
    audit: dict
    grammar: str = GRAMMAR

    def manifest(self) -> dict:
        """The complete declarative experiment; diagnostics do not affect identity."""
        return {
            "schema": 1,
            "grammar": self.grammar,
            "domain": TASK_DOMAIN,
            "source_digest": self.source_digest,
            "discovery_cases": [asdict(c) for c in self.discovery_inputs],
            "validation_cases": [asdict(c) for c in self.validation_inputs],
            "validation_distribution": {
                c.name: 1 / len(self.validation_inputs) for c in self.validation_inputs
            },
            "explanations": [asdict(h) for h in self.explanations],
            "selected": self.selected,
            "selection_rule": SELECTION_RULE,
        }

    @property
    def fingerprint(self) -> str:
        """Integrity hash of the frozen experiment, including held-out cases."""
        return digest(self.manifest())

    def save(self, path: Path) -> None:
        """Write before any held-out concrete executions."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**self.manifest(), "digest": self.fingerprint}, indent=2)
            + "\n",
            encoding="utf-8",
        )


def load_task_bank(path: Path) -> FrozenTaskBank:
    """Restore the same experiment without proposing, repairing or selecting."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return task_bank_from_data(payload)


def task_bank_from_data(payload: dict) -> FrozenTaskBank:
    """Restore a standalone bank or a bank embedded in a frozen route experiment."""
    payload = dict(payload)
    fingerprint = payload.pop("digest")
    if (
        digest(payload) != fingerprint
        or payload["schema"] != 1
        or payload["grammar"] not in (GRAMMAR, "unit-cost-priority-v1")
        or payload["domain"] != TASK_DOMAIN
    ):
        raise ValueError("Modified or unsupported task manifest")
    bank = FrozenTaskBank(
        payload["source_digest"],
        tuple(case_from_data(c) for c in payload["discovery_cases"]),
        tuple(case_from_data(c) for c in payload["validation_cases"]),
        tuple(from_data(h) for h in payload["explanations"]),
        payload["selected"],
        {},
        payload["grammar"],
    )
    check_partition(bank.discovery_inputs, bank.validation_inputs)
    if bank.fingerprint != fingerprint or any(
        not isinstance(h.spec, TaskSearchSpec) for h in bank.explanations
    ):
        raise ValueError("Unsupported task choices or ingredients")
    if bank.selected is not None and bank.selected not in {
        h.identifier for h in bank.explanations
    }:
        raise ValueError("Selected explanation is absent")
    return bank


def task_specifications(reference: TaskCase):
    """Bounded ingredient choices; fixed models come only from development data."""
    selections: tuple[Literal["fifo", "lifo"], ...] = ("fifo", "lifo")
    for model, goal, selection, order, fallback in itertools.product(
        ("input", "fixed"),
        ("input", "fixed"),
        selections,
        ((-1, 1), (1, -1)),
        (-1, 0, 1),
    ):
        yield TaskSearchSpec(
            selection=selection,
            order=order,
            fallback=fallback,
            model_source=model,
            goal_source=goal,
            fixed_transitions=reference.transitions,
            fixed_goal=reference.goal,
        )
    for rule in ("goal_direction", "lookup", "constant"):
        for fallback in ((-1, 0, 1) if rule == "constant" else (0,)):
            yield TaskSearchSpec(mode="reactive", reactive_rule=rule, fallback=fallback)


def discover_tasks(
    program: Program, development, validation, limits=Limits()
) -> FrozenTaskBank:
    """Enumerate on development traces only; held-out cases only enter the manifest."""
    development, validation = tuple(development), tuple(validation)
    check_partition(development, validation)
    runs = {case: program.observe(case) for case in development}
    graph, reactive, cuts = binding_candidates(
        program, runs, limits.bindings, limits.mapping_depth, limits.selectors
    )
    audit = {
        "limits": asdict(limits),
        "candidate_bindings": {"graph": len(graph), "reactive": len(reactive)},
        "tested": 0,
        **truncation_audit(cuts),
        "observation_checks": len(runs),
        "supported": [],
        "rejected": [],
        "ingredient_rejections": [],
    }
    proposals = (
        Explanation(spec, mapping)
        for spec in task_specifications(development[0])
        for mapping in (graph if spec.mode == "graph" else reactive)
    )
    return fit_task_candidates(
        program, development, validation, runs, proposals, limits, audit
    )


def fit_task_candidates(
    program, development, validation, runs, proposals, limits, audit, grammar=GRAMMAR
) -> FrozenTaskBank:
    """Shared bounded verifier and preselection, independent of the proposal family."""
    accepted, scores = [], {}
    for hypothesis in proposals:
        if audit["tested"] >= limits.candidates:
            exhaust_candidates(audit)
            break
        spec = hypothesis.spec
        audit["tested"] += 1
        evidence = verify(program, hypothesis, development, runs)
        # Preserve incorrect-model search as diagnostic mechanism evidence.
        if evidence["mechanism_status"] == "supported":
            accepted.append(hypothesis)
            audit["supported"].append(evidence)
            has_search = spec.mode == "graph" and any(
                r["search_expansions"] for r in evidence["ordinary"]
            )
            scores[hypothesis.identifier] = (
                -int(has_search),
                -sum(r["M"] for r in evidence["ordinary"]) / len(development),
                int(spec.model_source == "fixed")
                + int(spec.goal_source == "fixed")
                + int(getattr(spec, "heuristic_source", "input") == "fixed"),
                hypothesis.identifier,
            )
        else:
            audit["rejected"].append(
                {
                    "id": hypothesis.identifier,
                    "explanation": asdict(hypothesis),
                    "status": evidence["status"],
                    "reason": evidence.get("reason"),
                    "counterexample": evidence.get("counterexample", evidence),
                }
            )
    selected = min(scores, key=scores.__getitem__) if scores else None
    return FrozenTaskBank(
        program.digest,
        development,
        validation,
        tuple(accepted),
        selected,
        audit,
        grammar,
    )


def assess_task(program: Program, case: TaskCase) -> dict:
    """Separate memoryless closed-loop rollout; never supplies planning features."""
    current, seen = case.start, set()
    actions: list = []
    trajectory = [current]
    result = {
        "case": case.name,
        "goal": case.goal,
        "trajectory": trajectory,
        "actions": actions,
        "goal_reached": False,
        "attributed_policy_work": 0,
        "scope": "Separate original-policy rollout, outside the feature distribution",
    }
    while current != case.goal:
        if current in seen:
            return {**result, "status": "failed", "reason": "deterministic_cycle"}
        seen.add(current)
        try:
            action = program.original(case.at(current))
        except ERRORS as exc:
            return {
                **result,
                "status": "unresolved",
                "goal_reached": None,
                "reason": str(exc),
            }
        actions.append(action)
        try:
            current = case.move(current, action)
        except ValueError as exc:
            return {**result, "status": "failed", "reason": str(exc)}
        trajectory.append(current)
    return {**result, "status": "succeeded", "goal_reached": True}


def validate_tasks(program: Program, bank: FrozenTaskBank) -> dict:
    """Run exactly the frozen cases; never discover, repair, or reselect."""
    if program.digest != bank.source_digest:
        raise ValueError("Frozen source differs from program")
    check_partition(bank.discovery_inputs, bank.validation_inputs)
    cases = bank.validation_inputs
    weights = bank.manifest()["validation_distribution"]
    runs = {case: program.observe(case) for case in cases}
    results = evaluate_hypotheses(program, bank.explanations, cases, runs, weights)
    return {
        "frozen_digest": bank.fingerprint,
        "selected": bank.selected,
        "distribution": weights,
        "cases": [asdict(c) for c in cases],
        "explanations": results,
        "finding": summarize_findings(results),
        "observation_checks": len(runs),
        "task_performance": task_performance(program, cases, weights),
        "planningness": None,
        "counting": "Ordinary single-invocation features only; "
        "interventions, environment checks and success rollouts excluded",
    }


def evaluate_hypotheses(program, hypotheses, cases, runs, weights) -> list[dict]:
    """Verify an already frozen bank on its gate-assigned cases; never select."""
    results = []
    for hypothesis in hypotheses:
        evidence = verify(program, hypothesis, cases, runs)
        evidence.update(
            ordinary_mean_N=None, ordinary_mean_M=None, mechanism_mean_features=None
        )
        if evidence["mechanism_status"] == "supported":
            means = {
                feature: sum(
                    weights[r["case"]] * r[feature] for r in evidence["ordinary"]
                )
                for feature in FEATURES
            }
            evidence["mechanism_mean_features"] = means
            if evidence["status"] == "supported":
                evidence.update(ordinary_mean_N=means["N"], ordinary_mean_M=means["M"])
        results.append(evidence)
    return results


def task_performance(program, cases, weights) -> dict:
    """Assess success separately from all explanation feature populations."""
    outcomes = [assess_task(program, case) for case in cases]
    success_rate = (
        None
        if any(r["goal_reached"] is None for r in outcomes)
        else sum(weights[r["case"]] * r["goal_reached"] for r in outcomes)
    )
    return {"success_rate": success_rate, "cases": outcomes}
