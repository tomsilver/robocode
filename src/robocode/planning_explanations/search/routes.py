"""Development-discovered checkpoint-prefix gates and frozen per-route banks."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .benchmark import (
    FrozenTaskBank,
    check_partition,
    discover_tasks,
    evaluate_hypotheses,
    task_bank_from_data,
    task_performance,
)
from .checkpoints import Program, Trace
from .discover import Limits, digest
from .evidence import FEATURES, summarize_findings
from .heuristic_discovery import discover_heuristics
from .heuristics import HeuristicCase
from .tasks import TaskCase, case_from_data
from .verify import ERRORS

GATE_RULE = (
    "Exact checkpoint prefix through first while/for or pre-return; "
    "no values or outputs"
)


def route_prefix(trace: Trace, limit: int = 64) -> tuple[str, ...] | None:
    """Inspect only the ordinary prefix before a loop/return executes."""
    prefix: list[str] = []
    for snapshot in trace.snapshots:
        if snapshot.kind == "entry":
            continue
        if snapshot.kind == "exit" or len(prefix) >= limit:
            return None
        prefix.append(snapshot.site)
        if snapshot.site.endswith((":While", ":For", ":Return")):
            return tuple(prefix)
    return None


@dataclass(frozen=True)
class Route:
    """A fixed prefix and its development-discovered explanation bank."""

    prefix: tuple[str, ...]
    bank: FrozenTaskBank

    @property
    def identifier(self) -> str:
        """Stable structural route identity, independent of semantic names."""
        return digest({"prefix": self.prefix})[:12]

    def manifest(self) -> dict:
        """Embed all hypotheses and the preselected explanation, without the audit."""
        return {
            "prefix": list(self.prefix),
            "bank": {**self.bank.manifest(), "digest": self.bank.fingerprint},
        }


@dataclass(frozen=True)
class FrozenRouteBank:
    """One fixed distribution, gate language and bank for each discovered route."""

    source_digest: str
    discovery_inputs: tuple[TaskCase, ...]
    validation_inputs: tuple[TaskCase, ...]
    routes: tuple[Route, ...]
    limits: Limits
    max_routes: int = 8
    prefix_limit: int = 64
    audit: dict | None = None

    def manifest(self) -> dict:
        """Freeze the experiment before any held-out policy executions."""
        return {
            "schema": 1,
            "grammar": "task-prefix-routes-v1",
            "gate_rule": GATE_RULE,
            "source_digest": self.source_digest,
            "discovery_cases": [asdict(c) for c in self.discovery_inputs],
            "validation_cases": [asdict(c) for c in self.validation_inputs],
            "validation_distribution": {
                c.name: 1 / len(self.validation_inputs) for c in self.validation_inputs
            },
            "limits_per_route": asdict(self.limits),
            "max_routes": self.max_routes,
            "prefix_limit": self.prefix_limit,
            "routes": [route.manifest() for route in self.routes],
        }

    @property
    def fingerprint(self) -> str:
        """Hash gates and banks together, including their task partition."""
        return digest(self.manifest())

    def save(self, path: Path) -> None:
        """Persist the complete declaration before validation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**self.manifest(), "digest": self.fingerprint}, indent=2)
            + "\n",
            encoding="utf-8",
        )


def load_route_bank(path: Path) -> FrozenRouteBank:
    """Restore and check gates and all child banks without executing the source."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = payload.pop("digest")
    if digest(payload) != fingerprint:
        raise ValueError("Modified route manifest")
    bank = FrozenRouteBank(
        payload["source_digest"],
        tuple(case_from_data(c) for c in payload["discovery_cases"]),
        tuple(case_from_data(c) for c in payload["validation_cases"]),
        tuple(
            Route(tuple(r["prefix"]), task_bank_from_data(r["bank"]))
            for r in payload["routes"]
        ),
        Limits(**payload["limits_per_route"]),
        payload["max_routes"],
        payload["prefix_limit"],
    )
    check_partition(bank.discovery_inputs, bank.validation_inputs)
    if bank.fingerprint != fingerprint:
        raise ValueError("Unsupported route manifest fields or gate rule")
    if not 1 <= bank.max_routes <= 64 or not 1 <= bank.prefix_limit <= 256:
        raise ValueError("Invalid route budgets")
    if len(bank.routes) > bank.max_routes or len(
        {r.prefix for r in bank.routes}
    ) != len(bank.routes):
        raise ValueError("Duplicate or excessive routes")
    assigned: list[TaskCase] = []
    for route in bank.routes:
        if (
            not route.prefix
            or len(route.prefix) > bank.prefix_limit
            or route.bank.source_digest != bank.source_digest
            or route.bank.validation_inputs != bank.validation_inputs
            or any(c not in bank.discovery_inputs for c in route.bank.discovery_inputs)
        ):
            raise ValueError("Inconsistent child bank or route prefix")
        assigned.extend(route.bank.discovery_inputs)
    if len(set(assigned)) != len(assigned):
        raise ValueError("Development case assigned to multiple routes")
    return bank


def discover_routes(
    program: Program,
    development,
    validation,
    limits=Limits(),
    max_routes: int = 8,
    prefix_limit: int = 64,
) -> FrozenRouteBank:
    """Inventory prefixes on development runs, then discover within each partition."""
    if not 1 <= max_routes <= 64 or not 1 <= prefix_limit <= 256:
        raise ValueError("Route budgets exceed supported bounds")
    development, validation = tuple(development), tuple(validation)
    check_partition(development, validation)
    groups: dict[tuple[str, ...], list[TaskCase]] = {}
    unresolved = []
    for case in development:
        try:
            prefix = route_prefix(program.observe(case), prefix_limit)
        except ERRORS as exc:
            unresolved.append({"case": case.name, "reason": str(exc)})
            continue
        if prefix is None or (prefix not in groups and len(groups) >= max_routes):
            unresolved.append({"case": case.name, "reason": "route_inventory_budget"})
        else:
            groups.setdefault(prefix, []).append(case)
    discoverer = (
        discover_heuristics
        if all(isinstance(c, HeuristicCase) for c in development)
        else discover_tasks
    )
    routes = tuple(
        Route(prefix, discoverer(program, cases, validation, limits))
        for prefix, cases in groups.items()
    )
    return FrozenRouteBank(
        program.digest,
        development,
        validation,
        routes,
        limits,
        max_routes,
        prefix_limit,
        {"unresolved": unresolved},
    )


def validate_routes(program: Program, bank: FrozenRouteBank) -> dict:
    """Dispatch ordinary prefixes only; patched runs keep their original explanation."""
    if program.digest != bank.source_digest:
        raise ValueError("Frozen source differs from program")
    check_partition(bank.discovery_inputs, bank.validation_inputs)
    weights = bank.manifest()["validation_distribution"]
    by_prefix = {r.prefix: r for r in bank.routes}
    runs, assignments = {}, []
    for case in bank.validation_inputs:
        prefix, reason = None, None
        try:
            runs[case] = program.observe(case)
            prefix = route_prefix(runs[case], bank.prefix_limit)
        except ERRORS as exc:
            reason = str(exc)
        route = by_prefix.get(prefix) if prefix is not None else None
        assignments.append(
            {
                "case": case.name,
                "weight": weights[case.name],
                "prefix": prefix,
                "route": route.identifier if route else None,
                "reason": reason or (None if route else "unseen_or_unresolved_route"),
            }
        )
    results = []
    covered = selected_covered = 0.0
    totals = dict.fromkeys(FEATURES, 0.0)
    for route in bank.routes:
        names = {a["case"] for a in assignments if a["route"] == route.identifier}
        cases = tuple(c for c in bank.validation_inputs if c.name in names)
        mass = sum(weights[c.name] for c in cases)
        conditional = {c.name: weights[c.name] / mass for c in cases}
        evidence = (
            evaluate_hypotheses(
                program, route.bank.explanations, cases, runs, conditional
            )
            if cases
            else []
        )
        selected = next((r for r in evidence if r["id"] == route.bank.selected), None)
        means = (
            selected["mechanism_mean_features"]
            if selected and selected["status"] == "supported"
            else None
        )
        finding = summarize_findings(evidence)
        if any(r["status"] == "supported" for r in evidence):
            covered += mass
        if means is not None:
            selected_covered += mass
            for feature in FEATURES:
                totals[feature] += mass * means[feature]
        result = {
            "route": route.identifier,
            "prefix": route.prefix,
            "weight": mass,
            "frozen_digest": route.bank.fingerprint,
            "selected": route.bank.selected,
            "distribution": conditional,
            "cases": [asdict(c) for c in cases],
            "explanations": evidence,
            "finding": finding,
            "preselected_status": selected["status"] if selected else "unresolved",
            "preselected_mean_features": means,
            "observation_checks": len(cases),
            "planningness": None,
        }
        results.append(result)
        for assignment in assignments:
            if assignment["route"] != route.identifier:
                continue
            ordinary = (
                next(
                    (
                        r
                        for r in selected["ordinary"]
                        if r["case"] == assignment["case"]
                    ),
                    None,
                )
                if means and selected
                else None
            )
            assignment.update(
                finding=finding,
                preselected_status=result["preselected_status"],
                preselected_mechanism_finding=(
                    ordinary.get("mechanism_finding") if ordinary else "unresolved"
                ),
                features={f: ordinary[f] for f in FEATURES} if ordinary else None,
            )
    for assignment in assignments:
        if assignment["route"] is None:
            assignment.update(
                finding=summarize_findings([]),
                preselected_status="unresolved",
                preselected_mechanism_finding="unresolved",
                features=None,
            )
    codes = {r["finding"]["code"] for r in results if r["weight"]}
    complete = (
        all(a["route"] is not None for a in assignments) and "unresolved" not in codes
    )
    finding = (
        summarize_findings(
            [
                evidence
                for route in results
                if route["weight"]
                for evidence in route["explanations"]
            ]
        )
        if complete and len(codes) == 1
        else {
            "code": "mixed_routes" if complete else "partially_resolved_routes",
            "label": (
                "Different supported mechanisms across routes"
                if complete
                else "Some routes remain unresolved"
            ),
            "planningness": None,
        }
    )
    all_selected_supported = all(
        a["preselected_status"] == "supported" for a in assignments
    )
    return {
        "frozen_digest": bank.fingerprint,
        "distribution": weights,
        "assignments": assignments,
        "routes": results,
        "finding": finding,
        "supported_route_coverage": covered,
        "preselected_supported_coverage": selected_covered,
        "preselected_mean_features": totals if all_selected_supported else None,
        "task_performance": task_performance(program, bank.validation_inputs, weights),
        "planningness": None,
        "counting": "Ordinary invocations only; "
        "unknown routes have missing features, not zero",
    }
