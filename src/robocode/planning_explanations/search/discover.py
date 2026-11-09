"""Bounded proposals, immutable manifests, and fresh validation without refitting."""

# Bool is not an integer node ID or a valid experiment budget.
# pylint: disable=unidiomatic-typecheck

import hashlib
import itertools
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .checkpoints import Program
from .engine import SearchSpec
from .environment import DOMAIN, check_environment
from .evidence import FEATURES, summarize_findings
from .explanation import Explanation, Mapping, from_data
from .selectors import inventory, parse
from .verify import verify

GRAMMAR = "line-world-search-v1"
DISCOVERY_INPUTS = (0, 2, 4)
VALIDATION_INPUTS = (1, 3, 5)


def digest(value: dict) -> str:
    """Hash a portable JSON manifest, independent of dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Limits:
    """Explicit bounds on hypotheses and structural binding candidates."""

    candidates: int = 2000
    bindings: int = 64
    witness_length: int = 3
    mapping_depth: int = 2
    selectors: int = 64

    def __post_init__(self):
        if any(
            type(n) is not int or n < 1
            for name, n in asdict(self).items()
            if name != "mapping_depth"
        ):
            raise ValueError("Budgets must be positive integers")
        if type(self.mapping_depth) is not int or not 0 <= self.mapping_depth <= 4:
            raise ValueError("Mapping depth must be an integer between zero and four")


@dataclass(frozen=True)
class FrozenBank:
    """All supported discovery hypotheses and a development-selected candidate."""

    source_digest: str
    discovery_inputs: tuple[int, ...]
    explanations: tuple[Explanation, ...]
    selected: str | None
    audit: dict

    def manifest(self) -> dict:
        """Only fixed experiment choices, excluding diagnostic audit records."""
        return {
            "schema": 1,
            "grammar": GRAMMAR,
            "domain": DOMAIN,
            "source_digest": self.source_digest,
            "discovery_inputs": list(self.discovery_inputs),
            "explanations": [asdict(h) for h in self.explanations],
            "selected": self.selected,
            "selection_rule": "Largest development mean M; identifier breaks ties",
        }

    @property
    def fingerprint(self) -> str:
        """Freeze the source, inputs, grammar, hypotheses and selection together."""
        return digest(self.manifest())

    def save(self, path: Path) -> None:
        """Persist the bank before looking at validation executions."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**self.manifest(), "digest": self.fingerprint}, indent=2) + "\n"
        )


def load_frozen(path: Path) -> FrozenBank:
    """Load only a matching, intact declaration in the supported language."""
    payload = json.loads(path.read_text())
    fingerprint = payload.pop("digest")
    if digest(payload) != fingerprint:
        raise ValueError("Frozen manifest was modified")
    if (
        payload["schema"] != 1
        or payload["grammar"] != GRAMMAR
        or payload["domain"] != DOMAIN
    ):
        raise ValueError("Unsupported frozen experiment schema/domain")
    bank = FrozenBank(
        payload["source_digest"],
        tuple(payload["discovery_inputs"]),
        tuple(from_data(h) for h in payload["explanations"]),
        payload["selected"],
        {},
    )
    if bank.fingerprint != fingerprint:
        raise ValueError("Manifest contains unsupported fields or selection rules")
    if bank.selected is not None and bank.selected not in {
        h.identifier for h in bank.explanations
    }:
        raise ValueError("Selected explanation is absent from the frozen bank")
    return bank


def specifications() -> Iterator[SearchSpec]:
    """Finite search ingredients; none call the concrete policy or a simulator."""
    selections: tuple[Literal["fifo", "lifo"], ...] = ("fifo", "lifo")
    for block in (1, 2, 3):
        for goal in range(6 // block):
            for selection, order, fallback in itertools.product(
                selections, ((-1, 1), (1, -1)), (-1, 0, 1)
            ):
                yield SearchSpec(block, goal, selection, order, fallback)
            yield SearchSpec(block, goal, mode="reactive")


def truncation_audit(cuts: dict) -> dict:
    """Name each reached bound; any of them can hide an explanation."""
    return {
        "truncated": bool(cuts["inventory_sites"]) or cuts["binding_limit"],
        "truncation": {**cuts, "candidate_budget": False},
    }


def exhaust_candidates(audit: dict) -> None:
    """Record that untested proposals remained when the budget ran out."""
    audit["truncated"] = True
    audit["truncation"]["candidate_budget"] = True


def binding_candidates(
    program: Program, runs: dict, limit: int, depth: int = 2, selectors: int = 64
):
    """Inventory by concrete types/sites, never by variable or function meaning.

    Returns graph and reactive mappings plus the bounds that cut the inventory:
    the checkpoints whose value inventory exceeded ``selectors`` and whether
    the ``limit`` on bindings dropped candidates.
    """
    graph, reactive = [], []
    cut_sites: list[str] = []
    binding_limit = False
    for site, metadata in program.sites.items():
        events = [[s for s in t.snapshots if s.site == site] for t in runs.values()]
        if not all(events):
            continue
        inventories = [inventory(es[0].values, depth, selectors) for es in events]
        if any(cut for _, cut in inventories):
            cut_sites.append(site)
        values = [found for found, _ in inventories]
        common = set.intersection(*(set(found) for found in values))
        names = sorted(common, key=lambda name: (len(parse(name)[1]), name))

        def all_type(name: str, value_type: type, inventories=values) -> bool:
            return all(type(found[name]) is value_type for found in inventories)

        if metadata["kind"] == "loop":
            scalars = [n for n in names if all_type(n, int)]
            lists = [
                n
                for n in names
                if all_type(n, list)
                and all(all(type(v) is int for v in found[n]) for found in values)
            ]
            dictionaries = [
                n
                for n in names
                if all_type(n, dict)
                and all(
                    any(v is None for v in found[n].values())
                    and all(
                        type(k) is int and k >= 0 and (v is None or type(v) is int)
                        for k, v in found[n].items()
                    )
                    for found in values
                )
            ]
            for root, frontier, parents, offset in itertools.product(
                scalars, lists, dictionaries, (0, 10)
            ):
                # Every candidate engine starts with exactly the root in the
                # frontier and a singleton parent map. This is a shape/value
                # constraint of Q, not a semantic interpretation of local names.
                if not all(
                    found[frontier] == [found[root]]
                    and found[parents] == {found[root]: None}
                    for found in values
                ):
                    continue
                graph.append(Mapping(site, root, frontier, parents, offset=offset))
        elif metadata["kind"] == "statement" and all(len(es) == 1 for es in events):
            for name in names:
                # Ordinary reactive correspondence already requires this value
                # to be the returned action. This only prunes proposals; live
                # interventions must still establish the causal decision role.
                if all_type(name, int) and all(
                    found[name] == trace.output
                    for found, trace in zip(values, runs.values())
                ):
                    reactive.append(Mapping(site, action=name))

        # Prefer simpler paths across ALL sites, so new input-container paths
        # cannot starve a later top-level action checkpoint of the binding budget.
        def depth_cost(mapping):
            return sum(
                len(parse(n)[1])
                for n in (
                    mapping.root,
                    mapping.frontier,
                    mapping.parents,
                    mapping.action,
                )
                if n
            )

        graph.sort(key=depth_cost)
        reactive.sort(key=depth_cost)
        binding_limit |= len(graph) > limit or len(reactive) > limit
        graph, reactive = graph[:limit], reactive[:limit]
    cuts = {"inventory_sites": cut_sites, "binding_limit": binding_limit}
    return graph, reactive, cuts


def discover(
    program: Program, positions=DISCOVERY_INPUTS, limits=Limits()
) -> FrozenBank:
    """Discover explanations on development inputs only; preserve counterexamples."""
    positions = tuple(positions)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Provide distinct, nonempty discovery inputs")
    runs = {p: program.observe(p) for p in positions}
    graph, reactive, cuts = binding_candidates(
        program, runs, limits.bindings, limits.mapping_depth, limits.selectors
    )
    audit = {
        "limits": asdict(limits),
        "observation_checks": len(runs),
        "candidate_bindings": {"graph": len(graph), "reactive": len(reactive)},
        **truncation_audit(cuts),
        "tested": 0,
        "rejected": [],
        "ingredient_rejections": [],
        "supported": [],
    }
    accepted = []
    scores = {}
    exhausted = False
    for spec in specifications():
        environment = check_environment(spec, limits.witness_length)
        if environment["status"] != "supported":
            audit["ingredient_rejections"].append(
                {"spec": asdict(spec), "evidence": environment}
            )
            continue
        for mapping in graph if spec.mode == "graph" else reactive:
            if audit["tested"] >= limits.candidates:
                exhaust_candidates(audit)
                exhausted = True
                break
            hypothesis = Explanation(spec, mapping)
            audit["tested"] += 1
            evidence = verify(
                program, hypothesis, positions, runs, limits.witness_length
            )
            if evidence["status"] == "supported":
                accepted.append(hypothesis)
                audit["supported"].append(evidence)
                scores[hypothesis.identifier] = sum(
                    r["M"] for r in evidence["ordinary"]
                ) / len(positions)
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
        if exhausted:
            break
    selected = min(scores, key=lambda key: (-scores[key], key)) if scores else None
    audit["development_mean_M"] = scores
    return FrozenBank(program.digest, positions, tuple(accepted), selected, audit)


def validate(
    program: Program,
    bank: FrozenBank,
    positions=VALIDATION_INPUTS,
    weights=None,
    witness_limit: int = 3,
) -> dict:
    """Evaluate a frozen bank without discovery, repair, or reselection."""
    positions = tuple(positions)
    if program.digest != bank.source_digest:
        raise ValueError("Frozen source differs from the program")
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Provide distinct, nonempty validation inputs")
    if set(positions) & set(bank.discovery_inputs):
        raise ValueError("Validation inputs must be fresh")
    distribution = (
        {p: 1 / len(positions) for p in positions} if weights is None else dict(weights)
    )
    if (
        set(distribution) != set(positions)
        or any(
            not isinstance(w, (int, float)) or not 0 <= w <= 1
            for w in distribution.values()
        )
        or abs(sum(distribution.values()) - 1) > 1e-9
    ):
        raise ValueError("Weights must be probabilities over the validation inputs")
    runs = {p: program.observe(p) for p in positions}
    results = []
    for explanation in bank.explanations:
        evidence = verify(program, explanation, positions, runs, witness_limit)
        evidence["mode"] = explanation.spec.mode
        evidence["ordinary_mean_N"] = None
        evidence["ordinary_mean_M"] = None
        evidence["mechanism_mean_features"] = None
        if evidence["status"] == "supported":
            evidence["mechanism_mean_features"] = {
                feature: sum(
                    distribution[row["position"]] * row[feature]
                    for row in evidence["ordinary"]
                )
                for feature in FEATURES
            }
            for feature in ("N", "M"):
                evidence[f"ordinary_mean_{feature}"] = sum(
                    distribution[row["position"]] * row[feature]
                    for row in evidence["ordinary"]
                )
        results.append(evidence)
    return {
        "frozen_digest": bank.fingerprint,
        "selected": bank.selected,
        "distribution": distribution,
        "explanations": results,
        "finding": summarize_findings(results),
        "observation_checks": len(runs),
        "counting": "Ordinary distinct expanded environment nodes N; M=max(N-1,0). "
        "No intervention weights.",
        "planningness": None,
    }
