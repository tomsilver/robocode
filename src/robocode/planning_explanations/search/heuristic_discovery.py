"""Bounded joint proposals for priority-search ingredients and current-value maps."""

# Exact container/scalar types are the restricted representation contract.
# pylint: disable=unidiomatic-typecheck

import itertools
from collections.abc import Iterator
from dataclasses import asdict

from .benchmark import check_partition, fit_task_candidates, task_specifications
from .checkpoints import Program
from .discover import Limits, binding_candidates, truncation_audit
from .explanation import CostMapping, Explanation
from .heuristics import HeuristicCase, HeuristicSpec
from .selectors import inventory, select


def priority_specifications(reference: HeuristicCase):
    """Enumerate interpretable rules; these ingredients never run a solver."""
    for selection in ("fifo", "lifo", "greedy", "astar"):
        ties = ("first", "last") if selection in ("greedy", "astar") else ("first",)
        for tie, relaxation, timing, source, order, fallback in itertools.product(
            ties,
            ("first", "improve"),
            ("selection", "generation"),
            ("input", "fixed", "zero"),
            ((-1, 1), (1, -1)),
            (0, -1, 1),
        ):
            yield HeuristicSpec(
                selection=selection,
                tie=tie,
                relaxation=relaxation,
                goal_timing=timing,
                heuristic_source=source,
                order=order,
                fallback=fallback,
                fixed_heuristic=reference.heuristic,
            )


def cost_bindings(program, runs, limits) -> tuple:
    """Extend initial root/frontier/parent candidates by dictionary shapes only."""
    graph, reactive, cuts = binding_candidates(
        program, runs, limits.bindings, limits.mapping_depth, limits.selectors
    )
    cut_sites, binding_limit = list(cuts["inventory_sites"]), cuts["binding_limit"]
    result: list[CostMapping] = []
    for mapping in graph:
        # This slice uses identity state labels. Old shifted-label explanations
        # remain available in their existing suite.
        if mapping.offset:
            continue
        snapshots = [
            next(s for s in t.snapshots if s.site == mapping.site)
            for t in runs.values()
        ]
        inventories = [
            inventory(s.values, limits.mapping_depth, limits.selectors)
            for s in snapshots
        ]
        if any(cut for _, cut in inventories) and mapping.site not in cut_sites:
            cut_sites.append(mapping.site)
        common = sorted(set.intersection(*(set(found) for found, _ in inventories)))
        costs, heuristics = [], []
        for path in common:
            values = [found[path] for found, _ in inventories]
            if not all(
                type(v) is dict
                and all(
                    type(k) is int and type(x) is int and x >= 0 for k, x in v.items()
                )
                for v in values
            ):
                continue
            if all(
                v == {select(s.values, mapping.root): 0}
                for v, s in zip(values, snapshots)
            ):
                costs.append(path)
            if all(set(v) == set(range(6)) for v in values):
                heuristics.append(path)
        for cost, heuristic in itertools.product(costs, heuristics):
            if len(result) == limits.bindings:
                binding_limit = True
                break
            result.append(
                CostMapping(**asdict(mapping), costs=cost, heuristic=heuristic)
            )
    cuts = {"inventory_sites": cut_sites, "binding_limit": binding_limit}
    return result, reactive, cuts


# Late checkpoints in the heuristic fixtures hold up to 69 current value paths,
# mostly entries of the input transition and heuristic tables. 128 inventories
# them all, so a reported truncation means a bound actually hid candidates.
HEURISTIC_LIMITS = Limits(candidates=4000, selectors=128)


def discover_heuristics(
    program: Program, development, validation, limits=HEURISTIC_LIMITS
):
    """Discover on development traces; held-out task/heuristic data stay unexecuted."""
    development, validation = tuple(development), tuple(validation)
    check_partition(development, validation)
    if not all(isinstance(c, HeuristicCase) for c in (*development, *validation)):
        raise ValueError("Heuristic discovery requires declared HeuristicCases")
    runs = {case: program.observe(case) for case in development}
    graph, reactive, cuts = cost_bindings(program, runs, limits)
    audit = {
        "limits": asdict(limits),
        "candidate_bindings": {
            "graph_with_costs": len(graph),
            "reactive": len(reactive),
        },
        **truncation_audit(cuts),
        "tested": 0,
        "observation_checks": len(runs),
        "supported": [],
        "rejected": [],
        "ingredient_rejections": [],
        "grammar": (
            "Unit costs; identity states; input transitions and goal; "
            "bounded priority and update rules"
        ),
    }

    def proposals() -> Iterator[Explanation]:
        # Baselines get an explicit opportunity even under a small search budget.
        for spec in task_specifications(development[0]):
            if spec.mode == "reactive":
                for mapping in reactive:
                    yield Explanation(spec, mapping)
        for spec in priority_specifications(development[0]):
            for mapping in graph:
                yield Explanation(spec, mapping)

    return fit_task_candidates(
        program,
        development,
        validation,
        runs,
        proposals(),
        limits,
        audit,
        "unit-cost-priority-v1",
    )
