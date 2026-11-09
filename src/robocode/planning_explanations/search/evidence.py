"""Mechanism findings and computation features, independent of task success."""

from .engine import Action, Transition

FEATURES = (
    "N",
    "M",
    "max_discovery_depth",
    "max_expanded_depth",
    "generated_alternatives",
    "distinct_generated_states",
    "max_frontier_size",
    "search_expansions",
    "expansion_events",
    "reexpansions",
    "cost_improvements",
)

FINDING_LABELS = {
    "search_supported": "Search explanation supported",
    "search_mechanism_only": "Search mechanism supported; task grounding not supported",
    "reactive_only": "Only reactive explanations supported",
    "unresolved": "Unresolved",
}


def computation_features(transitions: list[Transition]) -> dict:
    """Count matched ordinary computations only; terminal Action leaves are separate."""
    expanded = {t.expanded for t in transitions if t.expanded is not None}
    generated = [
        n for t in transitions for n in t.generated if not isinstance(n, Action)
    ]
    depths = []
    expanded_depths = []
    for transition in transitions:
        parents = transition.state.parents
        for node in parents:
            if isinstance(node, Action):
                continue
            depth, current, seen = 0, node, set()
            while parents[current] is not None:
                if current in seen:
                    raise ValueError("Cyclic retained parent graph")
                seen.add(current)
                parent = parents[current]
                if not isinstance(parent, int):
                    raise ValueError("An environment parent must be a state ID")
                current = parent
                depth += 1
            depths.append(depth)
            if node == transition.expanded:
                expanded_depths.append(depth)
    return {
        "expanded_nodes": sorted(expanded),
        "N": len(expanded),
        "M": max(len(expanded) - 1, 0),
        "max_discovery_depth": max(depths, default=0),
        "max_expanded_depth": max(expanded_depths, default=0),
        "generated_alternatives": len(generated),
        "distinct_generated_states": len(set(generated)),
        "max_frontier_size": max([1] + [len(t.state.frontier) for t in transitions]),
        "search_expansions": sum(
            t.expanded is not None
            and any(not isinstance(n, Action) for n in t.generated)
            for t in transitions
        ),
        "expansion_events": sum(t.expanded is not None for t in transitions),
        "reexpansions": sum(t.expanded is not None for t in transitions)
        - len(expanded),
        "cost_improvements": sum(
            len(t.extra_data().get("improved", ())) for t in transitions
        ),
    }


def summarize_findings(results: list[dict]) -> dict:
    """Describe all frozen explanations without replacing the selected hypothesis.

    A failed or incomplete candidate does not become a zero-planning result.
    M is deliberately not used to decide whether search evidence exists.
    """
    graph: list[str] = []
    ungrounded: list[str] = []
    reactive: list[str] = []
    for result in results:
        if result.get("mechanism_status", result["status"]) != "supported":
            continue
        if result["mode"] == "reactive":
            if result["status"] == "supported":
                reactive.append(result["id"])
        elif any(
            r.get("search_expansions", 0) > 0 and r.get("decision_effects", 0) > 0
            for r in result["ordinary"]
        ):
            (graph if result["status"] == "supported" else ungrounded).append(
                result["id"]
            )
    code = (
        "search_supported"
        if graph
        else (
            "search_mechanism_only"
            if ungrounded
            else "reactive_only" if reactive else "unresolved"
        )
    )
    return {
        "code": code,
        "label": FINDING_LABELS[code],
        "search_explanations": graph,
        "ungrounded_search_mechanisms": ungrounded,
        "reactive_explanations": reactive,
        "scope": "Empirical support within the frozen bounded language; "
        "not proof of absence",
        "planningness": None,
    }
