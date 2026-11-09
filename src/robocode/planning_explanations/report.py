"""Readable and JSON evidence, with all populations and limitations explicit."""

# Keep literal Markdown table rows and report sentences together.
# pylint: disable=line-too-long

import json
from dataclasses import asdict
from math import isfinite
from pathlib import Path

from .discovery import Explanation, FrozenBank
from .instrument import Selector
from .task import Case, Model

LIMITATIONS = [
    "Empirical support within a restricted hypothesis and mapping family, not a proof.",
    "Hypotheses: exhaustive search to depth 1, 2 or 3 with a maximum backup and first-action ties. A path is valued by its last state's utility or by the sum of utilities along it (start state excluded); at depth 1 these coincide. Depth 1 compares its evaluations directly. Optionally the sound cutoff: stop expanding the second action once one of its paths beats the first action's score (the decision cannot change). Each expanded path needs its own value: a program that adds a shared term after taking the maximum is not covered. A route must expand the same nodes on all its inputs; best-first search, whose expansion order depends on values without any recorded branch, is not covered.",
    "Mappings: each coordinate selects one recorded numeric value (any expression, visit or container element) that equals the model's value in every discovery run of the route. Values are never combined or computed by a mapping.",
    "Python subset: allowlisted imports (heapq, math) and functions; the last top-level function is the policy, all others (top-level or nested, recursion allowed) are helpers. Loops, comprehensions, generator expressions, and allowlisted builtins and container methods are supported. No classes, lambdas, try/with, or global/nonlocal. Sets run but are never recorded. Each run may loop at most 10,000 times and record at most 5,000 expressions.",
    "Routes are executed if and while decisions and comprehension filters (not for-loop lengths). An input whose route has no supported explanation is also tested against frozen explanations from other routes (never refitted); if none passes it stays unresolved, with its reason.",
    "Each mapped state is replaced by every other state. Each mapped value is replaced by numbers far below and far above every utility, and by values just below, exactly at, and just above the point where the model's decision flips (0.125 away). A rule that differs from the model by less than that step is not detected. Joint patches (two coordinates at once) and values taken from other inputs are not tested.",
    'Every generated input has its own transition table. A program may instead search with its OWN fixed table: one is inferred from the successor states it computes (a fixed function of state and action on every input, differing from the environment somewhere, every entry seen on two inputs) and then tested like any model. A table that is only partly the program\'s own ("the environment, except ...") is not covered. Inputs needing a row the table lacks are not covered, and state patches into such rows are counted as untestable.',
    "Only the configured number of cases per route receives paired interventions.",
    "Interchangeable locations are copies of one quantity; each is validated as a single-coordinate swap, and combinations of swaps are not separately tested.",
    "A selected value is an observed occurrence, not a count of unique arithmetic operations.",
    "The action cache and analyzer rollouts are external preparation, never counted as planning.",
    "Supported mass is a lower bound on the share of states with verified planning. No lookahead means the action never changed when only the input's future changed (where actions lead, and how good the other states are), over every targeted change and 16 random futures: evidence, not proof. Its definition is behavioral: a program searching a fixed internal model with fixed internal values never responds to the environment and counts as no lookahead. Unresolved inputs look ahead (some change to the future moved the action) but have no verified model.",
    "Planningness is the weighted effective lookahead depth (log2 of complete paths evaluated) as an interval: no-lookahead inputs count 0, and unresolved inputs count their behavioral lower bound at the lower end and the family's MAXIMUM (3, the deepest model the analyzer can verify, not a true bound) at the upper end.",
    "The behavioral lower bound uses tree-shaped futures (the start state has no incoming moves; each first successor is entered only from it) in which the program takes the same route as on the input. A move whose redirection changes the action shows a compared path through it; only the first two moves of a path can be told apart, so the bound is at most 4 paths (2). It never exceeded a verified amount.",
    "Trusted small programs only: syntax restrictions and event budget are not a security sandbox.",
]


# 2: depth-1 models have no "scores" role, and explanations list interchangeable
# locations found by per-coordinate search.
# 3: patch records carry a "kind" (other state, low, high, below/at/above switch)
# and validated explanations count their informative interventions.
# 4: explanations name a path value as well as a depth; discovery audits are
# listed per model ("models") instead of per depth ("horizons").
# 5: models may prune: explanations name a pruning rule, and a mapping holds null
# for a node the model does not expand on that route.
# 6: an explanation may name its own inferred transition table; validation reports
# its agreement with the environment, uncovered inputs and untestable patches.
# 7: inputs carry a no-lookahead test and may have status "no_lookahead";
# planningness reports the average over inputs with a verified amount.
# 8: inputs that look ahead carry a behavioral lower bound on compared paths, which
# raises the lower end of planningness for unresolved inputs.
SCHEMA_VERSION = 8


def finite(value):
    """Strict JSON has no infinities: write them as the strings "inf" and "-inf".

    Programs may record them, for example a running maximum that starts at -inf.
    """
    if isinstance(value, float) and not isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def write_json(path, value):
    """Write deterministic, strict JSON suitable for diffing and reanalysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(finite(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def frozen_payload(bank):
    """Serialize exactly what is fixed before fresh validation starts."""
    return {
        "schema_version": SCHEMA_VERSION,
        "digest": bank.digest,
        "program_digest": bank.program_digest,
        "discovery_inputs": [asdict(c) for c in bank.discovery_inputs],
        "explanations": [asdict(e) for e in bank.explanations],
    }


def load_frozen(path):
    """Restore a saved bank for validation without repeating discovery."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported frozen-bank schema; rerun discovery")

    def selector(s):
        if s is None:
            return None
        return Selector(s["site"], s["occurrence"], tuple(s["path"]))

    explanations = tuple(
        Explanation(
            e["route"],
            e["horizon"],
            tuple(
                (role, tuple(selector(s) for s in selectors))
                for role, selectors in e["mapping"]
            ),
            tuple(
                (role, coordinate, tuple(selector(s) for s in locations))
                for role, coordinate, locations in e["interchangeable"]
            ),
            e["path_value"],
            e["pruning"],
            None if e["table"] is None else tuple(tuple(row) for row in e["table"]),
        )
        for e in payload["explanations"]
    )
    inputs = tuple(
        Case(
            c["label"],
            c["state"],
            tuple(tuple(r) for r in c["transitions"]),
            tuple(c["utilities"]),
            tuple(c["cached_actions"]),
        )
        for c in payload["discovery_inputs"]
    )
    bank = FrozenBank(payload["program_digest"], inputs, explanations, {})
    if bank.digest != payload["digest"]:
        raise ValueError("Frozen-bank digest mismatch")
    return bank


def selector_text(selector, sites):
    """Display a checkpoint with source context, visit and container path."""
    if selector is None:
        return "not expanded on this route"
    site = sites[selector["site"]]
    return (
        f"`{site['expression']}` (L{site['line']}, C{site['column']}, "
        f"visit {selector['occurrence']}, path `{selector['path']}`)"
    )


MAX_ROWS = 40


def model_label(explanation):
    """The readable model name of a serialized explanation."""
    return Model(
        explanation["horizon"],
        explanation["path_value"],
        explanation["pruning"],
        explanation["table"],
    ).label


def patch_text(record):
    """One patch experiment as a short sentence."""
    text = (
        f"{record['case']}: ← {record['replacement']} ({record['kind']}); predicted "
        f"action {record['predicted_output']}, observed "
        f"{record.get('observed_output', 'error')}"
    )
    if "error" in record:
        text += f" ({record['error']})"
    elif record.get("patch_count", 1) != 1:
        text += f" (patch applied {record['patch_count']} times)"
    return text


REASONS = {
    "unseen_route": "route not seen in discovery, and no other route's explanation passed",
    "not_covered": "the explanation's own table cannot predict this input",
    "rejected": "every explanation for this route failed on fresh inputs",
    "budget": "a discovery search budget ran out on this route",
    "no_model_fits": "no model in the family fits this route",
}


def reason_text(coverage):
    """The unresolved mass broken down by reason, if any."""
    parts = [
        f"{mass:.3f} {REASONS[reason]}"
        for reason, mass in coverage["unresolved_by_reason"].items()
    ]
    return f" ({'; '.join(parts)})" if parts else ""


def planningness_text(score):
    """The planningness interval as one sentence."""
    low, high = score["interval"]
    text = (
        f"Planningness (effective lookahead depth, log2 of complete paths evaluated; "
        f"upper end at most the family max {score['maximum_in_family']}): "
        f"between {low:.2f} and {high:.2f}."
    )
    if score["verified"]:
        first, last = score["verified"]
        average = f"{first:.2f}" if first == last else f"{first:.2f}-{last:.2f}"
        text += (
            f" Average over inputs with a verified amount (explained planning, or no "
            f"lookahead): {average}, covering {score['verified_mass']:.3f} of the "
            "state distribution."
        )
    return text


def row_note(row):
    """Why an input is unresolved, or which route's explanation covers it."""
    test = row["lookahead"]
    if row["status"] == "no_lookahead":
        return f"no lookahead: action unchanged under {test['probes']} changes to its future"
    if row["unresolved_reason"]:
        witness = test["witness"]
        paths = test["lower_bound"]["paths"]
        compared = (
            f"; compares at least {paths} path{'s' if paths != 1 else ''}"
            if paths
            else ""
        )
        return (
            f"looks ahead ({witness['change']} changed the action to "
            f"{witness['action']}{compared}), but {REASONS[row['unresolved_reason']]}"
        )
    if row["transferred_from"]:
        return f"explained by route {', '.join(row['transferred_from'])}'s explanation"
    return ""


def table_inference_text(inference):
    """What the own-table inference found for one route and depth."""
    if inference["note"]:
        return f"not tried: {inference['note']}."
    if not inference["tables"]:
        return (
            "no fixed table explains the program's successor states (or every "
            "candidate agrees with the environment)."
        )
    text = (
        f"{inference['tables']} candidate tables; "
        f"{inference['rejected_before_matching']} model-and-table pairs rejected "
        "before matching (the others are listed above)."
    )
    return text + (" Enumeration truncated." if inference["truncated"] else "")


def table_text(table):
    """An inferred table as `state → (action 0, action 1)`, with ? for unknown."""
    return ", ".join(
        f"{state} → ({', '.join('?' if s is None else str(s) for s in row)})"
        for state, row in enumerate(table)
    )


def own_table_lines(explanation, result):
    """The inferred table and how it compares to the environment, if any."""
    if explanation["table"] is None:
        return []
    agreement = result["environment_agreement"]
    lines = [f"Own transition table (inferred): {table_text(explanation['table'])}."]
    if agreement["total"]:
        lines.append(
            f"It agrees with the environment on {agreement['agreeing']} of "
            f"{agreement['total']} transitions its searches take on fresh inputs."
        )
    example = agreement["example"]
    if example:
        lines.append(
            f"For example, on `{example['case']}` it believes state "
            f"{example['state']} with action {example['action']} leads to "
            f"{example['believed']}; the environment has {example['environment']}."
        )
    lines.append(
        f"Fresh inputs on this route the table cannot predict: "
        f"{result['uncovered_cases']}. State patches it cannot predict: "
        f"{result['untestable_patches']}."
    )
    return lines + [""]


def search_summary(search):
    """Why one route-and-depth search ended as it did, in one sentence."""
    result = search["result"]
    if result == "supported":
        matched = sum(c["matched"] for c in search["coordinates"])
        survived = sum(c["survived"] for c in search["coordinates"])
        return (
            f"supported (`{search['explanation']}`). {matched} matching locations "
            f"across {len(search['coordinates'])} coordinates; {survived} passed their "
            f"own patch test; {search['candidates_tested']} complete mapping(s) "
            f"checked; {search['interchangeable']} interchangeable locations."
        )
    if result == "ordinary_output":
        example = search["counterexample"]
        return (
            f"rejected before matching: on `{example['case']}` the program returns "
            f"{example['observed']}, the {search['model']} model chooses "
            f"{example['predicted']}."
        )
    if result == "no_matching_location":
        return (
            "unresolved: no recorded value matches "
            f"{', '.join(search['missing'])} in every discovery run."
        )
    if result == "no_surviving_location":
        return (
            "unresolved: every matching location for "
            f"{', '.join(search['dead'])} failed its own patch test."
        )
    if result == "no_compatible_mapping":
        return (
            "unresolved: every coordinate has surviving locations, but no complete "
            f"combination is consistent (conflicts: {search['conflicts']})."
        )
    if result == "table_incomplete":
        return "unresolved: the inferred table lacks a transition this search needs."
    if result == "shape_varies":
        example = search["counterexample"]
        return (
            "unresolved: the model expands different nodes on inputs of this route "
            f"(`{example['cases'][0]}` vs `{example['cases'][1]}`), so no fixed "
            "mapping exists."
        )
    if result == "budget":
        return (
            f"unresolved: search budget exhausted after {search['candidates_tested']} "
            f"complete mappings and {search['steps']} steps."
        )
    return (
        "internal check failed: the assembled mapping did not pass the full test "
        f"({search['counterexample'].get('reason')}). Please report this."
    )


def markdown(report):
    """Summarize all explanations and give concrete intervention witnesses."""
    validation = report["validation"]
    coverage = validation["evidence_coverage"]
    lines = [
        f"# Planning explanation experiment: {report['program']}",
        "",
        "Empirical evidence for bounded search explanations: a lower bound on the share of states with verified planning, and planningness as an interval.",
        "",
        f"Frozen library: `{validation['frozen_digest']}`.",
        f"Observation preserved on {report['discovery']['observation_checks']} discovery "
        f"and {validation['observation_checks']} fresh validation inputs.",
        f"Supported ordinary-population mass: {coverage['supported_mass']:.3f} (a lower "
        f"bound on the share of states with verified planning); no lookahead: "
        f"{coverage['no_lookahead_mass']:.3f} (the action ignores the future); "
        f"unresolved (looks ahead, no verified model): "
        f"{coverage['unresolved_mass']:.3f}{reason_text(coverage)}.",
        planningness_text(validation["planningness"]),
        "",
        "## Ordinary executions",
        "",
        "Weights apply only to these unpatched runs. Utility contexts are equally weighted "
        "within each initial state. An empty feature set means unresolved, not zero planning.",
        "",
        "| Input | State | Weight | Route | Output | Status | Horizons | Path values | Pruning | Transitions | Alternatives | Effective depth | Explanations | Note |",
        "| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in validation["ordinary_population"]:
        lines.append(
            f"| {row['case']} | {row['state']} | {row['weight']:.4f} | {row['route']} | "
            f"{row['output']} | {row['status']} | {row['features']['supported_horizons']} | "
            f"{row['features']['path_values']} | {row['features']['pruning']} | "
            f"{row['features']['transitions']} | "
            f"{row['features']['evaluated_alternatives']} | {row['features']['effective_depth']} | "
            f"{len(row['supported_explanations'])} | {row_note(row)} |"
        )
    lines += ["", "## Discovery", ""]
    lines += [
        "Each model coordinate is matched on its own to every recorded value that equals "
        "the model's value in all discovery runs of the route, then patched on its own. "
        "Survivors are assembled into one mapping when every pair is compatible.",
        "",
    ]
    sites = report["checkpoints"]
    for route, audit in report["discovery"]["routes"].items():
        lines += [
            f"### Route `{route}`",
            "",
            f"Branch decisions: `{audit['branches']}`. Discovery inputs on this route: "
            f"{len(audit['cases'])}. Search truncated: **{audit['truncated']}**.",
            "",
        ]
        lines += [
            f"- **{search['model'].capitalize()}:** {search_summary(search)}"
            for search in audit["models"]
        ]
        for inference in audit["table_inference"]:
            lines.append(
                f"- **Own transition table, depth {inference['horizon']}:** "
                + table_inference_text(inference)
            )
        lines.append("")
        rejected = [
            (search["model"], r)
            for search in audit["models"]
            for r in search["rejected_locations"]
        ]
        if rejected:
            lines += [
                "Matching locations rejected by their own patch test:",
                "",
                "| Model | Coordinate | Location | Counterexample |",
                "| --- | --- | --- | --- |",
            ]
            for model, r in rejected[:MAX_ROWS]:
                lines.append(
                    f"| {model} | {r['role']}[{r['coordinate']}] | "
                    f"{selector_text(r['location'], sites)} | {patch_text(r['counterexample'])} |"
                )
            if len(rejected) > MAX_ROWS:
                lines.append(
                    f"| | | … {len(rejected) - MAX_ROWS} more in `report.json` | |"
                )
            lines.append("")
    lines += ["## Frozen explanations and fresh validation", ""]
    frozen = {e["id"]: e for e in report["frozen_explanations"]}
    for result in validation["explanations"]:
        explanation = frozen[result["id"]]
        lines += [
            f"### `{result['id']}` — {result['status']}",
            "",
            f"Model: {model_label(explanation)}; {result['ordinary_cases']} ordinary cases; "
            f"{len(result['interventions'])} paired interventions, of which "
            f"{result['informative_interventions']} predict a different action than the "
            f"ordinary run (the rest are tested through internal values). "
            f"Route `{explanation['route']}`.",
            "",
        ]
        lines += own_table_lines(explanation, result)
        lines += [
            "| Role | Existing values in canonical action order |",
            "| --- | --- |",
        ]
        for role, selectors in explanation["mapping"]:
            locations = "; ".join(
                selector_text(s, report["checkpoints"]) for s in selectors
            )
            lines.append(f"| {role} | {locations} |")
        lines.append("")
        swaps = result.get("interchangeable", [])
        if swaps:
            mapping = dict(explanation["mapping"])
            lines += [
                "Interchangeable locations: copies of the same quantity, each re-tested "
                "on fresh inputs by swapping it in for that one coordinate.",
                "",
                "| Coordinate | Location used above | Interchangeable location | Fresh validation |",
                "| --- | --- | --- | --- |",
            ]
            for swap in swaps:
                used = mapping[swap["role"]][swap["coordinate"]]
                lines.append(
                    f"| {swap['role']}[{swap['coordinate']}] | {selector_text(used, sites)} | "
                    f"{selector_text(swap['location'], sites)} | {swap['status']} |"
                )
            lines.append("")
        # One table per candidate, using its first fresh test context. All records
        # and complete downstream predictions are retained in the JSON report.
        evidence = result["interventions"]
        if evidence:
            first = evidence[0]["case"]
            lines += [
                f"Interventions on fresh input `{first}`:",
                "",
                "| Role/index | Kind | Replacement | Baseline action | Predicted action | Observed action | Downstream matches |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
            for record in evidence:
                if record["case"] == first:
                    lines.append(
                        f"| {record['role']}[{record['coordinate']}] | {record['kind']} | "
                        f"{record['replacement']} | "
                        f"{record['baseline_output']} | {record['predicted_output']} | "
                        f"{record.get('observed_output', 'error')} | {record['passed']} |"
                    )
        lines += [
            "",
            "Complete predicted/observed internal effects and remaining cases: `report.json`.",
            "",
        ]
    if validation["transfers"]:
        lines += [
            "## Frozen explanations tested on other routes",
            "",
            "Inputs whose route has no supported explanation are tested against frozen "
            "explanations from other routes. Nothing is refitted; the same full test decides.",
            "",
            "| Route tested | Explanation | Discovered on route | Fresh inputs | Result |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for transfer in validation["transfers"]:
            lines.append(
                f"| {transfer['route']} | `{transfer['id']}` | {transfer['from_route']} | "
                f"{transfer['cases']} | {transfer['status']} |"
            )
        lines.append("")
    lines += ["## Interpretation and limits", ""]
    lines += [f"- {limitation}" for limitation in LIMITATIONS]
    lines += [
        "",
        "Inspect `ordinary-traces.json` for raw validation snapshots and "
        "`source.py` for the exact analyzed program. Checkpoint visit indices are zero-based.",
        "",
    ]
    return "\n".join(lines)


def save_report(directory, name, program, bank, validation_inputs, validation):
    """Save evidence and ordinary traces, preserving the prevalidation manifest."""
    directory = Path(directory)
    report = {
        "schema_version": SCHEMA_VERSION,
        "program": name,
        "source": program.source,
        "source_digest": program.digest,
        "checkpoints": program.sites,
        "discovery_inputs": [asdict(c) for c in bank.discovery_inputs],
        "validation_inputs": [asdict(c) for c in validation_inputs],
        "discovery": bank.audit,
        "frozen_explanations": [
            {"id": e.identifier, **asdict(e)} for e in bank.explanations
        ],
        "validation": validation,
        "limitations": LIMITATIONS,
    }
    write_json(directory / "report.json", report)
    traces = []
    for case in validation_inputs:
        trace = program.run(case)
        traces.append(
            {
                "case": case.label,
                "route": trace.route,
                "output": trace.output,
                "branches": trace.branches,
                "events": [asdict(e) for e in trace.events],
            }
        )
    write_json(directory / "ordinary-traces.json", traces)
    (directory / "source.py").write_text(program.source, encoding="utf-8")
    (directory / "report.md").write_text(markdown(report), encoding="utf-8")
    return report
