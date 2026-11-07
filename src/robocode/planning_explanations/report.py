"""Readable and JSON evidence, with all populations and limitations explicit."""

# Keep literal Markdown table rows and report sentences together.
# pylint: disable=line-too-long

import json
from dataclasses import asdict
from pathlib import Path

from .discovery import Explanation, FrozenBank
from .instrument import Selector
from .task import Case

LIMITATIONS = [
    "Empirical support within a restricted hypothesis and mapping family, not a proof.",
    "Hypotheses: exhaustive depth-1/depth-2 search, maximum backup, first-action ties.",
    "Mappings: complete 2/4-value site-occurrence series or container leaves, permuted.",
    "No arbitrary cross-site scalar combinations, helpers, recursion, imports, or comprehensions.",
    "Routes are executed if-statement decisions; unseen or unvalidated routes stay unresolved.",
    "Two state replacements and low/high value replacements per mapped coordinate are tested.",
    "Only the configured number of cases per route receives paired interventions.",
    "Multiple alignments may describe the same computation; their feature counts are not added.",
    "A selected value is an observed occurrence, not a count of unique arithmetic operations.",
    "The action cache and analyzer rollouts are external preparation, never counted as planning.",
    "Evidence coverage is not planningness. Unresolved amount is null, not zero.",
    "Trusted small programs only: syntax restrictions and event budget are not a security sandbox.",
]


def write_json(path, value):
    """Write deterministic, strict JSON suitable for diffing and reanalysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def frozen_payload(bank):
    """Serialize exactly what is fixed before fresh validation starts."""
    return {
        "schema_version": 1,
        "digest": bank.digest,
        "program_digest": bank.program_digest,
        "discovery_inputs": [asdict(c) for c in bank.discovery_inputs],
        "explanations": [asdict(e) for e in bank.explanations],
    }


def load_frozen(path):
    """Restore a saved bank for validation without repeating discovery."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported frozen-bank schema")
    explanations = tuple(
        Explanation(
            e["route"],
            e["horizon"],
            tuple(
                (
                    role,
                    tuple(
                        Selector(s["site"], s["occurrence"], tuple(s["path"]))
                        for s in selectors
                    ),
                )
                for role, selectors in e["mapping"]
            ),
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
    site = sites[selector["site"]]
    return (
        f"`{site['expression']}` (L{site['line']}, C{site['column']}, "
        f"visit {selector['occurrence']}, path `{selector['path']}`)"
    )


def markdown(report):
    """Summarize all explanations and give concrete intervention witnesses."""
    validation = report["validation"]
    coverage = validation["evidence_coverage"]
    lines = [
        f"# Planning explanation experiment: {report['program']}",
        "",
        "Empirical evidence for a bounded search explanation. No scalar planningness metric.",
        "",
        f"Frozen library: `{validation['frozen_digest']}`.",
        f"Observation preserved on {report['discovery']['observation_checks']} discovery "
        f"and {validation['observation_checks']} fresh validation inputs.",
        f"Supported ordinary-population mass: {coverage['supported_mass']:.3f}; "
        f"unresolved mass: {coverage['unresolved_mass']:.3f}. These are evidence coverage.",
        "",
        "## Ordinary executions",
        "",
        "Weights apply only to these unpatched runs. Utility contexts are equally weighted "
        "within each initial state. An empty feature set means unresolved, not zero planning.",
        "",
        "| Input | State | Weight | Route | Output | Status | Horizons | Alternatives | Explanations |",
        "| --- | ---: | ---: | --- | ---: | --- | --- | --- | ---: |",
    ]
    for row in validation["ordinary_population"]:
        lines.append(
            f"| {row['case']} | {row['state']} | {row['weight']:.4f} | {row['route']} | "
            f"{row['output']} | {row['status']} | {row['features']['supported_horizons']} | "
            f"{row['features']['evaluated_alternatives']} | {len(row['supported_explanations'])} |"
        )
    lines += ["", "## Discovery and rejected candidates", ""]
    for route, audit in report["discovery"]["routes"].items():
        lines += [
            f"### Route `{route}`",
            "",
            f"Branch decisions: `{audit['branches']}`. Inventory: "
            f"{len(audit['inventory'])}/{audit['available_series']} series; "
            f"{audit['role_mappings_tested']} role mappings tested; "
            f"{audit['candidates_tested']} complete hypotheses tested; "
            f"search truncated: **{audit['truncated']}**.",
            "",
        ]
        if not audit["supported_candidates"]:
            lines += [
                "No supported explanation in the searched family. Amount remains unknown.",
                "",
            ]
        rejections = audit["rejected_candidates"]
        if rejections:
            lines += [
                "| Rejected candidate | Horizon | Counterexample |",
                "| --- | ---: | --- |",
            ]
            for rejection in rejections:
                counterexample = rejection.get("counterexample", {})
                detail = rejection["reason"]
                if counterexample:
                    detail += (
                        f": {counterexample['case']}, {counterexample['role']}"
                        f"[{counterexample['coordinate']}] ← {counterexample['replacement']}; "
                        f"predicted output {counterexample['predicted_output']}, "
                        f"observed {counterexample.get('observed_output', 'error')}"
                    )
                    if counterexample.get("predicted_output") == counterexample.get(
                        "observed_output"
                    ):
                        detail += " (internal downstream mismatch)"
                lines.append(
                    f"| {rejection['id']} | {rejection['explanation']['horizon']} | {detail} |"
                )
            lines.append("")
        lines += [
            "Role-filter rejection counts and sampled counterexamples are in `report.json`.",
            "",
        ]
    lines += ["## Frozen explanations and fresh validation", ""]
    frozen = {e["id"]: e for e in report["frozen_explanations"]}
    for result in validation["explanations"]:
        explanation = frozen[result["id"]]
        lines += [
            f"### `{result['id']}` — {result['status']}",
            "",
            f"Horizon {explanation['horizon']}; {result['ordinary_cases']} ordinary cases; "
            f"{len(result['interventions'])} paired interventions. Route `{explanation['route']}`.",
            "",
            "| Role | Existing values in canonical action order |",
            "| --- | --- |",
        ]
        for role, selectors in explanation["mapping"]:
            locations = "; ".join(
                selector_text(s, report["checkpoints"]) for s in selectors
            )
            lines.append(f"| {role} | {locations} |")
        lines.append("")
        # One table per candidate, using its first fresh test context. All records
        # and complete downstream predictions are retained in the JSON report.
        evidence = result["interventions"]
        if evidence:
            first = evidence[0]["case"]
            lines += [
                f"Interventions on fresh input `{first}`:",
                "",
                "| Role/index | Replacement | Baseline action | Predicted action | Observed action | Downstream matches |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
            for record in evidence:
                if record["case"] == first:
                    lines.append(
                        f"| {record['role']}[{record['coordinate']}] | {record['replacement']} | "
                        f"{record['baseline_output']} | {record['predicted_output']} | "
                        f"{record.get('observed_output', 'error')} | {record['passed']} |"
                    )
        lines += [
            "",
            "Complete predicted/observed internal effects and remaining cases: `report.json`.",
            "",
        ]
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
        "schema_version": 1,
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
