"""Portable audit reports and paired SVGs, with no plotting dependencies."""

# Exact types preserve raw key/container identity; long lines are HTML fragments.
# pylint: disable=unidiomatic-typecheck,line-too-long

import html
import json
from dataclasses import asdict
from pathlib import Path

from .benchmark import FrozenTaskBank
from .checkpoints import Program
from .discover import FrozenBank
from .evidence import summarize_findings
from .heuristics import HeuristicSpec
from .tasks import TaskCase, case_from_data
from .verify import ERRORS


def write_json(path: Path, value) -> None:
    """Write human-readable machine data."""
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def concrete_value(value) -> object:
    """Keep dictionary key types and tuple/list distinctions in raw trace exports."""
    if type(value) is dict:
        return {
            "$dict": [[concrete_value(k), concrete_value(v)] for k, v in value.items()]
        }
    if type(value) is tuple:
        return {"$tuple": [concrete_value(v) for v in value]}
    if type(value) is list:
        return [concrete_value(v) for v in value]
    return value


def trace_data(program: Program, positions) -> dict:
    """Export actual ordinary snapshots; alias groups supplement value encoding."""
    traces: list[dict] = []
    for position in positions:
        try:
            trace = program.observe(position)
        except ERRORS as exc:
            traces.append(
                {
                    "input": (
                        asdict(position) if isinstance(position, TaskCase) else position
                    ),
                    "status": "unresolved",
                    "reason": str(exc),
                    "snapshots": [],
                }
            )
            continue
        snapshots = []
        for snapshot in trace.snapshots:
            row = asdict(snapshot)
            row["values"] = concrete_value(snapshot.values)
            snapshots.append(row)
        traces.append(
            {
                "input": (
                    asdict(position) if isinstance(position, TaskCase) else position
                ),
                "position": trace.position,
                "output": trace.output,
                "snapshots": snapshots,
            }
        )
    return {"source_digest": program.digest, "sites": program.sites, "traces": traces}


def _text(x, y, value, size=13, color="#172b4d") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">'
        f"{html.escape(str(value))}</text>"
    )


def _configuration(state: dict | None, x: int, y: int) -> str:
    """Draw recorded nodes/current-parent edges; never complete missing trees."""
    if state is None:
        return _text(x, y + 35, "No decoded configuration recorded", color="#9c2626")
    nodes = [node for node, _ in state["parents"]]
    positions = {node: (x + 48 + i * 72, y + 70) for i, node in enumerate(nodes)}
    parts = [_text(x, y + 16, f'frontier (ordered): {state["frontier"]}')]
    for node, parent in state["parents"]:
        if parent not in positions:
            continue
        px, py = positions[parent]
        nx, ny = positions[node]
        parts.append(
            f'<path d="M {px} {py - 18} Q {(px + nx) / 2} {py - 55} {nx} {ny - 18}" '
            'fill="none" stroke="#677892" marker-end="url(#arrow)"/>'
        )
    for node, (nx, ny) in positions.items():
        fill = "#cde9ff" if node in state["frontier"] else "#eef1f5"
        parts.append(
            f'<circle cx="{nx}" cy="{ny}" r="22" fill="{fill}" stroke="#406080"/>'
        )
        parts.append(_text(nx - 17, ny + 4, node, 10))
        if node == state["root"]:
            parts.append(_text(nx - 14, ny + 38, "root", 11))
    parts.append(_text(x, y + 133, f'{state["phase"]}; output={state["output"]}', 12))
    if "costs" in state:
        for offset, key, title in (
            (153, "costs", "g: cost so far"),
            (171, "heuristic", "h: estimate"),
        ):
            values = ", ".join(f"{node}={value}" for node, value in state[key])
            parts.append(_text(x, y + offset, f"{title}: {values}", 12))
    return "".join(parts)


def paired_svg(comparison: dict, title: str) -> str:
    """Show independent predictions and decoded observations at every checkpoint."""
    rows = comparison.get("alignments", [])
    row_height = 245 if any("costs" in r["predicted"] for r in rows) else 190
    height = 100 + row_height * max(1, len(rows))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 {height}" '
        'role="img" aria-label="Predicted and observed search configurations">',
        '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6" fill="#677892"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif">',
        _text(20, 28, title, 18),
        _text(20, 54, "Predicted by independent Q", 15),
        _text(600, 54, "Observed through frozen mapping α", 15),
        _text(
            20,
            78,
            "Blue = frontier; arrows = current retained parents; root marker is not an expansion count.",
            12,
        ),
    ]
    for i, row in enumerate(rows):
        y = 103 + i * row_height
        status = "PASS" if row["passed"] else "FAIL"
        header = f'Step {row["step"]}: {row["operation"]} | {row["checkpoint"]} #{row["occurrence"]} | {status}'
        if "edit_correspondence" in row:
            lift = "PASS" if row["edit_correspondence"] else "FAIL"
            header += (
                f' | EDIT: lift {lift}; prior frontier {row["before_edit"]["frontier"]}'
            )
        parts += [
            _text(20, y, header, 13),
            _configuration(row["predicted"], 20, y + 8),
            _configuration(row.get("observed"), 600, y + 8),
            f'<line x1="20" y1="{y + row_height - 16}" x2="1140" y2="{y + row_height - 16}" stroke="#dce2eb"/>',
        ]
        if row_height > 190 and i:
            parts.append(
                _text(
                    20,
                    y + 207,
                    f'Selected: {row.get("selected")}; '
                    f'priorities before selection: {row.get("priorities", [])}; '
                    f'improved existing paths: {row.get("improved", [])}',
                    12,
                )
            )
    if not rows:
        parts.append(
            _text(
                20, 140, "No aligned configurations; inspect the counterexample JSON."
            )
        )
    parts.append("</g></svg>")
    return "".join(parts)


def describe_truncation(audit: dict) -> str:
    """Name each reached discovery bound; any of them can hide an explanation."""
    if "truncated" not in audit:
        return "audit unavailable (loaded bank)"
    if not audit["truncated"]:
        return "no bound was reached"
    detail = audit.get("truncation")
    if detail is None:
        return "a bound was reached (reason not recorded)"
    reasons = []
    if detail["inventory_sites"]:
        sites = detail["inventory_sites"]
        reasons.append(
            f"value inventory cut at {len(sites)} checkpoint(s): {', '.join(sites)}"
        )
    if detail["binding_limit"]:
        reasons.append("binding limit reached")
    if detail["candidate_budget"]:
        reasons.append("candidate budget exhausted")
    return "; ".join(reasons)


def _details(title: str, value) -> str:
    return (
        f"<details><summary>{html.escape(title)}</summary><pre>"
        f"{html.escape(json.dumps(value, indent=2))}</pre></details>"
    )


def _table(headers, rows) -> str:
    def cells(row, tag) -> str:
        return "".join(
            f'<{tag}>{html.escape(str(v)) if v is not None else "—"}</{tag}>'
            for v in row
        )

    return (
        "<table><thead><tr>"
        + cells(headers, "th")
        + "</tr></thead><tbody>"
        + "".join("<tr>" + cells(row, "td") + "</tr>" for row in rows)
        + "</tbody></table>"
    )


def save_unresolved(directory: Path, name: str, source: str, error: Exception) -> dict:
    """Write an explicit unknown, rather than leave a previous successful report."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename in (
        "ordinary.svg",
        "intervention.svg",
        "rejected-intervention.svg",
        "ordinary-traces.json",
    ):
        (directory / filename).unlink(missing_ok=True)
    (directory / "policy.py").write_text(source, encoding="utf-8")
    summary: dict = {
        "fixture": name,
        "status": "unresolved",
        "selected": None,
        "selected_mode": None,
        "finding": summarize_findings([]),
        "error": f"{type(error).__name__}: {error}",
        "supported_discovery": 0,
        "supported_validation": 0,
        "fixture_ground_truth": None,
        "ordinary_mean_N": None,
        "ordinary_mean_M": None,
        "task_success_rate": None,
        "supported_mean_feature_ranges": {},
        "planningness": None,
        "attributed_search_expansions": None,
        "surviving_explanations": [],
    }
    write_json(directory / "report.json", {"schema": 2, **summary})
    message = f'Unresolved: {summary["error"]}. No planning finding or numerical score is assigned.'
    (directory / "report.md").write_text(message + "\n", encoding="utf-8")
    (directory / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Unresolved experiment</h1><p>'
        + html.escape(message)
        + "</p>",
        encoding="utf-8",
    )
    return summary


def save_report(
    directory: Path,
    name: str,
    program: Program,
    bank: FrozenBank | FrozenTaskBank,
    validation: dict,
    fixture_truth: str | None = None,
) -> dict:
    """Export all ambiguity, all validation interventions, and rejection evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("ordinary.svg", "intervention.svg", "rejected-intervention.svg"):
        (directory / filename).unlink(missing_ok=True)
    program_path = directory / "policy.py"
    program_path.write_text(program.source, encoding="utf-8")
    report = {
        "schema": 2,
        "fixture": name,
        "fixture_ground_truth": {
            "label": fixture_truth,
            "source": "Known fixture construction; never supplied to discovery",
        },
        "frozen": bank.manifest(),
        "discovery": bank.audit,
        "validation": validation,
        "limits": [
            "Finite empirical evidence, not a universal equivalence proof",
            "Bounded ingredient and mapping language; absence is unresolved",
            "N/M describe each supported explanation, not a final planningness metric",
        ],
    }
    write_json(
        directory / "ordinary-traces.json",
        trace_data(
            program,
            (
                *bank.discovery_inputs,
                *(
                    tuple(case_from_data(c) for c in validation["cases"])
                    if isinstance(bank, FrozenTaskBank)
                    else map(int, validation["distribution"])
                ),
            ),
        ),
    )
    explanations = {h.identifier: h for h in bank.explanations}
    selected = next(
        (r for r in validation["explanations"] if r["id"] == bank.selected), None
    )
    summary = {
        "fixture": name,
        "supported_discovery": len(bank.explanations),
        "supported_validation": sum(
            r["status"] == "supported" for r in validation["explanations"]
        ),
        "selected": bank.selected,
        "status": selected["status"] if selected else "unresolved",
        "selected_mode": explanations[selected["id"]].spec.mode if selected else None,
        "ordinary_mean_N": selected["ordinary_mean_N"] if selected else None,
        "ordinary_mean_M": selected["ordinary_mean_M"] if selected else None,
        "discovery_truncated": bank.audit.get("truncated"),
        "discovery_truncation": bank.audit.get("truncation"),
        "planningness": None,
        "finding": validation["finding"],
        "fixture_ground_truth": fixture_truth,
        "task_success_rate": validation.get("task_performance", {}).get("success_rate"),
        "selected_mechanism_status": (
            selected.get("mechanism_status") if selected else "unresolved"
        ),
        "attributed_search_expansions": (
            (selected.get("mechanism_mean_features") or {}).get("search_expansions")
            if selected and selected["status"] == "supported"
            else None
        ),
        "surviving_explanations": [
            {
                "id": r["id"],
                "mode": explanations[r["id"]].spec.mode,
                "task_grounding": r["environment"]["status"],
                "joint_status": r["status"],
                "mean_features": r.get("mechanism_mean_features"),
                "selection": explanations[r["id"]].spec.selection,
                "goal_timing": getattr(
                    explanations[r["id"]].spec, "goal_timing", "selection"
                ),
                "relaxation": getattr(
                    explanations[r["id"]].spec, "relaxation", "first"
                ),
                "tie": getattr(explanations[r["id"]].spec, "tie", None),
            }
            for r in validation["explanations"]
            if r.get("mechanism_status", r["status"]) == "supported"
        ],
    }
    supported_means = [
        r["mechanism_mean_features"]
        for r in validation["explanations"]
        if r["status"] == "supported" and r.get("mechanism_mean_features") is not None
    ]
    summary["supported_mean_feature_ranges"] = (
        {
            feature: [
                min(r[feature] for r in supported_means),
                max(r[feature] for r in supported_means),
            ]
            for feature in (
                "N",
                "M",
                "search_expansions",
                "max_discovery_depth",
                "generated_alternatives",
                "expansion_events",
                "reexpansions",
                "cost_improvements",
            )
        }
        if supported_means
        else {}
    )
    report["summary"] = summary
    write_json(directory / "report.json", report)
    discovery_labels = [
        case.name if isinstance(case, TaskCase) else case
        for case in bank.discovery_inputs
    ]
    lines = [
        f"# Search explanations: {name}",
        "",
        f'**Finding across the frozen bank: {validation["finding"]["label"]}.**',
        "",
        f'Known fixture label: {fixture_truth or "not supplied"} (not an analyzer inference).',
        "",
        f'Validation of the preselected explanation `{bank.selected}`: **{summary["status"]}**.',
        "",
        f"Frozen digest: `{bank.fingerprint}`.",
        "",
        f'Discovery inputs: {discovery_labels}; validation distribution: {validation["distribution"]}.',
        "",
        "N counts distinct expanded environment nodes on aligned original execution intervals; "
        "M=max(N−1,0). Neither is a final planningness metric. Interventions and external witnesses "
        "have no distribution weight. A reactive explanation does not exclude richer explanations.",
        "",
    ]
    body = [
        f"<h1>Search explanations: {html.escape(name)}</h1>",
        f'<p><b>Finding across the frozen bank: {validation["finding"]["label"]}.</b></p>',
        f'<p>Known fixture label: {html.escape(fixture_truth or "not supplied")}. '
        "This label is from fixture construction and is never given to the analyzer.</p>",
        "<p>Compare ordinary executions and interventions using fixed mappings. "
        "All supported discovery explanations are retained; selection is frozen before fresh validation.</p>",
        f'<p>Validation of the preselected explanation <code>{bank.selected}</code>: <b>{summary["status"]}</b>. '
        f'Discovery inputs: {html.escape(str(discovery_labels))}. Fresh distribution: {html.escape(str(validation["distribution"]))}.</p>',
        "<p>Mechanism support, task grounding and task success are separate. M=0 does not imply "
        "reactivity. The finding considers all frozen hypotheses; the selected hypothesis is never replaced.</p>",
        "<p>N = distinct expanded environment nodes; M = max(N−1, 0). "
        "These are explanation-dependent features, not a final planningness metric. "
        "Goal selections are not expansions. Patched runs and witness generation do not enter the mean.</p>",
        "<p><b>Attributed search expansions</b> counts matched original-program expansions that generate "
        "environment successors. It excludes reactive Action leaves. It is a feature, not a universal planning score.</p>",
        "<p>Per-case counts are provisional until causal mechanism verification passes. "
        "Below, the range across fully supported frozen explanations describes ambiguity; "
        "it does not replace the preselected hypothesis or choose a different explanation per input.</p>",
        _details(
            "Mean feature ranges across supported frozen explanations",
            summary["supported_mean_feature_ranges"],
        ),
        '<p><a href="report.json">Full JSON</a> · <a href="frozen.json">Frozen bank</a> · '
        '<a href="ordinary-traces.json">Raw ordinary snapshots</a> · <a href="policy.py">Original policy</a></p>',
    ]
    if "route" in validation:
        body.insert(
            1,
            '<p><a href="../report.html">Back to route overview</a>. '
            "This bank is evaluated only on cases assigned by its frozen prefix gate.</p>",
        )
        body.insert(2, _details("Frozen prefix gate", validation["prefix"]))
    survivor_headers = (
        "Surviving explanation",
        "Family",
        "Selection / goal test / update",
        "Task grounding",
        "Joint validation",
        "Mean N",
        "Mean M",
        "Attributed search expansions",
    )
    survivor_rows = [
        (
            r["id"],
            r["mode"],
            f'{r["selection"]} / {r["goal_timing"]} / {r["relaxation"]}',
            r["task_grounding"],
            r["joint_status"],
            (r["mean_features"] or {}).get("N"),
            (r["mean_features"] or {}).get("M"),
            (r["mean_features"] or {}).get("search_expansions"),
        )
        for r in summary["surviving_explanations"]
    ]
    body += [
        "<h2>Surviving mechanism explanations</h2>",
        "<p>All rows were frozen before validation. Task grounding is checked separately. "
        "These rows do not replace a failed preselection.</p>",
        _table(survivor_headers, survivor_rows),
    ]
    lines += [
        "Surviving mechanism explanations (task grounding shown separately):",
        "",
        "| " + " | ".join(survivor_headers) + " |",
        "| " + " | ".join("---" for _ in survivor_headers) + " |",
    ]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in survivor_rows]
    lines.append("")
    if "task_performance" in validation:
        performance = validation["task_performance"]
        lines += [
            "",
            f'Separate closed-loop task success rate: {performance["success_rate"]}.',
            "",
        ]
        body += [
            f'<h2>Separate task success: {performance["success_rate"]}</h2>',
            "<p>These original-policy rollouts test reaching the goal. Their additional invocations "
            "do not enter the single-invocation N/M averages.</p>",
            _details("Frozen task cases and transitions", validation["cases"]),
            _details("Original-policy success rollouts", performance),
            _table(
                ("Case", "Goal", "First action", "Goal reached", "Outcome"),
                [
                    (
                        r["case"],
                        r["goal"],
                        r["actions"][0] if r["actions"] else None,
                        r["goal_reached"],
                        r.get("reason", r["status"]),
                    )
                    for r in performance["cases"]
                ],
            ),
        ]
    lines += [
        "| Explanation | Family | Mechanism | Task grounding | Joint validation | Mean N | Mean M | Probes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in validation["explanations"]:
        identifier = result["id"]
        hypothesis = explanations[identifier]
        lines.append(
            f'| `{identifier}` | {hypothesis.spec.mode} | {result.get("mechanism_status")} | '
            f'{result["environment"]["status"]} | {result["status"]} | '
            f'{result["ordinary_mean_N"]} | {result["ordinary_mean_M"]} | {len(result["interventions"])} |'
        )
        body += [
            f"<h2>{identifier} — {hypothesis.spec.mode}</h2>",
            f'<p>Mechanism: <b>{result.get("mechanism_status")}</b>; '
            f'task grounding: <b>{result["environment"]["status"]}</b>; joint validation: <b>{result["status"]}</b>.</p>',
            f'<p>Mean N: {result["ordinary_mean_N"]}; mean M: {result["ordinary_mean_M"]}. '
            f'{len(result["interventions"])} validation interventions.</p>',
            _details("Ingredients and mapping", asdict(hypothesis)),
            _details(
                "Ordinary mechanism feature means (separate from task grounding)",
                result.get("mechanism_mean_features"),
            ),
            _details(
                "Environment meaning and trajectory witnesses",
                result.get("environment"),
            ),
            _table(
                (
                    "Ordinary input",
                    "N",
                    "M",
                    "Attributed search expansions",
                    "Discovery depth",
                    "Expanded depth",
                    "Generated alternatives (occurrences)",
                    "Decision-changing probes",
                ),
                [
                    (
                        r.get("case", r["position"]),
                        r.get("N"),
                        r.get("M"),
                        r.get("search_expansions"),
                        r.get("max_discovery_depth"),
                        r.get("max_expanded_depth"),
                        r.get("generated_alternatives"),
                        r.get("decision_effects"),
                    )
                    for r in result["ordinary"]
                ],
            ),
        ]
        if isinstance(hypothesis.spec, HeuristicSpec):
            body += [
                f"<p><b>Selection: {hypothesis.spec.selection}; goal test: {hypothesis.spec.goal_timing}; "
                f"path update: {hypothesis.spec.relaxation}; priority ties: {hypothesis.spec.tie}.</b></p>",
                "<p>Unit-cost edges. Supplied heuristic values need not be admissible. "
                "Goal-on-generation stopping does not establish shortest-path optimality. "
                "Expansion events count revisits; N counts distinct expanded states.</p>",
                _table(
                    (
                        "Ordinary case",
                        "Expansion events",
                        "Distinct expanded states N",
                        "Re-expansions",
                        "Cost improvements",
                    ),
                    [
                        (
                            r.get("case", r["position"]),
                            r.get("expansion_events"),
                            r.get("N"),
                            r.get("reexpansions"),
                            r.get("cost_improvements"),
                        )
                        for r in result["ordinary"]
                    ],
                ),
            ]
        for ordinary in result["ordinary"]:
            title = f'Ordinary {ordinary.get("case", "position " + str(ordinary["position"]))}; N={ordinary.get("N")}, M={ordinary.get("M")}'
            graphic = paired_svg(ordinary, title)
            body.append(
                f'<details><summary>{html.escape(title)}</summary>{graphic}{_details("Alignment data", ordinary)}</details>'
            )
            if identifier == bank.selected and ordinary is result["ordinary"][0]:
                (directory / "ordinary.svg").write_text(graphic, encoding="utf-8")
        representative = next(
            (r for r in result["interventions"] if r["informative"] and r["step"] > 0),
            next(iter(result["interventions"]), None),
        )
        for number, intervention in enumerate(result["interventions"], 1):
            edit = intervention["edit"]
            value = (
                f' ({edit["value"]})' if edit["operation"] in ("set", "drop") else ""
            )
            title = (
                f'Intervention {number}/{len(result["interventions"])}: {intervention.get("case", "p=" + str(intervention["position"]))}; '
                f'{edit["operation"]} {edit["variable"]}{value} at step {intervention["step"]}'
            )
            graphic = paired_svg(intervention.get("comparison", {}), title)
            effect = (
                f'Baseline output {intervention["baseline_output"]}; predicted {intervention["predicted_output"]}; '
                f'observed {intervention.get("observed_output")}. '
                f'Prefix preserved: {intervention.get("prefix_preserved")}; passed: {intervention["passed"]}.'
            )
            body.append(
                f"<details><summary>{html.escape(title)}</summary><p>{effect}</p>{graphic}"
                f'{_details("Full intervention evidence", intervention)}</details>'
            )
            if identifier == bank.selected and intervention is representative:
                (directory / "intervention.svg").write_text(graphic, encoding="utf-8")
    lines += [
        "",
        f"Ambiguity: {len(bank.explanations)} supported discovery explanations retained.",
        "",
        f"Discovery bounds: {describe_truncation(bank.audit)}.",
        "",
        "Open **report.html** for every ordinary trace and every validation intervention, with "
        "predicted/observed trees side by side. **ordinary.svg** and **intervention.svg** are "
        "representative views of the frozen selected explanation. **report.json** also contains "
        "all development probes and rejected-candidate counterexamples. **ordinary-traces.json** "
        "contains original snapshots, checkpoint source lines, and alias groups.",
    ]
    body += [
        "<h2>Rejected candidates and search limits</h2>",
        "<p>No supported candidate within these bounds means unresolved. Rejected graph search "
        "does not establish that a program never plans. Reactive baselines are intentionally retained.</p>",
        f"<p>Discovery bounds: {html.escape(describe_truncation(bank.audit))}.</p>",
        _details(
            "Discovery budgets and counts",
            {
                k: v
                for k, v in bank.audit.items()
                if k not in ("supported", "rejected", "ingredient_rejections")
            },
        ),
        _details("Rejected computational candidates", bank.audit.get("rejected", [])),
        _details(
            "Rejected environment ingredients",
            bank.audit.get("ingredient_rejections", []),
        ),
    ]
    rejection = next(
        (
            r
            for r in bank.audit.get("rejected", [])
            if r["reason"] == "intervention"
            and r["explanation"]["spec"]["mode"] == "graph"
            and r["explanation"]["spec"]["fallback"] == 0
        ),
        None,
    )
    if rejection:
        counterexample = rejection["counterexample"]
        graphic = paired_svg(
            counterexample.get("comparison", {}),
            "Rejected graph explanation: intervention counterexample",
        )
        (directory / "rejected-intervention.svg").write_text(graphic, encoding="utf-8")
        body += [
            "<h2>One rejected graph explanation</h2>",
            "<p>This candidate matched ordinary executions but failed a paired intervention.</p>",
            graphic,
            _details("Candidate and counterexample", rejection),
        ]
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>Search explanations: {html.escape(name)}</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#172b4d}"
        "details{border:1px solid #ccd5df;border-radius:6px;padding:12px;margin:12px 0}"
        "summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "svg{width:100%;height:auto}a{color:#1460a0}"
        "table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}"
        "th,td{text-align:left;padding:8px;border-bottom:1px solid #ccd5df}th{background:#eef4fa}</style>"
        + "\n".join(body)
        + "</html>"
    )
    (directory / "report.html").write_text(page, encoding="utf-8")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
