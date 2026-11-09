"""Route overview linked to complete per-route mappings and intervention reports."""

# Long lines are HTML fragments, as in the shared paired-SVG report writer.
# pylint: disable=line-too-long

import html
from pathlib import Path

from .checkpoints import Program
from .report import _details, _table, save_report, trace_data, write_json
from .routes import FrozenRouteBank


def save_route_report(
    directory: Path,
    name: str,
    program: Program,
    bank: FrozenRouteBank,
    validation: dict,
    fixture_truth: str | None = None,
) -> dict:
    """Retain unknown route mass and show each frozen bank without test-time selection."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "policy.py").write_text(program.source, encoding="utf-8")
    summaries = []
    for route, result in zip(bank.routes, validation["routes"]):
        child_dir = directory / route.identifier
        route.bank.save(child_dir / "frozen.json")
        summaries.append(
            save_report(
                child_dir,
                f"{name}: route {route.identifier}",
                program,
                route.bank,
                result,
                fixture_truth,
            )
        )
    means = validation["preselected_mean_features"] or {}
    assignments = validation["assignments"]
    statuses = {a["preselected_status"] for a in assignments}
    summary = {
        "fixture": name,
        "fixture_ground_truth": fixture_truth,
        "finding": validation["finding"],
        "selected": None,
        "status": (
            "supported"
            if statuses == {"supported"}
            else ("unresolved" if "unresolved" in statuses else "rejected")
        ),
        "supported_discovery": sum(len(r.bank.explanations) for r in bank.routes),
        "supported_validation": sum(s["supported_validation"] for s in summaries),
        "ordinary_mean_N": means.get("N"),
        "ordinary_mean_M": means.get("M"),
        "attributed_search_expansions": means.get("search_expansions"),
        "task_success_rate": validation["task_performance"]["success_rate"],
        "supported_route_coverage": validation["supported_route_coverage"],
        "preselected_supported_coverage": validation["preselected_supported_coverage"],
        "routes": summaries,
        "planningness": None,
    }
    write_json(
        directory / "report.json",
        {
            "schema": 3,
            "summary": summary,
            "frozen": bank.manifest(),
            "discovery": bank.audit,
            "validation": validation,
        },
    )
    # The per-route reports include detailed ordinary traces. Keep an overview
    # copy including unseen routes so their observed prefixes can be inspected.
    write_json(
        directory / "ordinary-traces.json",
        trace_data(program, (*bank.discovery_inputs, *bank.validation_inputs)),
    )
    route_headers = (
        "Route",
        "Weight",
        "Finding across frozen bank",
        "Validation of the preselected explanation",
        "Surviving mechanisms",
        "Mean N",
        "Mean M",
        "Attributed search expansions",
    )
    route_rows = [
        (
            r["route"],
            r["weight"],
            r["finding"]["label"],
            r["preselected_status"],
            len(s["surviving_explanations"]),
            s["ordinary_mean_N"],
            s["ordinary_mean_M"],
            s["attributed_search_expansions"],
        )
        for r, s in zip(validation["routes"], summaries)
    ]
    case_headers = (
        "Ordinary case",
        "Weight",
        "Frozen route",
        "Finding for route",
        "Preselected invocation finding",
        "Preselected N",
        "Preselected M",
        "Attributed search expansions",
    )
    case_rows = [
        (
            a["case"],
            a["weight"],
            a["route"] or "UNSEEN / UNRESOLVED",
            a["finding"]["label"],
            a["preselected_mechanism_finding"],
            (a["features"] or {}).get("N"),
            (a["features"] or {}).get("M"),
            (a["features"] or {}).get("search_expansions"),
        )
        for a in assignments
    ]
    introduction = (
        f'Supported-route coverage: {validation["supported_route_coverage"]:.0%}. '
        f'Preselected-explanation coverage: {validation["preselected_supported_coverage"]:.0%}. '
        "Coverage is the weight of ordinary cases assigned to a route with support. "
        "Unknown routes have missing counts, never zero. Means below use the preselected "
        "explanation per route and require support on every validation case."
    )
    body = [
        f"<h1>Execution routes: {html.escape(name)}</h1>",
        f'<p><b>{html.escape(validation["finding"]["label"])}</b></p>',
        f'<p>Known fixture construction: {html.escape(fixture_truth or "not supplied")}. '
        "This label is never supplied to discovery.</p>",
        f"<p>{introduction}</p>",
        _details(
            "Distribution-wide preselected features",
            validation["preselected_mean_features"],
        ),
        "<p>N and M describe expansions under an explanation. Attributed search expansions "
        "counts expansions generating environment successors; reactive Action leaves contribute zero. "
        "No universal planningness score is assigned.</p>",
        '<p><a href="frozen.json">Frozen routes and banks</a> · <a href="report.json">Full JSON</a> · '
        '<a href="ordinary-traces.json">Original snapshots</a> · <a href="policy.py">Original policy</a></p>',
        "<h2>Route banks and ambiguity</h2>",
        _table(route_headers, route_rows),
        "<p>Route means are conditional on that route. The distribution-wide mean uses the "
        "frozen route weights. Surviving alternatives never replace a failed preselection.</p>",
    ]
    lines = [
        f"# Execution routes: {name}",
        "",
        validation["finding"]["label"],
        "",
        introduction,
        "",
        f"Distribution-wide preselected features: {validation['preselected_mean_features']}",
        "",
        "| " + " | ".join(route_headers) + " |",
        "| " + " | ".join("---" for _ in route_headers) + " |",
    ]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in route_rows]
    for route, result in zip(bank.routes, validation["routes"]):
        link = f"{route.identifier}/report.html"
        body += [
            f'<h3><a href="{link}">Route {route.identifier}: mappings, ordinary trees and every intervention</a></h3>',
            _details(
                "Frozen prefix gate (source checkpoint locations)",
                [{"site": site, **program.sites[site]} for site in route.prefix],
            ),
            _details(
                "Preselected explanation",
                {"id": result["selected"], "status": result["preselected_status"]},
            ),
        ]
        lines += ["", f"[Route {route.identifier}: full evidence]({link})"]
    body += [
        "<h2>Ordinary cases</h2>",
        _table(case_headers, case_rows),
        _details("Observed prefixes and assignments", assignments),
        "<h2>Separate closed-loop task success</h2>",
        _details(
            "Original-policy rollouts, excluded from all feature averages",
            validation["task_performance"],
        ),
        "<h2>Limits</h2><p>Gates inspect only the prefix through the first loop or pre-return checkpoint. "
        "Routes that diverge later are not separately discovered. Interventions retain the ordinary route's explanation. "
        "Finite agreement is empirical support within bounded grammars, not a proof that no other planning exists.</p>",
    ]
    lines += [
        "",
        f"Separate task success: {summary['task_success_rate']}",
        "",
        "The HTML overview links every route's full mappings, ordinary SVGs and all paired interventions.",
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen execution routes</title>'
        "<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#172b4d}"
        "table{border-collapse:collapse;width:100%}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd5df}"
        "th{background:#eef4fa}details{margin:16px 0;padding:12px;border:1px solid #ccd5df}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        + "\n".join(body)
        + "</html>",
        encoding="utf-8",
    )
    return summary
