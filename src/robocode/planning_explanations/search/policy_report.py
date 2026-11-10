"""Milestone 8 reports: per-part tables and a program row with stated rules."""

# Long lines are HTML fragments.
# pylint: disable=line-too-long

import html
from dataclasses import asdict
from pathlib import Path

from .policy_discovery import DEPTH, EFFORT, FrozenPolicyBank
from .policy_runner import PolicyProgram
from .report import _details, _table, write_json

FINDINGS = {
    "search_supported": "Search explanations supported for every search-shaped run",
    "partially_explained": "Some search-shaped runs are unresolved; totals withheld",
    "no_search_parts": "No search-shaped loops within the grammar",
}

RULES = (
    "Effort counts (expansion events, N, generated alternatives, re-expansions, "
    "cost improvements) are summed over an episode's runs; each run's expansions "
    "are counted once, in the call where they happen. Depths are maxima. Planning "
    "events count runs with at least one expansion. Shares are fractions of the "
    "episode's action calls. Means are taken only across validation episodes. A "
    "total is withheld (shown as —) when any search-shaped run is unresolved: an "
    "unresolved run is not zero. Attributed effort is supported effort, not an "
    "estimate of total planning: no action witness means unresolved use, not zero "
    "planning. Cross-call attribution requires the same search intervention to "
    "predict the complete stored plan and a later action, with plan-only and "
    "plan-restoration controls. Probes are excluded from ordinary counts."
)


def _finding(bank: FrozenPolicyBank, validation: dict) -> dict:
    statuses = [p["status"] for p in bank.parts.values()]
    unseen = any(
        r["part_status"] == "unseen_part"
        for e in validation["episodes"]
        for r in e["runs"]
    )
    if "search" not in statuses and "no_explanation" not in statuses and not unseen:
        code = "no_search_parts"
    elif validation["summary"]["complete"]:
        code = "search_supported"
    else:
        code = "partially_explained"
    return {"code": code, "label": FINDINGS[code], "planningness": None}


def _fmt(value, digits: int = 2):
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, digits)
    return value


def _lower_bound(value, untested: int):
    """Runs without an action witness can only add supported attributed effort."""
    if value is None or not untested:
        return value
    return f"≥ {_fmt(value)}"


def _spec_row(site: str, part: dict, identifier: str | None) -> list:
    chosen = next((e for e in part["explanations"] if e.identifier == identifier), None)
    if chosen is None:
        return [site] + [None] * 9
    spec, mapping = chosen.spec, chosen.mapping
    return [
        site,
        spec.selection,
        spec.priority,
        spec.heuristic,
        spec.goal_timing,
        spec.relaxation,
        spec.stale,
        spec.neighborhood,
        f"{spec.model} ({mapping.model})" if mapping.model else spec.model,
        f"root={mapping.root}, frontier={mapping.frontier}, parents={mapping.parents}, "
        f"costs={mapping.costs}" + (f", goal={mapping.goal}" if mapping.goal else ""),
    ]


def save_policy_report(
    directory: Path,
    name: str,
    program: PolicyProgram,
    bank: FrozenPolicyBank,
    validation: dict,
    fixture_truth: str | None,
) -> dict:
    """Write policy source, JSON audit, Markdown summary and an HTML report."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "policy.py").write_text(program.source, encoding="utf-8")
    finding = _finding(bank, validation)
    summary = validation["summary"]
    part_rows = []
    for site, part in sorted(bank.parts.items(), key=lambda i: int(i[0].split(":")[0])):
        info = program.sites[site]
        selected = part.get("selected")
        part_rows.append(
            [
                site,
                info["function"],
                info["source"][:60],
                part["status"],
                len(part["explanations"]),
                selected,
                part.get("readout") or None,
                (
                    validation["bank_validation"].get(site, {}).get(selected)
                    if selected
                    else None
                ),
            ]
        )
    spec_rows = [
        _spec_row(site, part, part.get("selected"))
        for site, part in sorted(bank.parts.items())
        if part["status"] == "search"
    ]
    episode_rows, part_episode_rows = [], []
    for episode in validation["episodes"]:
        program_row = episode["accounting"]["program"]
        label = f'{episode["case"][0]} obstacles, seed {episode["case"][1]}'
        episode_rows.append(
            [
                label,
                "yes" if episode["solved"] else "no",
                episode["steps"],
                program_row["planning_events"],
                program_row["initialization_expansion_events"],
                program_row["expansion_events"],
                program_row["max_expanded_depth"],
                _lower_bound(
                    program_row["attributed_expansion_events"],
                    program_row["runs_without_action_witness"],
                ),
                f'{program_row["followed_linked_plan"]} ({_fmt(program_row["followed_plan_share"])})',
                program_row["ran_search"],
                program_row["neither"] + program_row["followed_unlinked_plan"],
                _fmt(program_row["world_insensitive_share"]),
                "yes" if program_row["complete"] else "no",
            ]
        )
        for site, row in sorted(episode["accounting"]["parts"].items()):
            if row["status"] == "not_search_shaped":
                continue
            part_episode_rows.append(
                [
                    label,
                    site,
                    row["runs"],
                    row["supported"],
                    row["initialization_runs"],
                    row["expansion_events"],
                    row["N"],
                    row["max_expanded_depth"],
                    row["reexpansions"],
                    row["cost_improvements"],
                    row["attributed_expansion_events"],
                    row["plan_versions_linked"],
                ]
            )
    mean = summary["mean"]
    chain_headers = [
        "Episode",
        "Producer call",
        "Cache",
        "Readout status",
        "Edit visit / kind",
        "Stored plan matches / changed",
        "Probe outcome",
        "Later call",
        "Ordinary action",
        "Predicted action",
        "Observed action",
        "Restored action",
    ]
    chain_rows = []
    for episode in validation["episodes"]:
        rows = []
        for version in episode["versions"]:
            for chain in version["chains"]:
                for probe in chain["probes"]:
                    future = probe.get("future", [])
                    witness = future[-1] if future else {}
                    rows.append(
                        [
                            str(episode["case"]),
                            version["call"],
                            version["attribute"],
                            chain["status"],
                            f'{probe["edit"]["visit"]} / {probe["edit"]["operation"]}',
                            f'{probe.get("prediction_matches")} / {probe.get("changed")}',
                            probe.get("reason"),
                            witness.get("call"),
                            witness.get("ordinary_action"),
                            witness.get("predicted_action"),
                            witness.get("observed_action"),
                            witness.get("restored_action"),
                        ]
                    )
        chain_rows.extend(rows[:24])
    mean_rows = [[k, _fmt(v)] for k, v in mean.items()] + [
        [k, v] for k, v in summary["max"].items()
    ]
    part_headers = [
        "Part",
        "Function",
        "Source",
        "Development status",
        "Bank size",
        "Preselected",
        "Plan read-out",
        "Validation of preselected",
    ]
    spec_headers = [
        "Part",
        "Selection",
        "Priority",
        "Heuristic",
        "Goal test",
        "Updates",
        "Stale entries",
        "Moves",
        "Model",
        "Mapping",
    ]
    episode_headers = [
        "Episode",
        "Solved",
        "Steps",
        "Planning events",
        "At initialization",
        "Expansion events (sum)",
        "Max expanded depth",
        "Attributed to actions",
        "Followed a causally linked plan (share)",
        "Ran a search",
        "Neither",
        "World-insensitive share",
        "Complete",
    ]
    part_episode_headers = [
        "Episode",
        "Part",
        "Runs",
        "Supported",
        "At initialization",
        "Expansion events",
        "N (sum)",
        "Max expanded depth",
        "Re-expansions",
        "Cost improvements",
        "Attributed to actions",
        "Structural plan matches",
    ]

    def markdown(headers, rows) -> list[str]:
        lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
        lines += [
            "| " + " | ".join("—" if v is None else str(v) for v in row) + " |"
            for row in rows
        ]
        return lines

    episodes = ", ".join(f"({c}, {s})" for c, s in bank.validation)
    lines = [
        f"# Search explanations of a stateful policy: {name}",
        "",
        f"Known fixture label: {fixture_truth or 'not supplied'} (never given to the analyzer).",
        "",
        f"**Finding: {finding['label']}.**",
        "",
        f"Validation episodes (obstacles, seed): {episodes}, uniform weights. "
        f"Frozen bank `{bank.fingerprint[:16]}` was written before any of them ran.",
        "",
        "## Parts",
        "",
        *markdown(part_headers, part_rows),
        "",
        "## Preselected explanations",
        "",
        *markdown(spec_headers, spec_rows),
        "",
        "## Program level, per validation episode",
        "",
        *markdown(episode_headers, episode_rows),
        "",
        "## Search parts, per validation episode",
        "",
        *markdown(part_episode_headers, part_episode_rows),
        "",
        "## Across validation episodes",
        "",
        *markdown(["Quantity", "Value"], mean_rows),
        "",
        "## Search → stored plan → later action",
        "",
        "First 24 probes per episode below; complete predicted/observed plans, "
        "frozen readout templates, rejected candidates and future calls are in "
        "`report.json` under `validation.episodes[].versions[].chains[]`. "
        "A tree-shaped cache and a successful rotation probe alone earn no search credit. "
        "No row means no applicable frozen readout or no sampled producer probe.",
        "",
        *markdown(chain_headers, chain_rows),
        "",
        f"Solved {summary['solved']} of {summary['episodes']} validation episodes "
        "(separate from every count above).",
        "",
        RULES,
        "",
    ]
    report_json = {
        "schema": 2,
        "fixture": name,
        "fixture_ground_truth": fixture_truth,
        "finding": finding,
        "frozen": bank.manifest(),
        "fingerprint": bank.fingerprint,
        "sites": program.sites,
        "discovery_audit": bank.audit,
        "validation": validation,
    }
    write_json(directory / "report.json", report_json)
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    body = [
        f"<h1>Search explanations of a stateful policy: {html.escape(name)}</h1>",
        f"<p><b>Finding: {html.escape(finding['label'])}.</b></p>",
        f"<p>Known fixture label: {html.escape(fixture_truth or 'not supplied')}. "
        "It comes from construction and is never given to the analyzer.</p>",
        f"<p>Validation episodes (obstacles, seed): {html.escape(episodes)}. The frozen bank "
        f"<code>{bank.fingerprint[:16]}</code> was written before any of them ran.</p>",
        "<p>A <b>part</b> is one <code>while</code> loop; a <b>run</b> is one execution of it. "
        "A run's inputs are read at its first loop checkpoint; every later iteration is predicted "
        "by the abstract engine and compared. The transition model is the policy's own free-cell "
        "predicate, accepted as given and not checked against the simulator.</p>",
        "<h2>Parts</h2>",
        _table(part_headers, part_rows),
        "<h2>Preselected explanations</h2>",
        _table(spec_headers, spec_rows),
        "<h2>Program level, per validation episode</h2>",
        f"<p>{html.escape(RULES)}</p>",
        _table(episode_headers, episode_rows),
        "<p><b>Followed a linked plan</b>: rotating the cached plan 90° about the robot rotated the "
        "planar command (cosine ≥ 0.9), and that plan version has a complete search-to-action witness. "
        "<b>World-insensitive</b>: shifting every obstacle left that command unchanged. "
        "<b>Ran a search</b>: the call contained a search-shaped run; its expansions are counted in "
        "that run only. <b>Attributed to actions</b> counts runs whose plan was followed or whose "
        "paired edit changed the same call's command; ≥ marks a lower bound when some runs have "
        "no action witness. Missing causal support is unresolved use, not zero planning.</p>",
        "<h2>Search → stored plan → later action</h2>",
        "<p>The same search edit must predict the entire changed cache. On later ordinary "
        "observations, installing only that predicted cache must reproduce the action change, "
        "and restoring the ordinary cache must restore the ordinary action. Replays carry "
        "policy state across at most 32 calls; they stop on replanning or cache replacement. "
        "The readout holds the ordinary endpoint and optional final target fixed; it does not "
        "explain destination selection. First 24 probes per episode shown; full plans and "
        "counterexamples are in the plan-version details below and report.json.</p>",
        _table(chain_headers, chain_rows),
        "<h2>Search parts, per validation episode</h2>",
        _table(part_episode_headers, part_episode_rows),
        "<h2>Across validation episodes</h2>",
        _table(["Quantity", "Value"], mean_rows),
        f"<p>Solved {summary['solved']} of {summary['episodes']} validation episodes "
        "(separate from every count above).</p>",
        "<h2>Banks and evidence</h2>",
    ]
    for site, part in sorted(bank.parts.items()):
        if not part["explanations"]:
            continue
        body.append(
            _details(
                f"Part {site}: {len(part['explanations'])} frozen explanations",
                [{"id": e.identifier, **asdict(e)} for e in part["explanations"]],
            )
        )
    for episode in validation["episodes"]:
        runs = [r for r in episode["runs"] if r["part_status"] != "not_search_shaped"]
        sample = runs[:3] + [r for r in runs if r.get("status") != "supported"][:3]
        body.append(
            _details(
                f"Episode {episode['case']}: {len(runs)} search-shaped runs "
                "(first three and up to three unsupported shown; all in report.json)",
                sample,
            )
        )
        body.append(
            _details(
                f"Episode {episode['case']}: plan versions",
                [dict(version) for version in episode["versions"]],
            )
        )
    body.append(_details("Discovery audit", bank.audit))
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>Policy search explanations: {html.escape(name)}</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#172b4d}"
        "details{border:1px solid #ccd5df;border-radius:6px;padding:12px;margin:12px 0}"
        "summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#1460a0}"
        "table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}"
        "th,td{text-align:left;padding:8px;border-bottom:1px solid #ccd5df}th{background:#eef4fa}</style>"
        + "\n".join(body)
        + "</html>"
    )
    (directory / "report.html").write_text(page, encoding="utf-8")
    return {
        "fixture": name,
        "fixture_ground_truth": fixture_truth,
        "finding": finding,
        "complete": summary["complete"],
        "solved": f"{summary['solved']}/{summary['episodes']}",
        **{
            f"mean_{k}": _fmt(mean.get(k))
            for k in (
                "planning_events",
                "expansion_events",
                "attributed_expansion_events",
                "runs_with_untested_use",
                "runs_without_action_witness",
                "initialization_expansion_events",
                "followed_plan_share",
                "searched_share",
            )
        },
        "max_expanded_depth": summary["max"]["max_expanded_depth"],
        "parts": {s: p["status"] for s, p in bank.parts.items()},
        "effort_features": list(EFFORT),
        "depth_features": list(DEPTH),
    }
