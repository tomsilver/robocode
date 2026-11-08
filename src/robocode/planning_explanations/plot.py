"""Plot a saved ordinary explanation and side-by-side intervention experiments.

Run from the repository root:
    PYTHONPATH=src python -m robocode.planning_explanations.plot --help
"""

import argparse
import json
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .task import Model

Row = dict[str, Any]
Roles = dict[str, list]
DEFAULT = Path("outputs/planning-explanations/two_step")


def model(explanation: Row) -> Model:
    """The model a serialized explanation names."""
    return Model(
        explanation["horizon"], explanation["path_value"], explanation["pruning"]
    )


def leaf_label(explanation: Row) -> str:
    """Leaf utilities are U; path sums are V."""
    return "U" if explanation["path_value"] == "leaf" else "V"


def read_roles(explanation: Row, trace: Row) -> Roles:
    """Apply the frozen selectors to recorded snapshots, without rolling out a model."""
    events = {(e["site"], e["occurrence"]): e["value"] for e in trace["events"]}
    roles: Roles = {}
    for role, selectors in explanation["mapping"]:
        roles[role] = []
        for selector in selectors:
            if selector is None:  # A node this route's search does not expand.
                roles[role].append(None)
                continue
            value = events[selector["site"], selector["occurrence"]]
            for key in selector["path"]:
                # JSON object keys are strings, even for Python integer dict keys.
                value = value[str(key)] if isinstance(value, dict) else value[key]
            roles[role].append(value)
    return roles


def pick_case(report: Row, case: str | None, identifier: str | None) -> Row:
    """Prefer a supported case; an explicit case must never silently change."""
    rows = report["validation"]["ordinary_population"]
    if case:
        for row in rows:
            if row["case"] == case:
                return row
        raise ValueError(f"Unknown validation case {case!r}; use --list")
    if identifier and not any(identifier in r["supported_explanations"] for r in rows):
        raise ValueError(f"No supported validation case for explanation {identifier}")
    return next(
        (
            r
            for r in rows
            if r["supported_explanations"]
            and (not identifier or identifier in r["supported_explanations"])
        ),
        rows[0],
    )


def selected_probes(
    probes: list[Row], numbers: list[int] | None, all_probes: bool
) -> list[tuple[int, Row]]:
    """Use case-local, one-based numbers; default to two patches of the same leaf."""
    numbered = list(enumerate(probes, start=1))
    if numbers is not None:
        if any(i < 1 or i > len(probes) for i in numbers):
            raise ValueError(
                f"Intervention numbers must be in 1..{len(probes)}; use --list"
            )
        return [(i, probes[i - 1]) for i in dict.fromkeys(numbers)]
    if all_probes:
        return numbered
    leaf = [
        p for p in numbered if p[1]["role"] == "evaluations" and p[1]["coordinate"] == 0
    ]
    return (leaf or numbered)[:2]


def draw_tree(
    ax: Axes,
    start: int,
    roles: Roles,
    action: int,
    baseline: Roles,
    original_action: int,
    target: tuple[str, int] | None = None,
    value_label: str = "U",
    unobserved: frozenset = frozenset(),
) -> None:
    """Draw action paths, keeping distinct paths to the same task state separate."""
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    depth = sum(role.startswith("states_") for role in roles)

    def position(level: int, index: int) -> tuple[float, float]:
        return (index + 0.5) / 2**level, 0.86 - 0.72 * level / depth

    for level in range(depth + 1):
        for i in range(2**level):
            x, y = position(level, i)
            # A node the search did not expand (a cutoff) is drawn faded.
            skipped = bool(level) and roles[f"states_{level}"][i] is None
            if level:
                px, py = position(level - 1, i // 2)
                chosen = level == 1 and i == action
                ax.plot(
                    [px, x],
                    [py, y],
                    color="#007668" if chosen else "#a5b2bd",
                    lw=2.5 if chosen else 1,
                    linestyle="--" if skipped else "-",
                    zorder=1,
                )
                ax.text(
                    (px + x) / 2,
                    (py + y) / 2,
                    str(i % 2),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#334451",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                )
            fields = []
            if level:
                fields.append(("State", f"states_{level}", i))
                if level == depth:
                    fields.append((value_label, "evaluations", i))
                if level == 1 and "scores" in roles:
                    # Depth-1 models compare their leaf utilities directly.
                    fields.append(("Score", "scores", i))
            lines = []
            for label, role, index in fields:
                value = roles[role][index]
                changed = " Δ" if value != baseline[role][index] else ""
                lines.append(f"{label} {value}{changed}")
            if skipped and (f"states_{level}", i) in unobserved:
                # The route has no location for this node, so the program's value
                # cannot be read; it is not evidence that the node was skipped.
                lines = ["not observed"]
            elif skipped:
                was_expanded = baseline[f"states_{level}"][i] is not None
                lines = ["not expanded" + (" Δ" if was_expanded else "")]
            if not level:
                lines = [
                    f"State {start}",
                    f"Action {action}" + (" Δ" if action != original_action else ""),
                ]
            patched = any((role, j) == target for _, role, j in fields)
            ax.text(
                x,
                y,
                "\n".join(lines),
                ha="center",
                va="center",
                zorder=2,
                fontsize=10 if depth < 3 else 7,
                color="#8a99a5" if skipped else "#152f3b",
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "facecolor": (
                        "white" if skipped else ("#fff4e2" if patched else "#f3f8f8")
                    ),
                    "edgecolor": (
                        "#b66511" if patched else ("#c9d3da" if skipped else "#a4b6b8")
                    ),
                    "linewidth": 1.8 if patched else 0.8,
                    "linestyle": "--" if skipped else "-",
                },
            )


def ordinary_figure(report: Row, row: Row, explanation: Row, roles: Roles) -> Figure:
    """The unpatched trace, with expression provenance for this specific mapping."""
    fig = Figure(figsize=(7.2, 4.7))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.77, bottom=0.17)
    fig.text(0.04, 0.93, "Ordinary explanation tree", fontsize=18, weight="bold")
    fig.text(
        0.04,
        0.86,
        f"{report['program']}  ·  {row['case']}  ·  "
        f"{model(explanation).label}  ·  mapping {explanation['id']}",
        fontsize=10,
    )
    draw_tree(
        fig.subplots(),
        row["state"],
        roles,
        row["output"],
        roles,
        row["output"],
        value_label=leaf_label(explanation),
    )
    sites = report["checkpoints"]
    sources = []
    for role, selectors in explanation["mapping"]:
        exprs = dict.fromkeys(
            sites[s["site"]]["expression"] for s in selectors if s is not None
        )
        sources.append(f"{role}: {', '.join(exprs)}")
    fig.text(
        0.04,
        0.10,
        (
            "U = terminal utility"
            if leaf_label(explanation) == "U"
            else "V = sum of utilities along the path"
        )
        + "   ·   Edges = actions   ·   Teal = returned action",
        fontsize=9,
        color="#334451",
    )
    fig.text(0.04, 0.035, "   ·   ".join(sources), fontsize=8, wrap=True)
    return fig


def comparison_figure(
    report: Row,
    row: Row,
    explanation: Row,
    baseline: Roles,
    group: list[tuple[int, Row]],
) -> Figure:
    """One intervention per column, with model and program values in separate rows."""
    fig = Figure(figsize=(5.3 * len(group), 7.6))
    fig.subplots_adjust(
        left=0.025, right=0.975, top=0.80, bottom=0.07, wspace=0.08, hspace=0.30
    )
    axes = fig.subplots(2, len(group), squeeze=False)
    unmapped = frozenset(
        (role, i)
        for role, selectors in explanation["mapping"]
        for i, selector in enumerate(selectors)
        if selector is None
    )
    fig.text(
        0.025, 0.95, "Interventions on the same execution", fontsize=17, weight="bold"
    )
    fig.text(
        0.025,
        0.90,
        f"{report['program']}  ·  {row['case']}  ·  "
        f"ordinary action {row['output']}  ·  mapping {explanation['id']}",
        fontsize=10,
    )
    for col, (num, record) in enumerate(group):
        role, coord = record["role"], record["coordinate"]
        title = (
            f"#{num}   {role}[{coord}] ({record['kind']}): "
            f"{record['old_value']} → {record['replacement']}"
        )
        for line, side in enumerate(("predicted", "observed")):
            ax = axes[line, col]
            description = (
                "Model prediction"
                if line == 0
                else (
                    "Program observation  ·  "
                    + ("match" if record["passed"] else "failed check")
                )
            )
            ax.set_title((title + "\n" if line == 0 else "") + description, fontsize=11)
            if side + "_roles" not in record:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    "No observation recorded\n" + record.get("error", ""),
                    transform=ax.transAxes,
                    ha="center",
                    wrap=True,
                )
                continue
            draw_tree(
                ax,
                row["state"],
                record[side + "_roles"],
                record[side + "_output"],
                baseline,
                row["output"],
                (role, coord),
                leaf_label(explanation),
                unmapped if side == "observed" else frozenset(),
            )
    fig.text(
        0.025,
        0.022,
        "Outlined node = patch target   ·   Δ = changed from ordinary"
        "   ·   Teal edge = selected action"
        + (
            "   ·   not observed = no location for that node on this route"
            if unmapped
            else ""
        ),
        fontsize=9,
    )
    return fig


def run(args: argparse.Namespace) -> None:
    """Read a fixture report and write separate ordinary and comparison figures."""
    path = args.report / "report.json" if args.report.is_dir() else args.report
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["schema_version"] != 5:
        raise ValueError("Unsupported report schema; regenerate the report")
    if args.list:
        for r in report["validation"]["ordinary_population"]:
            print(
                f"{r['case']}: {r['status']}  {', '.join(r['supported_explanations'])}"
            )
    row = pick_case(report, args.case, args.explanation)
    ids = row["supported_explanations"]
    if not ids:
        print(f"{row['case']}: unresolved; no supported tree to plot.")
        return
    identifier = args.explanation or ids[0]
    if identifier not in ids:
        raise ValueError(f"Mapping not supported for {row['case']}; choose from {ids}")
    explanation = next(
        e for e in report["frozen_explanations"] if e["id"] == identifier
    )
    validation = next(
        e for e in report["validation"]["explanations"] if e["id"] == identifier
    )
    probes = [p for p in validation["interventions"] if p["case"] == row["case"]]
    print(f"{row['case']}: mapping {identifier} ({len(ids)} supported for this case)")
    if args.list:
        for i, probe in enumerate(probes, start=1):
            print(
                f"#{i}: {probe['role']}[{probe['coordinate']}] ({probe['kind']})  "
                f"{probe['old_value']} → {probe['replacement']}  |  action: "
                f"predicted {probe['predicted_output']}, "
                f"observed {probe.get('observed_output', 'unavailable')}"
            )
        if not probes:
            print("No paired interventions recorded for this case.")
        return
    selected = selected_probes(probes, args.interventions, args.all_interventions)
    traces = json.loads(
        (path.parent / "ordinary-traces.json").read_text(encoding="utf-8")
    )
    trace = next((t for t in traces if t["case"] == row["case"]), None)
    if not trace or trace["route"] != row["route"] or trace["output"] != row["output"]:
        raise ValueError("Ordinary trace is missing or inconsistent with report")
    roles = read_roles(explanation, trace)
    directory = args.output_dir or path.parent / "plots" / row["case"] / identifier
    directory.mkdir(parents=True, exist_ok=True)

    def save(fig: Figure, stem: str) -> None:
        destination = directory / f"{stem}.{args.format}"
        fig.savefig(destination, dpi=150, bbox_inches="tight", pad_inches=0.18)
        print(destination)

    save(ordinary_figure(report, row, explanation, roles), "ordinary-tree")
    if not selected:
        print("No paired interventions recorded; wrote only the ordinary tree.")
    for i in range(0, len(selected), 3):
        group = selected[i : i + 3]
        numbers = "-".join(str(n) for n, _ in group)
        save(
            comparison_figure(report, row, explanation, roles, group),
            f"interventions-{numbers}",
        )


def main() -> None:
    """Select cases, mappings and interventions without executing fixture code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=DEFAULT,
        help="fixture directory or report.json (default: two_step)",
    )
    parser.add_argument(
        "--case", help="validation case label; default: first supported case"
    )
    parser.add_argument(
        "--explanation", help="frozen mapping ID; default: first supported"
    )
    parser.add_argument(
        "--list", action="store_true", help="list cases, mappings, and patch numbers"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--interventions",
        type=int,
        nargs="+",
        metavar="N",
        help="case-local patch numbers printed by --list",
    )
    selection.add_argument(
        "--all-interventions",
        action="store_true",
        help="plot all recorded patches in groups of up to three",
    )
    parser.add_argument("--format", choices=("png", "svg", "pdf"), default="png")
    parser.add_argument("--output-dir", type=Path, help="override the plots directory")
    try:
        run(parser.parse_args())
    except (OSError, ValueError, KeyError, IndexError) as exc:
        parser.exit(2, f"Could not plot: {exc}\n")


if __name__ == "__main__":
    main()
