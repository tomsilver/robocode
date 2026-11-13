"""`--suite policy`: milestone 8 on class-based policies with state across calls."""

import argparse
import json
from pathlib import Path

from ..instrument import UnsupportedProgram
from .policy_discovery import (
    PolicyLimits,
    discover_policy,
    load_policy_bank,
    validate_policy,
)
from .policy_domains import DEVELOPMENT, VALIDATION, DiscWorld, KinderDomain
from .policy_fixtures import GROUND_TRUTH, KINDER_POLICY, POLICY_FIXTURES, kinder_source
from .policy_report import at_least, finding_text, save_policy_report, share_range
from .policy_runner import PolicyProgram
from .report import save_unresolved, write_json

FAILURES = (
    UnsupportedProgram,
    ValueError,
    RuntimeError,
    AssertionError,
    SyntaxError,
    KeyError,
    TypeError,
    IndexError,
    NameError,
    ArithmeticError,
)


def policy_sources(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    """Name -> (source, domain kind); KinDER for the real and custom policies."""
    if args.source:
        return {args.source.stem: (args.source.read_text(encoding="utf-8"), "kinder")}
    every = {name: (source, "disc") for name, source in POLICY_FIXTURES.items()}
    every[KINDER_POLICY] = (kinder_source(), "kinder")
    return every if args.fixture == "all" else {args.fixture: every[args.fixture]}


def policy_limits(args: argparse.Namespace) -> PolicyLimits:
    """Suite defaults unless a flag overrides them."""
    defaults = PolicyLimits()
    return PolicyLimits(
        candidates=args.max_candidates or defaults.candidates,
        bindings=args.max_bindings,
        mapping_depth=args.mapping_depth,
        selectors=args.max_selectors or defaults.selectors,
        edits_per_run=args.edits_per_run,
        probed_runs=args.probed_runs,
    )


def _bounded(row: dict, key: str, mean: bool = True, partial: bool = False):
    """A summary value, or "≥" its lower bound when the value is withheld."""
    value = row.get(f"mean_{key}" if mean else key)
    return at_least(value, row.get(f"lower_bound_{key}"), partial)


def run_policy_suite(args: argparse.Namespace) -> int:
    """Discover, persist each bank, validate, and write one report per policy."""
    limits = policy_limits(args)
    config = (
        json.loads(args.env_config.read_text(encoding="utf-8"))
        if args.env_config
        else None
    )
    kinder = None
    summaries, failed = [], False
    for name, (source, kind) in policy_sources(args).items():
        directory = args.output_dir / name
        try:
            if kind == "kinder":
                kinder = kinder or KinderDomain(config)
            domain = kinder if kind == "kinder" else DiscWorld()
            program = PolicyProgram(source)
            bank = (
                load_policy_bank(args.frozen)
                if args.frozen
                else discover_policy(program, domain, DEVELOPMENT, VALIDATION, limits)
            )
            # Persist the bank BEFORE any validation episode is executed.
            bank.save(directory / "frozen.json")
            validation = validate_policy(program, domain, bank)
            summary = save_policy_report(
                directory,
                name,
                program,
                bank,
                validation,
                None if args.source else GROUND_TRUTH[name],
            )
        except FAILURES as exc:
            summary = save_unresolved(directory, name, source, exc)
            failed = True
        summaries.append(summary)
        score = at_least(
            summary.get("mean_search_score"),
            summary.get("lower_bound_search_score"),
        )
        planning = at_least(
            summary.get("mean_planning_events"),
            summary.get("lower_bound_planning_events"),
        )
        expansions = at_least(
            summary.get("mean_expansion_events"),
            summary.get("lower_bound_expansion_events"),
        )
        print(
            f'{name}: {finding_text(summary["finding"])}; '
            f"search score={score}, "
            f"planning events={planning}, "
            f"expansion events={expansions}, "
            "followed-plan share="
            f'{summary.get("mean_followed_plan_share")}'
            f' (max {summary.get("mean_followed_plan_share_max")})'
        )
    write_json(args.output_dir / "summary.json", summaries)
    rows = [
        "# Search explanations of stateful policies (milestone 8)",
        "",
        "Means are uniform over each policy's frozen validation episodes. The search "
        "score is the mean, per policy call, of states expanded beyond the start "
        "(0 for a call without a search), in the policy's own grid cells. Effort is "
        "summed within an episode, depth is a maximum, and the followed-plan share is "
        "the fraction of action calls whose command followed a cached plan with a "
        "complete search-to-action witness; its max adds calls whose plan evidence is "
        "unresolved. "
        "These are features of the frozen explanations, not a planningness score.",
        "",
        "| Policy | Known fixture label | Finding | Search score (per call) | "
        "Planning events | Expansion events | Attributed to actions | "
        "At initialization | Followed-plan share (witnessed – max) | "
        "Searched share | Max expanded depth | Solved |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        cells = [
            f'[{row["fixture"]}]({row["fixture"]}/report.html)',
            row.get("fixture_ground_truth"),
            finding_text(row["finding"]),
            _bounded(row, "search_score"),
            _bounded(row, "planning_events"),
            _bounded(row, "expansion_events"),
            _bounded(
                row,
                "attributed_expansion_events",
                partial=bool(row.get("mean_runs_without_action_witness")),
            ),
            _bounded(row, "initialization_expansion_events"),
            share_range(
                row.get("mean_followed_plan_share"),
                row.get("mean_followed_plan_share_max"),
            ),
            row.get("mean_searched_share"),
            _bounded(row, "max_expanded_depth", mean=False),
            row.get("solved"),
        ]
        rows.append(
            "| " + " | ".join("—" if c is None else str(c) for c in cells) + " |"
        )
    rows += [
        "",
        "≥ marks a lower bound. For effort, the total counts only runs the analyzer "
        "could read; the other runs are unknown (never zero) and can only add to it. "
        "For attributed effort, it also marks runs with no complete action witness, "
        "which is not zero planning. — marks a value with no bound. "
        "A share range's max counts plans "
        "whose search provenance is unresolved; it is the most the evidence allows, "
        "not an estimate. Cross-call credit requires a search "
        "edit to predict the stored plan and later action, with mediation controls.",
    ]
    (args.output_dir / "summary.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(f'Reports: {args.output_dir.resolve() / "summary.md"}')
    return 1 if failed else 0


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags used only by the policy suite."""
    parser.add_argument(
        "--env-config",
        type=Path,
        help="KinDER env config JSON for --suite policy --source (default: the "
        "stored ClutteredRetrieval2D config)",
    )
    parser.add_argument(
        "--edits-per-run",
        type=int,
        default=5,
        help="Policy suite: paired edits per run",
    )
    parser.add_argument(
        "--probed-runs",
        type=int,
        default=8,
        help="Policy suite: runs probed per part per episode",
    )
