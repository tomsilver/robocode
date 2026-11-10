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
from .policy_report import save_policy_report, share_range
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
        print(
            f'{name}: {summary["finding"]["label"]}; '
            f'planning events={summary.get("mean_planning_events")}, '
            f'expansion events={summary.get("mean_expansion_events")}, '
            "followed-plan share="
            f'{summary.get("mean_followed_plan_share")}'
            f' (max {summary.get("mean_followed_plan_share_max")})'
        )
    write_json(args.output_dir / "summary.json", summaries)
    rows = [
        "# Search explanations of stateful policies (milestone 8)",
        "",
        "Means are uniform over each policy's frozen validation episodes. Effort is "
        "summed within an episode, depth is a maximum, and the followed-plan share is "
        "the fraction of action calls whose command followed a cached plan with a "
        "complete search-to-action witness; its max adds calls whose plan evidence is "
        "unresolved. "
        "These are features of the frozen explanations, not a planningness score.",
        "",
        "| Policy | Known fixture label | Finding | Planning events | "
        "Expansion events | Attributed to actions | At initialization | "
        "Followed-plan share (witnessed – max) | Searched share | "
        "Max expanded depth | Solved |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        cells = [
            f'[{row["fixture"]}]({row["fixture"]}/report.html)',
            row.get("fixture_ground_truth"),
            row["finding"]["label"],
            row.get("mean_planning_events"),
            row.get("mean_expansion_events"),
            (
                f'≥ {row.get("mean_attributed_expansion_events")}'
                if row.get("mean_runs_without_action_witness")
                else row.get("mean_attributed_expansion_events")
            ),
            row.get("mean_initialization_expansion_events"),
            share_range(
                row.get("mean_followed_plan_share"),
                row.get("mean_followed_plan_share_max"),
            ),
            row.get("mean_searched_share"),
            row.get("max_expanded_depth"),
            row.get("solved"),
        ]
        rows.append(
            "| " + " | ".join("—" if c is None else str(c) for c in cells) + " |"
        )
    rows += [
        "",
        "— marks a withheld value: an unresolved search-shaped run is not zero. "
        "≥ marks supported effort with unresolved use in other runs: no complete "
        "action witness is not zero planning. A share range's max counts plans "
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
