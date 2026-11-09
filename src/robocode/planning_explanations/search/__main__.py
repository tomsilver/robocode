"""Run search-explanation milestones 1–7 without robotics or LLMs."""

import argparse
from pathlib import Path

from ..instrument import UnsupportedProgram
from .benchmark import discover_tasks, load_task_bank, validate_tasks
from .checkpoints import Program
from .discover import Limits, discover, load_frozen, validate
from .fixtures import FIXTURES, GROUND_TRUTH
from .heuristic_discovery import HEURISTIC_LIMITS, discover_heuristics
from .heuristic_fixtures import GROUND_TRUTH as HEURISTIC_TRUTH
from .heuristic_fixtures import HEURISTIC_FIXTURES, heuristic_cases
from .report import save_report, save_unresolved, write_json
from .route_fixtures import GROUND_TRUTH as ROUTE_TRUTH
from .route_fixtures import ROUTE_FIXTURES, route_cases
from .route_report import save_route_report
from .routes import discover_routes, load_route_bank, validate_routes
from .task_fixtures import GROUND_TRUTH as TASK_TRUTH
from .task_fixtures import TASK_FIXTURES, benchmark_cases


def _run_experiment(name: str, source: str, args: argparse.Namespace, limits: Limits):
    directory = args.output_dir / name
    program = Program(source)
    if args.suite == "routes":
        development, heldout = route_cases(name if not args.source else "mixed")
        route_bank = (
            load_route_bank(args.frozen)
            if args.frozen
            else discover_routes(
                program,
                development,
                heldout,
                limits,
                args.max_routes,
                args.prefix_limit,
            )
        )
        route_bank.save(directory / "frozen.json")
        validation = validate_routes(program, route_bank)
        return save_route_report(
            directory,
            name,
            program,
            route_bank,
            validation,
            None if args.source else ROUTE_TRUTH[name],
        )
    if args.suite in ("tasks", "heuristics"):
        development, heldout = (
            heuristic_cases()
            if args.suite == "heuristics"
            else benchmark_cases(name if not args.source else "adaptive")
        )
        discoverer = (
            discover_heuristics if args.suite == "heuristics" else discover_tasks
        )
        task_bank = (
            load_task_bank(args.frozen)
            if args.frozen
            else discoverer(program, development, heldout, limits)
        )
        # Persist the bank BEFORE any held-out concrete policy is executed.
        task_bank.save(directory / "frozen.json")
        validation = validate_tasks(program, task_bank)
        return save_report(
            directory,
            name,
            program,
            task_bank,
            validation,
            (
                None
                if args.source
                else (HEURISTIC_TRUTH if args.suite == "heuristics" else TASK_TRUTH)[
                    name
                ]
            ),
        )
    bank = load_frozen(args.frozen) if args.frozen else discover(program, limits=limits)
    bank.save(directory / "frozen.json")
    validation = validate(program, bank)
    return save_report(
        directory,
        name,
        program,
        bank,
        validation,
        None if args.source else GROUND_TRUTH[name],
    )


def main(argv=None) -> int:
    """Discover, persist the bank, then validate and export inspectable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("line", "tasks", "routes", "heuristics"), default="line"
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--fixture",
        choices=(
            "all",
            *sorted(
                set(FIXTURES)
                | set(TASK_FIXTURES)
                | set(ROUTE_FIXTURES)
                | set(HEURISTIC_FIXTURES)
            ),
        ),
        default="all",
    )
    sources.add_argument(
        "--source", type=Path, help="Trusted one-argument Python policy"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a separate output directory for each suite",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Default: 4000 for heuristics, 2000 otherwise",
    )
    parser.add_argument("--max-bindings", type=int, default=64)
    parser.add_argument("--mapping-depth", type=int, default=2)
    parser.add_argument(
        "--max-selectors",
        type=int,
        help="Value paths inventoried per checkpoint. Default: 128 for heuristics, "
        "64 otherwise",
    )
    parser.add_argument("--max-routes", type=int, default=8)
    parser.add_argument("--prefix-limit", type=int, default=64)
    parser.add_argument(
        "--frozen", type=Path, help="Validate an existing bank; skip discovery"
    )
    args = parser.parse_args(argv)
    fixtures = {
        "line": FIXTURES,
        "tasks": TASK_FIXTURES,
        "routes": ROUTE_FIXTURES,
        "heuristics": HEURISTIC_FIXTURES,
    }[args.suite]
    if not args.source and args.fixture != "all" and args.fixture not in fixtures:
        parser.error("Fixture does not belong to the selected suite")
    if args.output_dir is None:
        args.output_dir = Path(
            {
                "line": "outputs/search-explanations",
                "tasks": "outputs/search-task-benchmark",
                "routes": "outputs/search-route-benchmark",
                "heuristics": "outputs/search-heuristic-benchmark",
            }[args.suite]
        )
    if args.frozen and not args.source and args.fixture == "all":
        parser.error("--frozen requires --source or a single --fixture")
    defaults = HEURISTIC_LIMITS if args.suite == "heuristics" else Limits()
    try:
        limits = Limits(
            (
                args.max_candidates
                if args.max_candidates is not None
                else defaults.candidates
            ),
            args.max_bindings,
            mapping_depth=args.mapping_depth,
            selectors=(
                args.max_selectors
                if args.max_selectors is not None
                else defaults.selectors
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.source:
        sources_to_run = {args.source.stem: args.source.read_text(encoding="utf-8")}
    else:
        sources_to_run = (
            fixtures
            if args.fixture == "all"
            else {args.fixture: fixtures[args.fixture]}
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    execution_error = False
    for name, source in sources_to_run.items():
        try:
            summary = _run_experiment(name, source, args, limits)
        except (
            ValueError,
            RuntimeError,
            AssertionError,
            SyntaxError,
            KeyError,
            TypeError,
            IndexError,
            NameError,
            ArithmeticError,
        ) as exc:
            summary = save_unresolved(args.output_dir / name, name, source, exc)
            execution_error = True
        summaries.append(summary)
        print(
            f'{name}: {summary["finding"]["label"]}; '
            f'validation of the preselected explanation: {summary["status"]}; '
            f'{summary["supported_validation"]} explanations; '
            f'mean N={summary["ordinary_mean_N"]}, M={summary["ordinary_mean_M"]}, '
            f'attributed search expansions={summary.get("attributed_search_expansions")}'
        )
    write_json(args.output_dir / "summary.json", summaries)
    rows = [
        "# Search explanation demonstration",
        "",
        (
            "Uniform over each fixture's frozen validation task cases. "
            if args.suite != "line"
            else "Uniform fresh-input distribution: positions 1, 3, 5. "
        )
        + "N/M are features of the frozen selected explanation, "
        "not a universal planningness score.",
        "",
        "| Program | Known fixture label | Finding across frozen bank | "
        "Validation of the preselected explanation | Task success | Selected mean N | "
        "Selected mean M | Attributed search expansions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        rows.append(
            f'| [{row["fixture"]}]({row["fixture"]}/report.html) | '
            f'{row["fixture_ground_truth"]} | {row["finding"]["label"]} | '
            f'{row["status"]} | {row["task_success_rate"]} | '
            f'{row["ordinary_mean_N"]} | {row["ordinary_mean_M"]} | '
            f'{row.get("attributed_search_expansions")} |'
        )
    rows += [
        "",
        "Each HTML report includes **all** validation interventions. "
        "Standalone SVGs are representative views; JSON contains the complete audit.",
        "",
        "An unresolved or rejected selected explanation gets no numerical average. "
        "A supported reactive explanation does not rule out other explanations. "
        "Fixture labels come from construction and are never inputs to the analyzer. "
        "Task success comes from separate original-policy rollouts. "
        "Individual reports show feature ranges across supported frozen explanations.",
    ]
    (args.output_dir / "summary.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(f'Reports: {args.output_dir.resolve() / "summary.md"}')
    return 1 if execution_error else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (UnsupportedProgram, ValueError, RuntimeError, AssertionError) as error:
        raise SystemExit(
            f"Unresolved experiment: {error}. No planning score assigned."
        ) from error
