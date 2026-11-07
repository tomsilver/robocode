"""Run the fixture demonstration without an LLM or robotics dependencies."""

import argparse
from pathlib import Path

from .discovery import Limits, discover, validate
from .fixtures import FIXTURES
from .instrument import Program
from .report import frozen_payload, save_report, write_json
from .task import cases


def main():
    """Discover, persist the freeze, then generate fresh validation inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/planning-explanations")
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fixture", choices=list(FIXTURES))
    selection.add_argument(
        "--source", type=Path, help="One restricted four-argument Python function"
    )
    parser.add_argument("--discovery-seed", type=int, default=17)
    parser.add_argument("--validation-seed", type=int, default=29)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument(
        "--state-weights",
        nargs=6,
        type=float,
        help="Probabilities for initial states 0 through 5; must sum to 1",
    )
    args = parser.parse_args()
    if args.discovery_seed == args.validation_seed:
        parser.error("Discovery and validation seeds must differ")
    limits = Limits(candidates=args.max_candidates)
    sources = FIXTURES
    if args.fixture:
        sources = {args.fixture: FIXTURES[args.fixture]}
    if args.source:
        sources = {args.source.stem: args.source.read_text(encoding="utf-8")}
    weights = dict(enumerate(args.state_weights)) if args.state_weights else None
    summary = []
    for name, source in sources.items():
        program = Program(source)
        bank = discover(program, cases(args.discovery_seed, "discovery"), limits)
        directory = args.output_dir / name
        # This file is written BEFORE fresh inputs are generated or evaluated.
        write_json(directory / "frozen.json", frozen_payload(bank))
        fresh = cases(args.validation_seed, "validation")
        validation = validate(program, bank, fresh, weights)
        save_report(directory, name, program, bank, fresh, validation)
        coverage = validation["evidence_coverage"]
        supported = sum(e["status"] == "supported" for e in validation["explanations"])
        summary.append(
            {
                "program": name,
                "frozen_explanations": len(bank.explanations),
                "validated_explanations": supported,
                "evidence_coverage": coverage,
                "search_truncated": any(
                    r["truncated"] for r in bank.audit["routes"].values()
                ),
                "report": str(directory / "report.md"),
            }
        )
        print(
            f"{name:20} supported explanations={supported:2}  "
            f"ordinary supported mass={coverage['supported_mass']:.3f}  "
            f"unresolved={coverage['unresolved_mass']:.3f}"
        )
    write_json(args.output_dir / "summary.json", summary)
    lines = [
        "# Planning explanation prototype",
        "",
        "Coverage below is evidence coverage, not a planningness score. "
        "Unresolved mass has unknown planning amount.",
        "",
        "| Program | Supported explanations | Supported mass | "
        "Unresolved mass | Truncated |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        name = row["program"]
        coverage = row["evidence_coverage"]
        lines.append(
            f"| [{name}]({name}/report.md) | {row['validated_explanations']} | "
            f"{coverage['supported_mass']:.3f} | {coverage['unresolved_mass']:.3f} | "
            f"{row['search_truncated']} |"
        )
    (args.output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"\nReports: {args.output_dir.resolve() / 'README.md'}")


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
