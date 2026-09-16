"""End-to-end CC versus CC + source policy-efficiency analysis.

Given a directory containing Blackbox and Whitebox ZIP collections (either
``Blackbox.zip`` / ``Whitebox.zip`` or unpacked ``Blackbox/`` / ``Whitebox/``
directories), this script discovers completed runs, replays the final policies
for timing, identifies identical seeds perfect in both settings, and writes all
tables and figures in one invocation.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from tqdm import tqdm  # type: ignore[import-untyped]

from experiments.render_cc_source_matched_summary import (
    _load,
    _render,
    _summaries,
    _write_seed_tables,
)
from experiments.run_final_timing_sample import (
    Job,
    _run_one,
    _scan_nested_directory,
    _scan_outer,
)


def _source_jobs(archive_root: Path, cache: Path) -> list[Job]:
    jobs: list[Job] = []
    for label in ("blackbox", "whitebox"):
        outer_archives = sorted(
            path
            for path in archive_root.rglob("*.zip")
            if path.name.lower() == f"{label}.zip"
        )
        for archive in outer_archives:
            jobs.extend(_scan_outer(archive, label, cache))

        directories = sorted(
            path
            for path in archive_root.rglob("*")
            if path.is_dir()
            and path.name.lower() == label
            and "__MACOSX" not in path.parts
        )
        for directory in directories:
            jobs.extend(_scan_nested_directory(directory, label))
    return jobs


def _select(jobs: list[Job]) -> list[Job]:
    selected: dict[tuple[str, str, int], Job] = {}
    for job in jobs:
        if job.method not in {"claude", "claude_whitebox"}:
            continue
        key = (job.method, job.environment, job.seed)
        if key not in selected or job.rank > selected[key].rank:
            selected[key] = job
    return sorted(
        selected.values(), key=lambda job: (job.environment, job.method, job.seed)
    )


def _write_manifest(path: Path, jobs: list[Job]) -> None:
    rows = [
        {
            "method": job.method,
            "environment": job.environment,
            "seed": job.seed,
            "solve_rate": job.solve_rate,
            "source": str(job.local_run or job.outer),
        }
        for job in jobs
    ]
    path.write_text(json.dumps(rows, indent=2) + "\n")


def _replay(jobs: list[Job], args: argparse.Namespace) -> None:
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_run_one, job, args) for job in jobs]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="timing replays",
            unit="run",
        ):
            job, code, detail = future.result()
            if code:
                failures.append(
                    f"{job.method}/{job.environment}/seed_{job.seed}: {detail}"
                )
    if failures:
        formatted = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"{len(failures)} timing replay(s) failed:\n{formatted}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--tmpdir", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.cache = args.cache or args.output / ".archive-cache"
    args.tmpdir = args.tmpdir or args.output / ".tmp"
    for path in (args.cache, args.tmpdir):
        path.mkdir(parents=True, exist_ok=True)

    jobs = _select(_source_jobs(args.archive_root, args.cache))
    if not jobs:
        raise RuntimeError(
            "No Claude Code blackbox/whitebox runs found under "
            f"{args.archive_root}"
        )
    print(f"Discovered {len(jobs)} unique completed CC/CC + source runs")
    _write_manifest(args.output / "manifest.json", jobs)
    _replay(jobs, args)

    matched, timings_by_seed = _load(args.output)
    if not matched:
        raise RuntimeError("No identical seeds were perfect in both settings")
    summaries = _summaries(matched, timings_by_seed)
    matched_count = sum(map(len, matched.values()))
    environment_count = len(matched)
    _write_seed_tables(
        args.output / "matched_seeds.csv",
        args.output / "matched_seeds.md",
        matched,
        timings_by_seed,
    )
    _render(
        args.output / "cc_source_efficiency.png",
        summaries,
        matched_count,
        environment_count,
    )
    print(
        f"Wrote all deliverables to {args.output} "
        f"({matched_count} matched seeds across {environment_count} environments)"
    )


if __name__ == "__main__":
    main()
