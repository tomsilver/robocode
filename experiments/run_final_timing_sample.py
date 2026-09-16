"""Replay archived final policies on a small held-out timing sample."""

# pylint: disable=line-too-long,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml
from tqdm import tqdm  # type: ignore[import-untyped]


@dataclass(frozen=True)
class Job:
    method: str
    environment: str
    seed: int
    outer: Path | None
    inner_member: str | None
    result_member: str | None
    local_run: Path | None
    rank: tuple[int, str]
    solve_rate: float | None = None


def _method(outer_label: str, member: str, experiment_id: str = "") -> str | None:
    if experiment_id:
        if "__llm_genplan__" in experiment_id:
            return "genplan"
        if "__bilevel_planning__" in experiment_id:
            return "planner"
        if "__agentic__" in experiment_id and "__whitebox" in experiment_id:
            if "claude" in experiment_id:
                return "claude_whitebox"
        if "__agentic__" in experiment_id and "__blackbox" in experiment_id:
            if "codex" in experiment_id:
                return "codex"
            if "claude" in experiment_id:
                return "claude"
        return None
    name = member.replace("\\", "/")
    if outer_label == "baselines":
        if name.startswith("Baselines/Planner/"):
            return "planner"
        if name.startswith("Baselines/GenPlan/"):
            return "genplan"
    if outer_label == "blackbox":
        if name.startswith("Blackbox/Codex GPT5.6 Sol/"):
            return "codex"
        if name.startswith("Blackbox/Claude Code Opus 5/"):
            return "claude"
    if outer_label == "whitebox" and name.startswith("Whitebox/"):
        return "claude_whitebox"
    return None


def _timestamp(text: str) -> str:
    found = re.findall(r"20\d\d-\d\d-\d\d[_T-]\d\d[-:]\d\d[-:]\d\d", text)
    return found[-1] if found else ""


def _scan_outer(path: Path, label: str, cache: Path) -> list[Job]:
    jobs: list[Job] = []
    with zipfile.ZipFile(path) as outer:
        for member in outer.namelist():
            method = _method(label, member)
            if method is None or not member.endswith(".zip") or "__MACOSX" in member:
                continue
            nested_dir = cache / "nested" / label / method
            nested_dir.mkdir(parents=True, exist_ok=True)
            nested = nested_dir / Path(member).name
            if not nested.exists() or nested.stat().st_size != outer.getinfo(member).file_size:
                temporary = nested.with_suffix(nested.suffix + ".partial")
                with outer.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 << 20)
                temporary.replace(nested)
            try:
                with zipfile.ZipFile(nested) as inner:
                    for result_member in inner.namelist():
                        if not result_member.endswith("/results.json"):
                            continue
                        result = json.loads(inner.read(result_member))
                        if result.get("eval_complete") is False:
                            continue
                        experiment_id = str(result.get("experiment_id", ""))
                        environment = experiment_id.split("__", 1)[0]
                        if not environment:
                            # Early archives did not store experiment_id in
                            # results.json; recover it from the nested archive.
                            environment = result_member.split("/", 1)[0].split("__", 1)[0]
                        seed = result.get("replicate_seed")
                        if environment and isinstance(seed, int):
                            jobs.append(Job(
                                method, environment, seed, nested, None,
                                result_member, None,
                                (int("outdated" not in member.lower()), _timestamp(result_member)),
                                float(result["solve_rate"]) if isinstance(result.get("solve_rate"), (int, float)) else None,
                            ))
            except (zipfile.BadZipFile, json.JSONDecodeError, KeyError):
                continue
    return jobs


def _scan_nested_directory(path: Path, label: str) -> list[Job]:
    """Scan an unpacked Blackbox/Whitebox directory of result ZIPs."""
    jobs: list[Job] = []
    for nested in path.rglob("*.zip"):
        relative = nested.relative_to(path).as_posix()
        synthetic_member = f"{label.title()}/{relative}"
        method = _method(label, synthetic_member)
        if method is None:
            continue
        try:
            with zipfile.ZipFile(nested) as inner:
                for result_member in inner.namelist():
                    if not result_member.endswith("/results.json"):
                        continue
                    result = json.loads(inner.read(result_member))
                    if result.get("eval_complete") is False:
                        continue
                    experiment_id = str(result.get("experiment_id", ""))
                    environment = experiment_id.split("__", 1)[0]
                    if not environment:
                        environment = result_member.split("/", 1)[0].split("__", 1)[0]
                    seed = result.get("replicate_seed")
                    if environment and isinstance(seed, int):
                        jobs.append(Job(
                            method, environment, seed, nested, None,
                            result_member, None,
                            (int("outdated" not in nested.name.lower()), _timestamp(result_member)),
                            float(result["solve_rate"]) if isinstance(result.get("solve_rate"), (int, float)) else None,
                        ))
        except (zipfile.BadZipFile, json.JSONDecodeError, KeyError):
            continue
    return jobs


def _scan_local(root: Path) -> list[Job]:
    jobs: list[Job] = []
    for result_path in root.rglob("results.json"):
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        experiment_id = str(result.get("experiment_id", ""))
        method = _method("", "", experiment_id)
        seed = result.get("replicate_seed")
        environment = experiment_id.split("__", 1)[0]
        if method and environment and isinstance(seed, int) and result.get("eval_complete") is not False:
            jobs.append(Job(
                method, environment, seed, None, None, None, result_path.parent,
                (1, _timestamp(str(result_path))),
                float(result["solve_rate"]) if isinstance(result.get("solve_rate"), (int, float)) else None,
            ))
    return jobs


def _safe_output(root: Path, relative: PurePosixPath) -> Path | None:
    if not relative.parts or ".." in relative.parts:
        return None
    return root.joinpath(*relative.parts)


def _extract_job(job: Job, cache: Path) -> Path:
    if job.local_run is not None:
        return job.local_run
    assert job.outer and job.result_member
    destination = cache / job.method / job.environment / f"seed_{job.seed}"
    marker = destination / ".ready"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    run_prefix = job.result_member.rsplit("/", 1)[0] + "/"
    with zipfile.ZipFile(job.outer) as inner:
        for name in inner.namelist():
            if not name.startswith(run_prefix) or name.endswith("/"):
                continue
            relative = PurePosixPath(name[len(run_prefix):])
            if relative.parts[:2] in {
                ("sandbox", ".git"), ("sandbox", ".agent_sessions"),
                ("sandbox", ".mcp"), ("sandbox", "__pycache__"),
            }:
                continue
            output = _safe_output(destination, relative)
            if output is None:
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(inner.read(name))
    marker.touch()
    return destination


def _command(job: Job, run_dir: Path, output: Path, episodes: int) -> list[str]:
    overrides_path = run_dir / ".hydra" / "overrides.yaml"
    overrides = yaml.safe_load(overrides_path.read_text()) or []
    drop = ("num_eval_tasks=", "num_eval_workers=", "experiment_id=", "hydra.")
    overrides = [str(value) for value in overrides if not str(value).startswith(drop)]
    overrides.extend([
        f"num_eval_tasks={episodes}", "num_eval_workers=1", "experiment_id=null",
        "render_videos=false", "record_approach_history=false",
        f"hydra.run.dir={output}",
    ])
    if job.method != "planner":
        overrides.append(f"++approach.load_dir={run_dir}")
    return ["uv", "run", "python", "experiments/run_experiment.py", *overrides]


def _run_one(job: Job, args: argparse.Namespace) -> tuple[Job, int, str]:
    output = args.output / job.method / job.environment / f"seed_{job.seed}"
    result_path = output / "results.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
            required = (
                ("planning_time",)
                if job.method == "planner"
                else ("policy_time_s", "env_time_s")
            )
            if len(result.get("per_episode", [])) == args.episodes and all(
                all(field in episode for field in required)
                for episode in result["per_episode"]
            ):
                return job, 0, "already complete"
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    run_dir = _extract_job(job, args.cache)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TMPDIR"] = str(args.tmpdir)
    log_path = output / "timing_replay.log"
    with log_path.open("w") as log:
        process = subprocess.run(
            _command(job, run_dir, output, args.episodes), cwd=args.repo,
            env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    return job, process.returncode, str(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--blackbox", type=Path, required=True)
    parser.add_argument("--blackbox-dir", type=Path)
    parser.add_argument("--whitebox", type=Path)
    parser.add_argument("--local-results-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tmpdir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument(
        "--methods", nargs="+",
        help="Only replay these method labels (for example: claude_whitebox)",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    for path in (args.output, args.cache, args.tmpdir):
        path.mkdir(parents=True, exist_ok=True)
    print("Materializing and indexing nested result archives...", flush=True)
    discovered = _scan_outer(args.baselines, "baselines", args.cache)
    discovered.extend(_scan_outer(args.blackbox, "blackbox", args.cache))
    if args.blackbox_dir:
        discovered.extend(_scan_nested_directory(args.blackbox_dir, "blackbox"))
    if args.whitebox:
        discovered.extend(_scan_outer(args.whitebox, "whitebox", args.cache))
    if args.local_results_root:
        discovered.extend(_scan_local(args.local_results_root))
    selected: dict[tuple[str, str, int], Job] = {}
    for job in discovered:
        if args.methods and job.method not in args.methods:
            continue
        key = (job.method, job.environment, job.seed)
        if key not in selected or job.rank > selected[key].rank:
            selected[key] = job
    jobs = sorted(selected.values(), key=lambda job: (job.environment, job.method, job.seed))
    manifest: list[dict[str, object]] = []
    manifest_path = args.output / "manifest.json"
    if args.methods and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        manifest.extend(row for row in previous if row.get("method") not in args.methods)
    manifest.extend([
        {"method": job.method, "environment": job.environment, "seed": job.seed,
         "solve_rate": job.solve_rate,
         "source": str(job.local_run or job.outer)}
        for job in jobs
    ])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Replaying {len(jobs)} final policies with {args.jobs} workers", flush=True)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_run_one, job, args) for job in jobs]
        completed = concurrent.futures.as_completed(futures)
        for index, future in enumerate(
            tqdm(completed, total=len(futures), desc="timing replays", unit="run"), 1
        ):
            job, code, detail = future.result()
            failures += code != 0
            print(
                f"[{index}/{len(jobs)}] {job.method} {job.environment} seed={job.seed} "
                f"{'ok' if code == 0 else f'FAILED({code})'} {detail}", flush=True,
            )
    print(f"Complete: {len(jobs) - failures} succeeded, {failures} failed", flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
