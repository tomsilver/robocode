"""Evaluate one sandbox checkpoint of a stored run on the held-out episode suite.

The checkpoint's tree is exported from the run's bundle into a scratch directory and
scored by ``experiments/run_experiment.py`` with ``approach.load_dir``, so no agent or
container is started. The run's own environment config is used verbatim, since object
counts differ between campaigns and the repository's environment files.

Example:
    REEVAL_EVAL_SEED=<seed> python -m experiments.reeval.evaluate \
        --store <store> --out <out> --run <run_key> --tree <tree sha>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from experiments.reeval.store import (
    BUNDLE_FILE,
    Task,
    load_run,
    output_path,
    write_json_atomic,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_SCHEMA = 1
_LOG_TAIL_LINES = 40
# One eval process per core: numerical libraries must not spawn their own threads,
# because the policy time budget is measured in wall-clock time.
EVAL_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MUJOCO_GL": "disable",
    "DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def eval_seed_from_env() -> int:
    """The private evaluation master seed, supplied by the operator."""
    value = os.environ.get("REEVAL_EVAL_SEED")
    if not value:
        raise SystemExit("set REEVAL_EVAL_SEED to the team's evaluation seed")
    return int(value)


def cpu_model() -> str:
    """CPU model name, recorded because the policy budget is wall-clock time."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def harness_revision() -> str:
    """Revision of this repository checkout (baked into the image at build)."""
    baked = REPO_ROOT / "REVISION"
    if baked.is_file():
        return baked.read_text().strip()
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def export_tree(bundle: Path, tree: str, dest: Path) -> None:
    """Extract ``tree`` from a git bundle into ``dest``."""
    repo = dest.parent / "repo.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(bundle), str(repo)],
        check=True,
        capture_output=True,
    )
    dest.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "--git-dir", str(repo), "archive", "--format=tar", tree],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)


def build_command(
    run: dict[str, Any], conf_dir: Path, load_dir: Path, run_dir: Path, seed: int
) -> list[str]:
    """``run_experiment.py`` invocation that only evaluates ``load_dir/sandbox``."""
    return [
        sys.executable,
        str(REPO_ROOT / "experiments" / "run_experiment.py"),
        "environment=reeval_env",
        f"hydra.searchpath=[file://{conf_dir}]",
        "approach=agentic",
        "primitive_level=none",
        f"approach.blackbox={run['blackbox']}",
        f"approach.blackbox_strict={run['blackbox_strict']}",
        # Checked by name only; loading a saved approach starts no container.
        "approach.container_backend=apptainer",
        f"approach.load_dir={load_dir}",
        f"eval_timeout={run['eval_timeout']}",
        f"num_eval_tasks={run['num_eval_tasks']}",
        f"max_steps={run['max_steps']}",
        f"replicate_seed={run['replicate_seed']}",
        f"eval_seed={seed}",
        "render_videos=false",
        "record_approach_history=false",
        "mcp_tools=[]",
        f"hydra.run.dir={run_dir}",
    ]


def subprocess_timeout(run: dict[str, Any]) -> float:
    """Upper bound on one checkpoint: every episode at the harness's wall cap."""
    return float(run["num_eval_tasks"]) * (10 * float(run["eval_timeout"]) + 60)


def evaluate(task: Task, store: Path, scratch: Path, seed: int) -> dict[str, Any]:
    """Score one checkpoint and return its record (without writing it)."""
    run = load_run(store, task.run_key)
    commits = [c["sha"] for c in run["commits"] if c["tree"] == task.tree]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="reeval-", dir=scratch))
    try:
        load_dir = work / "checkpoint"
        bundle = store / "runs" / task.run_key / BUNDLE_FILE
        export_tree(bundle, task.tree, load_dir / "sandbox")
        env_conf = work / "conf" / "environment" / "reeval_env.yaml"
        env_conf.parent.mkdir(parents=True)
        env_conf.write_text(yaml.safe_dump(run["environment"]))
        run_dir = work / "run"
        cmd = build_command(run, work / "conf", load_dir, run_dir, seed)
        log_path = work / "eval.log"
        status = "ok"
        with open(log_path, "wb") as log:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=work,
                    env={**os.environ, **EVAL_ENV},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=subprocess_timeout(run),
                    check=False,
                )
                returncode: int | None = proc.returncode
            except subprocess.TimeoutExpired:
                returncode, status = None, "timeout"
        results_path = run_dir / "results.json"
        results = None
        if results_path.is_file():
            results = json.loads(results_path.read_text())
            results.pop("eval_seed", None)
        elif status == "ok":
            status = "failed"
        log_tail = None
        if status != "ok":
            lines = log_path.read_text(errors="replace").splitlines()
            log_tail = "\n".join(lines[-_LOG_TAIL_LINES:])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {
        "schema": RECORD_SCHEMA,
        "run_key": task.run_key,
        "tree": task.tree,
        "commits": commits,
        "status": status,
        "returncode": returncode,
        "results": results,
        "log_tail": log_tail,
        "host": platform.node(),
        "cpu_model": cpu_model(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "harness_revision": harness_revision(),
        "started_at": started,
        "wall_seconds": round(time.monotonic() - t0, 1),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", required=True, help="run key in the store")
    parser.add_argument("--tree", required=True, help="sandbox tree sha")
    parser.add_argument(
        "--scratch", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp"))
    )
    args = parser.parse_args()
    task = Task(args.run, args.tree, 0.0)
    record = evaluate(task, args.store, args.scratch, eval_seed_from_env())
    write_json_atomic(output_path(args.out, task), record)
    results = record["results"] or {}
    print(f"{record['status']} solve_rate={results.get('solve_rate')}")


if __name__ == "__main__":
    main()
