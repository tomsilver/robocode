"""Plan and run the checkpoint evaluations, one process per core.

``plan`` freezes the pending checkpoints into a manifest and splits them into shards
by longest-processing-time-first on the original eval's wall time. ``run`` evaluates
one shard, so SLURM array task ``i`` runs ``--shard i``. Every submission plans
afresh from what is still missing, so resubmitting resumes and rebalances.

Example:
    python -m experiments.reeval.worker plan --store S --out O --manifest M \
        --num-shards 40 --jobs 32
    REEVAL_EVAL_SEED=<seed> python -m experiments.reeval.worker run --store S \
        --out O --manifest M --shard 0 --jobs 32
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.reeval.evaluate import eval_seed_from_env, evaluate
from experiments.reeval.store import Task, list_tasks, output_path, write_json_atomic

logger = logging.getLogger(__name__)


def assign_shards(tasks: list[Task], num_shards: int) -> list[list[Task]]:
    """Greedy LPT split; each shard lists its tasks longest first."""
    ordered = sorted(tasks, key=lambda t: (-t.est_seconds, t.run_key, t.tree))
    heap = [(0.0, i) for i in range(num_shards)]
    shards: list[list[Task]] = [[] for _ in range(num_shards)]
    for task in ordered:
        load, i = heapq.heappop(heap)
        shards[i].append(task)
        heapq.heappush(heap, (load + task.est_seconds, i))
    return shards


def is_done(out: Path, task: Task, retry_failed: bool) -> bool:
    """A record exists (and, with ``retry_failed``, is a success)."""
    path = output_path(out, task)
    if not path.is_file():
        return False
    if not retry_failed:
        return True
    return json.loads(path.read_text())["status"] == "ok"


def plan(args: argparse.Namespace) -> None:
    """Write the manifest of pending checkpoints and print the cost estimate."""
    tasks = list_tasks(args.store, args.select)
    if args.match:
        tasks = [t for t in tasks if re.search(args.match, t.run_key)]
    pending = [t for t in tasks if not is_done(args.out, t, args.retry_failed)]
    if args.limit is not None:
        pending = sorted(pending, key=lambda t: (t.est_seconds, t.run_key))
        pending = pending[: args.limit]
    shards = assign_shards(pending, args.num_shards)
    write_json_atomic(
        args.manifest, [[dataclasses.asdict(t) for t in shard] for shard in shards]
    )
    loads = [sum(t.est_seconds for t in s) / 3600 for s in shards]
    longest = max((t.est_seconds for t in pending), default=0.0) / 3600
    print(f"pending: {len(pending)} of {len(tasks)} checkpoints")
    print(
        f"estimated core-hours: {sum(loads):.0f} (longest checkpoint {longest:.1f} h)"
    )
    print(
        f"{args.num_shards} shards x {args.jobs} cores: busiest shard needs about "
        f"{max(loads, default=0) / args.jobs:.1f} h, mean "
        f"{sum(loads) / args.num_shards / args.jobs:.1f} h"
    )


def run(args: argparse.Namespace) -> None:
    """Evaluate one shard of the manifest."""
    seed = eval_seed_from_env()
    shards = json.loads(args.manifest.read_text())
    mine = [Task(**t) for t in shards[args.shard]]
    mine = [t for t in mine if not is_done(args.out, t, args.retry_failed)]
    logger.info(
        "shard %d/%d: %d pending checkpoints, %d parallel",
        args.shard,
        len(shards),
        len(mine),
        args.jobs,
    )
    args.scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(evaluate, task, args.store, args.scratch, seed): task
            for task in mine
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("%s %s: harness error", task.run_key, task.tree)
                continue
            write_json_atomic(output_path(args.out, task), record)
            done += 1
            logger.info(
                "[%d/%d] %s %s %s solve_rate=%s %.0fs",
                done,
                len(mine),
                record["status"],
                task.run_key,
                task.tree[:12],
                (record["results"] or {}).get("solve_rate"),
                record["wall_seconds"],
            )
    logger.info("shard finished in %.0f s", time.monotonic() - t0)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--store", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
        p.add_argument("--retry-failed", action="store_true")
    plan_parser = sub.choices["plan"]
    plan_parser.add_argument("--select", choices=["all", "final"], default="all")
    plan_parser.add_argument("--num-shards", type=int, default=1)
    plan_parser.add_argument("--match", help="regex on run keys to keep")
    plan_parser.add_argument(
        "--limit", type=int, help="keep only the N cheapest checkpoints (smoke tests)"
    )
    run_parser = sub.choices["run"]
    run_parser.add_argument("--shard", type=int, default=0)
    run_parser.add_argument(
        "--scratch", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp"))
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "plan":
        plan(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
