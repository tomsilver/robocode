"""Assemble the reset-distribution readout table.

Scans run directories and prints solve rates per (condition, seed), split by the
evaluation suite each run was scored on. Point it at one or more roots:

    python experiments/analyze_reset_dist.py outputs/reset_dist_a_2026-08-01
"""

import json
import sys
from pathlib import Path


def _load(results_path: Path) -> dict | None:
    """Read a results.json, or None if it is missing or unparseable."""
    try:
        with open(results_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def collect(root: Path) -> list[dict]:
    """Gather one row per scored run found under *root*."""
    rows = []
    for results_path in sorted(root.rglob("results.json")):
        data = _load(results_path)
        if data is None or "solve_rate" not in data:
            continue
        rel = results_path.relative_to(root).parts
        rows.append(
            {
                "group": "/".join(rel[:-3]) if len(rel) > 3 else "",
                "cond": rel[-3] if len(rel) >= 3 else "?",
                "seed": rel[-2],
                "solve_rate": data["solve_rate"],
                "eval_seed": data.get("eval_seed"),
                "crashed": data.get("num_crashed_episodes", 0),
                "by_count": data.get("by_count"),
                "cost": data.get("agent_cost_usd"),
            }
        )
    return rows


def main() -> None:
    """Print the solve-rate table for every run under the given roots."""
    roots = [Path(a) for a in sys.argv[1:]] or [Path("outputs")]
    rows: list[dict] = []
    for root in roots:
        rows.extend(collect(root))
    if not rows:
        print("no results.json with a solve_rate found")
        return

    width = max(max(len(r["group"]) for r in rows), len("group")) + 2
    print(
        f"{'group'.ljust(width)}{'cond':10}{'seed':7}{'solve':>7}"
        f"{'crash':>7}{'eval_seed':>11}{'cost':>8}"
    )
    print("-" * (width + 50))
    for r in sorted(rows, key=lambda r: (r["group"], r["cond"], r["seed"])):
        cost = f"{r['cost']:.2f}" if r["cost"] is not None else "-"
        print(
            f"{r['group'].ljust(width)}{r['cond']:10}{r['seed']:7}"
            f"{r['solve_rate']:>7.2f}{r['crashed']:>7}"
            f"{str(r['eval_seed']):>11}{cost:>8}"
        )

    print("\nby object count")
    for r in sorted(rows, key=lambda r: (r["group"], r["cond"], r["seed"])):
        if not r["by_count"]:
            continue
        counts = " ".join(
            f"{c}:{v['solve_rate']:.2f}" if isinstance(v, dict) else f"{c}:{v}"
            for c, v in sorted(r["by_count"].items(), key=lambda kv: int(kv[0]))
        )
        print(f"  {r['group']}/{r['cond']}/{r['seed']}: {counts}")


if __name__ == "__main__":
    main()
