"""Test whether a policy's failures concentrate on buttons only a bottom grasp reaches.

A side grasp carries the stick lower than a bottom grasp does, so buttons above the
side-grasp ceiling are pressable only after grasping the stick from below. This
regenerates each evaluated instance (deterministic given eval_seed and the object
count schedule) and splits the per-episode outcomes on whether the instance holds
such a button.

The instances are rebuilt from the environment as currently configured, so only
point this at runs scored on that same distribution.

    python experiments/analyze_reach_failures.py <run_dir> [--ceiling 2.43]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from robocode.environments.variable_object_count_env import VariableObjectCountEnv

EVAL_COUNTS = [1, 2, 3, 5, 10]
SIDE_GRASP_CEILING = 2.43


def instance_features(eval_seed: int, num_eval: int) -> list[dict]:
    env = VariableObjectCountEnv(
        constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:"
        "StickButton2DEnv",
        count_kwarg="num_buttons",
        count_object_prefix="button",
        design_counts=[1, 2, 3],
        eval_counts=EVAL_COUNTS,
        bilevel_env_name="stickbutton2d",
    )
    task_rng = np.random.default_rng(eval_seed)
    seeds = [int(task_rng.integers(0, 2**63)) for _ in range(num_eval)]
    counts = [EVAL_COUNTS[i % len(EVAL_COUNTS)] for i in range(num_eval)]
    out = []
    for seed, count in zip(seeds, counts):
        state, _ = env.reset(seed=seed, options={"object_count": count})
        ys = [
            float(state.get(state.get_object_from_name(n), "y"))
            for n in state.get_object_names()
            if n.startswith("button")
        ]
        out.append(
            {
                "seed": seed,
                "count": count,
                "max_button_y": max(ys),
                "stick_x": float(state.get(state.get_object_from_name("stick"), "x")),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ceiling", type=float, default=SIDE_GRASP_CEILING)
    args = parser.parse_args()

    with open(args.run_dir / "results.json", encoding="utf-8") as f:
        results = json.load(f)
    per_episode = results["per_episode"]
    eval_seed = results.get("eval_seed")
    if eval_seed is None:
        raise ValueError("results.json has no eval_seed; cannot rebuild the suite")

    feats = instance_features(eval_seed, len(per_episode))
    for ep, ft in zip(per_episode, feats):
        if ep.get("object_count") is not None and ep["object_count"] != ft["count"]:
            raise ValueError("rebuilt suite does not match the recorded object counts")

    high = [
        (e, f) for e, f in zip(per_episode, feats) if f["max_button_y"] > args.ceiling
    ]
    low = [
        (e, f) for e, f in zip(per_episode, feats) if f["max_button_y"] <= args.ceiling
    ]

    def rate(pairs: list) -> tuple[float, int]:
        if not pairs:
            return float("nan"), 0
        sol = sum(1 for e, _ in pairs if e.get("solved"))
        return sol / len(pairs), len(pairs)

    hi_rate, hi_n = rate(high)
    lo_rate, lo_n = rate(low)
    print(f"run: {args.run_dir}")
    print(
        f"overall solve rate: {results['solve_rate']:.3f}  "
        f"(eval_seed={eval_seed}, n={len(per_episode)})"
    )
    print(f"side-grasp reach ceiling: {args.ceiling}")
    print(
        f"  instances with a button ABOVE the ceiling: "
        f"{hi_rate:.3f} solved  (n={hi_n})"
    )
    print(
        f"  instances with all buttons below:          "
        f"{lo_rate:.3f} solved  (n={lo_n})"
    )

    failures = [(e, f) for e, f in zip(per_episode, feats) if not e.get("solved")]
    if failures:
        above = sum(1 for _, f in failures if f["max_button_y"] > args.ceiling)
        print(
            f"\n{len(failures)} failures, {above} of them on instances with a "
            f"button above the ceiling"
        )
        print("  failures (count, max_button_y, stick_x):")
        for _, feat in failures[:20]:
            flag = " <- above ceiling" if feat["max_button_y"] > args.ceiling else ""
            print(
                f"    count={feat['count']:2d}  max_button_y={feat['max_button_y']:.4f}"
                f"  stick_x={feat['stick_x']:.4f}{flag}"
            )


if __name__ == "__main__":
    main()
