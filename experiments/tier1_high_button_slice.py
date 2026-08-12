"""Build and score high-button slices of StickButton2D (reach-ceiling probe).

A slice pins (seed, count) instances rejection-sampled from a chosen reset
distribution so that every instance holds at least one button above a height
ceiling, optionally with the stick restricted to an x range. Frozen policies are
then scored on the fixed slice. Environments are built from pinned configs, so
either distribution can be scored regardless of the kinder checkout:

    python experiments/tier1_high_button_slice.py --build-slice s.json \
        --distribution standard --stick-min 0.5 --stick-max 3.0 --n 100
    python experiments/tier1_high_button_slice.py --slice s.json \
        --policy <run_dir> --cond noprims --out <dir>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from robocode.approaches.base_approach import BaseApproach
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.primitives import build_primitives
from robocode.utils.episode import load_generated_approach, run_episode_with_timeout

EVAL_COUNTS = [1, 2, 3, 5, 10]
SIDE_GRASP_CEILING = 2.43
PRIMITIVES = {
    "noprims": [],
    "lowlevel": ["check_action_collision", "BiRRT"],
    "bilevel": ["bilevel_models"],
}
ENV_PATHS = {
    "standard": "kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",
    "band": "robocode.environments.stickbutton2d_a:StickButton2DEnv",
}


def make_env(distribution: str) -> VariableObjectCountEnv:
    """The variable-count StickButton2D env over the named spawn distribution."""
    return VariableObjectCountEnv(
        constant_object_env_path=ENV_PATHS[distribution],
        count_kwarg="num_buttons",
        count_object_prefix="button",
        design_counts=[1, 2, 3],
        eval_counts=EVAL_COUNTS,
        bilevel_env_name="stickbutton2d",
    )


def _instance_geometry(state: Any) -> tuple[float, float]:
    ys = [
        float(state.get(state.get_object_from_name(b), "y"))
        for b in state.get_object_names()
        if b.startswith("button")
    ]
    stick_x = float(state.get(state.get_object_from_name("stick"), "x"))
    return max(ys), stick_x


def build_slice(args: argparse.Namespace) -> None:
    """Collect instances with a button above the ceiling and the stick in range."""
    env = make_env(args.distribution)
    rng = np.random.default_rng(args.search_seed)
    chosen: list[dict] = []
    tried = 0
    while len(chosen) < args.n:
        tried += 1
        seed = int(rng.integers(0, 2**63))
        count = EVAL_COUNTS[len(chosen) % len(EVAL_COUNTS)]
        state, _ = env.reset(seed=seed, options={"object_count": count})
        max_by, stick_x = _instance_geometry(state)
        if max_by > args.ceiling and args.stick_min <= stick_x <= args.stick_max:
            chosen.append(
                {
                    "seed": seed,
                    "count": count,
                    "max_button_y": max_by,
                    "stick_x": stick_x,
                }
            )
            if len(chosen) % 20 == 0:
                print(f"  {len(chosen)}/{args.n} after {tried} draws", flush=True)
    payload = {
        "ceiling": args.ceiling,
        "distribution": args.distribution,
        "stick_min": args.stick_min,
        "stick_max": args.stick_max,
        "search_seed": args.search_seed,
        "acceptance_rate": len(chosen) / tried,
        "instances": chosen,
    }
    args.build_slice.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {len(chosen)} instances to {args.build_slice} "
        f"(acceptance {payload['acceptance_rate']:.4f})"
    )


class _GeneratedApproachAdapter(BaseApproach):
    """Drive a generated policy through the BaseApproach interface the runner uses."""

    def __init__(
        self,
        action_space: Any,
        observation_space: Any,
        seed: int,
        primitives: dict,
        generated: Any,
    ) -> None:
        super().__init__(action_space, observation_space, seed, primitives)
        self._generated = generated

    def reset(self, state: Any, info: dict) -> None:
        super().reset(state, info)
        self._generated.reset(state, info)

    def update(self, state: Any, reward: float, done: bool, info: dict) -> None:
        super().update(state, reward, done, info)
        if hasattr(self._generated, "update"):
            self._generated.update(state, reward, done, info)

    def _get_action(self) -> Any:
        return self._generated.get_action(self._last_state)


def evaluate(args: argparse.Namespace) -> None:
    """Score one frozen policy on an existing slice."""
    payload = json.loads(args.slice.read_text())
    env = make_env(payload["distribution"])
    primitives = build_primitives(env, PRIMITIVES[args.cond])
    generated = load_generated_approach(
        args.policy / "sandbox" / "approach.py",
        env.action_space,
        env.observation_space,
        primitives,
    )
    approach = _GeneratedApproachAdapter(
        env.action_space, env.observation_space, 0, {}, generated
    )

    per_episode = []
    for inst in payload["instances"]:
        max_steps = env.max_steps_for_count(inst["count"])
        try:
            result, _, _ = run_episode_with_timeout(
                env,
                approach,
                inst["seed"],
                max_steps,
                timeout=args.timeout,
                render=False,
                count=inst["count"],
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            per_episode.append(
                {
                    "solved": False,
                    "crashed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    **inst,
                }
            )
            continue
        per_episode.append({**result, **inst})

    scored = [e for e in per_episode if not e.get("crashed")]
    solve_rate = (
        float(np.mean([e["solved"] for e in scored])) if scored else float("nan")
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "slice": str(args.slice),
                "ceiling": payload["ceiling"],
                "distribution": payload["distribution"],
                "policy": str(args.policy),
                "solve_rate": solve_rate,
                "num_eval_tasks": len(payload["instances"]),
                "num_crashed_episodes": len(per_episode) - len(scored),
                "per_episode": per_episode,
            },
            indent=2,
        )
    )
    by_count: dict[int, list] = {}
    for e in scored:
        by_count.setdefault(e["count"], []).append(e["solved"])
    counts = " ".join(f"{c}:{np.mean(v):.2f}" for c, v in sorted(by_count.items()))
    print(f"{args.policy}: slice solve rate {solve_rate:.3f}  [{counts}]", flush=True)


def main() -> None:
    """Build or evaluate the configured high-button task slice."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-slice", type=Path)
    parser.add_argument("--distribution", choices=sorted(ENV_PATHS), default="standard")
    parser.add_argument("--stick-min", type=float, default=0.0)
    parser.add_argument("--stick-max", type=float, default=3.45)
    parser.add_argument("--slice", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--cond", choices=sorted(PRIMITIVES))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--ceiling", type=float, default=SIDE_GRASP_CEILING)
    parser.add_argument("--search-seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.build_slice:
        build_slice(args)
        return
    if not (args.slice and args.policy and args.cond and args.out):
        parser.error("evaluation needs --slice, --policy, --cond and --out")
    evaluate(args)


if __name__ == "__main__":
    main()
