"""Score a frozen policy on a reset distribution pinned by argument, not by checkout.

The reset-distribution experiment needs a policy scored on a distribution other than
the one the working tree currently carries, while synthesis runs are still using that
tree. Passing the stick spawn bounds explicitly keeps both possible at once, and
reproduces the instances exactly: the two variants differ only in that one config
field, so the sampler consumes the same randomness given the same bounds.

Results are additionally split on the side-grasp reach ceiling, which is the failure
mode a policy that only ever grasps the stick from the side should show.

    python experiments/eval_on_pinned_distribution.py --policy <run_dir> \
        --cond noprims --distribution standard --out <dir>
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from kinder.envs.kinematic2d.stickbutton2d import (
    StickButton2DEnv,
    StickButton2DEnvConfig,
)
from kinder.envs.kinematic2d.structs import SE2Pose

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

_BASE = StickButton2DEnvConfig()
_TABLE_Y = _BASE.table_pose.y
_STICK_H = _BASE.stick_shape[1]
_TOP = SE2Pose(_BASE.world_max_x - _BASE.stick_shape[0], _TABLE_Y - _STICK_H / 10, 0)

STICK_BOUNDS = {
    "standard": (SE2Pose(_BASE.world_min_x, _TABLE_Y - _STICK_H / 2, 0), _TOP),
    "band": (SE2Pose(3.435, _TABLE_Y - _STICK_H / 2, 0), _TOP),
}


def make_env_class(distribution: str) -> type:
    """A StickButton2DEnv whose stick spawn range is pinned regardless of the tree."""
    config = replace(_BASE, stick_init_pose_bounds=STICK_BOUNDS[distribution])

    class PinnedStickButton2DEnv(StickButton2DEnv):
        """StickButton2DEnv with the stick spawn range fixed at construction."""

        def __init__(self, num_buttons: int = 3, **kwargs):
            super().__init__(num_buttons=num_buttons, config=config, **kwargs)

    return PinnedStickButton2DEnv


def make_env(distribution: str) -> VariableObjectCountEnv:
    """Build the variable-count env over a pinned stick spawn range."""
    env_cls = make_env_class(distribution)
    module = f"{__name__}:_pinned_{distribution}"
    globals()[f"_pinned_{distribution}"] = env_cls
    return VariableObjectCountEnv(
        constant_object_env_path=module,
        count_kwarg="num_buttons",
        count_object_prefix="button",
        design_counts=[1, 2, 3],
        eval_counts=EVAL_COUNTS,
        bilevel_env_name="stickbutton2d",
    )


class _GeneratedApproachAdapter(BaseApproach):
    """Drive a generated policy through the BaseApproach interface the runner uses.

    A generated ``approach.py`` exposes ``reset``/``get_action``/``update``; the
    episode runner calls ``step()``. This mirrors what ``GeneratedProgramApproach``
    does in the real eval path so scores are directly comparable.
    """

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


def main() -> None:
    """Score one frozen policy on the pinned distribution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--cond", choices=sorted(PRIMITIVES), required=True)
    parser.add_argument("--distribution", choices=sorted(STICK_BOUNDS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--ceiling", type=float, default=SIDE_GRASP_CEILING)
    args = parser.parse_args()

    env = make_env(args.distribution)
    generated = load_generated_approach(
        args.policy / "sandbox" / "approach.py",
        env.action_space,
        env.observation_space,
        build_primitives(env, PRIMITIVES[args.cond]),
    )
    approach = _GeneratedApproachAdapter(
        env.action_space, env.observation_space, 0, {}, generated
    )

    task_rng = np.random.default_rng(args.eval_seed)
    seeds = [int(task_rng.integers(0, 2**63)) for _ in range(args.num_eval)]
    counts = [EVAL_COUNTS[i % len(EVAL_COUNTS)] for i in range(args.num_eval)]

    per_episode = []
    for seed, count in zip(seeds, counts):
        state, _ = env.reset(seed=seed, options={"object_count": count})
        max_button_y = max(
            float(state.get(state.get_object_from_name(n), "y"))
            for n in state.get_object_names()
            if n.startswith("button")
        )
        stick_x = float(state.get(state.get_object_from_name("stick"), "x"))
        max_steps = env.max_steps_for_count(count)
        try:
            result, _, _ = run_episode_with_timeout(
                env,
                approach,
                seed,
                max_steps,
                timeout=args.timeout,
                render=False,
                count=count,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = {
                "solved": False,
                "crashed": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        per_episode.append(
            {
                **result,
                "seed": seed,
                "count": count,
                "max_button_y": max_button_y,
                "stick_x": stick_x,
            }
        )

    scored = [e for e in per_episode if not e.get("crashed")]
    solve = float(np.mean([e["solved"] for e in scored])) if scored else float("nan")
    high = [e for e in scored if e["max_button_y"] > args.ceiling]
    low = [e for e in scored if e["max_button_y"] <= args.ceiling]

    def rate(v: list) -> float:
        return float(np.mean([e["solved"] for e in v])) if v else float("nan")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "policy": str(args.policy),
                "distribution": args.distribution,
                "eval_seed": args.eval_seed,
                "solve_rate": solve,
                "ceiling": args.ceiling,
                "solve_rate_above_ceiling": rate(high),
                "solve_rate_below_ceiling": rate(low),
                "n_above_ceiling": len(high),
                "num_crashed_episodes": len(per_episode) - len(scored),
                "per_episode": per_episode,
            },
            indent=2,
        )
    )

    print(f"{args.policy} on {args.distribution}: solve {solve:.3f}")
    print(
        f"   above ceiling {rate(high):.3f} (n={len(high)})  "
        f"below {rate(low):.3f} (n={len(low)})"
    )
    fails = [e for e in scored if not e["solved"]]
    if fails:
        above = sum(1 for e in fails if e["max_button_y"] > args.ceiling)
        print(f"   {len(fails)} failures, {above} with a button above the ceiling")
        for e in fails[:12]:
            flag = " <- above ceiling" if e["max_button_y"] > args.ceiling else ""
            print(
                f"     count={e['count']:2d} max_button_y={e['max_button_y']:.4f} "
                f"stick_x={e['stick_x']:.4f}{flag}"
            )


if __name__ == "__main__":
    main()
