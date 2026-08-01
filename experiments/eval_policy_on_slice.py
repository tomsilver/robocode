"""Evaluate frozen policies on a targeted slice of the instance distribution.

The regular eval suite draws instances uniformly, so a rare configuration lands in
only a handful of the 100 episodes and the suite has almost no power to resolve
whether a policy handles it. This builds a slice by rejection-sampling seeds that
satisfy a predicate (e.g. "holds a button above the side-grasp reach ceiling"), then
scores each policy on that fixed slice.

    python experiments/eval_policy_on_slice.py --build-slice out.json --n 100
    python experiments/eval_policy_on_slice.py --slice out.json --policy <run_dir> \
        --cond noprims --out <dir>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnvConfig

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


def make_env() -> VariableObjectCountEnv:
    """Build the variable-count StickButton2D env used for slice scoring."""
    return VariableObjectCountEnv(
        constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:"
        "StickButton2DEnv",
        count_kwarg="num_buttons",
        count_object_prefix="button",
        design_counts=[1, 2, 3],
        eval_counts=EVAL_COUNTS,
        bilevel_env_name="stickbutton2d",
    )


def build_slice(path: Path, n: int, ceiling: float, search_seed: int) -> None:
    """Collect n instances that hold a button above the reach ceiling."""
    env = make_env()
    rng = np.random.default_rng(search_seed)
    chosen: list[dict] = []
    tried = 0
    while len(chosen) < n:
        tried += 1
        seed = int(rng.integers(0, 2**63))
        count = EVAL_COUNTS[len(chosen) % len(EVAL_COUNTS)]
        state, _ = env.reset(seed=seed, options={"object_count": count})
        ys = [
            float(state.get(state.get_object_from_name(b), "y"))
            for b in state.get_object_names()
            if b.startswith("button")
        ]
        if max(ys) > ceiling:
            chosen.append({"seed": seed, "count": count, "max_button_y": max(ys)})
            if len(chosen) % 20 == 0:
                print(f"  {len(chosen)}/{n} after {tried} draws", flush=True)
    payload = {
        "ceiling": ceiling,
        "stick_x_lower_bound": round(
            StickButton2DEnvConfig().stick_init_pose_bounds[0].x, 4
        ),
        "search_seed": search_seed,
        "acceptance_rate": len(chosen) / tried,
        "instances": chosen,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {len(chosen)} instances to {path} "
        f"(acceptance {payload['acceptance_rate']:.3f})"
    )


class _GeneratedApproachAdapter(BaseApproach):
    """Drive a generated policy through the BaseApproach interface the runner uses.

    A generated ``approach.py`` exposes ``reset``/``get_action``/``update``, while the
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


def evaluate(
    slice_path: Path, policy_dir: Path, cond: str, out_dir: Path, timeout: float
) -> None:
    payload = json.loads(slice_path.read_text())
    instances = payload["instances"]
    # A slice pins seeds, but a seed only names the same instance under the reset
    # distribution it was drawn from. Fail loudly rather than score a different one.
    expected = payload["stick_x_lower_bound"]
    actual = round(StickButton2DEnvConfig().stick_init_pose_bounds[0].x, 4)
    if actual != expected:
        raise ValueError(
            f"slice was built with stick x lower bound {expected}, but the "
            f"environment currently has {actual}; check out the matching kinder source"
        )
    env = make_env()
    primitives = build_primitives(env, PRIMITIVES[cond])
    generated = load_generated_approach(
        policy_dir / "sandbox" / "approach.py",
        env.action_space,
        env.observation_space,
        primitives,
    )
    approach = _GeneratedApproachAdapter(
        env.action_space, env.observation_space, 0, {}, generated
    )

    per_episode = []
    for inst in instances:
        max_steps = env.max_steps_for_count(inst["count"])
        try:
            result, _, _ = run_episode_with_timeout(
                env,
                approach,
                inst["seed"],
                max_steps,
                timeout=timeout,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "slice": str(slice_path),
                "ceiling": payload["ceiling"],
                "solve_rate": solve_rate,
                "num_eval_tasks": len(instances),
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
    print(f"{policy_dir}: slice solve rate {solve_rate:.3f}  [{counts}]")


def main() -> None:
    """Build a slice, or score one frozen policy on an existing slice."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-slice", type=Path)
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
        build_slice(args.build_slice, args.n, args.ceiling, args.search_seed)
        return
    if not (args.slice and args.policy and args.cond and args.out):
        parser.error("evaluation needs --slice, --policy, --cond and --out")
    evaluate(args.slice, args.policy, args.cond, args.out, args.timeout)


if __name__ == "__main__":
    main()
