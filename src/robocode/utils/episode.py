"""Episode-running, video-saving, and approach-loading utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach

logger = logging.getLogger(__name__)

# Symbols that would let a generated program run the SeSamE planner instead of
# composing the bilevel MODELS itself. Deployed programs given the bilevel_models
# primitive must NOT call the planner: doing so reintroduces the per-instance
# search the generalized program is meant to amortize away.
_FORBIDDEN_PLANNER_REFS = (
    "run_sesame",
    "BacktrackingRefiner",
    "BilevelPlanningGraph",
    "bilevel_planning.sesame",
)


def _reject_planner_references(source: str, primitives: dict[str, Any]) -> None:
    """Reject a generated approach that invokes the bilevel planner (anti-cheat).

    Only enforced when the ``bilevel_models`` primitive is in play; other
    approaches are unaffected.
    """
    if "bilevel_models" not in primitives:
        return
    # Substring (not AST) matching is deliberate. This is a cooperative guardrail
    # plus a clear error message, not an adversarial sandbox: aliased imports still
    # carry the module path so they are caught, and defeating it needs deliberate
    # importlib obfuscation the instructed agent will not produce by accident. AST
    # parsing to dodge comment/string false positives would be over-engineering.
    hits = [ref for ref in _FORBIDDEN_PLANNER_REFS if ref in source]
    if hits:
        raise ValueError(
            f"Generated approach references the bilevel planner ({', '.join(hits)}); "
            "with the bilevel_models primitive you must compose the models yourself, "
            "not run SeSamE search."
        )


def load_generated_approach(
    path: Path,
    action_space: Any,
    observation_space: Any,
    primitives: dict[str, Any],
) -> Any:
    """Load a ``GeneratedApproach`` class from the given file.

    Temporarily adds the parent directory of *path* to ``sys.path`` so that
    ``approach.py`` can import sibling modules written by the agent, then
    removes it to avoid polluting the global import path.
    """
    sandbox_dir = str(path.parent.resolve())
    if sandbox_dir not in sys.path:
        sys.path.insert(0, sandbox_dir)
    try:
        source = path.read_text()
        _reject_planner_references(source, primitives)
        # Set __file__ so the exec'd code can use it (e.g. to locate
        # sibling modules via os.path.dirname(__file__)).  exec() does
        # not set this automatically unlike a normal module import.
        namespace: dict[str, Any] = {"__file__": str(path)}
        exec(compile(source, str(path), "exec"), namespace)  # pylint: disable=exec-used
    finally:
        sys.path.remove(sandbox_dir)
    cls = namespace["GeneratedApproach"]
    instance = cls(action_space, observation_space, primitives=primitives)
    logger.info("Loaded generated approach from %s", path)
    return instance


def run_episode(
    env: Any,
    approach: BaseApproach,
    seed: int,
    max_steps: int,
    render: bool = False,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]], Any]:
    """Run a single evaluation episode; return metrics, frames, final state."""
    state, info = env.reset(seed=seed)
    approach.reset(state, info)

    frames: list[NDArray[np.uint8]] = []

    def _capture() -> None:
        rendered: Any = env.render()
        if isinstance(rendered, np.ndarray):
            frames.append(rendered)

    if render:
        _capture()

    total_reward = 0.0
    num_steps = 0
    terminated = False
    for _ in range(max_steps):
        action = approach.step()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        num_steps += 1
        approach.update(state, float(reward), terminated or truncated, info)
        if render:
            _capture()
        if terminated or truncated:
            break

    metrics = {
        "total_reward": total_reward,
        "num_steps": num_steps,
        "solved": bool(terminated),
    }
    return metrics, frames, state


def run_per_instance_eval(
    env: Any,
    approach: BaseApproach,
    eval_seeds: list[int],
    *,
    max_budget_usd: float,
    output_dir: Path,
    max_budget_per_instance_usd: float | None = None,
    render: bool = False,
) -> dict[str, Any]:
    """Evaluate a per-instance approach, spending one global budget across seeds.

    Each seed gets its own budget-bounded agent run via ``approach.solve_instance``
    (which scores on its own configured eval horizon) until the global
    ``max_budget_usd`` is exhausted; remaining seeds are then left unattempted.
    ``max_budget_per_instance_usd`` optionally caps a single seed's spend
    (``None`` = a seed may use all remaining budget).

    Returns a results dict ready to merge into ``results.json``. ``solve_rate`` is
    over *all* eval seeds (unattempted, unsolved, and crashed attempts all count as
    failures); reward/step means are taken only over attempted, non-crashed,
    scored episodes.
    """
    per_episode: list[dict[str, Any]] = []
    remaining = max_budget_usd
    total_cost = 0.0
    num_solved = 0
    num_attempted = 0
    for i, seed in enumerate(eval_seeds):
        if remaining <= 0:
            per_episode.append(
                {
                    "seed": seed,
                    "attempted": False,
                    "solved": False,
                    "crashed": False,
                    "total_reward": None,
                    "num_steps": None,
                    "cost_usd": 0.0,
                }
            )
            continue
        budget_i = (
            remaining
            if max_budget_per_instance_usd is None
            else min(max_budget_per_instance_usd, remaining)
        )
        result = approach.solve_instance(
            env=env,
            seed=seed,
            budget_usd=budget_i,
            output_subdir=output_dir / f"instance_{i}",
            render=render,
        )
        num_attempted += 1
        remaining -= result.cost_usd
        total_cost += result.cost_usd
        if result.solved:
            num_solved += 1
        if render and result.frames:
            video_dir = output_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            save_video(result.frames, video_dir / f"episode_{i}.gif")
        per_episode.append(
            {
                # Approach-specific per-instance metrics first; the fixed keys
                # below take precedence over any colliding extras key.
                **result.extras,
                "seed": seed,
                "attempted": True,
                "solved": result.solved,
                "crashed": result.crashed,
                "total_reward": result.total_reward,
                "num_steps": result.num_steps,
                "cost_usd": result.cost_usd,
            }
        )

    scored = [e for e in per_episode if e["attempted"] and not e["crashed"]]
    num_crashed = sum(1 for e in per_episode if e["crashed"])

    def _mean(key: str) -> float:
        vals = [e[key] for e in scored if e[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    num_eval = len(eval_seeds)
    results = {
        "mean_eval_reward": _mean("total_reward"),
        "mean_eval_steps": _mean("num_steps"),
        "solve_rate": num_solved / num_eval if num_eval else float("nan"),
        "num_eval_tasks": num_eval,
        "num_attempted": num_attempted,
        "num_solved": num_solved,
        "num_evaluated_episodes": len(scored),
        "num_crashed_episodes": num_crashed,
        "per_episode": per_episode,
        "total_cost_usd": total_cost,
    }

    # Average any numeric approach-specific extras (e.g. planning_time) over
    # scored episodes, surfaced as mean_<key> so they flow into results.json and
    # become analyze_results columns automatically. Extras keys may vary across
    # episodes (a failed plan has fewer), so aggregate per key over whoever has it.
    reserved = {
        "seed",
        "attempted",
        "solved",
        "crashed",
        "total_reward",
        "num_steps",
        "cost_usd",
    }

    def _mean_extra(key: str) -> float:
        vals = [
            v
            for e in scored
            if isinstance((v := e.get(key)), (int, float)) and not isinstance(v, bool)
        ]
        return float(np.mean(vals)) if vals else float("nan")

    extra_keys = {
        k
        for e in scored
        for k, v in e.items()
        if k not in reserved and isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    for key in sorted(extra_keys):
        results[f"mean_{key}"] = _mean_extra(key)
    return results


def save_video(frames: list[NDArray[np.uint8]], path: Path, fps: int = 10) -> None:
    """Save a list of RGB frames as a gif."""
    duration = 1000.0 / fps  # ms per frame
    iio.imwrite(str(path), frames, duration=duration, loop=0)
    logger.info("Saved video to %s", path)


def save_frames(
    frames: list[NDArray[np.uint8]],
    output_dir: Path,
    max_frames: int | None = None,
) -> list[str]:
    """Save frames as individual PNGs, returning the list of filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    to_save = frames[:max_frames] if max_frames is not None else frames
    filenames: list[str] = []
    for i, frame in enumerate(to_save):
        filename = f"frame_{i:04d}.png"
        iio.imwrite(str(output_dir / filename), frame)
        filenames.append(filename)
    logger.info("Saved %d frames to %s", len(filenames), output_dir)
    return filenames
