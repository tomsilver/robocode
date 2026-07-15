"""Episode-running, video-saving, and approach-loading utilities."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sys
import traceback
from collections.abc import Callable
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
    "BilevelPlanningAgent",
)


def _reject_planner_references(source: str, primitives: dict[str, Any]) -> None:
    """Reject a generated approach that invokes the bilevel planner (anti-cheat).

    Only enforced when the ``bilevel_models`` primitive is in play; other
    approaches are unaffected.
    """
    if "bilevel_models" not in primitives:
        return
    # A cooperative guardrail plus a clear error message, not an adversarial
    # sandbox: substring matching suffices because aliased imports still carry the
    # module path, and only deliberate importlib obfuscation would evade it.
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
    count: int | None = None,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]], Any]:
    """Run a single evaluation episode; return metrics, frames, final state.

    ``count`` pins the object count for a variable-count env (via
    ``reset(options={"object_count": count})``); ``None`` leaves the env to sample.
    """
    if count is not None:
        state, info = env.reset(seed=seed, options={"object_count": count})
    else:
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
    if "object_count" in info:
        metrics["object_count"] = info["object_count"]
    return metrics, frames, state


def run_in_forked_worker(
    ctx: mp.context.ForkContext,
    target: Callable[..., None],
    args: tuple[Any, ...],
    timeout: float,
) -> tuple[str, int | None]:
    """Run ``target(*args)`` in a forked worker, killing it if it overruns ``timeout``.

    ``target`` writes its outcome into a shared dict it is handed in ``args`` (the
    caller reads that dict afterward). Returns ``(outcome, exitcode)``: ``"timeout"``
    means the worker overran and got a SIGINT, a short grace period, then SIGKILL;
    ``"finished"`` means it exited on its own (an empty shared dict then means it died
    before reporting).
    """
    proc = ctx.Process(target=target, args=args)
    proc.start()
    assert proc.pid is not None
    proc.join(timeout)
    if proc.is_alive():
        os.kill(proc.pid, signal.SIGINT)
        proc.join(3)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return "timeout", proc.exitcode
    return "finished", proc.exitcode


def run_episode_with_timeout(
    env: Any,
    approach: BaseApproach,
    seed: int,
    max_steps: int,
    *,
    timeout: float | None,
    render: bool = False,
    count: int | None = None,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]], Any]:
    """Run one eval episode under a hard per-instance wall-clock ``timeout``.

    The rollout runs in a forked worker (inheriting the live env and approach) so a
    hung or too-slow policy can be killed; on overrun it is scored unsolved
    (``metrics["timed_out"]``). A worker that dies before reporting (policy
    exception, OOM, native crash) re-raises here. ``timeout=None`` runs in-process.
    """
    if timeout is None:
        return run_episode(env, approach, seed, max_steps, render=render, count=count)
    ctx = mp.get_context("fork")  # fork: the worker inherits the live env + approach
    with ctx.Manager() as manager:
        result = manager.dict()
        outcome, exitcode = run_in_forked_worker(
            ctx,
            _timed_episode_worker,
            (env, approach, seed, max_steps, render, count, result),
            timeout,
        )
        if outcome == "timeout":
            metrics: dict[str, Any] = {
                "total_reward": 0.0,
                "num_steps": 0,
                "solved": False,
                "timed_out": True,
            }
            if count is not None:
                metrics["object_count"] = count
            return metrics, [], None
        if "metrics" in result:
            return result["metrics"], list(result["frames"]), result["final_state"]
        # The worker died before reporting; surface it so the caller scores a crash.
        raise RuntimeError(
            result.get(
                "error",
                f"eval episode worker (seed {seed}) exited with code "
                f"{exitcode} before reporting a result",
            )
        )


def _timed_episode_worker(
    env: Any,
    approach: BaseApproach,
    seed: int,
    max_steps: int,
    render: bool,
    count: int | None,
    result: Any,
) -> None:
    """Run one episode in a subprocess and stash its outcome in ``result``.

    The try/except deliberately carries a policy crash across the process boundary
    as ``result["error"]`` for the parent to re-raise. ``metrics`` is written last,
    so its presence means the run finished (a partial dict means the worker died).
    """
    try:
        metrics, frames, final_state = run_episode(
            env, approach, seed, max_steps, render=render, count=count
        )
        result["frames"] = frames
        result["final_state"] = final_state
        result["metrics"] = metrics
    except Exception:  # pylint: disable=broad-exception-caught
        result["error"] = traceback.format_exc()


def summarize_by_count(
    scheduled_counts: list[int], per_episode: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], int | None, int | None]:
    """Break results down by object count for a variable-count scaling curve.

    ``scheduled_counts[i]`` is the object count assigned to ``per_episode[i]`` (the two
    lists are parallel). The per-count ``solve_rate`` uses the **full scheduled**
    denominator: crashed, unattempted, and unsolved episodes all count as failures, so
    the curve does not flatter a policy that crashes on larger counts.

    Returns ``(by_count, largest_count_all_solved, largest_count_any_solved)`` where the
    two scalars are the largest count solved on every / at least one scheduled episode
    (``None`` if no count qualifies).
    """
    if len(scheduled_counts) != len(per_episode):
        raise ValueError(
            "scheduled_counts and per_episode must be parallel, got "
            f"{len(scheduled_counts)} and {len(per_episode)}; a length mismatch would "
            "silently drop episodes and break the full-scheduled denominator."
        )
    reserved = {"seed", "attempted", "crashed", "solved", "object_count"}
    buckets: dict[int, dict[str, Any]] = {}
    for count, episode in zip(scheduled_counts, per_episode):
        bucket = buckets.setdefault(
            count, {"n": 0, "n_solved": 0, "steps": [], "extras": {}}
        )
        bucket["n"] += 1
        if episode.get("solved"):
            bucket["n_solved"] += 1
        steps = episode.get("num_steps")
        if steps is not None:
            bucket["steps"].append(steps)
        for key, value in episode.items():
            if key in reserved or key in {"total_reward", "num_steps", "cost_usd"}:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bucket["extras"].setdefault(key, []).append(value)

    by_count: dict[int, dict[str, Any]] = {}
    for count in sorted(buckets):
        bucket = buckets[count]
        entry: dict[str, Any] = {
            "n": bucket["n"],
            "n_solved": bucket["n_solved"],
            "solve_rate": (
                bucket["n_solved"] / bucket["n"] if bucket["n"] else float("nan")
            ),
        }
        if bucket["steps"]:
            entry["mean_num_steps"] = float(np.mean(bucket["steps"]))
        for key, values in sorted(bucket["extras"].items()):
            entry[f"mean_{key}"] = float(np.mean(values))
        by_count[count] = entry

    solved_counts = [c for c, e in by_count.items() if e["solve_rate"] > 0]
    fully_solved = [c for c, e in by_count.items() if e["solve_rate"] >= 1.0]
    largest_all = max(fully_solved) if fully_solved else None
    largest_any = max(solved_counts) if solved_counts else None
    return by_count, largest_all, largest_any


def run_per_instance_eval(
    env: Any,
    approach: BaseApproach,
    eval_seeds: list[int],
    *,
    max_budget_usd: float,
    output_dir: Path,
    max_budget_per_instance_usd: float | None = None,
    render: bool = False,
    eval_counts: list[int] | None = None,
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
    if eval_counts is not None and len(eval_counts) != len(eval_seeds):
        raise ValueError(
            "eval_counts and eval_seeds must be parallel, got "
            f"{len(eval_counts)} and {len(eval_seeds)}; scheduled counts would "
            "otherwise misalign with evaluated seeds."
        )
    per_episode: list[dict[str, Any]] = []
    remaining = max_budget_usd
    total_cost = 0.0
    num_solved = 0
    num_attempted = 0
    for i, seed in enumerate(eval_seeds):
        # The scheduled count for this episode, attached to every per_episode entry
        # (unattempted and crashed included) so by-count reporting keeps the honest
        # full-scheduled denominator once the budget runs out; None for fixed-count.
        count_i = eval_counts[i] if eval_counts is not None else None
        count_field = {"object_count": count_i} if count_i is not None else {}
        if remaining <= 0:
            per_episode.append(
                {
                    **count_field,
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
            count=count_i,
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
                # below take precedence over any colliding extras key. The scheduled
                # count is set explicitly so a crashed attempt (empty extras) still
                # carries it, matching what a solved attempt reports.
                **result.extras,
                **count_field,
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

    if eval_counts is not None:
        by_count, largest_all, largest_any = summarize_by_count(
            eval_counts, per_episode
        )
        results["by_count"] = by_count
        results["largest_count_all_solved"] = largest_all
        results["largest_count_any_solved"] = largest_any
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
