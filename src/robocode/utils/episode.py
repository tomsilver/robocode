"""Episode-running, video-saving, and approach-loading utilities."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterator

import imageio.v3 as iio
import imageio_ffmpeg
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


# A frozen GeneratedApproach is scored through reset()/get_action() only; referencing
# set_state or sample_next_state mutates the scored env and can fake a solve. Match
# ".name" with an identifier boundary so get_state, reset_state, and set_stateful pass.
_FORBIDDEN_STATE_MUTATIONS = ("set_state", "sample_next_state")


def _reject_state_mutation(source: str) -> None:
    """Reject a generated approach that mutates the scored env (anti-cheat)."""
    hits = [n for n in _FORBIDDEN_STATE_MUTATIONS if re.search(rf"\.{n}\b", source)]
    if hits:
        raise ValueError(
            f"Generated approach references {', '.join(hits)}; approach.py is scored "
            "through reset()/get_action() only and must reach the goal via the actions "
            "it returns, not by mutating the environment's state."
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
        _reject_state_mutation(source)
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
    progress_callback: Callable[[int, int], None] | None = None,
    frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]], Any]:
    """Run a single evaluation episode; return metrics, frames, final state.

    ``count`` pins the object count for a variable-count env (via
    ``reset(options={"object_count": count})``); ``None`` leaves the env to sample.
    ``frame_sink`` receives each rendered frame as it is captured and the returned
    frame list stays empty; long episodes then need only one frame in memory.
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
            if frame_sink is not None:
                frame_sink(rendered)
            else:
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
        if progress_callback is not None:
            progress_callback(num_steps, max_steps)
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


# A forked child cannot touch MuJoCo on macOS. Constructing an env initializes
# CGL/Metal in this process, and Apple's GPU frameworks are undefined in a child that
# forks without exec'ing, so the child dies with SIGSEGV inside ``mjr_freeContext`` the
# moment it resets the env. That holds even when the parent never rendered and even
# when the child builds its own env, so there is nothing to hand over or tear down
# first: the rollout has to stay in this process. See
# integration_tests/check_eval_timeout.py.
_EPISODE_FORK_SAFE = sys.platform != "darwin"


class EpisodeTimeout(BaseException):
    """Raised in-process when an episode overruns its wall-clock budget.

    Deliberately a :class:`BaseException`, for the reason :class:`KeyboardInterrupt`
    is one: a generated policy that wraps its body in ``except Exception`` would
    otherwise swallow the timeout and run unbounded. That shape is common in
    generated code::

        while True:
            try:
                ...
            except Exception:
                continue

    The forked path is immune because it stops such a policy with SIGINT and then
    SIGKILL, so making this catchable would have quietly dropped the guarantee that
    path provides rather than matching it.
    """


def _timed_out_metrics(count: int | None) -> dict[str, Any]:
    """Score an episode stopped at its budget: unsolved, and flagged as timed out."""
    metrics: dict[str, Any] = {
        "total_reward": 0.0,
        "num_steps": 0,
        "solved": False,
        "timed_out": True,
    }
    if count is not None:
        metrics["object_count"] = count
    return metrics


@contextlib.contextmanager
def _sigalrm_guard(timeout: float) -> Iterator[None]:
    """Raise :class:`EpisodeTimeout` in this thread once *timeout* seconds pass.

    This is what catches a policy that hangs *inside* a single call, which a deadline
    checked between steps cannot see. ``setitimer`` is main-thread-only, so off the
    main thread this is a no-op and the per-step deadline is the only guard.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(_signum: int, _frame: Any) -> None:
        raise EpisodeTimeout

    previous = signal.signal(signal.SIGALRM, _fire)
    # Repeating, not one-shot. A policy that catches BaseException could still
    # swallow the first alarm; re-arming means it keeps being interrupted rather
    # than escaping the budget outright on a single catch.
    signal.setitimer(signal.ITIMER_REAL, timeout, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _run_episode_in_process(
    env: Any,
    approach: BaseApproach,
    seed: int,
    max_steps: int,
    *,
    timeout: float,
    render: bool,
    count: int | None,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]], Any]:
    """Run one episode here, aborting it if it overruns ``timeout``.

    The path for platforms where ``fork`` cannot carry the env (see
    :data:`_EPISODE_FORK_SAFE`). Scoring matches the forked path exactly: an overrun
    returns :func:`_timed_out_metrics` and a policy exception propagates for the
    caller to score as a crash.

    The one thing it cannot do is contain a *native* crash in the policy, which takes
    this process down instead of one worker. That is the deliberate trade: on macOS
    the alternative is not a safer rollout but no rollout at all.
    """
    deadline = time.monotonic() + timeout

    def _abort_if_overrun(_step: int, _total: int) -> None:
        if time.monotonic() > deadline:
            raise EpisodeTimeout

    try:
        with _sigalrm_guard(timeout):
            return run_episode(
                env,
                approach,
                seed,
                max_steps,
                render=render,
                count=count,
                progress_callback=_abort_if_overrun,
            )
    except EpisodeTimeout:
        return _timed_out_metrics(count), [], None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Re-raise as the forked path does -- a RuntimeError carrying the traceback --
        # so the error strings a caller records in results.json have the same shape on
        # every platform rather than the concrete type here and RuntimeError there.
        raise RuntimeError(traceback.format_exc()) from exc


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
    timeout: float,
    render: bool = False,
    count: int | None = None,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]], Any]:
    """Run one eval episode under a hard per-instance wall-clock ``timeout``.

    The rollout runs in a forked worker (inheriting the live env and approach) so a
    hung or too-slow policy can be killed; on overrun it is scored unsolved
    (``metrics["timed_out"]``). A worker that dies before reporting (policy
    exception, OOM, native crash) re-raises here.

    Where forking cannot carry the env, the rollout runs in this process instead and
    is bounded the same way; :func:`_run_episode_in_process` covers what that costs.
    """
    if not _EPISODE_FORK_SAFE or getattr(approach, "requires_in_process_eval", False):
        return _run_episode_in_process(
            env,
            approach,
            seed,
            max_steps,
            timeout=timeout,
            render=render,
            count=count,
        )

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
            return _timed_out_metrics(count), [], None
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


def summarize_count_regimes(
    by_count: dict[int, dict[str, Any]], design_counts: list[int]
) -> dict[str, dict[str, Any]]:
    """Aggregate count-sweep results into design and held-out regimes."""
    design = set(design_counts)
    regimes: dict[str, dict[str, Any]] = {}
    for name, counts in (
        ("design", [count for count in by_count if count in design]),
        ("held_out", [count for count in by_count if count not in design]),
    ):
        scheduled = sum(int(by_count[count]["n"]) for count in counts)
        solved = sum(int(by_count[count]["n_solved"]) for count in counts)
        regimes[name] = {
            "counts": sorted(counts),
            "n": scheduled,
            "n_solved": solved,
            "solve_rate": solved / scheduled if scheduled else float("nan"),
        }
    return regimes


def summarize_eval_episodes(per_episode: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate an eval suite's per-episode records into headline metrics.

    ``per_episode`` is parallel to the scheduled eval seeds: every scheduled episode
    contributes exactly one entry, whether it ran, crashed, or was never attempted.

    ``solve_rate`` uses the **full scheduled** denominator -- a crashed or unattempted
    episode is an unsolved one -- because the alternative silently rescales the headline
    to whatever survived, so a policy that crashes on the hard instances scores higher
    the more of them it kills. That is also the denominator
    :func:`summarize_by_count` already uses, so a run's headline and its scaling curve
    now agree instead of disagreeing exactly when something went wrong.

    Reward and step means stay over scored episodes only. There is no per-env worst-case
    return to charge a crash with, and inventing one would flatter a crashy policy in
    the other direction; ``eval_complete`` is what tells a reader the mean is partial.
    """
    scored = [
        e for e in per_episode if e.get("attempted", True) and not e.get("crashed")
    ]
    num_crashed = sum(1 for e in per_episode if e.get("crashed"))
    num_unattempted = sum(1 for e in per_episode if not e.get("attempted", True))
    num_eval = len(per_episode)

    def _mean(key: str) -> float:
        vals = [e[key] for e in scored if e.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "mean_eval_reward": _mean("total_reward"),
        "mean_eval_steps": _mean("num_steps"),
        "solve_rate": (
            sum(1 for e in per_episode if e.get("solved")) / num_eval
            if num_eval
            else float("nan")
        ),
        "num_eval_tasks": num_eval,
        "num_evaluated_episodes": len(scored),
        "num_crashed_episodes": num_crashed,
        # Every scheduled episode produced a real score. False means the means above
        # cover only part of the suite, which is otherwise indistinguishable from a
        # clean run once these numbers reach a summary table.
        "eval_complete": num_crashed == 0 and num_unattempted == 0,
    }


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
    results = {
        **summarize_eval_episodes(per_episode),
        "num_attempted": num_attempted,
        "num_solved": num_solved,
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


@contextlib.contextmanager
def open_video_writer(
    path: Path, fps: int = 10
) -> Iterator[Callable[[NDArray[np.uint8]], None]]:
    """Yield an ``append(frame)`` callable that streams RGB frames into a GIF.

    Frames are piped to an ffmpeg subprocess and encoded as they arrive, so peak
    memory stays at one frame regardless of episode length. (Pillow-based GIF
    writers buffer every frame until close, which can OOM on long episodes.)
    All frames must share the first frame's shape.
    """
    proc: subprocess.Popen[bytes] | None = None
    size: tuple[int, int] | None = None

    def _append(frame: NDArray[np.uint8]) -> None:
        nonlocal proc, size
        rgb = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        if proc is None:
            size = (rgb.shape[1], rgb.shape[0])
            # stats_mode=single + new=1 builds a palette per frame, keeping the
            # whole pipeline single-pass and streaming.
            proc = subprocess.Popen(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-v",
                    "error",
                    *("-f", "rawvideo", "-pix_fmt", "rgb24"),
                    *("-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-"),
                    *(
                        "-vf",
                        "split[a][b];[a]palettegen=stats_mode=single[p];"
                        "[b][p]paletteuse=new=1",
                    ),
                    # Explicit format: the target may be a temp name without
                    # a .gif extension.
                    *("-loop", "0", "-f", "gif"),
                    str(path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        if (rgb.shape[1], rgb.shape[0]) != size:
            raise ValueError(f"frame shape changed mid-video: {rgb.shape}")
        assert proc.stdin is not None
        try:
            proc.stdin.write(rgb.tobytes())
        except BrokenPipeError:
            _raise_encode_failure(proc)

    try:
        yield _append
        if proc is not None:
            assert proc.stdin is not None
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            if proc.wait() != 0:
                _raise_encode_failure(proc)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


def _raise_encode_failure(proc: subprocess.Popen[bytes]) -> None:
    """Kill a failed ffmpeg process and raise with its stderr."""
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    assert proc.stderr is not None
    err = proc.stderr.read().decode(errors="replace")
    raise RuntimeError(f"ffmpeg GIF encode failed: {err.strip()}")


def save_video(frames: list[NDArray[np.uint8]], path: Path, fps: int = 10) -> None:
    """Save a list of RGB frames as a gif, encoding one frame at a time."""
    if not frames:
        raise ValueError("no frames to save")
    with open_video_writer(path, fps=fps) as append_frame:
        for frame in frames:
            append_frame(frame)
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
