"""Run a generated ``GeneratedApproach`` on training tasks and classify failures.

Used by the LLM-GenPlan debug loop. Each episode runs in a ``multiprocessing``
worker so an infinite-looping ``get_action`` can be killed by timeout. When the
loop runs inside the sandbox container (``container_backend`` docker/apptainer),
this runs there too.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Callable
from multiprocessing.managers import SyncManager
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, NamedTuple

import gymnasium
import hydra
from gymnasium.spaces import Space
from omegaconf import OmegaConf

from robocode.approaches.base_approach import BaseApproach
from robocode.primitives import build_primitives
from robocode.utils.episode import (
    load_generated_approach,
    run_episode,
    run_episode_with_timeout,
    run_in_forked_worker,
)

logger = logging.getLogger(__name__)


def render_state(observation_space: Space[Any], obs: Any) -> str:
    """Human-readable view of an observation (object-centric if possible)."""
    devectorize = getattr(observation_space, "devectorize", None)
    if devectorize is not None:
        return str(devectorize(obs))
    return repr(obs)


class _InvalidActionError(Exception):
    """The generated policy returned an action outside the action space."""

    def __init__(self, step: int, action: Any) -> None:
        super().__init__(f"invalid action at step {step}: {action!r}")
        self.step = step
        self.action = action


class _ValidationPolicy(BaseApproach[Any, Any]):
    """Adapt a raw ``GeneratedApproach`` to ``run_episode``, checking actions.

    Mirrors the delegation in ``LLMGenPlanApproach`` so validation runs the
    generated code exactly as it runs at eval, through the same rollout loop.
    """

    def __init__(
        self,
        generated: Any,
        action_space: Space[Any],
        observation_space: Space[Any],
        primitives: dict[str, Callable[..., Any]],
    ) -> None:
        super().__init__(action_space, observation_space, seed=0, primitives=primitives)
        self._generated = generated
        self._num_steps = 0

    def reset(self, state: Any, info: dict[str, Any]) -> None:
        super().reset(state, info)
        self._num_steps = 0
        self._generated.reset(state, info)

    def update(
        self, state: Any, reward: float, done: bool, info: dict[str, Any]
    ) -> None:
        super().update(state, reward, done, info)
        self._num_steps += 1
        if hasattr(self._generated, "update"):
            self._generated.update(state, reward, done, info)

    def _get_action(self) -> Any:
        action = self._generated.get_action(self._last_state)
        if not self._action_space.contains(action):
            raise _InvalidActionError(self._num_steps, action)
        return action


def validate_tasks(
    env: gymnasium.Env,
    approach_path: Path,
    action_space: Space[Any],
    observation_space: Space[Any],
    primitives: dict[str, Callable[..., Any]],
    seeds: list[int],
    max_steps: int,
    timeout: float,
) -> dict[str, str] | None:
    """Run the policy on each task; return the first failure, or None if all pass."""
    ctx = mp.get_context("fork")  # fork: workers inherit the live env
    with ctx.Manager() as manager:
        for seed in seeds:
            result = _validate_episode(
                env,
                approach_path,
                action_space,
                observation_space,
                primitives,
                seed,
                max_steps,
                timeout,
                ctx,
                manager,
            )
            if not result["solved"]:
                return {
                    "error_type": result["error_type"],
                    "feedback": result["feedback"],
                }
    return None


class TaskScore(NamedTuple):
    """Aggregate score of a policy over a set of tasks, for ranking candidates."""

    num_solved: int
    num_completed: int  # rollouts that finished without crashing / timing out
    num_total: int
    mean_reward: float  # mean reward over completed rollouts (0.0 if none)


class TaskEvaluation(NamedTuple):
    """First seed-ordered failure and aggregate score from one set of rollouts."""

    failure: dict[str, str] | None
    score: TaskScore | None


def evaluate_held_out_tasks_parallel(
    environment: dict[str, Any],
    approach_path: Path,
    primitive_names: list[str],
    seeds: list[int],
    max_steps: list[int],
    counts: list[int | None],
    timeout: float,
    max_workers: int,
) -> list[dict[str, Any]]:
    """Evaluate a saved generated policy on independent tasks concurrently.

    Each spawned worker owns an independently constructed environment and policy,
    reusing them across its episodes just as serial evaluation does. Results are
    returned in the caller's seed order, regardless of worker completion order.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if len(seeds) != len(max_steps) or len(seeds) != len(counts):
        raise ValueError("seeds, max_steps, and counts must have equal lengths")

    ctx = mp.get_context("spawn")
    results: list[dict[str, Any] | None] = [None] * len(seeds)
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = min(max_workers, len(seeds))
    workers = [
        ctx.Process(
            target=_held_out_worker_loop,
            args=(
                environment,
                approach_path,
                primitive_names,
                timeout,
                task_queue,
                result_queue,
            ),
        )
        for _ in range(num_workers)
    ]
    for worker in workers:
        worker.start()
    for index, seed in enumerate(seeds):
        task_queue.put((index, seed, max_steps[index], counts[index]))
    for _ in workers:
        task_queue.put(None)

    completed = 0
    while completed < len(seeds) and any(worker.is_alive() for worker in workers):
        try:
            index, result = result_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        results[index] = result
        completed += 1
        logger.info(
            "Held-out episode %d/%d complete: solved=%s, steps=%s",
            completed,
            len(seeds),
            result["solved"],
            result["num_steps"],
        )

    for worker in workers:
        worker.join()
    while True:
        try:
            index, result = result_queue.get_nowait()
        except queue.Empty:
            break
        if results[index] is None:
            results[index] = result

    exitcodes = [worker.exitcode for worker in workers if worker.exitcode]
    for index, result in enumerate(results):
        if result is not None:
            continue
        results[index] = {
            "total_reward": None,
            "num_steps": None,
            "solved": False,
            "crashed": True,
            "error": (
                "evaluation worker exited before reporting; " f"exit codes {exitcodes}"
            ),
            **({"object_count": counts[index]} if counts[index] is not None else {}),
        }

    ordered_results = [result for result in results if result is not None]
    assert len(ordered_results) == len(seeds)
    return ordered_results


def _held_out_worker_loop(
    environment: dict[str, Any],
    approach_path: Path,
    primitive_names: list[str],
    timeout: float,
    tasks: Any,
    results: Any,
) -> None:
    """Construct one isolated evaluator and consume held-out episode tasks."""
    env = hydra.utils.instantiate(OmegaConf.create(environment))
    try:
        primitives = build_primitives(env, primitive_names)
        generated = load_generated_approach(
            approach_path, env.action_space, env.observation_space, primitives
        )
        policy = _ValidationPolicy(
            generated, env.action_space, env.observation_space, primitives
        )
        while (task := tasks.get()) is not None:
            index, seed, max_steps, count = task
            try:
                metrics, _, _ = run_episode_with_timeout(
                    env,
                    policy,
                    seed,
                    max_steps,
                    timeout=timeout,
                    count=count,
                )
                results.put((index, metrics))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                results.put(
                    (
                        index,
                        {
                            "total_reward": None,
                            "num_steps": None,
                            "solved": False,
                            "crashed": True,
                            "error": f"{type(exc).__name__}: {exc}",
                            **({"object_count": count} if count is not None else {}),
                        },
                    )
                )
    finally:
        env.close()


def evaluate_tasks(
    env: gymnasium.Env,
    approach_path: Path,
    action_space: Space[Any],
    observation_space: Space[Any],
    primitives: dict[str, Callable[..., Any]],
    seeds: list[int],
    max_steps: int,
    timeout: float,
) -> TaskEvaluation:
    """Evaluate every task once, returning feedback and a score from those outcomes.

    Feedback remains the first failure in the caller's seed order. A policy-load
    failure is candidate-wide, so no additional seeds are run and no score is
    reported, matching the previous validation-then-scoring behavior.
    """
    ctx = mp.get_context("fork")  # fork: workers inherit the live env
    results: list[dict[str, Any]] = []
    with ctx.Manager() as manager:
        for seed in seeds:
            result = _validate_episode(
                env,
                approach_path,
                action_space,
                observation_space,
                primitives,
                seed,
                max_steps,
                timeout,
                ctx,
                manager,
            )
            results.append(result)
            if result.get("error_type") == "policy-load-error":
                return _summarize_results(results)
    return _summarize_results(results)


def evaluate_tasks_parallel(
    environment: dict[str, Any],
    approach_path: Path,
    primitive_names: list[str],
    seeds: list[int],
    max_steps: int,
    timeout: float,
    max_workers: int,
) -> TaskEvaluation:
    """Evaluate tasks concurrently in isolated, freshly constructed environments.

    ``spawn`` is required because Kindergarden's MuJoCo contexts are not fork-safe.
    Each process constructs and closes its own environment. Results are returned in
    the caller's seed order so scheduling cannot change feedback or scoring.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any] | None] = [None] * len(seeds)
    pending = iter(enumerate(seeds))
    active: dict[int, tuple[BaseProcess, Any, int, float | None]] = {}
    with ctx.Manager() as manager:
        shared_results = manager.dict()

        def start_next() -> bool:
            try:
                index, seed = next(pending)
            except StopIteration:
                return False
            ready = ctx.Event()
            process = ctx.Process(
                target=_isolated_episode_worker,
                args=(
                    environment,
                    approach_path,
                    primitive_names,
                    seed,
                    max_steps,
                    ready,
                    shared_results,
                    index,
                ),
            )
            process.start()
            active[index] = (process, ready, seed, None)
            return True

        while len(active) < min(max_workers, len(seeds)) and start_next():
            pass

        while active:
            now = time.monotonic()
            for index, (process, ready, seed, deadline) in list(active.items()):
                if deadline is None and ready.is_set():
                    deadline = now + timeout
                    active[index] = (process, ready, seed, deadline)
                if not process.is_alive():
                    process.join()
                    results[index] = shared_results.get(
                        index, _worker_crashed_result(seed, process.exitcode)
                    )
                    del active[index]
                    start_next()
                elif deadline is not None and now >= deadline:
                    process.terminate()
                    process.join()
                    results[index] = _timeout_result(seed, timeout)
                    del active[index]
                    start_next()
            if active:
                time.sleep(0.01)

    ordered_results = [result for result in results if result is not None]
    assert len(ordered_results) == len(seeds)
    return _summarize_results(ordered_results)


def _isolated_episode_worker(
    environment: dict[str, Any],
    approach_path: Path,
    primitive_names: list[str],
    seed: int,
    max_steps: int,
    ready: Any,
    results: Any,
    index: int,
) -> None:
    """Construct one environment, run one scoring episode, and close it."""
    env = hydra.utils.instantiate(OmegaConf.create(environment))
    try:
        primitives = build_primitives(env, primitive_names)
        ready.set()
        results[index] = _classify_episode(
            env,
            approach_path,
            env.action_space,
            env.observation_space,
            primitives,
            seed,
            max_steps,
        )
    finally:
        env.close()


def _summarize_results(results: list[dict[str, Any]]) -> TaskEvaluation:
    """Produce seed-ordered feedback and aggregate scoring from task outcomes."""
    first_failure: dict[str, str] | None = None
    completed_rewards: list[float] = []
    for result in results:
        if not result["solved"] and first_failure is None:
            first_failure = {
                "error_type": result["error_type"],
                "feedback": result["feedback"],
            }
        if result["solved"] or result.get("error_type") == "not-solved":
            completed_rewards.append(float(result["total_reward"]))
    if first_failure is not None and first_failure["error_type"] == "policy-load-error":
        return TaskEvaluation(first_failure, None)
    mean_reward = (
        sum(completed_rewards) / len(completed_rewards) if completed_rewards else 0.0
    )
    return TaskEvaluation(
        first_failure,
        TaskScore(
            sum(int(result["solved"]) for result in results),
            len(completed_rewards),
            len(results),
            mean_reward,
        ),
    )


def score_tasks(
    env: gymnasium.Env,
    approach_path: Path,
    action_space: Space[Any],
    observation_space: Space[Any],
    primitives: dict[str, Callable[..., Any]],
    seeds: list[int],
    max_steps: int,
    timeout: float,
) -> TaskScore:
    """Run the policy on every task and aggregate the outcomes into a ``TaskScore``.

    Unlike :func:`validate_tasks` (which stops at the first failure to produce
    debug feedback), this runs all seeds so callers can rank partially-successful
    policies. A rollout is *completed* if it solved or ran to the step limit;
    invalid actions, exceptions, timeouts, and crashes are not. Ranking on
    ``(num_solved, num_completed, mean_reward)`` prefers a policy that solves
    more, then one that runs without crashing, before comparing reward, so a
    crashing policy never outranks a runnable unsolved one. ``mean_reward`` is
    averaged over completed rollouts only (crashes have no meaningful reward).
    """
    ctx = mp.get_context("fork")  # fork: workers inherit the live env
    num_solved = 0
    completed_rewards: list[float] = []
    with ctx.Manager() as manager:
        for seed in seeds:
            result = _validate_episode(
                env,
                approach_path,
                action_space,
                observation_space,
                primitives,
                seed,
                max_steps,
                timeout,
                ctx,
                manager,
            )
            num_solved += int(result["solved"])
            if result["solved"] or result.get("error_type") == "not-solved":
                completed_rewards.append(float(result["total_reward"]))
    mean_reward = (
        sum(completed_rewards) / len(completed_rewards) if completed_rewards else 0.0
    )
    return TaskScore(num_solved, len(completed_rewards), len(seeds), mean_reward)


def _validate_episode(
    env: gymnasium.Env,
    approach_path: Path,
    action_space: Space[Any],
    observation_space: Space[Any],
    primitives: dict[str, Callable[..., Any]],
    seed: int,
    max_steps: int,
    timeout: float,
    ctx: mp.context.ForkContext,
    manager: SyncManager,
) -> dict[str, Any]:
    """Run one episode (timeout-guarded) and return its classified result.

    Always returns a dict with ``solved`` / ``total_reward`` / ``num_steps``; on
    failure it also carries ``error_type`` / ``feedback``. Consumed by both
    :func:`validate_tasks` and :func:`score_tasks`.
    """
    result = manager.dict()
    outcome, exitcode = run_in_forked_worker(
        ctx,
        _episode_worker,
        (
            env,
            approach_path,
            action_space,
            observation_space,
            primitives,
            seed,
            max_steps,
            result,
        ),
        timeout,
    )
    if outcome == "timeout":
        return _timeout_result(seed, timeout)
    if "solved" not in result:
        # The worker died before reporting (OOM kill, segfault in native code,
        # os._exit, ...), so there is no traceback to forward.
        return _worker_crashed_result(seed, exitcode)
    return dict(result)


def _timeout_result(seed: int, timeout: float) -> dict[str, Any]:
    """Classify an episode process that exceeded its wall-clock budget."""
    return {
        "solved": False,
        "total_reward": 0.0,
        "num_steps": 0,
        "error_type": "timeout",
        "feedback": (
            f"On the task with seed {seed}, get_action did not finish within "
            f"{timeout:g}s. The code likely has an infinite loop or is far too slow."
        ),
    }


def _worker_crashed_result(seed: int, exitcode: int | None) -> dict[str, Any]:
    """Classify an episode process that exited without returning an outcome."""
    return {
        "solved": False,
        "total_reward": 0.0,
        "num_steps": 0,
        "error_type": "worker-crashed",
        "feedback": (
            f"On the task with seed {seed}, the episode worker died with exit code "
            f"{exitcode} before reporting a result (e.g. out of memory or a crash "
            "in native code). Make the code terminate normally and reduce memory use."
        ),
    }


def _episode_worker(
    env: gymnasium.Env,
    approach_path: Path,
    action_space: Space[Any],
    observation_space: Space[Any],
    primitives: dict[str, Callable[..., Any]],
    seed: int,
    max_steps: int,
    result: Any,
) -> None:
    """Run one episode in a subprocess and report the classified outcome.

    The single ``update`` keeps the shared dict all-or-nothing, so the parent
    can treat a partial result as a crashed worker.
    """
    result.update(
        _classify_episode(
            env,
            approach_path,
            action_space,
            observation_space,
            primitives,
            seed,
            max_steps,
        )
    )


def _classify_episode(
    env: gymnasium.Env,
    approach_path: Path,
    action_space: Space[Any],
    observation_space: Space[Any],
    primitives: dict[str, Callable[..., Any]],
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Roll out one episode via ``run_episode`` and classify the outcome.

    The try/except is deliberate: a crash in generated code becomes LLM feedback.
    """
    try:
        generated = load_generated_approach(
            approach_path, action_space, observation_space, primitives
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return {
            "solved": False,
            "total_reward": 0.0,
            "num_steps": 0,
            "error_type": "policy-load-error",
            "feedback": (
                f"On the task with seed {seed}, the submitted module could not be "
                "loaded as a GeneratedApproach policy:\n" + traceback.format_exc()
            ),
        }
    try:
        policy = _ValidationPolicy(
            generated, action_space, observation_space, primitives
        )
        metrics, _, final_state = run_episode(env, policy, seed, max_steps)
        if metrics["solved"]:
            return {
                "solved": True,
                "total_reward": metrics["total_reward"],
                "num_steps": metrics["num_steps"],
            }
        return {
            "solved": False,
            "total_reward": metrics["total_reward"],
            "num_steps": metrics["num_steps"],
            "error_type": "not-solved",
            "feedback": (
                f"On the task with seed {seed}, the policy ran for "
                f"{metrics['num_steps']} steps without reaching the goal (total "
                f"reward {metrics['total_reward']:g}). The final state was:\n"
                f"{render_state(observation_space, final_state)}\n"
                "Compare it to the goal/termination condition in the source."
            ),
        }
    except _InvalidActionError as e:
        return {
            "solved": False,
            "total_reward": 0.0,
            "num_steps": e.step,
            "error_type": "invalid-action",
            "feedback": (
                f"On the task with seed {seed}, at step {e.step} get_action "
                f"returned {e.action!r}, which is not a valid action. Valid "
                f"actions must lie in the action space: {action_space}."
            ),
        }
    except Exception:  # pylint: disable=broad-exception-caught
        return {
            "solved": False,
            "total_reward": 0.0,
            "num_steps": 0,
            "error_type": "python-exception",
            "feedback": (
                f"On the task with seed {seed}, the code raised an exception:\n"
                + traceback.format_exc()
            ),
        }
