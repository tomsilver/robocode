"""Run a generated ``GeneratedApproach`` on training tasks and classify failures.

Used by the LLM-GenPlan debug loop. Each episode runs in a ``multiprocessing``
worker so an infinite-looping ``get_action`` can be killed by timeout. When the
loop runs inside the sandbox container (``container_backend`` docker/apptainer),
this runs there too.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Callable
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any, NamedTuple

import gymnasium
from gymnasium.spaces import Space

from robocode.approaches.base_approach import BaseApproach
from robocode.utils.episode import (
    load_generated_approach,
    run_episode,
    run_in_forked_worker,
)


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
        return {
            "solved": False,
            "total_reward": 0.0,
            "num_steps": 0,
            "error_type": "timeout",
            "feedback": (
                f"On the task with seed {seed}, get_action did not finish within "
                f"{timeout:g}s. The code likely has an infinite loop or is far too "
                "slow."
            ),
        }
    if "solved" not in result:
        # The worker died before reporting (OOM kill, segfault in native code,
        # os._exit, ...), so there is no traceback to forward.
        return {
            "solved": False,
            "total_reward": 0.0,
            "num_steps": 0,
            "error_type": "worker-crashed",
            "feedback": (
                f"On the task with seed {seed}, the episode worker died with "
                f"exit code {exitcode} before reporting a result (e.g. out "
                "of memory or a crash in native code). Make the code terminate "
                "normally and reduce memory use."
            ),
        }
    return dict(result)


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
