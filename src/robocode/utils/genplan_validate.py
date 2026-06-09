"""Run a generated ``GeneratedApproach`` on training tasks and classify failures.

Used by the LLM-GenPlan debug loop. Each episode runs in a ``multiprocessing``
worker so an infinite-looping ``get_action`` can be killed by timeout. When the
loop runs inside the sandbox container (``use_docker``), this runs there too.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium
from gymnasium.spaces import Space

from robocode.utils.episode import load_generated_approach


def render_state(observation_space: Space[Any], obs: Any) -> str:
    """Human-readable view of an observation (object-centric if possible)."""
    devectorize = getattr(observation_space, "devectorize", None)
    if devectorize is not None:
        return str(devectorize(obs))
    return repr(obs)


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
    for seed in seeds:
        failure = _validate_episode(
            env,
            approach_path,
            action_space,
            observation_space,
            primitives,
            seed,
            max_steps,
            timeout,
        )
        if failure is not None:
            return failure
    return None


def _validate_episode(
    env: gymnasium.Env,
    approach_path: Path,
    action_space: Space[Any],
    observation_space: Space[Any],
    primitives: dict[str, Callable[..., Any]],
    seed: int,
    max_steps: int,
    timeout: float,
) -> dict[str, str] | None:
    ctx = mp.get_context("fork")  # fork: worker inherits the live env
    result = ctx.Manager().dict()
    proc = ctx.Process(
        target=_episode_worker,
        args=(
            env,
            approach_path,
            action_space,
            observation_space,
            primitives,
            seed,
            max_steps,
            result,
        ),
    )
    proc.start()
    assert proc.pid is not None
    proc.join(timeout)
    if proc.is_alive():
        os.kill(proc.pid, signal.SIGINT)
        proc.join(3)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "error_type": "timeout",
            "feedback": (
                f"On the task with seed {seed}, get_action did not finish within "
                f"{timeout:g}s. The code likely has an infinite loop or is far too "
                "slow."
            ),
        }
    if result.get("solved"):
        return None
    return {"error_type": result["error_type"], "feedback": result["feedback"]}


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
    """Run one episode in a subprocess and classify the outcome.

    The try/except is deliberate: a crash in generated code becomes LLM feedback.
    """
    try:
        approach = load_generated_approach(
            approach_path, action_space, observation_space, primitives
        )
        state, info = env.reset(seed=seed)
        approach.reset(state, info)
        total_reward = 0.0
        for step in range(max_steps):
            action = approach.get_action(state)
            if not action_space.contains(action):
                result["solved"] = False
                result["error_type"] = "invalid-action"
                result["feedback"] = (
                    f"On the task with seed {seed}, at step {step} get_action "
                    f"returned {action!r}, which is not a valid action. Valid "
                    f"actions must lie in the action space: {action_space}."
                )
                return
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if hasattr(approach, "update"):
                approach.update(state, float(reward), terminated or truncated, info)
            if terminated:
                result["solved"] = True
                return
            if truncated:
                break
        result["solved"] = False
        result["error_type"] = "not-solved"
        result["feedback"] = (
            f"On the task with seed {seed}, the policy ran for {max_steps} steps "
            f"without reaching the goal (total reward {total_reward:g}). The final "
            f"state was:\n{render_state(observation_space, state)}\n"
            "Compare it to the goal/termination condition in the source."
        )
    except Exception:  # pylint: disable=broad-exception-caught
        result["solved"] = False
        result["error_type"] = "python-exception"
        result["feedback"] = (
            f"On the task with seed {seed}, the code raised an exception:\n"
            + traceback.format_exc()
        )
