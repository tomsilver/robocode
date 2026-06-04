"""LLM-GenPlan baseline (Silver et al. 2023, arXiv:2305.11014).

A non-agentic baseline: a plain LLM is shown the environment source, asked to
summarize it, propose a strategy, and implement a ``GeneratedApproach`` policy.
The *framework* (not the model) then runs that policy on training tasks and, on
failure, re-prompts the model with structured error feedback (automated
debugging). Contrast with ``AgenticApproach``, where an autonomous coding agent
writes and self-tests the policy with tools.
"""

from __future__ import annotations

import inspect
import logging
import multiprocessing as mp
import os
import signal
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import gymnasium
from gymnasium.spaces import Space
from omegaconf import DictConfig

from robocode.approaches.base_approach import BaseApproach
from robocode.utils.episode import load_generated_approach
from robocode.utils.llm import LLMResponse, create_llm_client
from robocode.utils.source_deps import collect_local_deps

logger = logging.getLogger(__name__)

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")

_INTERFACE_SPEC = """\
Implement the strategy as a Python class named `GeneratedApproach` in a single \
code block:

```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        \"\"\"action_space and observation_space are the gym spaces above.\"\"\"
        ...

    def reset(self, state, info):
        \"\"\"Called at the start of each episode with the initial observation.\"\"\"
        ...

    def get_action(self, state):
        \"\"\"Return a valid action (matching action_space) for this state.\"\"\"
        ...
```

`state` is a numpy observation matching the observation space. `get_action` is \
called every step and must return an action inside the action space. The class \
may keep internal state between calls (e.g. a precomputed plan). Return ONLY \
the code block; do not write tests or explanations."""

_SUMMARY_PROMPT = "Write a short summary of this environment in words."

_STRATEGY_PROMPT = (
    "There is a simple strategy for solving all instances of this environment "
    "without using search or learning. What is that strategy? Describe it in "
    "words; do not write code yet."
)


class LLMGenPlanApproach(BaseApproach[_ObsType, _ActType]):
    """Generate a policy with a plain LLM and debug it against training tasks."""

    def __init__(
        self,
        action_space: Space[_ActType],
        observation_space: Space[_ObsType],
        seed: int,
        primitives: dict[str, Callable[..., Any]],
        completion: DictConfig,
        env: gymnasium.Env | None = None,
        env_description_path: str | None = None,
        output_dir: str = ".",
        load_dir: str | None = None,
        max_steps: int = 100,
        num_train_tasks: int = 10,
        num_prompt_tasks: int = 2,
        max_debug_attempts: int = 4,
        skip_chain_of_thought: bool = False,
        episode_timeout_s: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_space, observation_space, seed, primitives, env_description_path
        )
        self._client = create_llm_client(completion)
        self._env = env
        self._output_dir = Path(output_dir)
        self._load_dir = Path(load_dir) if load_dir is not None else None
        self._max_steps = max_steps
        self._num_train_tasks = num_train_tasks
        self._num_prompt_tasks = num_prompt_tasks
        self._max_debug_attempts = max_debug_attempts
        self._skip_chain_of_thought = skip_chain_of_thought
        self._episode_timeout_s = episode_timeout_s
        self._generated: Any = None
        self.total_cost_usd: float | None = None

    def train(self) -> None:
        if self._load_dir is not None:
            self._load_generated(self._load_dir / "sandbox" / "approach.py")
            return

        assert self._env is not None, "LLMGenPlanApproach needs the env during train()"
        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        approach_path = sandbox_dir / "approach.py"

        train_seeds = list(range(self._num_train_tasks))
        context = self._build_context(train_seeds[: self._num_prompt_tasks])

        messages: list[dict[str, str]] = []
        if self._skip_chain_of_thought:
            messages.append(
                {"role": "user", "content": f"{context}\n\n{_INTERFACE_SPEC}"}
            )
        else:
            self._exchange(messages, f"{context}\n\n{_SUMMARY_PROMPT}", sandbox_dir, 0)
            self._exchange(messages, _STRATEGY_PROMPT, sandbox_dir, 1)
            messages.append({"role": "user", "content": _INTERFACE_SPEC})

        for t in range(self._max_debug_attempts + 1):
            response = self._complete(messages, sandbox_dir, f"impl{t}")
            messages.append({"role": "assistant", "content": response})
            # Each response is a complete GeneratedApproach class, so the latest
            # one replaces the previous. Do NOT accumulate: appending would stack
            # several class definitions in approach.py (only the last would run).
            approach_path.write_text(_parse_python_code(response))

            failure = self._validate(approach_path, train_seeds)
            if failure is None:
                logger.info(
                    "All %d training tasks solved at attempt %d", len(train_seeds), t
                )
                break
            logger.info("Attempt %d failed (%s)", t, failure["error_type"])
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{failure['feedback']}\nFix the code. Return the complete, "
                        "corrected GeneratedApproach class as a single code block."
                    ),
                }
            )

        self._load_generated(approach_path)

    # ------------------------------------------------------------------ prompt

    def _build_context(self, prompt_seeds: list[int]) -> str:
        """Env description + source + a couple of example initial states."""
        assert self._env is not None
        parts = ["You are writing a policy for the environment below.\n"]
        if self._env_description_path is not None:
            parts.append(Path(self._env_description_path).read_text(encoding="utf-8"))
        parts.append("## Full source code\n\n" + _gather_env_source(self._env))
        parts.append(
            "## Example initial states\n\n"
            "Each instance below is one task (env.reset(seed=...)). Only the "
            "initial state varies across instances; the GOAL is fixed by the "
            "environment (see the Reward/termination logic in the source above) "
            "and any goal-relevant features are part of the observation itself."
        )
        for s in prompt_seeds:
            obs, _ = self._env.reset(seed=s)
            parts.append(f"### seed {s}\n\n{_render_state(self._state_space, obs)}")
        return "\n\n".join(parts)

    def _exchange(
        self, messages: list[dict[str, str]], prompt: str, sandbox: Path, idx: int
    ) -> None:
        messages.append({"role": "user", "content": prompt})
        response = self._complete(messages, sandbox, f"cot{idx}")
        messages.append({"role": "assistant", "content": response})

    def _complete(self, messages: list[dict[str, str]], sandbox: Path, tag: str) -> str:
        result: LLMResponse = self._client.complete(messages)
        if result.cost_usd is not None:
            self.total_cost_usd = (self.total_cost_usd or 0.0) + result.cost_usd
        (sandbox / f"{tag}_prompt.txt").write_text(messages[-1]["content"])
        (sandbox / f"{tag}_response.txt").write_text(result.text)
        return result.text

    # -------------------------------------------------------------- validation

    def _validate(self, approach_path: Path, seeds: list[int]) -> dict[str, str] | None:
        """Run the policy on each training task; return the first failure."""
        for seed in seeds:
            failure = self._validate_one(approach_path, seed)
            if failure is not None:
                return failure
        return None

    def _validate_one(self, approach_path: Path, seed: int) -> dict[str, str] | None:
        ctx = mp.get_context("fork")  # fork: worker inherits the live env
        result = ctx.Manager().dict()
        proc = ctx.Process(
            target=_run_episode_worker,
            args=(
                self._env,
                approach_path,
                self._action_space,
                self._state_space,
                self._primitives,
                seed,
                self._max_steps,
                result,
            ),
        )
        proc.start()
        proc.join(self._episode_timeout_s)
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGINT)  # type: ignore[arg-type]
            proc.join(3)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return {
                "error_type": "timeout",
                "feedback": (
                    f"On the task with seed {seed}, get_action did not finish "
                    f"within {self._episode_timeout_s:g}s. The code likely has an "
                    "infinite loop or is far too slow."
                ),
            }
        if result.get("solved"):
            return None
        return {"error_type": result["error_type"], "feedback": result["feedback"]}

    # ------------------------------------------------------------- delegation

    def _load_generated(self, path: Path) -> None:
        self._generated = load_generated_approach(
            path, self._action_space, self._state_space, self._primitives
        )

    def reset(self, state: _ObsType, info: dict[str, Any]) -> None:
        super().reset(state, info)
        if self._generated is not None:
            self._generated.reset(state, info)

    def update(
        self, state: _ObsType, reward: float, done: bool, info: dict[str, Any]
    ) -> None:
        super().update(state, reward, done, info)
        if self._generated is not None and hasattr(self._generated, "update"):
            self._generated.update(state, reward, done, info)

    def _get_action(self) -> _ActType:
        if self._generated is not None:
            action: _ActType = self._generated.get_action(self._last_state)
            return action
        return self._action_space.sample()


def _run_episode_worker(
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

    The single try/except here is deliberate: it turns a crash in generated code into
    structured LLM feedback (the point of the debugging loop).
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
            f"state was:\n{_render_state(observation_space, state)}\n"
            "Compare it to the goal/termination condition in the source."
        )
    except Exception:  # pylint: disable=broad-exception-caught
        result["solved"] = False
        result["error_type"] = "python-exception"
        result["feedback"] = (
            f"On the task with seed {seed}, the code raised an exception:\n"
            + traceback.format_exc()
        )


def _gather_env_source(env: gymnasium.Env) -> str:
    """Best-effort bundle of the env's local source (robocode + kinder)."""
    files: list[Path] = []
    for obj, package in _source_targets(env):
        src = Path(inspect.getsourcefile(type(obj)))  # type: ignore[arg-type]
        root = src.parents[len(type(obj).__module__.split(".")) - 1]
        files.extend(collect_local_deps(src, root, package))
    seen: set[Path] = set()
    blocks: list[str] = []
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        blocks.append(f"### {path.name}\n```python\n{path.read_text()}\n```")
    return "\n\n".join(blocks)


def _source_targets(env: gymnasium.Env) -> list[tuple[Any, str]]:
    """The objects whose source to bundle, with their top-level package."""
    targets: list[tuple[Any, str]] = [(env, type(env).__module__.split(".")[0])]
    underlying = getattr(env, "_kinder_env", None)
    if underlying is not None:
        targets.append((underlying, type(underlying).__module__.split(".")[0]))
    return targets


def _render_state(observation_space: Space[Any], obs: Any) -> str:
    """Human-readable view of an observation (object-centric if possible)."""
    if hasattr(observation_space, "devectorize"):
        return str(observation_space.devectorize(obs))  # type: ignore[attr-defined]
    return repr(obs)


def _parse_python_code(response: str) -> str:
    """Extract the first ```python fenced block, else the raw response."""
    marker = "```python"
    if marker in response:
        rest = response[response.index(marker) + len(marker) :]
        end = rest.index("```") if "```" in rest else len(rest)
        return rest[:end].strip()
    return response.strip()
