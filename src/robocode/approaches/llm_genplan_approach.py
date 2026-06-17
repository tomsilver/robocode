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
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import gymnasium
from gymnasium.spaces import Space
from omegaconf import DictConfig, OmegaConf

from robocode.approaches.base_approach import BaseApproach
from robocode.primitives import format_primitives_description
from robocode.utils.apptainer_sandbox import _DEFAULT_SIF, run_genplan_in_apptainer
from robocode.utils.docker_sandbox import run_genplan_in_docker
from robocode.utils.episode import load_generated_approach
from robocode.utils.genplan_validate import render_state, validate_tasks
from robocode.utils.llm import LLMClient, LLMResponse, create_llm_client
from robocode.utils.sandbox_types import resolve_container_backend
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
    "without using search. What is that strategy?"
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
        env_cfg: str | None = None,  # JSON env config, for the docker driver
        env_description_path: str | None = None,
        output_dir: str = ".",
        load_dir: str | None = None,
        max_steps: int = 100,
        num_train_tasks: int = 10,
        num_prompt_tasks: int = 2,
        max_debug_attempts: int | None = 4,
        max_budget_usd: float | None = 5.0,
        chain_of_thought: bool = True,
        episode_timeout_s: float = 30.0,
        use_docker: bool = True,
        container_backend: str | None = None,
        docker_image: str = "robocode-sandbox",
        sif_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_space, observation_space, seed, primitives, env_description_path
        )
        self._seed = seed
        self._completion_cfg = completion
        self._container_backend = resolve_container_backend(
            container_backend, use_docker
        )
        # Sandboxed runs build the client inside the container, so the host
        # needs no client/key.
        self._client: LLMClient | None = (
            create_llm_client(completion) if container_backend == "local" else None
        )
        self._env = env
        self._env_cfg = env_cfg
        self._output_dir = Path(output_dir)
        self._load_dir = Path(load_dir) if load_dir is not None else None
        self._max_steps = max_steps
        self._num_train_tasks = num_train_tasks
        self._num_prompt_tasks = num_prompt_tasks
        self._max_debug_attempts = max_debug_attempts
        self._max_budget_usd = max_budget_usd
        self._chain_of_thought = chain_of_thought
        self._episode_timeout_s = episode_timeout_s
        self._docker_image = docker_image
        self._sif_path = Path(sif_path) if sif_path is not None else _DEFAULT_SIF
        self._generated: Any = None
        self.total_cost_usd: float | None = None
        # Number of LLM generations made (debug attempts for genplan, candidates
        # for best-of-k); recorded into results.json.
        self.num_generations: int | None = None

    def train(self) -> None:
        if self._load_dir is not None:
            self._load_generated(self._load_dir / "sandbox" / "approach.py")
            return

        assert self._max_generations is not None or self._max_budget_usd is not None, (
            "set a loop bound: max_budget_usd and/or the generation-step cap "
            "(max_debug_attempts / max_generation_steps)"
        )

        # Sandboxed: run the whole loop inside one container (docker/apptainer)
        # via the genplan driver; the driver reruns train() locally inside.
        if self._container_backend in ("docker", "apptainer"):
            self._train_in_container()
            self._load_generated(self._output_dir / "sandbox" / "approach.py")
            return

        assert self._env is not None, "LLMGenPlanApproach needs the env during train()"
        assert self._client is not None
        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        approach_path = sandbox_dir / "approach.py"
        train_seeds = list(range(self._num_train_tasks))
        self._generate_local(approach_path, sandbox_dir, train_seeds)
        self._load_generated(approach_path)

    def _generate_local(
        self, approach_path: Path, sandbox_dir: Path, train_seeds: list[int]
    ) -> None:
        """Build the prompt and debug it against the training tasks (in-process)."""
        context = self._build_context(train_seeds[: self._num_prompt_tasks])
        messages = self._build_initial_messages(context, sandbox_dir)
        self._debug_loop(messages, approach_path, sandbox_dir, train_seeds)

    def _build_initial_messages(
        self, context: str, sandbox_dir: Path
    ) -> list[dict[str, str]]:
        """Initial message list: the strategy CoT, or a single no-CoT prompt.

        Best-of-K reuses this verbatim, calling it fresh per candidate so a CoT
        run resamples the summary/strategy exchanges independently.
        """
        messages: list[dict[str, str]] = []
        if not self._chain_of_thought:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\nThere is a simple strategy for solving "
                        "all instances of this environment without using "
                        f"search. {_INTERFACE_SPEC}"
                    ),
                }
            )
        else:
            self._exchange(messages, f"{context}\n\n{_SUMMARY_PROMPT}", sandbox_dir, 0)
            self._exchange(messages, _STRATEGY_PROMPT, sandbox_dir, 1)
            messages.append({"role": "user", "content": _INTERFACE_SPEC})
        return messages

    def _driver_config(self, completion: dict[str, Any]) -> dict[str, Any]:
        """Config the in-container driver reads to rebuild and run this approach.

        Subclasses override to set ``approach`` and add their own fields.
        """
        assert self._env_cfg is not None, "container runs need env_cfg"
        return {
            "approach": "genplan",
            "completion": completion,
            "environment": json.loads(self._env_cfg),
            "seed": self._seed,
            "primitive_names": list(self._primitives),
            "max_steps": self._max_steps,
            "num_train_tasks": self._num_train_tasks,
            "num_prompt_tasks": self._num_prompt_tasks,
            "max_debug_attempts": self._max_debug_attempts,
            "max_budget_usd": self._max_budget_usd,
            "chain_of_thought": self._chain_of_thought,
            "episode_timeout_s": self._episode_timeout_s,
        }

    def _train_in_container(self) -> None:
        """Write the driver config and run the loop in one sandbox container."""
        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        completion = cast(
            dict[str, Any], OmegaConf.to_container(self._completion_cfg, resolve=True)
        )
        config = self._driver_config(completion)
        (sandbox_dir / "genplan_config.json").write_text(json.dumps(config))
        if self._container_backend == "apptainer":
            run_genplan_in_apptainer(sandbox_dir, completion, sif_path=self._sif_path)
        else:
            run_genplan_in_docker(sandbox_dir, completion, image=self._docker_image)
        cost = json.loads((sandbox_dir / "cost.json").read_text(encoding="utf-8"))
        self.total_cost_usd = cost["total_cost_usd"]
        self.num_generations = cost.get("num_generations")

    # ---------------------------------------------------------------- budgeting

    @property
    def _max_generations(self) -> int | None:
        """Total generations the step cap allows, or None for no step cap.

        GenPlan does one initial attempt plus ``max_debug_attempts`` debug
        attempts; Best-of-K overrides this with its own candidate count.
        """
        if self._max_debug_attempts is None:
            return None
        return self._max_debug_attempts + 1

    def _within_budget(self, steps: int) -> bool:
        """Whether another generation is allowed by the step cap and dollar budget."""
        under_steps = self._max_generations is None or steps < self._max_generations
        under_cost = (
            self._max_budget_usd is None
            or (self.total_cost_usd or 0.0) < self._max_budget_usd
        )
        return under_steps and under_cost

    def _assert_loop_bounded(self) -> None:
        """Fail loudly if neither a step cap nor a tracked cost can bound the loop."""
        if self._max_generations is None and not (
            self.total_cost_usd and self.total_cost_usd > 0
        ):
            raise RuntimeError(
                "No step cap is set and the completion backend reports no cost, so "
                "the generation loop is unbounded. Set max_debug_attempts / "
                "max_generation_steps, or use a cost-reporting backend (e.g. the "
                "Claude CLI) together with max_budget_usd."
            )

    def _debug_loop(
        self,
        messages: list[dict[str, str]],
        approach_path: Path,
        sandbox_dir: Path,
        seeds: list[int],
    ) -> None:
        """Generate, validate, re-prompt with feedback until solved or out of budget.

        The first implementation attempt always runs (chain-of-thought may have
        already consumed the budget); the budget/step cap bounds only the further
        debug attempts, so an approach file is always written.
        """
        t = 0
        while True:
            response = self._complete(messages, sandbox_dir, f"impl{t}")
            messages.append({"role": "assistant", "content": response})
            # Overwrite, don't accumulate: each response is a full class.
            approach_path.write_text(_parse_python_code(response))
            if t == 0:
                self._assert_loop_bounded()

            failure = self._validate(approach_path, seeds)
            if failure is None:
                logger.info(
                    "All %d training tasks solved at attempt %d",
                    self._num_train_tasks,
                    t,
                )
                self.num_generations = t + 1
                return
            logger.info("Attempt %d failed (%s)", t, failure["error_type"])
            t += 1
            if not self._within_budget(t):
                break
            messages.append(
                {"role": "user", "content": f"{failure['feedback']}\nFix the code."}
            )
        self.num_generations = t  # budget/step cap reached without solving

    # ------------------------------------------------------------------ prompt

    def _build_context(self, prompt_seeds: list[int]) -> str:
        """Env description + source + a couple of example initial states."""
        assert self._env is not None
        parts = ["You are writing a policy for the environment below.\n"]
        if self._env_description_path is not None:
            parts.append(Path(self._env_description_path).read_text(encoding="utf-8"))
        parts.append("## Full source code\n\n" + _gather_env_source(self._env))
        parts.append(
            "## Primitives\n\n" + format_primitives_description(list(self._primitives))
        )
        parts.append(
            "## Example initial states\n\n"
            "Each instance below is one task (env.reset(seed=...))."
        )
        for s in prompt_seeds:
            obs, _ = self._env.reset(seed=s)
            parts.append(f"### seed {s}\n\n{render_state(self._state_space, obs)}")
        return "\n\n".join(parts)

    def _exchange(
        self, messages: list[dict[str, str]], prompt: str, sandbox: Path, idx: int
    ) -> None:
        messages.append({"role": "user", "content": prompt})
        response = self._complete(messages, sandbox, f"cot{idx}")
        messages.append({"role": "assistant", "content": response})

    def _complete(self, messages: list[dict[str, str]], sandbox: Path, tag: str) -> str:
        assert self._client is not None
        result: LLMResponse = self._client.complete(messages)
        if result.cost_usd is not None:
            self.total_cost_usd = (self.total_cost_usd or 0.0) + result.cost_usd
        (sandbox / f"{tag}_prompt.txt").write_text(messages[-1]["content"])
        (sandbox / f"{tag}_response.txt").write_text(result.text)
        return result.text

    # -------------------------------------------------------------- validation

    def _validate(self, approach_path: Path, seeds: list[int]) -> dict[str, str] | None:
        """Run the policy on the training tasks in-process; return first failure.

        When ``container_backend`` is docker/apptainer this runs inside the
        sandbox container (the whole loop does), so it is always isolated from
        the host there.
        """
        assert self._env is not None
        return validate_tasks(
            self._env,
            approach_path,
            self._action_space,
            self._state_space,
            self._primitives,
            seeds,
            self._max_steps,
            self._episode_timeout_s,
        )

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


def _gather_env_source(env: gymnasium.Env) -> str:
    """Best-effort bundle of the env's local source (robocode + kinder)."""
    files: list[Path] = []
    for obj, package in _source_targets(env):
        source_file = inspect.getsourcefile(type(obj))
        assert source_file is not None
        src = Path(source_file)
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


def _parse_python_code(response: str) -> str:
    """Extract the first ```python fenced block, else the raw response."""
    marker = "```python"
    if marker in response:
        rest = response[response.index(marker) + len(marker) :]
        end = rest.index("```") if "```" in rest else len(rest)
        return rest[:end].strip()
    return response.strip()
