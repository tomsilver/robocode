"""An approach that uses a Claude agent to generate behavior-based solutions.

Instead of writing a monolithic policy, the agent is guided to decompose the
task into a fixed sequence of behaviors (CDL-style), each with an explicit
precondition, subgoal, and a feedforward policy body.  The agent must:

1.  Reason about the high-level behavior decomposition first.
2.  Implement and test each behavior in isolation before composing them.
3.  Chain the behaviours into a final ``GeneratedApproach``.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space

from robocode.approaches.base_approach import BaseApproach
from robocode.mcp import MCP_TOOL_DESCRIPTIONS
from robocode.primitives import PRIMITIVE_DESCRIPTIONS
from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    DockerSandboxConfig,
)
from robocode.utils.episode import load_generated_approach
from robocode.utils.rate_limit import run_with_rate_limit_retry
from robocode.utils.sandbox import SandboxConfig

logger = logging.getLogger(__name__)

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert at writing purely imperative, feedforward policies for "
    "gymnasium environments. You decompose tasks into a fixed sequence of "
    "BEHAVIORS, where each behavior is a small, self-contained module with an "
    "explicit precondition, subgoal, and a deterministic policy body. "
    "IMPORTANT: You MUST write ALL files (approach.py, test scripts, etc.) "
    "to the current working directory using RELATIVE paths only. "
    "Never use absolute paths when writing files. "
    "IMPORTANT: Write code often to approach.py as you iterate. You may be "
    "interrupted at any time, so you should make sure that approach.py is "
    "your best current attempt at all times. "
    "VERSION CONTROL: This directory is a git repo. After each meaningful "
    "change to approach.py or supporting modules, run "
    "`git add -A && git commit -m '<describe what you changed and why>'`. "
    "Commit often, do not batch everything into one final commit. "
    "You should commit the approach every time before you test it in the environment. "
    "Use the Task tool to explore source code in parallel — e.g. spawn "
    "subagents to read environment dynamics, reward functions, and object "
    "types simultaneously rather than sequentially."
)

_MCP_TOOLS_SYSTEM_PROMPT_SUFFIX = (
    " IMPORTANT: You have visual debugging tools (render_state, render_policy). "
    "Start by calling render_state to see the environment before writing code. "
    "When your approach fails, call render_policy to visually diagnose the "
    "failure BEFORE guessing at fixes. "
    "CRITICAL: MCP tools are only available to YOU directly — they CANNOT be "
    "called from inside Task subagents. Always call MCP tools yourself, then "
    "delegate image reading to a Task subagent."
)

_CDL_DECOMPOSITION_PROMPT = """\

BEFORE writing any low-level code, you MUST first reason about the HIGH-LEVEL \
BEHAVIOR DECOMPOSITION of this task. Think of the task as a fixed sequence of \
phases/behaviors that, when executed in order, solve the environment.

For each behavior, explicitly define:
1. **Name**: A descriptive name.
2. **Precondition** (``initializable(state) -> bool``): Under what state conditions can \
this behavior start? This is a boolean predicate on the observation. This should always \
be true after the previous behavior's subgoal is achieved. For the initial behavior, \
it should be satisfied by the initial state of the environment (env.reset()). \
3. **Subgoal** (``terminated(state) -> bool``): What condition must be true for this \
behavior to be considered complete? Another boolean predicate.
4. **Policy body**: A high-level description of the strategy (e.g., "move above the \
object, lower the arm, activate vacuum, retract arm").
5. **Why this ordering**: Explain why the previous behavior's subgoal satisfies this \
behavior's precondition.

Example decomposition for a pick-and-place task with obstructions:
- **Behavior 1: ClearRegion** — Precondition: obstructions overlap the goal region. \
Subgoal: no obstructions overlap the goal region. Policy: for each obstruction on the \
surface, pick it up and place it in an empty area.
- **Behavior 2: PickAndPlace** — Precondition: goal region is clear. Subgoal: target \
block is on the goal surface. Policy: pick the block, carry it to the surface, place it.

The approach should determine which behavior to start from by checking preconditions \
BACKWARDS from the last behavior. If the last behavior's precondition is already \
satisfied, skip all earlier behaviors.

Write out your full decomposition BEFORE writing any code. This decomposition is the \
most important part of your solution.
"""

_BEHAVIOR_IMPLEMENTATION_PROMPT = """\

IMPORTANT: You MUST follow this EXACT file structure. Do NOT put everything in \
one file. Do NOT put helper functions inside approach.py or behavior files.

Required files:
- ``obs_helpers.py`` — ALL functions that parse/interpret the observation vector. \
This includes extracting object positions, computing geometric predicates \
(overlaps, is_on, etc.), and any named constants for observation indices. \
Every "magic number" related to observation parsing (index offsets, tolerances, \
physics constants like table height) MUST be a named constant here.
- ``act_helpers.py`` — ALL functions that help generate actions. This includes \
waypoint interpolation, action clipping, proportional controllers, etc. \
Every "magic number" related to action generation (step limits, arm extension \
rate, etc.) MUST be a named constant here.
- ``behaviors.py`` — ALL behavior classes. Each behavior inherits from the \
``Behavior`` base class (provided in ``behavior.py``). Behaviors import from \
``obs_helpers`` and ``act_helpers`` but contain NO magic numbers themselves.
- ``approach.py`` — ONLY the ``GeneratedApproach`` class. It imports behaviors \
from ``behaviors.py`` and does NOTHING except: (1) in ``reset``, determine \
the behavior sequence by checking ``initializable`` backwards, (2) in \
``get_action``, delegate to the current behavior's ``step()`` and advance \
when ``terminated()`` returns True. NO control logic, NO observation parsing, \
NO action generation in this file.

CRITICAL RULES:
- NO magic numbers anywhere except as named constants in obs_helpers.py or \
act_helpers.py. Every numeric literal (tolerances, offsets, indices, limits) \
must have a descriptive name. ``0.05`` is WRONG; ``DX_LIMIT = 0.05`` is RIGHT.
- Behaviors must use obs_helpers for ALL observation access. Never index into \
the observation array directly inside a behavior — use named extraction \
functions like ``extract_robot(obs)``, ``extract_rect(obs, "target_block")``.
- approach.py must be THIN. Its reset() only builds a behavior deque using \
backward precondition checking. Its get_action() only delegates to the \
current behavior and advances on termination. Nothing else.

A ``Behavior`` base class is provided in your working directory as ``behavior.py``:

```python
from behavior import Behavior

class MyBehavior(Behavior):
    def reset(self, x):
        \"\"\"Initialize internal state for a new execution.\"\"\"
        ...
    def initializable(self, x) -> bool:
        \"\"\"Return True if the precondition is met.\"\"\"
        ...
    def terminated(self, x) -> bool:
        \"\"\"Return True if the subgoal has been achieved.\"\"\"
        ...
    def step(self, x):
        \"\"\"Return the next action.\"\"\"
        ...
```

Testing protocol — for EACH behavior, write and run a test script that:
1. Resets the environment (try multiple seeds: 0, 1, 2, 3, 42).
2. If needed, manually sets up the state so the behavior's precondition is met \
(e.g., move obstructions away for a "pick" behavior).
3. Calls ``behavior.reset(state)``, then loops ``behavior.step(state)`` until \
``behavior.terminated(state)`` returns True.
4. **Asserts** that the subgoal is actually achieved and the behavior completes \
within a reasonable number of steps.
5. Only after ALL behaviors pass their individual tests, compose them into the \
final ``approach.py``.

IMPORTANT: If a behavior's action plan is exhausted but the subgoal is not \
reached, re-generate the plan from the current observation instead of \
repeating the same failed plan.
"""

_INTERFACE_SPEC = """\
Write `approach.py` containing a class `GeneratedApproach` with the following \
interface:

```python
from collections import deque
from behaviors import BehaviorA, BehaviorB  # your behavior classes

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self._behaviors = deque()
        self._current = None

    def reset(self, state, info):
        # Determine behavior sequence by checking preconditions BACKWARDS.
        # This is the ONLY logic allowed here.
        b_last = BehaviorB()
        b_first = BehaviorA()
        if b_last.initializable(state):
            self._behaviors = deque([b_last])
        else:
            self._behaviors = deque([b_first, b_last])
        self._current = self._behaviors.popleft()
        self._current.reset(state)

    def get_action(self, state):
        # Advance to next behavior when subgoal reached.
        if self._current.terminated(state) and self._behaviors:
            self._current = self._behaviors.popleft()
            self._current.reset(state)
        return self._current.step(state)
```

The ``reset`` method MUST ONLY build a behavior deque using backward \
precondition checking. The ``get_action`` method MUST ONLY delegate to the \
current behavior and advance on termination. No other logic is allowed in \
approach.py — all intelligence lives in the behaviors and helpers.

{primitives_description}

Write the best approach you can — ideally one that solves the environment \
optimally. Your `approach.py` should only use packages available in the \
current environment.

IMPORTANT: Use `{python_executable}` to run your test scripts, since that \
interpreter has all required packages installed. For example:
```bash
{python_executable} test_behavior_[behavior_name].py
{python_executable} test_approach.py
```

You can also inspect the source code of any imported module to understand \
the environment's dynamics in detail (reward function, transition logic, \
termination conditions, etc.). To locate a module's source file:
```bash
{python_executable} -c "import some_module; print(some_module.__file__)"
```
Then read the source to inform your approach.\
"""

_GEOMETRY_PROMPT = """\

BEFORE writing any code, you MUST first reason in detail about the geometry \
of this environment. Think carefully and qualitatively about spatial \
relationships, shapes, motions, and constraints.

CRITICAL: Your geometric reasoning must be PURELY QUALITATIVE. Do NOT use \
any numbers AT ALL — not in your reasoning, not in your thinking, not \
anywhere in your geometric analysis. Instead of saying "the object is at \
position (x, y)" say "the object is near the boundary". Instead of "move \
0.1 units" say "move a small step".

Your geometric reasoning should cover:
- What shapes are involved and how they interact.
- Key spatial relationships (above, inside, overlapping, adjacent, etc.).
- What geometric constraints exist (boundaries, clearances, collision).
- What motions/transformations are needed (translate, rotate, extend arm).
- What makes a configuration "good" or "bad" geometrically.

This qualitative analysis should directly inform your behavior decomposition \
and low-level policy design.
"""

_PROMPT_WITH_DESCRIPTION = """\
You are writing a behavior-based approach for the environment described below.

Your approach should be general enough to solve any instance of this environment \
(env.reset()), but it does NOT need to be adaptable to different other environments.

{env_description}
{geometry_prompt}
{cdl_decomposition_prompt}
{interface_spec}
{behavior_implementation_prompt}\
"""

_PROMPT_WITH_SOURCE = """\
Read the environment source files in this directory to understand the state \
type, action space, and dynamics.
{cdl_decomposition_prompt}
{interface_spec}
{behavior_implementation_prompt}\
"""


class AgenticCDLApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses a Claude agent to write behavior-decomposed code."""

    def __init__(
        self,
        action_space: Space[_ActType],
        observation_space: Space[_ObsType],
        seed: int,
        primitives: dict[str, Callable[..., Any]],
        env_description_path: str | None = None,
        model: str = "sonnet",
        max_budget_usd: float = 5.0,
        output_dir: str = ".",
        load_dir: str | None = None,
        use_docker: bool = False,
        geometry_prompt: bool = True,
        mcp_tools: tuple[str, ...] = (),
        max_output_tokens: int = 16384,
        autocompact_pct: int = 80,
    ) -> None:
        super().__init__(
            action_space,
            observation_space,
            seed,
            primitives,
            env_description_path,
        )
        self._model = model
        self._max_budget_usd = max_budget_usd
        self._output_dir = Path(output_dir)
        self._load_dir = Path(load_dir) if load_dir is not None else None
        self._use_docker = use_docker
        self._geometry_prompt = geometry_prompt
        self._mcp_tools = mcp_tools
        self._max_output_tokens = max_output_tokens
        self._autocompact_pct = autocompact_pct
        self._generated: Any = None
        self.total_cost_usd: float | None = None

    def train(self) -> None:  # noqa: C901 — mirrors AgenticApproach.train
        """Generate the behavior-based approach via a sandboxed Claude agent."""
        if self._load_dir is not None:
            approach_file = self._load_dir / "sandbox" / "approach.py"
            if not approach_file.exists():
                raise FileNotFoundError(f"No approach file at {approach_file}")
            self._load_generated(approach_file)
            return

        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Seed the sandbox with the Behavior base class so the agent can
        # inherit from it.
        behavior_src = (
            Path(__file__).resolve().parent.parent / "primitives" / "behavior.py"
        )
        init_files = {"behavior.py": behavior_src}

        # Build primitives description.
        if self._primitives:
            lines = ["`primitives` is a dict with these callables:\n"]
            for name in sorted(self._primitives):
                desc = PRIMITIVE_DESCRIPTIONS.get(name, f"`{name}`")
                lines.append(f"- {desc}")
            primitives_desc = "\n".join(lines)
            names = ", ".join(f"`{n}`" for n in sorted(self._primitives))
            primitives_desc += (
                f"\n\nIMPORTANT: Your approach MUST use the following "
                f"primitives: {names}. These are essential for solving "
                f"this environment. Read their descriptions above and "
                f"integrate them into your solution."
            )
        else:
            primitives_desc = "`primitives` is an empty dict."

        if self._mcp_tools:
            mcp_lines = [
                "\n\nYou also have MCP tools for visual debugging (they do NOT "
                "affect your test scripts):\n",
            ]
            for name in self._mcp_tools:
                if name in MCP_TOOL_DESCRIPTIONS:
                    mcp_lines.append(f"- {MCP_TOOL_DESCRIPTIONS[name]}")
            primitives_desc += "\n".join(mcp_lines)

        python_exe = DOCKER_PYTHON if self._use_docker else sys.executable
        interface_spec = _INTERFACE_SPEC.format(
            python_executable=python_exe,
            primitives_description=primitives_desc,
        )

        geometry = _GEOMETRY_PROMPT if self._geometry_prompt else ""

        if self._env_description_path is not None:
            env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
            prompt = _PROMPT_WITH_DESCRIPTION.format(
                env_description=env_desc,
                geometry_prompt=geometry,
                cdl_decomposition_prompt=_CDL_DECOMPOSITION_PROMPT,
                interface_spec=interface_spec,
                behavior_implementation_prompt=_BEHAVIOR_IMPLEMENTATION_PROMPT,
            )
        else:
            prompt = _PROMPT_WITH_SOURCE.format(
                cdl_decomposition_prompt=_CDL_DECOMPOSITION_PROMPT,
                interface_spec=interface_spec,
                behavior_implementation_prompt=_BEHAVIOR_IMPLEMENTATION_PROMPT,
            )

        system_prompt = _SYSTEM_PROMPT
        if self._mcp_tools:
            system_prompt += _MCP_TOOLS_SYSTEM_PROMPT_SUFFIX

        docker_config: DockerSandboxConfig | None = None
        config: SandboxConfig | None = None
        if self._use_docker:
            docker_config = DockerSandboxConfig(
                sandbox_dir=sandbox_dir,
                init_files=init_files,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=system_prompt,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
                primitive_names=tuple(self._primitives),
                mcp_tools=self._mcp_tools,
                max_output_tokens=self._max_output_tokens,
                autocompact_pct=self._autocompact_pct,
            )
            sandbox_logger = logging.getLogger("robocode.utils.docker_sandbox")
        else:
            config = SandboxConfig(
                sandbox_dir=sandbox_dir,
                init_files=init_files,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=system_prompt,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
                max_output_tokens=self._max_output_tokens,
                autocompact_pct=self._autocompact_pct,
            )
            sandbox_logger = logging.getLogger("robocode.utils.sandbox")

        log_path = sandbox_dir / "agent_log.txt"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        sandbox_logger.addHandler(file_handler)
        try:
            result = run_with_rate_limit_retry(
                docker_config if self._use_docker else None,
                config if not self._use_docker else None,
            )
        finally:
            sandbox_logger.removeHandler(file_handler)
            file_handler.close()

        self.total_cost_usd = result.total_cost_usd

        if result.success and result.output_file is not None:
            self._load_generated(result.output_file)
        else:
            logger.warning("Agent failed to generate approach: %s", result.error)

    def _load_generated(self, path: Path) -> None:
        """Load a GeneratedApproach class from the given file."""
        self._generated = load_generated_approach(
            path, self._action_space, self._state_space, self._primitives
        )

    def reset(self, state: _ObsType, info: dict[str, Any]) -> None:
        """Start a new episode."""
        super().reset(state, info)
        if self._generated is not None:
            self._generated.reset(state, info)

    def update(
        self,
        state: _ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Record the reward and next state following an action."""
        super().update(state, reward, done, info)
        if self._generated is not None and hasattr(self._generated, "update"):
            self._generated.update(state, reward, done, info)

    def _get_action(self) -> _ActType:
        if self._generated is not None:
            try:
                return self._generated.get_action(self._last_state)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Generated approach failed, using random")
        return self._action_space.sample()
