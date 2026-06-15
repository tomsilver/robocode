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
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space
from omegaconf import DictConfig

from robocode.approaches.base_approach import BaseApproach
from robocode.mcp import (
    MCP_TOOLS_SYSTEM_PROMPT_SUFFIX,
    MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX,
    mcp_tool_descriptions,
)
from robocode.primitives import (
    blackbox_primitive_manifest,
    format_primitives_description,
)
from robocode.utils.apptainer_sandbox import ApptainerSandboxConfig
from robocode.utils.backends import (
    CLAUDE_PROMPT_SUFFIX,
    OPENCODE_PROMPT_SUFFIX,
    create_backend,
)
from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    DockerSandboxConfig,
)
from robocode.utils.env_server import (
    ENV_CLIENT_SRC,
    env_server_running,
    serialize_space,
    write_env_spaces,
)
from robocode.utils.episode import load_generated_approach
from robocode.utils.rate_limit import run_with_rate_limit_retry
from robocode.utils.sandbox import SandboxConfig
from robocode.utils.sandbox_types import resolve_container_backend

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
    "Use subagents to explore source code in parallel, e.g. spawn "
    "subagents to read environment dynamics, reward functions, and object "
    "types simultaneously rather than sequentially. "
    "TOKEN BUDGET: You have a limited output-token budget per turn. Be concise. "
    "Do NOT write lengthy prose reasoning about geometry or arithmetic — put "
    "all numerical calculations in code (Python scripts or inline print "
    "statements) and read the results. Your text should be SHORT: state what "
    "you will do, then immediately write code. Never narrate step-by-step "
    "arithmetic in text."
)

_SYSTEM_PROMPT_BLACKBOX = (
    "You are an expert at writing purely imperative, feedforward policies for "
    "gymnasium environments. You decompose tasks into a fixed sequence of "
    "BEHAVIORS, where each behavior is a small, self-contained module with an "
    "explicit precondition, subgoal, and a deterministic policy body. "
    "The environment is a black box: its source code is not available, so "
    "you will discover the dynamics, reward structure, and termination "
    "conditions empirically by interacting with a live environment instance. "
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
    "Use subagents to run exploration experiments in parallel, e.g. spawn "
    "subagents to probe action effects, reward structure, and termination "
    "conditions simultaneously rather than sequentially; each test script "
    "gets its own fresh environment instance. "
    "TOKEN BUDGET: You have a limited output-token budget per turn. Be concise. "
    "Do NOT write lengthy prose reasoning about geometry or arithmetic — put "
    "all numerical calculations in code (Python scripts or inline print "
    "statements) and read the results. Your text should be SHORT: state what "
    "you will do, then immediately write code. Never narrate step-by-step "
    "arithmetic in text."
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
This includes extracting object positions, computing geometric predicates, \
and any named constants for observation indices. \
Every "magic number" related to observation parsing MUST be a named constant here. \
{obs_inspection_note}\
{obs_helpers_note}
- ``act_helpers.py`` — ALL functions that help generate actions. This includes \
waypoint interpolation, action clipping, proportional controllers, etc. \
Every "magic number" related to action generation (step limits, arm extension \
rate, etc.) MUST be a named constant here. \
{act_helpers_note}
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

_OBS_INSPECTION_NOTE = """\
BEFORE writing this file, you MUST run: \
``feats = env.unwrapped.observation_space.devectorize(obs)`` \
to inspect the observation structure. This returns a dictionary mapping \
feature names to their values, so you can see exactly what each part of \
the observation vector represents. Use this to determine the correct \
indices, feature names, and semantics — do NOT guess the observation layout. \
"""

_OBS_INSPECTION_NOTE_BLACKBOX = """\
BEFORE writing this file, you MUST map the observation layout empirically \
with env_client: reset with several seeds, perturb the state with \
``set_state``, and observe which observation entries change as you act. \
Do NOT guess the observation layout. \
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
```\
"""

_INSPECT_SOURCE_SUFFIX = """

You can also inspect the source code of any imported module to understand \
the environment's dynamics in detail (reward function, transition logic, \
termination conditions, etc.). To locate a module's source file:
```bash
{python_executable} -c "import some_module; print(some_module.__file__)"
```
Then read the source to inform your approach.\
"""

_BLACKBOX_INTERACTION_SPEC = """\
The environment is a BLACK BOX. You CANNOT read its source code; it is not \
available anywhere on this machine. A live environment server is running. \
Interact with it through `env_client.py` in your working directory:

```python
from env_client import make_env

env = make_env()  # fresh environment instance per call
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
state = env.get_state()  # numpy snapshot of the full environment state
env.set_state(state)  # restore a snapshot
env.close()
```

`env.observation_space` and `env.action_space` expose `shape`, `low`, \
`high`, `dtype`, and `sample()`; the same metadata is in `env_spaces.json`. \
`env.max_steps` is the episode step limit used at evaluation time.

`env.make_primitives()` returns the SAME `primitives` dict the evaluation \
harness passes to `GeneratedApproach.__init__` (env-dependent primitives run \
on the host); use it in test scripts so behaviors exercise the real \
primitives.

You can also call `env.observation_space.devectorize(obs)` to get an \
object-centric view of an observation. It returns an `ObjectCentricState` \
with `get_object_names()`, `get_object_from_name(name)`, `get_objects(type)`, \
and `get(obj, feature)`; iterate objects via `get_objects(...)`, NOT \
`for obj in ocs`. Call `env.observation_space.vectorize(ocs)` to go back to a \
flat array. The evaluation harness passes the SAME `observation_space` to \
`GeneratedApproach`, so `observation_space.devectorize(obs)` works identically \
there, and `approach.py` can use it directly.

Parallel test scripts are fine: every `make_env()` call creates an \
independent environment instance. Use `set_state` to put the environment \
into the state a behavior's precondition requires when testing it in \
isolation.

Start by exploring systematically: reset with several seeds, apply \
controlled actions, and study how the observation vector changes to \
identify what each dimension means and how actions affect the state.

CRITICAL: `approach.py` itself must NOT import `env_client`. It will be \
evaluated against the real environment by a separate harness that calls \
`reset(state, info)` and `get_action(state)` directly. Use `env_client` \
ONLY in test and exploration scripts.\
"""

_GEOMETRY_PROMPT = """\

BEFORE writing any code, briefly describe (in 5-10 bullet points) the key \
geometric relationships in this environment:
- What shapes are involved and how they interact spatially.
- What motions/transformations are needed (translate, rotate, extend arm).
- What collision constraints exist (clearances, boundaries).

Keep this analysis SHORT and QUALITATIVE — no numbers, no arithmetic. \
If you need to compute specific positions or offsets, write a Python \
script to do it and print the results. NEVER do arithmetic in text.
"""

_INITIAL_HELPERS_PROMPT = """\

IMPORTANT: You have been provided with initial versions of ``obs_helpers.py`` \
and ``act_helpers.py`` in your working directory. These files contain CORRECT \
observation parsing (feature indices, object layout, extraction functions) and \
action generation helpers (waypoint interpolation, action limits) for this \
environment. You MUST use them as your starting point:

- **DO NOT** rewrite observation parsing from scratch — the provided \
``obs_helpers.py`` already has the correct feature layout and extraction \
functions.
- **DO NOT** rewrite action helpers from scratch — the provided \
``act_helpers.py`` already has correct action limits and waypoint utilities.
- You MAY and SHOULD add new helper functions, constants, or geometric \
predicates to these files as needed for your behaviors.
- You MAY modify existing functions if you need to extend them, but do not \
change the core observation layout or action format — they are correct.

Start by reading these files to understand the observation structure and \
available utilities before writing any behaviors.
"""

_PROMPT_WITH_DESCRIPTION = """\
You are writing a behavior-based approach for the environment described below.

Your approach should be general enough to solve any instance of this environment \
(env.reset()), but it does NOT need to be adaptable to different other environments.

{env_description}
{initial_helpers_prompt}
{geometry_prompt}
{cdl_decomposition_prompt}
{interface_spec}
{behavior_implementation_prompt}\
"""

_PROMPT_WITH_SOURCE = """\
Read the environment source files in this directory to understand the state \
type, action space, and dynamics.
{initial_helpers_prompt}
{cdl_decomposition_prompt}
{interface_spec}
{behavior_implementation_prompt}\
"""

_PROMPT_BLACKBOX = """\
You are writing a behavior-based approach for an environment that you can \
only access as a black box.

Your approach should be general enough to solve any instance of this \
environment (each reset gives a new instance), but it does NOT need to be \
adaptable to different other environments.
{env_description_section}
{blackbox_interaction_spec}
{initial_helpers_prompt}
{geometry_prompt}
{cdl_decomposition_prompt}
{interface_spec}
{behavior_implementation_prompt}\
"""

_BLACKBOX_DESCRIPTION_PREFIX = """\

The environment is described below.

"""


class AgenticCDLApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses a Claude agent to write behavior-decomposed code."""

    def __init__(
        self,
        action_space: Space[_ActType],
        observation_space: Space[_ObsType],
        seed: int,
        primitives: dict[str, Callable[..., Any]],
        backend: DictConfig,  # Hydra backend config (backend name, model, etc.)
        env_description_path: str | None = None,
        max_budget_usd: float = 5.0,
        max_turns: int = 0,
        output_dir: str = ".",
        load_dir: str | None = None,
        use_docker: bool = False,
        container_backend: str | None = None,
        geometry_prompt: bool = True,
        mcp_tools: tuple[str, ...] = (),
        max_output_tokens: int = 16384,
        autocompact_pct: int = 80,
        env_name: str | None = None,
        blackbox: bool = False,
        env_cfg: str | None = None,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_space,
            observation_space,
            seed,
            primitives,
            env_description_path,
            **kwargs,
        )
        self._backend_cfg = backend
        self._backend = create_backend(backend)
        self._model = backend["model"]
        self._max_budget_usd = max_budget_usd
        self._max_turns = max_turns
        self._output_dir = Path(output_dir)
        self._load_dir = Path(load_dir) if load_dir is not None else None
        self._container_backend = resolve_container_backend(
            container_backend, use_docker
        )
        self._geometry_prompt = geometry_prompt
        self._mcp_tools = mcp_tools
        self._max_output_tokens = max_output_tokens
        self._autocompact_pct = autocompact_pct
        self._env_name = env_name
        self._blackbox = blackbox
        self._env_cfg = env_cfg
        self._max_steps = max_steps
        if blackbox:
            if env_cfg is None:
                raise ValueError("blackbox mode requires env_cfg")
            # Fail fast on spaces the blackbox protocol cannot serialize.
            serialize_space(action_space)
            serialize_space(observation_space)
            if self._container_backend == "local":
                logger.warning(
                    "blackbox with the local backend is best-effort only: "
                    "the OS sandbox cannot prevent reading env source from "
                    "the host filesystem"
                )
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
        init_files: dict[str, Path] = {"behavior.py": behavior_src}
        if self._blackbox:
            init_files["env_client.py"] = ENV_CLIENT_SRC

        # If env-specific helper files exist, copy them into the sandbox so
        # the agent starts with correct obs/act helpers instead of writing
        # them from scratch. Skipped in blackbox mode: these helpers spell out
        # the observation layout (feature names and indices), which is exactly
        # what the agent must discover empirically when the source is withheld.
        has_initial_helpers = False
        if self._env_name is not None and not self._blackbox:
            helpers_dir = (
                Path(__file__).resolve().parent.parent / "primitives" / self._env_name
            )
            for helper_name in ("obs_helpers.py", "act_helpers.py"):
                helper_path = helpers_dir / helper_name
                if helper_path.exists():
                    init_files[helper_name] = helper_path
                    has_initial_helpers = True

        # Build primitives description (shared with AgenticApproach; the
        # blackbox flag appends per-primitive black-box notes, e.g. how to feed
        # the CRV planners the devectorized state).
        primitives_desc = format_primitives_description(
            list(self._primitives), blackbox=self._blackbox
        )

        if self._mcp_tools:
            backend_name = self._backend_cfg["backend"]
            tool_descs = mcp_tool_descriptions(backend_name, blackbox=self._blackbox)
            mcp_lines = [
                "\n\nYou also have MCP tools for visual debugging (they do NOT "
                "affect your test scripts):\n",
            ]
            for name in self._mcp_tools:
                if name in tool_descs:
                    mcp_lines.append(f"- {tool_descs[name]}")
            primitives_desc += "\n".join(mcp_lines)

        python_exe = (
            DOCKER_PYTHON if self._container_backend != "local" else sys.executable
        )
        interface_spec = _INTERFACE_SPEC.format(
            python_executable=python_exe,
            primitives_description=primitives_desc,
        )
        if not self._blackbox:
            interface_spec += _INSPECT_SOURCE_SUFFIX.format(
                python_executable=python_exe
            )

        geometry = _GEOMETRY_PROMPT if self._geometry_prompt else ""
        initial_helpers = _INITIAL_HELPERS_PROMPT if has_initial_helpers else ""

        if has_initial_helpers:
            provided_note = (
                "This file is ALREADY PROVIDED in your working directory with "
                "correct parsing logic. Read it first, then extend it with any "
                "additional helpers you need. Do NOT rewrite it from scratch."
            )
            obs_helpers_note = provided_note
            act_helpers_note = provided_note
        else:
            obs_helpers_note = ""
            act_helpers_note = ""

        behavior_impl_prompt = _BEHAVIOR_IMPLEMENTATION_PROMPT.format(
            obs_inspection_note=(
                _OBS_INSPECTION_NOTE_BLACKBOX
                if self._blackbox
                else _OBS_INSPECTION_NOTE
            ),
            obs_helpers_note=obs_helpers_note,
            act_helpers_note=act_helpers_note,
        )

        if self._blackbox:
            env_description_section = ""
            if self._env_description_path is not None:
                env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
                env_description_section = _BLACKBOX_DESCRIPTION_PREFIX + env_desc + "\n"
            prompt = _PROMPT_BLACKBOX.format(
                env_description_section=env_description_section,
                blackbox_interaction_spec=_BLACKBOX_INTERACTION_SPEC,
                initial_helpers_prompt=initial_helpers,
                geometry_prompt=geometry,
                cdl_decomposition_prompt=_CDL_DECOMPOSITION_PROMPT,
                interface_spec=interface_spec,
                behavior_implementation_prompt=behavior_impl_prompt,
            )
        elif self._env_description_path is not None:
            env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
            prompt = _PROMPT_WITH_DESCRIPTION.format(
                env_description=env_desc,
                geometry_prompt=geometry,
                cdl_decomposition_prompt=_CDL_DECOMPOSITION_PROMPT,
                interface_spec=interface_spec,
                behavior_implementation_prompt=behavior_impl_prompt,
                initial_helpers_prompt=initial_helpers,
            )
        else:
            prompt = _PROMPT_WITH_SOURCE.format(
                initial_helpers_prompt=initial_helpers,
                cdl_decomposition_prompt=_CDL_DECOMPOSITION_PROMPT,
                interface_spec=interface_spec,
                behavior_implementation_prompt=behavior_impl_prompt,
            )

        system_prompt = _SYSTEM_PROMPT_BLACKBOX if self._blackbox else _SYSTEM_PROMPT
        backend_name = self._backend_cfg["backend"]
        if backend_name == "opencode":
            system_prompt += OPENCODE_PROMPT_SUFFIX
        else:
            system_prompt += CLAUDE_PROMPT_SUFFIX
        if self._mcp_tools:
            system_prompt += (
                MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX
                if self._blackbox
                else MCP_TOOLS_SYSTEM_PROMPT_SUFFIX
            )

        docker_config: DockerSandboxConfig | None = None
        apptainer_config: ApptainerSandboxConfig | None = None
        config: SandboxConfig | None = None
        if self._container_backend == "docker":
            docker_config = DockerSandboxConfig(
                sandbox_dir=sandbox_dir,
                init_files=init_files,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=system_prompt,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
                max_turns=self._max_turns,
                primitive_names=tuple(self._primitives),
                mcp_tools=self._mcp_tools,
                max_output_tokens=self._max_output_tokens,
                autocompact_pct=self._autocompact_pct,
                blackbox=self._blackbox,
            )
            sandbox_logger = logging.getLogger("robocode.utils.docker_sandbox")
        elif self._container_backend == "apptainer":
            apptainer_config = ApptainerSandboxConfig(
                sandbox_dir=sandbox_dir,
                init_files=init_files,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=system_prompt,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
                max_turns=self._max_turns,
                primitive_names=tuple(self._primitives),
                mcp_tools=self._mcp_tools,
                max_output_tokens=self._max_output_tokens,
                autocompact_pct=self._autocompact_pct,
                blackbox=self._blackbox,
            )
            sandbox_logger = logging.getLogger("robocode.utils.apptainer_sandbox")
        else:
            config = SandboxConfig(
                sandbox_dir=sandbox_dir,
                init_files=init_files,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=system_prompt,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
                max_turns=self._max_turns,
                mcp_tools=self._mcp_tools,
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
            with ExitStack() as stack:
                if self._blackbox:
                    assert self._env_cfg is not None  # validated in __init__
                    port, token = stack.enter_context(
                        env_server_running(self._env_cfg, sandbox_dir)
                    )
                    write_env_spaces(
                        sandbox_dir,
                        container_backend=self._container_backend,
                        port=port,
                        token=token,
                        observation_space=self._state_space,
                        action_space=self._action_space,
                        max_steps=self._max_steps,
                        primitives_manifest=blackbox_primitive_manifest(
                            list(self._primitives)
                        ),
                    )
                result = run_with_rate_limit_retry(
                    docker_config,
                    config,
                    backend=self._backend,
                    apptainer_config=apptainer_config,
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
