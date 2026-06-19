"""An approach that uses an LLM coding agent to generate approach code."""

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

_SYSTEM_PROMPT = (
    "You are an expert at writing policies for gymnasium environments. "
    "You will read environment source code, understand the dynamics, "
    "and write an optimal approach class. "
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
    "types simultaneously rather than sequentially."
)

_SYSTEM_PROMPT_BLACKBOX = (
    "You are an expert at writing policies for gymnasium environments. "
    "The environment is a black box: its source code is not available, so "
    "you will discover the dynamics, reward structure, and termination "
    "conditions empirically by interacting with a live environment "
    "instance, and write an optimal approach class. "
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
    "gets its own fresh environment instance."
)

_INTERFACE_SPEC = """\
Write `approach.py` containing a class `GeneratedApproach` with the following \
interface:

```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        \"\"\"Initialize with the environment's gym spaces.\"\"\"
        ...

    def reset(self, state, info):
        \"\"\"Called at the start of each episode with the initial state.\"\"\"
        ...

    def get_action(self, state):
        \"\"\"Return a valid action for the given state.\"\"\"
        ...
```

The class can maintain internal state between calls (e.g., a computed plan). \
The `reset` method is called at the start of each episode. The `get_action` \
method is called each step and must return a valid action.

{primitives_description}

Write the best approach you can \u2014 ideally one that solves the environment \
optimally. Your `approach.py` should only use packages available in the \
current environment. Write test scripts that use the real environment to \
verify your approach works.

IMPORTANT: Use `{python_executable}` to run your test scripts, since that \
interpreter has all required packages installed. For example:
```bash
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
harness passes to `GeneratedApproach.__init__`. Use it in test scripts so \
they exercise the real primitives (env-dependent ones run on the host):

```python
from env_client import make_env

env = make_env()
primitives = env.make_primitives()
```

You can also call `env.observation_space.devectorize(obs)` to get an \
object-centric view of an observation. It returns an `ObjectCentricState` \
with `get_object_names()`, `get_object_from_name(name)`, `get_objects(type)`, \
and `get(obj, feature)`; iterate objects via `get_objects(...)`, NOT \
`for obj in ocs`. Call `env.observation_space.vectorize(ocs)` to go back to a \
flat array. The evaluation harness passes the SAME `observation_space` to \
`GeneratedApproach`, so `observation_space.devectorize(obs)` works identically \
there, and `approach.py` can use it directly.

Parallel test scripts are fine: every `make_env()` call creates an \
independent environment instance.

Start by exploring systematically: reset with several seeds, apply \
controlled actions, and study how the observation vector changes to \
identify what each dimension means and how actions affect the state.

CRITICAL: `approach.py` itself must NOT import `env_client`. It will be \
evaluated against the real environment by a separate harness that calls \
`reset(state, info)` and `get_action(state)` directly. Use `env_client` \
ONLY in test and exploration scripts.\
"""

_GEOMETRY_PROMPT = """\

BEFORE writing any code, you MUST first reason in detail about the geometry of this environment. \
Think carefully and qualitatively about spatial relationships, shapes, motions, and constraints.

CRITICAL: Your geometric reasoning must be PURELY QUALITATIVE. Do NOT use any numbers AT ALL — \
not in your reasoning, not in your thinking, not anywhere in your geometric analysis. This means \
NO coordinates, NO distances, NO angles, NO dimensions, NO sizes, NO counts of objects, NO \
thresholds, NO numeric constants, NO array indices, NO velocities, NO ratios, NO percentages. \
Not even "2D" or "3D" — say "two-dimensional" or "three-dimensional" instead. If you catch \
yourself about to write a number, stop and rephrase using purely relational, qualitative language. \
Instead of saying "the object is at position (x, y)" say "the object is near the boundary". \
Instead of "move 0.1 units" say "move a small step". Instead of "the angle is 90 degrees" say \
"the surfaces are perpendicular".

Your geometric reasoning should cover topics like:
- What kinds of geometric shapes are involved (e.g. rectangles, circles, polygons, cuboids, \
spheres, cylinders, capsules)? How do their shapes affect interactions — for instance, \
rectangles tile differently than circles, spheres roll while cuboids don't, narrow corridors \
between rectangular obstacles require precise alignment.
- What are the key spatial relationships between objects? Reason about relative orientations \
(parallel, perpendicular, oblique, aligned, tangent, skewed, transverse), relative positions \
(adjacent, opposite, coplanar, collinear, concentric, coaxial, symmetric), and topological \
relations (inside, outside, overlapping, enclosing, intersecting, touching, disjoint). Which \
of these relationships matter for solving the task?
- What geometric constraints exist? Are there boundaries, obstacles, containment relationships, \
or clearance requirements? How do the shapes of obstacles create narrow passages, dead ends, \
or open regions? Are surfaces flush or offset? Are relevant surfaces convex or concave?
- What kind of motions or transformations are involved? Are objects translating, rotating, or \
both? Are motions continuous or discrete? Does an object's shape affect how it can move \
(e.g. a rectangle rotating requires more swept area than a circle of similar size)?
- What makes a configuration "good" or "bad" geometrically? Think about reachability, \
collision-freeness, coverage, proximity to goals.
- What is the overall geometric strategy? For example:
  - "The agent needs to navigate around obstacles to reach a goal region, so it must find a \
path that threads between blocked areas while staying within bounds. When two rectangular \
obstacles have parallel edges with a gap between them, the agent can pass through \
perpendicular to those edges."
  - "Objects must be packed tightly without overlapping, so the approach needs to find \
placements where each new object fits into remaining gaps while respecting clearance from \
existing objects. Rectangular objects can be aligned with parallel edges flush against each \
other for dense packing, while circular objects leave unavoidable gaps."
  - "The robot arm must move its end effector to a grasp pose, which means planning a \
sequence of joint motions that avoids self-collision and keeps the kinematic chain valid. \
The shape of the target object determines viable grasp orientations — a sphere can be \
grasped from any direction, while a flat rectangular object requires approaching \
perpendicular to one of its faces."
  - "The agent must push an object toward a target, which requires approaching from the \
opposite side so that the push direction is aligned with the line from object to goal. \
A cylindrical object may roll unpredictably under pushes that are oblique to its axis."

This qualitative geometric analysis should directly inform your code. Write your reasoning \
out before you start coding. Do NOT skip this step. Remember: your geometric reasoning must \
contain ZERO numbers. If any number appears in your geometric analysis, you have failed the task.
"""

_MODULAR_CODE_PROMPT = """\

IMPORTANT: Write MODULAR code, like a skilled software engineer:
- Break your solution into small, self-contained modules in separate .py files.
- Each module should be minimal and focused on a single responsibility, small enough to \
reason about, test, and reuse independently.
- Write and run a test script for each module BEFORE composing them together. Verify each \
piece works in isolation. The tests should play out the modules in the actual environment \
if possible, verifying the conditions before and after execution and ensuring these match \
the expectations, under multiple conditions and edge cases, and should not just rely \
on mock objects or simplified assumptions.
- Your final `approach.py` should import from these modules and compose them into the \
complete solution. Keep `approach.py` itself as thin as possible, it should primarily \
orchestrate your tested modules.
- Prefer many small files over one large file. If a function could be useful in multiple \
contexts, it belongs in its own module.
IMPORTANT: be careful about repeated behavior! If an action or strategy in your \
approach fails, you should design your code to avoid repeating that failure.
"""

_PROMPT_WITH_DESCRIPTION = """\
You are writing an approach for the environment described below.

Your approach should be general enough to solve any instance of this environment (env.reset()), \
but it does NOT need to be adaptable to different other environments.

{env_description}
{geometry_prompt}
{interface_spec}
{modular_code_prompt}\
"""

_PROMPT_WITH_SOURCE = """\
Read the environment source files in this directory to understand the state \
type, action space, and dynamics.
{interface_spec}
{modular_code_prompt}\
"""

_PROMPT_BLACKBOX = """\
You are writing an approach for an environment that you can only access as \
a black box.

Your approach should be general enough to solve any instance of this \
environment (each reset gives a new instance), but it does NOT need to be \
adaptable to different other environments.
{env_description_section}
{blackbox_interaction_spec}
{geometry_prompt}
{interface_spec}
{modular_code_prompt}\
"""

_BLACKBOX_DESCRIPTION_PREFIX = """\

The environment is described below.

"""


class AgenticApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses an LLM coding agent to write approach code."""

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
        modular_code_prompt: bool = False,
        mcp_tools: tuple[str, ...] = (),
        max_output_tokens: int = 16384,
        autocompact_pct: int = 80,
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
        self._modular_code_prompt = modular_code_prompt
        self._mcp_tools = mcp_tools
        self._max_output_tokens = max_output_tokens
        self._autocompact_pct = autocompact_pct
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

    def train(self) -> None:
        if self._load_dir is not None:
            approach_file = self._load_dir / "sandbox" / "approach.py"
            if not approach_file.exists():
                raise FileNotFoundError(f"No approach file at {approach_file}")
            self._load_generated(approach_file)
            return

        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Build the prompt. If we have an env description, inline it so the
        # agent knows exactly which environment to target.  Otherwise fall
        # back to asking the agent to read source files.
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

        modular = _MODULAR_CODE_PROMPT if self._modular_code_prompt else ""

        if self._blackbox:
            env_description_section = ""
            if self._env_description_path is not None:
                env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
                env_description_section = _BLACKBOX_DESCRIPTION_PREFIX + env_desc + "\n"
            geometry = _GEOMETRY_PROMPT if self._geometry_prompt else ""
            prompt = _PROMPT_BLACKBOX.format(
                env_description_section=env_description_section,
                blackbox_interaction_spec=_BLACKBOX_INTERACTION_SPEC,
                geometry_prompt=geometry,
                interface_spec=interface_spec,
                modular_code_prompt=modular,
            )
        elif self._env_description_path is not None:
            env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
            geometry = _GEOMETRY_PROMPT if self._geometry_prompt else ""
            prompt = _PROMPT_WITH_DESCRIPTION.format(
                env_description=env_desc,
                geometry_prompt=geometry,
                modular_code_prompt=modular,
                interface_spec=interface_spec,
            )
        else:
            prompt = _PROMPT_WITH_SOURCE.format(
                modular_code_prompt=modular,
                interface_spec=interface_spec,
            )

        docker_config: DockerSandboxConfig | None = None
        apptainer_config: ApptainerSandboxConfig | None = None
        config: SandboxConfig | None = None
        system_prompt = _SYSTEM_PROMPT_BLACKBOX if self._blackbox else _SYSTEM_PROMPT
        init_files: dict[str, Path] = {}
        if self._blackbox:
            init_files["env_client.py"] = ENV_CLIENT_SRC
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
                blackbox=self._blackbox,
            )
            sandbox_logger = logging.getLogger("robocode.utils.sandbox")

        # Write agent logs to a file in the sandbox directory.
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
            if result.error:
                logger.info(
                    "Agent stopped early (%s) but committed an approach; "
                    "evaluating it.",
                    result.error,
                )
            self._load_generated(result.output_file)
        else:
            raise RuntimeError(f"Agent failed to generate an approach: {result.error}")

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
        # Never silently fall back to random: a random eval would propagate a
        # misleading 0 through the results as if the generated approach had been
        # measured. Fail loudly instead. Use approach=random for a random
        # baseline. A runtime error in the generated approach also propagates.
        if self._generated is None:
            raise RuntimeError(
                "No generated approach is loaded (the agent did not produce a "
                "usable approach.py). Refusing to evaluate a silent random "
                "policy; use approach=random for a random baseline."
            )
        return self._generated.get_action(self._last_state)
