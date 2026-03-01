"""An approach that uses a Claude agent to generate approach code."""

import asyncio
import logging
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space

from robocode.approaches.base_approach import BaseApproach
from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    DockerSandboxConfig,
    run_agent_in_docker_sandbox,
)
from robocode.utils.sandbox import SandboxConfig, SandboxResult, run_agent_in_sandbox

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
    "your best current attempt at all times."
    "Use the Task tool to explore source code in parallel — e.g. spawn "
    "subagents to read environment dynamics, reward functions, and object "
    "types simultaneously rather than sequentially."
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
```

You can also inspect the source code of any imported module to understand \
the environment's dynamics in detail (reward function, transition logic, \
termination conditions, etc.). To locate a module's source file:
```bash
{python_executable} -c "import some_module; print(some_module.__file__)"
```
Then read the source to inform your approach.\
"""

_PRIMITIVE_DESCRIPTIONS: dict[str, str] = {
    "check_action_collision": (
        "`check_action_collision(state, action) -> bool` returns True when "
        "taking `action` in `state` would cause a collision (i.e. the agent "
        "stays in place). Use it to avoid wasted steps \u2014 e.g. in search or "
        "planning algorithms, skip actions that collide."
    ),
    "render_state": (
        "`render_state(state, ax_callback=None) -> np.ndarray` renders the "
        "given `state` as an RGB image (H\u00d7W\u00d73 uint8 numpy array). "
        "Optionally pass `ax_callback`, a function that takes a matplotlib "
        "`Axes` and draws on it. Use this to add markers, lines, "
        "annotations, or any other matplotlib drawing. Examples:\n"
        "  `render_state(state, ax_callback=lambda ax: ax.plot(1.5, 2.0, 'ro'))`\n"
        "  `render_state(state, ax_callback=lambda ax: ax.annotate('goal', (3, 1)))`\n"
        "Save to disk with "
        '`imageio.imwrite("state.png", render_state(state))` and read the '
        "file to visually understand the spatial layout."
    ),
    "csp": (
        "`csp` is a module providing a constraint satisfaction problem (CSP) "
        "solver. Use it to sample configurations (e.g. placements, grasps) "
        "that satisfy constraints (e.g. collision-free). Key classes:\n"
        "  - `csp.CSPVariable(name, domain)` \u2014 a variable with a "
        "`gymnasium.spaces.Space` domain.\n"
        "  - `csp.FunctionalCSPConstraint(name, variables, fn)` \u2014 a "
        "constraint where `fn(*vals) -> bool`.\n"
        "  - `csp.CSP(variables, constraints, cost=None)` \u2014 the problem.\n"
        "  - `csp.FunctionalCSPSampler(fn, csp, sampled_vars)` \u2014 a "
        "sampler where `fn(current_vals, rng) -> dict | None`.\n"
        "  - `csp.RandomWalkCSPSolver(seed)` \u2014 solver; call "
        "`.solve(csp, initialization, samplers)` to get a satisfying "
        "assignment or None.\n"
        "  - `csp.CSPCost(name, variables, cost_fn)` \u2014 optional cost to "
        "minimize.\n"
        "  - `csp.LogProbCSPConstraint(name, variables, logprob_fn, "
        "threshold)` \u2014 constraint from log probabilities.\n"
        "Access via `primitives['csp']`, e.g. "
        "`primitives['csp'].CSPVariable(...)`."
    ),
    "BiRRT": (
        "`BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng, "
        "num_attempts, num_iters, smooth_amt)` \u2014 Bidirectional RRT motion "
        "planner. Construct one, then call `birrt.query(start, goal)` to get "
        "a collision-free path (list of states) or None. "
        "`sample_fn(state) -> state` samples a random state, "
        "`extend_fn(s1, s2) -> Iterable[state]` interpolates between states, "
        "`collision_fn(state) -> bool` returns True if state is in collision, "
        "`distance_fn(s1, s2) -> float` returns distance between states, "
        "`rng` is a `np.random.Generator`."
    ),
}

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
- Break your solution into small, self-contained modules in separate .py files \
(e.g., `pathfinding.py`, `state_utils.py`, `planning.py`).
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
- Modules should be organized by functionality, and organized in directories if needed. \
For example, if you have multiple modules related to geometry, put them in a `geometry/` subdirectory. \
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


_DEFAULT_RESET_HOUR = 3  # fallback hour if we can't parse the reset time


def _parse_reset_hour(reset_str: str) -> int:
    """Parse a reset time like '3am' or '11pm' into a 24-hour int."""
    reset_str = reset_str.strip().lower()
    match = re.match(r"(\d{1,2})(am|pm)", reset_str)
    if not match:
        return _DEFAULT_RESET_HOUR
    hour = int(match.group(1))
    period = match.group(2)
    if period == "am":
        return 0 if hour == 12 else hour
    return hour if hour == 12 else hour + 12


def _seconds_until_reset(reset_hour: int) -> float:
    """Return seconds until the given hour (local time), plus a small buffer."""
    now = datetime.now()
    target = now.replace(hour=reset_hour, minute=5, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


class AgenticApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses a Claude agent to write approach code."""

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
        modular_code_prompt: bool = False,
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
        self._modular_code_prompt = modular_code_prompt
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
        if self._primitives:
            lines = ["`primitives` is a dict with these callables:\n"]
            for name in sorted(self._primitives):
                desc = _PRIMITIVE_DESCRIPTIONS.get(name, f"`{name}`")
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

        python_exe = DOCKER_PYTHON if self._use_docker else sys.executable
        interface_spec = _INTERFACE_SPEC.format(
            python_executable=python_exe,
            primitives_description=primitives_desc,
        )

        modular = _MODULAR_CODE_PROMPT if self._modular_code_prompt else ""

        if self._env_description_path is not None:
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
        config: SandboxConfig | None = None
        if self._use_docker:
            docker_config = DockerSandboxConfig(
                sandbox_dir=sandbox_dir,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
                primitive_names=tuple(self._primitives),
            )
            sandbox_logger = logging.getLogger("robocode.utils.docker_sandbox")
        else:
            config = SandboxConfig(
                sandbox_dir=sandbox_dir,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
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
            result = self._run_with_rate_limit_retry(
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

    def _run_with_rate_limit_retry(
        self,
        docker_config: DockerSandboxConfig | None,
        local_config: SandboxConfig | None,
    ) -> SandboxResult:
        """Run the sandbox, retrying on rate-limit by sleeping until reset."""
        while True:
            if docker_config is not None:
                result = asyncio.run(run_agent_in_docker_sandbox(docker_config))
            else:
                assert local_config is not None
                result = asyncio.run(run_agent_in_sandbox(local_config))

            if result.rate_limit_reset is None:
                return result

            reset_hour = _parse_reset_hour(result.rate_limit_reset)
            wait_secs = _seconds_until_reset(reset_hour)
            hours = wait_secs / 3600
            logger.warning(
                "Rate-limited (%s). Sleeping %.1f hours until %d:05 ...",
                result.error,
                hours,
                reset_hour,
            )
            time.sleep(wait_secs)
            logger.info("Woke up after rate-limit sleep, retrying...")

    def _load_generated(self, path: Path) -> None:
        """Load a GeneratedApproach class from the given file.

        Temporarily adds the parent directory of *path* to ``sys.path`` so
        that ``approach.py`` can import sibling modules written by the agent,
        then removes it to avoid polluting the global import path.
        """
        sandbox_dir = str(path.parent.resolve())
        added = sandbox_dir not in sys.path
        if added:
            sys.path.insert(0, sandbox_dir)
        try:
            source = path.read_text()
            namespace: dict[str, Any] = {}
            exec(  # pylint: disable=exec-used
                compile(source, str(path), "exec"), namespace
            )
        finally:
            if added:
                sys.path.remove(sandbox_dir)
        cls = namespace["GeneratedApproach"]
        self._generated = cls(
            self._action_space,
            self._state_space,
            primitives=self._primitives,
        )
        logger.info("Loaded generated approach from %s", path)

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
