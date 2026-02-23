"""An approach that uses a Claude agent to generate approach code."""

import asyncio
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space

from robocode.approaches.base_approach import BaseApproach
from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    DockerSandboxConfig,
    run_agent_in_docker_sandbox,
)
from robocode.utils.sandbox import SandboxConfig, run_agent_in_sandbox

logger = logging.getLogger(__name__)

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")

_SYSTEM_PROMPT = (
    "You are an expert at writing policies for gymnasium environments. "
    "You will read environment source code, understand the dynamics, "
    "and write an optimal approach class. "
    "IMPORTANT: You MUST write ALL files (approach.py, test scripts, etc.) "
    "to the current working directory using RELATIVE paths only. "
    "Never use absolute paths when writing files."
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

_PROMPT_WITH_DESCRIPTION = """\
You are writing an approach for the environment is described below.

Your approach should be general enough to solve any instance of this environment (env.reset()), \
but it does NOT need to be adaptable to different other environments.

{env_description}

{interface_spec}\
"""

_PROMPT_WITH_SOURCE = """\
Read the environment source files in this directory to understand the state \
type, action space, and dynamics.

{interface_spec}\
"""


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

        if self._env_description_path is not None:
            env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
            prompt = _PROMPT_WITH_DESCRIPTION.format(
                env_description=env_desc, interface_spec=interface_spec
            )
        else:
            prompt = _PROMPT_WITH_SOURCE.format(interface_spec=interface_spec)

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
            if self._use_docker:
                result = asyncio.run(run_agent_in_docker_sandbox(docker_config))
            else:
                result = asyncio.run(run_agent_in_sandbox(config))
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
        source = path.read_text()
        namespace: dict[str, Any] = {}
        exec(compile(source, str(path), "exec"), namespace)  # pylint: disable=exec-used
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
