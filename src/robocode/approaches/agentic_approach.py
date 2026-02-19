"""An approach that uses a Claude agent to generate approach code."""

import asyncio
import inspect
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space

from robocode.approaches.base_approach import BaseApproach
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
    "Never use absolute paths when writing files. "
    "IMPORTANT: Write code often to approach.py as you iterate. You may be "
    "interrupted at any time, so you should make sure that approach.py is "
    "your best current attempt at all times. "
    "IMPORTANT: Keep CLAUDE.md updated with your plan and progress. It is "
    "loaded into your context every turn and acts as your persistent memory. "
    "Update the Progress checklist as you complete each step. "
    "IMPORTANT: Write modular code. Develop test-first. Keep your tests "
    "in separate test_*.py files and run them frequently as you develop. "
    "Debug with tests. When you encounter a bug or unexpected "
    "behavior, do NOT jump straight to fixing the code. First, write a "
    "test that reproduces the issue. Then debug by iterating until that "
    "test passes."
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

`primitives` is a dict of helper callables. Their source code is in the \
`primitives/` directory — read each file to understand what is available. \
{required_primitives_note}
Write the best approach you can \u2014 ideally one that solves the environment \
optimally. Your `approach.py` should only use packages available in the \
current environment.

Follow this workflow:

STEP 0 — PLAN: Before writing any code, read the environment source to \
understand the state type, action space, dynamics, reward function, and \
termination conditions. Then update CLAUDE.md with these sections:
- **Plan**: Your approach strategy (algorithm, key insights, edge cases).
- **Curriculum**: 3-5 test scenarios ordered simple→complex. For each, \
note what aspect it tests and how to construct the state.
- **Progress**: A markdown checklist tracking your work — plan written, \
each curriculum test passing, integration test passing. Update this \
checklist as you complete each step.

STEP 1 — CURRICULUM: Following your plan, write the test files. For each \
scenario in your curriculum, write a separate file: `test_curriculum_1.py`, \
`test_curriculum_2.py`, etc. Construct each state directly using the state \
class (do NOT use env.reset()). Each file should construct the state, import \
GeneratedApproach from approach.py, instantiate it, call reset(state, {{}}) \
and get_action(state), and assert the expected behavior.

STEP 2 — IMPLEMENT: Write approach.py incrementally. Get \
test_curriculum_1.py to pass first, then test_curriculum_2.py, and so on. \
Run each test to verify before moving to the next.

STEP 3 — INTEGRATION TEST: After all curriculum tests pass, write a final \
integration test that runs the full approach in the real environment.

Structure your code modularly:
- Break complex logic into small helper functions with clear names.
- Each function should have a single responsibility.
- Avoid deeply nested logic; extract inner blocks into named functions.

IMPORTANT: Use `{python_executable}` to run your test scripts, since that \
interpreter has all required packages installed. For example:
```bash
{python_executable} test_curriculum_1.py
```

You can also inspect the source code of any imported module to understand \
the environment's dynamics in detail (reward function, transition logic, \
termination conditions, etc.). To locate a module's source file:
```bash
{python_executable} -c "import some_module; print(some_module.__file__)"
```
Then read the source to inform your approach.\
"""


def _get_primitive_source(obj: Any) -> Path | None:
    """Resolve the source file for a primitive callable or module."""
    # Unwrap functools.partial to get the underlying function.
    while isinstance(obj, partial):
        obj = obj.func
    try:
        source_file = inspect.getfile(obj)
    except (TypeError, OSError):
        return None
    path = Path(source_file)
    return path if path.exists() else None


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
        required_primitives: list[str] | None = None,
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
        self._required_primitives = required_primitives or []
        self._generated: Any = None

    def train(self) -> None:
        if self._load_dir is not None:
            approach_file = self._load_dir / "sandbox" / "approach.py"
            if not approach_file.exists():
                raise FileNotFoundError(f"No approach file at {approach_file}")
            self._load_generated(approach_file)
            return

        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Resolve primitive source files and copy them into the sandbox.
        init_files: dict[str, Path] = {}
        for _, obj in self._primitives.items():
            source_path = _get_primitive_source(obj)
            if source_path is not None:
                init_files[f"primitives/{source_path.name}"] = source_path

        if self._required_primitives:
            names = ", ".join(f"`{n}`" for n in self._required_primitives)
            required_note = (
                f"Your approach MUST use these primitives: {names}. "
                f"They are essential for solving this environment."
            )
        else:
            required_note = ""

        python_exe = sys.executable
        interface_spec = _INTERFACE_SPEC.format(
            python_executable=python_exe,
            required_primitives_note=required_note,
        )

        if self._env_description_path is not None:
            env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
            prompt = _PROMPT_WITH_DESCRIPTION.format(
                env_description=env_desc, interface_spec=interface_spec
            )
        else:
            prompt = _PROMPT_WITH_SOURCE.format(interface_spec=interface_spec)

        config = SandboxConfig(
            sandbox_dir=sandbox_dir,
            init_files=init_files,
            output_filename="approach.py",
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            model=self._model,
            max_budget_usd=self._max_budget_usd,
        )

        # Write agent logs to a file in the sandbox directory.
        sandbox_logger = logging.getLogger("robocode.utils.sandbox")
        sandbox_logger.setLevel(logging.DEBUG)
        log_path = sandbox_dir / "agent_log.txt"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        sandbox_logger.addHandler(file_handler)
        try:
            result = asyncio.run(run_agent_in_sandbox(config))
        finally:
            sandbox_logger.removeHandler(file_handler)
            file_handler.close()

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
