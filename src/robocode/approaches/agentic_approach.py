"""An approach that uses a Claude agent to generate approach code."""

import asyncio
import logging
import sys
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
    "and write an optimal approach class."
)

_INTERFACE_SPEC = """\
Write `approach.py` containing a class `GeneratedApproach` with the following \
interface:

```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space):
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

_PROMPT_WITH_DESCRIPTION = """\
You are writing an approach for ONE specific environment. The environment is \
fully described below \u2014 this is the ONLY environment your code will be \
tested on. Do NOT try to handle other environments or be generic.

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
        visible_filepaths: list[str] | None = None,
        env_description_path: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 50,
        output_dir: str = ".",
        load_dir: str | None = None,
    ) -> None:
        super().__init__(
            action_space,
            observation_space,
            seed,
            visible_filepaths,
            env_description_path,
        )
        self._model = model
        self._max_turns = max_turns
        self._output_dir = Path(output_dir)
        self._load_dir = Path(load_dir) if load_dir is not None else None
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

        # Build the prompt. If we have an env description, inline it so the
        # agent knows exactly which environment to target.  Otherwise fall
        # back to asking the agent to read source files.
        python_exe = sys.executable
        interface_spec = _INTERFACE_SPEC.format(python_executable=python_exe)

        if self._env_description_path is not None:
            env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
            prompt = _PROMPT_WITH_DESCRIPTION.format(
                env_description=env_desc, interface_spec=interface_spec
            )
        else:
            prompt = _PROMPT_WITH_SOURCE.format(interface_spec=interface_spec)

        config = SandboxConfig(
            sandbox_dir=sandbox_dir,
            output_filename="approach.py",
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            model=self._model,
            max_turns=self._max_turns,
        )

        # Write agent logs to a file in the sandbox directory.
        sandbox_logger = logging.getLogger("robocode.utils.sandbox")
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
        self._generated = cls(self._action_space, self._state_space)
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
