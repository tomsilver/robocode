"""An agentic approach that evolves an existing approach for a new environment.

Instead of writing an approach from scratch, the agent receives:
- A working approach from a previous (related) environment
- The source code / description of that previous environment
- The description / source of the current target environment

The agent must understand the differences between environments, then
refactor and evolve the previous approach to solve the new one.
"""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space

from robocode.approaches.agentic_approach import (
    _GEOMETRY_PROMPT,
    _INTERFACE_SPEC,
    _MODULAR_CODE_PROMPT,
    _PRIMITIVE_DESCRIPTIONS,
    AgenticApproach,
)
from robocode.utils.docker_sandbox import DOCKER_PYTHON

logger = logging.getLogger(__name__)

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")

_PROGRESSIVE_SYSTEM_PROMPT = (
    "You are an expert at writing policies for gymnasium environments. "
    "You are given an existing approach that works well in a related "
    "environment. Your job is to understand both environments, identify "
    "the differences, and evolve the existing approach to solve the new "
    "environment — reusing as much of the existing code as possible. "
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

_PROGRESSIVE_PROMPT = """\
You are evolving an existing approach to solve a NEW environment. You have access to:

1. **Previous approach** — A working approach from a related environment. All files \
from the previous sandbox are in the `prev_approach/` directory. Start by reading \
`prev_approach/approach.py` and any modules it imports.

2. **Previous environment** — The environment the previous approach was built for. \
{prev_env_info}

3. **Current environment** — The NEW environment you must solve. \
{curr_env_info}

## Your workflow — follow these steps IN ORDER:

### Step 1: Understand the previous approach
Read `prev_approach/approach.py` and all its supporting modules. Run the previous \
approach in the CURRENT environment to see what works and what breaks. Write a test \
script that instantiates the current environment and runs the previous approach's \
`GeneratedApproach` class (importing from `prev_approach/`). Observe the behavior, \
reward, and any errors.

### Step 2: Understand the environment differences
{env_diff_instructions}

### Step 3: Plan your evolution strategy
Based on the differences, decide:
- Which parts of the previous approach can be REUSED as-is (copy or import them)
- Which parts need ADAPTATION (same logic, different parameters or structure)
- Which parts need to be NEWLY WRITTEN (functionality that didn't exist before)

The goal is MAXIMUM REUSE. Do not rewrite from scratch — evolve.

### Step 4: Implement and test iteratively
- Copy reusable modules from `prev_approach/` to your working directory and adapt them.
- Write new modules only for genuinely new functionality.
- Test each module against the CURRENT environment before composing.
- Write `approach.py` that composes everything into a working solution.
- Run full integration tests with the current environment.

### Step 5: Verify and refine
Run your approach on multiple episodes. Check that it achieves good reward \
consistently. If it fails on some episodes, debug and fix.

{geometry_prompt}
{interface_spec}
{modular_code_prompt}\
"""

_ENV_DIFF_WITH_DESCRIPTIONS = """\
Compare the two environment descriptions carefully. Look for differences in:
- State/observation space (different features, dimensions, ranges)
- Action space (different actions, ranges, semantics)
- Reward structure (different reward functions, sparse vs dense)
- Dynamics (different transition rules, constraints)
- Objects and their properties (new object types, changed shapes, etc.)
- Task objectives (same goal? harder variant? different goal?)

Write out the key differences explicitly before you start coding.\
"""

_ENV_DIFF_WITH_SOURCE = """\
Read the source code of both environments. Compare them to find differences in:
- State/observation space
- Action space
- Reward structure and dynamics
- Objects and constraints
- Task objectives

Write out the key differences explicitly before you start coding.\
"""


class ProgressiveAgenticApproach(AgenticApproach[_ObsType, _ActType]):
    """An approach that evolves a previous approach for a new environment."""

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
        resume_dir: str | None = None,
        resume_env: str | None = None,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            seed=seed,
            primitives=primitives,
            env_description_path=env_description_path,
            model=model,
            max_budget_usd=max_budget_usd,
            output_dir=output_dir,
            load_dir=load_dir,
            use_docker=use_docker,
            geometry_prompt=geometry_prompt,
            modular_code_prompt=modular_code_prompt,
        )
        self._resume_dir = Path(resume_dir) if resume_dir is not None else None
        self._resume_env = resume_env

    def train(self) -> None:
        # If no resume_dir, fall back to the standard agentic approach.
        if self._resume_dir is None:
            logger.info("No resume_dir set, falling back to standard agentic approach")
            super().train()
            return

        if self._load_dir is not None:
            approach_file = self._load_dir / "sandbox" / "approach.py"
            if not approach_file.exists():
                raise FileNotFoundError(f"No approach file at {approach_file}")
            self._load_generated(approach_file)
            return

        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Copy previous approach files into prev_approach/ subdirectory.
        prev_sandbox = self._resume_dir / "sandbox"
        if not prev_sandbox.exists():
            raise FileNotFoundError(
                f"Previous sandbox not found at {prev_sandbox}. "
                f"Expected resume_dir to contain a 'sandbox/' subdirectory."
            )
        prev_dest = sandbox_dir / "prev_approach"
        if prev_dest.exists():
            shutil.rmtree(prev_dest)
        # Copy everything except .git and .claude dirs.
        shutil.copytree(
            prev_sandbox,
            prev_dest,
            ignore=shutil.ignore_patterns(".git", ".claude"),
        )
        logger.info("Copied previous approach from %s to %s", prev_sandbox, prev_dest)

        # Also copy previous environment description if available.
        prev_env_desc_path = self._resume_dir / "env_description.md"
        if prev_env_desc_path.exists():
            shutil.copy2(prev_env_desc_path, sandbox_dir / "prev_env_description.md")

        # Build the prompt.
        prompt = self._build_progressive_prompt(sandbox_dir)

        from robocode.utils.docker_sandbox import (  # pylint: disable=import-outside-toplevel
            DockerSandboxConfig,
        )
        from robocode.utils.sandbox import (  # pylint: disable=import-outside-toplevel
            SandboxConfig,
        )

        docker_config = None
        config = None
        if self._use_docker:
            docker_config = DockerSandboxConfig(
                sandbox_dir=sandbox_dir,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=_PROGRESSIVE_SYSTEM_PROMPT,
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
                system_prompt=_PROGRESSIVE_SYSTEM_PROMPT,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
            )
            sandbox_logger = logging.getLogger("robocode.utils.sandbox")

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

    def _build_progressive_prompt(self, sandbox_dir: Path) -> str:
        """Build the progressive evolution prompt."""
        import sys as _sys  # pylint: disable=import-outside-toplevel

        python_exe = DOCKER_PYTHON if self._use_docker else _sys.executable

        # Build primitives description.
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

        interface_spec = _INTERFACE_SPEC.format(
            python_executable=python_exe,
            primitives_description=primitives_desc,
        )

        # Previous environment info.
        prev_env_desc_file = sandbox_dir / "prev_env_description.md"
        if prev_env_desc_file.exists():
            prev_env_info = "Its description is in `prev_env_description.md`. Read it."
            if self._resume_env:
                prev_env_info += f" The environment config was `{self._resume_env}`."
        elif self._resume_env:
            prev_env_info = (
                f"The previous environment config was `{self._resume_env}`. "
                f"Read the environment source code to understand it."
            )
        else:
            prev_env_info = (
                "Read the previous approach code to infer what environment "
                "it was designed for."
            )

        # Current environment info.
        if self._env_description_path is not None:
            curr_env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
            curr_env_info = "Here is its description:\n\n" + curr_env_desc
            env_diff_instructions = _ENV_DIFF_WITH_DESCRIPTIONS
        else:
            curr_env_info = (
                "Read the environment source files in this directory to "
                "understand the state type, action space, and dynamics."
            )
            env_diff_instructions = _ENV_DIFF_WITH_SOURCE

        geometry = _GEOMETRY_PROMPT if self._geometry_prompt else ""
        modular = _MODULAR_CODE_PROMPT if self._modular_code_prompt else ""

        return _PROGRESSIVE_PROMPT.format(
            prev_env_info=prev_env_info,
            curr_env_info=curr_env_info,
            env_diff_instructions=env_diff_instructions,
            geometry_prompt=geometry,
            interface_spec=interface_spec,
            modular_code_prompt=modular,
        )
