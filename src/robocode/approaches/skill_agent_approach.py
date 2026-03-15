"""An approach that uses a Claude agent to iteratively develop skills.

Unlike AgenticApproach which asks the agent to write the entire solution
from scratch, SkillAgenticApproach starts from an initial skill library
and approach, then asks the agent to:

1. Test the initial approach and identify failure modes.
2. Write NEW skills (subclasses of Kinematic2dRobotController) that
   address the failures, with unit tests for each skill.
3. Update the approach to use the augmented skill library.

The user provides an ``initial_skill_dir`` — a directory containing:
  - ``approach.py`` with a ``GeneratedApproach`` class that composes the
    initial skills.
  - One or more skill files (e.g. ``pick_skill.py``, ``push_skill.py``),
    each defining a ``Kinematic2dRobotController`` subclass.

All ``.py`` files from this directory are copied into the sandbox so the
agent can read, run, and modify them.
"""

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space

from robocode.approaches.agentic_approach import (
    _GEOMETRY_PROMPT,
    _run_async,
    _parse_reset_hour,
    _seconds_until_reset,
)
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


_SKILL_AGENT_SYSTEM_PROMPT = """\
You are an expert robot-control engineer. You write parameterized skill \
controllers (subclasses of Kinematic2dRobotController) that are tested \
individually before being composed into a full approach.

IMPORTANT: Write ALL files to the current working directory using RELATIVE \
paths only. Never use absolute paths when writing files.

IMPORTANT: Write code often. You may be interrupted at any time, so make \
sure approach.py always reflects your best current attempt.\
"""

_SKILL_INTERFACE_SPEC = """\
## Skill interface

Every skill MUST be a subclass of `Kinematic2dRobotController`.  The base \
class is already available in the sandbox (see `skills/` directory).

A skill subclass must implement:

```python
class MySkill(Kinematic2dRobotController):
    def __init__(self, objects, action_space, init_constant_state=None):
        super().__init__(objects, action_space, init_constant_state)
        # Store any extra object references from `objects`.

    def sample_parameters(self, x, rng):
        \"\"\"Sample continuous parameters for this skill.
        Return a tuple of floats.\"\"\"
        ...

    def _get_vacuum_actions(self):
        \"\"\"Return (vacuum_during_movement, vacuum_after_movement).
        0.0 = off, 1.0 = on.\"\"\"
        ...

    def _generate_waypoints(self, state):
        \"\"\"Return list[tuple[SE2Pose, float]] — waypoints as
        (robot_pose, arm_length) pairs.\"\"\"
        ...
```

### How to use a skill at runtime

```python
controller = MySkill(objects, action_space, init_constant_state)
params = controller.sample_parameters(state, rng)
controller.reset(state, params)
while not controller.terminated():
    action = controller.step()        # NDArray[float32], shape (5,)
    state, _, terminated, _, _ = env.step(action)
    controller.observe(state)
    if terminated:
        break
```

`sample_parameters` may produce infeasible parameters. Wrap the \
reset+step loop in `try/except TrajectorySamplingFailure` and retry \
with new parameters.

### Utilities available in the sandbox

- `skills/utils.py` — contains `run_motion_planning_for_crv_robot` \
(BiRRT-based motion planner) and `TrajectorySamplingFailure`.
- `SE2Pose(x, y, theta)` — rigid-body pose.  `SE2Pose.inverse` is a \
cached property (NOT a method — no parentheses). Compose with `*`.
- `state_2d_has_collision(state, moving_objects, static_objects, cache)` \
— collision checker.\
"""

_SKILL_AGENT_PROMPT = """\
## Your task

You are given a directory with an initial skill library and an \
`approach.py` that composes them. The approach may fail on some \
environment instances.

### Initial files in the sandbox

The following files have been provided:
{initial_file_list}

Read ALL of them to understand the existing skills and how the approach \
uses them.

### Step 1 — Evaluate the current approach

Run the initial approach on the environment with the \
provided successful initial states and failed initial states. \
Write a test script (e.g. `test_approach.py`) \
that runs the approach and reports success/failure. Summarize \
the common failure modes in the failure instances and how could introducing a new skill help fix them.

### Step 2 — Design and implement a new skill

Based on the failures, design ONE new skill controller (a subclass of \
`Kinematic2dRobotController`) that addresses the most impactful failure.

Write the skill to a new file (e.g. `my_new_skill.py`). Follow the \
conventions in the existing skill files.

### Step 3 — Write a unit test for the new skill

Write a test script (e.g. `test_my_new_skill.py`) that:
1. Creates the environment and resets with the provided initial states.
2. Sets up any preconditions (e.g. if your skill assumes the hook is \
   grasped, run the pick skill first).
3. Instantiates your new skill, samples parameters, and steps it.
4. Asserts a concrete success condition.

Run the test and iterate until it passes. Use `{python_executable}` to \
run scripts.

### Step 4 — Update the approach

Update `approach.py` to incorporate your new skill into the existing \
approach logic. Keep the same `GeneratedApproach` interface:

```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, skills):
        ...
    def reset(self, state, info):
        ...
    def get_action(self, state):
        ...
```

`skills` is a dict mapping skill class names to classes. Your approach \
should instantiate and compose these skills to solve the task.

Re-run your evaluation script from Step 1 to verify the updated \
approach improves over the initial one.

{env_description}
{geometry_prompt}
{skill_interface_spec}
"""

_TARGETED_SKILL_AGENT_PROMPT = """\
## Your task

You are given a directory with an initial skill library and an \
`approach.py` that composes them. The approach currently succeeds on \
some tasks but fails on others. Your job is to analyze the failures, \
write a new skill named **{skill_name}** to fix them, and make sure \
the successful tasks still work.

### Initial files in the sandbox

The following files have been provided:
{initial_file_list}

Read ALL of them to understand the existing skills and how the approach \
uses them.

### Pre-saved initial states

Initial environment states have been provided as `.npy` files in two \
directories:

**Failed tasks** (the approach currently fails on these):
- Directory: `failed_states/`
{failed_state_files}

**Successful tasks** (the approach currently solves these — do NOT break them):
- Directory: `success_states/`
{success_state_files}

IMPORTANT: When creating the environment, you MUST pass \
`allow_state_access=True` so that `set_state` works:

```python
env = ObjectCentricPushPullHook2DEnv(render_mode="rgb_array", allow_state_access=True)
env.reset()  # must call reset once before set_state
```

To load a saved state and reset the environment with it:

```python
import numpy as np
state = np.load("failed_states/init_episode_3.npy", allow_pickle=True).item()
env.unwrapped.set_state(state)
```

### Step 1 — Analyze the failures

Write a test script (e.g. `test_failures.py`) that loads EACH failed \
state from `failed_states/`, runs the current approach, and reports \
what goes wrong. Visualize or log the object positions to understand \
why the approach fails. Summarize the common failure modes.

### Step 2 — Design and implement the `{skill_name}` skill

Based on the failure analysis, design a new skill controller called \
`{skill_name}` (a subclass of `Kinematic2dRobotController`) that \
addresses the most impactful failure mode.

Write the skill to a new file called `{skill_filename}`. Follow the \
conventions in the existing skill files.

### Step 3 — Write a unit test for the new skill

Write a test script (e.g. `test_{skill_filename}`) that:
1. Loads a failed state from `failed_states/`.
2. Resets the environment with that state.
3. Sets up any preconditions (e.g. if your skill assumes the hook is \
   grasped, run the pick skill first).
4. Instantiates your `{skill_name}` skill, samples parameters, and \
   steps it.
5. Asserts a concrete success condition.

Run the test and iterate until it passes. Use `{python_executable}` to \
run scripts.

### Step 4 — Update the approach

Update `approach.py` to incorporate `{skill_name}` into the existing \
approach logic. Keep the same `GeneratedApproach` interface:

```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, skills, initial_constant_state):
        ...
    def reset(self, state, info):
        ...
    def get_action(self, state):
        ...
```

`skills` is a dict mapping skill class names to classes. Your approach \
should instantiate and compose these skills to solve the task.

### Step 5 — Verify

Write a final test script (e.g. `test_all.py`) that:
1. Loads ALL failed states from `failed_states/` and verifies the \
   updated approach now solves them.
2. Loads ALL success states from `success_states/` and verifies the \
   updated approach still solves them (no regressions).

Report the results. The goal is 100% on both sets.

{env_description}
{geometry_prompt}
{skill_interface_spec}
"""


class SkillAgenticApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses a Claude agent to develop new skills.

    Parameters
    ----------
    initial_skill_dir:
        Directory containing the initial ``approach.py`` and skill files.
        Every ``.py`` file in this directory is copied into the sandbox.
    """

    def __init__(
        self,
        action_space: Space[_ActType],
        observation_space: Space[_ObsType],
        seed: int,
        initial_skill_dir: str,
        env_description_path: str | None = None,
        model: str = "sonnet",
        max_budget_usd: float = 5.0,
        output_dir: str = ".",
        load_dir: str | None = None,
        geometry_prompt: bool = False,
        max_output_tokens: int = 16384,
        autocompact_pct: int = 80,
        use_docker: bool = False,
        failed_state_dir: str | None = None,
        success_state_dir: str | None = None,
        primitives: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        super().__init__(
            action_space,
            observation_space,
            seed,
            primitives={},
            env_description_path=env_description_path,
        )
        self._initial_skill_dir = Path(initial_skill_dir)
        self._model = model
        self._max_budget_usd = max_budget_usd
        self._output_dir = Path(output_dir)
        self._load_dir = Path(load_dir) if load_dir is not None else None
        self._geometry_prompt = geometry_prompt
        self._max_output_tokens = max_output_tokens
        self._autocompact_pct = autocompact_pct
        self._use_docker = use_docker
        self._failed_state_dir = (
            Path(failed_state_dir) if failed_state_dir is not None else None
        )
        self._success_state_dir = (
            Path(success_state_dir) if success_state_dir is not None else None
        )
        self._generated: Any = None
        self._skill_classes: dict[str, type] = {}
        self.total_cost_usd: float | None = None

    # ------------------------------------------------------------------
    # Training: run the agent to develop skills
    # ------------------------------------------------------------------

    def train(self) -> None:
        if self._load_dir is not None:
            self._load_from_dir(Path(self._load_dir) / "sandbox")
            return

        sandbox_dir = self._output_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Collect files to seed into the sandbox.
        init_files: dict[str, Path] = {}

        # Copy the entire src/robocode/skills directory into the sandbox
        # so the agent has access to all skill utilities and existing
        # skills.  prpl-mono (kinder, relational_structs, etc.) is
        # already bind-mounted in Docker mode.
        skills_src = Path(__file__).parent.parent / "skills"
        skills_dest = sandbox_dir / "skills"
        if skills_dest.exists():
            shutil.rmtree(skills_dest)
        shutil.copytree(
            skills_src,
            skills_dest,
            ignore=shutil.ignore_patterns("__pycache__"),
        )

        # Copy every .py file from the initial skill directory into the
        # sandbox root so the agent can directly read and modify them.
        if not self._initial_skill_dir.is_dir():
            raise FileNotFoundError(
                f"initial_skill_dir not found: {self._initial_skill_dir}"
            )
        initial_py_files: list[str] = []
        for py_file in sorted(self._initial_skill_dir.glob("*.py")):
            init_files[py_file.name] = py_file
            initial_py_files.append(py_file.name)

        if "approach.py" not in init_files:
            raise FileNotFoundError(
                f"No approach.py found in {self._initial_skill_dir}"
            )

        # Build the file listing for the prompt.
        file_list_str = "\n".join(f"- `{name}`" for name in initial_py_files)

        # Copy init state .npy files into the sandbox if provided.
        failed_state_files_str = ""
        success_state_files_str = ""
        use_targeted = (
            self._failed_state_dir is not None
            or self._success_state_dir is not None
        )

        if self._failed_state_dir is not None:
            dst = sandbox_dir / "failed_states"
            dst.mkdir(exist_ok=True)
            npy_files = sorted(self._failed_state_dir.glob("*.npy"))
            for f in npy_files:
                shutil.copy2(f, dst / f.name)
            failed_state_files_str = "\n".join(
                f"- `failed_states/{f.name}`" for f in npy_files
            )

        if self._success_state_dir is not None:
            dst = sandbox_dir / "success_states"
            dst.mkdir(exist_ok=True)
            npy_files = sorted(self._success_state_dir.glob("*.npy"))
            for f in npy_files:
                shutil.copy2(f, dst / f.name)
            success_state_files_str = "\n".join(
                f"- `success_states/{f.name}`" for f in npy_files
            )

        # Build the prompt.
        env_desc = ""
        if self._env_description_path is not None:
            env_desc = Path(self._env_description_path).read_text(
                encoding="utf-8"
            )
        geometry = _GEOMETRY_PROMPT if self._geometry_prompt else ""

        python_exe = DOCKER_PYTHON if self._use_docker else sys.executable
        if use_targeted:
            # Derive skill name from the last component of the failed
            # state dir (e.g. "failure1" from "init_states/.../failure1").
            if self._failed_state_dir is not None:
                raw_name = self._failed_state_dir.name
            else:
                raw_name = "new_skill"
            # Convert folder name to CamelCase class name and snake_case
            # filename, e.g. "failure1" -> "Failure1Controller" /
            # "failure1_skill.py".
            skill_name = (
                raw_name.replace("_", " ").title().replace(" ", "")
                + "Controller"
            )
            skill_filename = f"{raw_name}_skill.py"

            prompt = _TARGETED_SKILL_AGENT_PROMPT.format(
                python_executable=python_exe,
                initial_file_list=file_list_str,
                failed_state_files=failed_state_files_str,
                success_state_files=success_state_files_str,
                skill_name=skill_name,
                skill_filename=skill_filename,
                env_description=env_desc,
                geometry_prompt=geometry,
                skill_interface_spec=_SKILL_INTERFACE_SPEC,
            )
        else:
            prompt = _SKILL_AGENT_PROMPT.format(
                python_executable=python_exe,
                initial_file_list=file_list_str,
                env_description=env_desc,
                geometry_prompt=geometry,
                skill_interface_spec=_SKILL_INTERFACE_SPEC,
            )

        docker_config: DockerSandboxConfig | None = None
        config: SandboxConfig | None = None
        if self._use_docker:
            docker_config = DockerSandboxConfig(
                sandbox_dir=sandbox_dir,
                init_files=init_files,
                output_filename="approach.py",
                prompt=prompt,
                system_prompt=_SKILL_AGENT_SYSTEM_PROMPT,
                model=self._model,
                max_budget_usd=self._max_budget_usd,
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
                system_prompt=_SKILL_AGENT_SYSTEM_PROMPT,
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
            result = self._run_with_rate_limit_retry(docker_config, config)
        finally:
            sandbox_logger.removeHandler(file_handler)
            file_handler.close()

        self.total_cost_usd = result.total_cost_usd

        if result.success and result.output_file is not None:
            self._load_from_dir(sandbox_dir)
        else:
            logger.warning(
                "Skill agent failed to generate approach: %s", result.error
            )

    def _run_with_rate_limit_retry(
        self,
        docker_config: DockerSandboxConfig | None,
        local_config: SandboxConfig | None,
    ) -> SandboxResult:
        """Run the sandbox, retrying on rate-limit."""
        while True:
            if docker_config is not None:
                result = _run_async(
                    lambda: run_agent_in_docker_sandbox(docker_config)
                )
            else:
                assert local_config is not None
                result = _run_async(
                    lambda: run_agent_in_sandbox(local_config)
                )

            if result.rate_limit_reset is None:
                return result

            reset_hour = _parse_reset_hour(result.rate_limit_reset)
            wait_secs = _seconds_until_reset(reset_hour)
            logger.warning(
                "Rate-limited (%s). Sleeping %.1f hours ...",
                result.error,
                wait_secs / 3600,
            )
            time.sleep(wait_secs)

    # ------------------------------------------------------------------
    # Loading: discover and load skills + approach from the sandbox dir
    # ------------------------------------------------------------------

    def _load_from_dir(self, sandbox_dir: Path) -> None:
        """Load all skill classes and the approach from the sandbox dir."""
        approach_file = sandbox_dir / "approach.py"
        if not approach_file.exists():
            raise FileNotFoundError(f"No approach file at {approach_file}")

        sandbox_str = str(sandbox_dir.resolve())
        if sandbox_str not in sys.path:
            sys.path.insert(0, sandbox_str)
        try:
            self._skill_classes = self._discover_skills(sandbox_dir)
            self._load_approach(approach_file)
        finally:
            if sandbox_str in sys.path:
                sys.path.remove(sandbox_str)

    def _discover_skills(self, sandbox_dir: Path) -> dict[str, type]:
        """Find all Kinematic2dRobotController subclasses in the sandbox.

        Scans every .py file (except tests, approach.py, and utility
        modules) for classes that subclass Kinematic2dRobotController.
        """
        from kinder_models.kinematic2d.utils import (  # pylint: disable=import-outside-toplevel
            Kinematic2dRobotController,
        )

        skip = {"approach.py"}
        skills: dict[str, type] = {}

        for py_file in sorted(sandbox_dir.glob("*.py")):
            if py_file.name in skip or py_file.name.startswith("test_"):
                continue
            try:
                ns: dict[str, Any] = {}
                source = py_file.read_text(encoding="utf-8")
                exec(  # pylint: disable=exec-used
                    compile(source, str(py_file), "exec"), ns
                )
                for obj in ns.values():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, Kinematic2dRobotController)
                        and obj is not Kinematic2dRobotController
                    ):
                        skills[obj.__name__] = obj
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to load skills from %s", py_file, exc_info=True
                )

        logger.info("Discovered skills: %s", list(skills.keys()))
        return skills

    def _load_approach(self, path: Path) -> None:
        """Load the GeneratedApproach class from approach.py."""
        source = path.read_text(encoding="utf-8")
        namespace: dict[str, Any] = {}
        exec(  # pylint: disable=exec-used
            compile(source, str(path), "exec"), namespace
        )
        cls = namespace["GeneratedApproach"]
        self._generated = cls(
            self._action_space,
            self._state_space,
            skills=self._skill_classes,
        )
        logger.info("Loaded approach from %s", path)

    # ------------------------------------------------------------------
    # Episode execution: delegate to the generated approach
    # ------------------------------------------------------------------

    def reset(self, state: _ObsType, info: dict[str, Any]) -> None:
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
