"""An approach that uses multiple sequential Claude agent sessions to
iteratively improve generated approach code.

Session 0 generates the initial approach (approach_v0.py).
Session x (x >= 1) loads approach_v(x-1).py, tests it on seeds 0-9,
identifies a failure, debugs and improves it, and saves approach_vx.py.
"""

import logging
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from gymnasium.spaces import Space

from robocode.approaches.base_approach import BaseApproach
from robocode.utils.claude_reset import parse_reset_hour, run_async, seconds_until_reset
from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    DockerSandboxConfig,
    run_agent_in_docker_sandbox,
)
from robocode.utils.sandbox import SandboxConfig, SandboxResult, run_agent_in_sandbox

logger = logging.getLogger(__name__)

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_INITIAL = (
    "You are an expert at writing policies for gymnasium environments. "
    "You will read environment source code, understand the dynamics, "
    "and write an optimal approach class. "
    "IMPORTANT: You MUST write ALL files (approach.py, test scripts, etc.) "
    "to the current working directory using RELATIVE paths only. "
    "Never use absolute paths when writing files. "
    "IMPORTANT: Write code often to approach.py as you iterate. You may be "
    "interrupted at any time, so you should make sure that approach.py is "
    "your best current attempt at all times. "
    "Use the Task tool to explore source code in parallel — e.g. spawn "
    "subagents to read environment dynamics, reward functions, and object "
    "types simultaneously rather than sequentially."
)

_SYSTEM_PROMPT_IMPROVE = (
    "You are an expert at debugging and improving policies for gymnasium "
    "environments. You will be given an existing approach, test it across "
    "multiple seeds, identify a failure case, and fix it. "
    "IMPORTANT: You MUST write ALL files to the current working directory "
    "using RELATIVE paths only. Never use absolute paths when writing files. "
    "IMPORTANT: Write code often to approach.py as you iterate. You may be "
    "interrupted at any time, so you should make sure that approach.py is "
    "your best current attempt at all times. "
    "Use the Task tool to explore source code in parallel."
)

# ---------------------------------------------------------------------------
# Interface spec (shared)
# ---------------------------------------------------------------------------

_INTERFACE_SPEC = """\
Your output must be `approach.py` containing a class `GeneratedApproach` with \
the following interface:

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
        "stays in place). Use it to avoid wasted steps — e.g. in search or "
        "planning algorithms, skip actions that collide."
    ),
    "render_state": (
        "`render_state(state, ax_callback=None) -> np.ndarray` renders the "
        "given `state` as an RGB image (H×W×3 uint8 numpy array). "
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
        "  - `csp.CSPVariable(name, domain)` — a variable with a "
        "`gymnasium.spaces.Space` domain.\n"
        "  - `csp.FunctionalCSPConstraint(name, variables, fn)` — a "
        "constraint where `fn(*vals) -> bool`.\n"
        "  - `csp.CSP(variables, constraints, cost=None)` — the problem.\n"
        "  - `csp.FunctionalCSPSampler(fn, csp, sampled_vars)` — a "
        "sampler where `fn(current_vals, rng) -> dict | None`.\n"
        "  - `csp.RandomWalkCSPSolver(seed)` — solver; call "
        "`.solve(csp, initialization, samplers)` to get a satisfying "
        "assignment or None.\n"
        "  - `csp.CSPCost(name, variables, cost_fn)` — optional cost to "
        "minimize.\n"
        "  - `csp.LogProbCSPConstraint(name, variables, logprob_fn, "
        "threshold)` — constraint from log probabilities.\n"
        "Access via `primitives['csp']`, e.g. "
        "`primitives['csp'].CSPVariable(...)`."
    ),
    "BiRRT": (
        "`BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng, "
        "num_attempts, num_iters, smooth_amt)` — Bidirectional RRT motion "
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
- What kinds of geometric shapes are involved?
- What are the key spatial relationships between objects?
- What geometric constraints exist?
- What kind of motions or transformations are involved?
- What makes a configuration "good" or "bad" geometrically?
- What is the overall geometric strategy?

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
- Write and run a test script for each module BEFORE composing them together.
- Your final `approach.py` should import from these modules and compose them into the \
complete solution. Keep `approach.py` itself as thin as possible.
- Prefer many small files over one large file.
"""

# ---------------------------------------------------------------------------
# Session-specific prompts
# ---------------------------------------------------------------------------

_INITIAL_SESSION_PROMPT = """\
You are writing an approach for the environment described below.

Your approach should be general enough to solve any instance of this environment (env.reset()), \
but it does NOT need to be adaptable to different other environments.

{env_description}
{geometry_prompt}
{interface_spec}
{modular_code_prompt}

This is session 0 of an iterative improvement process. Write the best initial \
approach you can. Your output will be saved as approach_v0.py and improved in \
subsequent sessions.\
"""

_INITIAL_SESSION_PROMPT_NO_DESC = """\
Read the environment source files in this directory to understand the state \
type, action space, and dynamics.
{interface_spec}
{modular_code_prompt}

This is session 0 of an iterative improvement process. Write the best initial \
approach you can. Your output will be saved as approach_v0.py and improved in \
subsequent sessions.\
"""

_IMPROVE_SESSION_PROMPT = """\
This is session {session_id} of an iterative improvement process.

The file `approach_v{prev_id}.py` contains the current best approach (version \
{prev_id}). Your job is to improve it.

Follow these steps IN ORDER:

1. **Read the previous approach**: Read `approach_v{prev_id}.py` and all its \
   helper modules carefully. Understand what it does and how.

2. **Test on multiple seeds**: Write and run a test script that evaluates \
   `approach_v{prev_id}.py` on seeds 0 through 9 (inclusive). For each seed, \
   reset the environment with that seed and run a full episode. Record the \
   total reward (or success/failure) for each seed. Example:
   ```python
   import gymnasium as gym
   # ... set up env and approach ...
   results = {{}}
   for seed in range(10):
       obs, info = env.reset(seed=seed)
       approach.reset(obs, info)
       total_reward = 0
       done = False
       while not done:
           action = approach.get_action(obs)
           obs, reward, terminated, truncated, info = env.step(action)
           total_reward += reward
           done = terminated or truncated
       results[seed] = total_reward
       print(f"Seed {{seed}}: reward={{total_reward}}")
   ```

3. **Identify a failure**: From the test results, pick a seed where the \
   approach performs worst (lowest reward or outright failure). Focus on \
   that specific case.

4. **Debug the failure**: Investigate WHY the approach fails on that seed. \
   Use render_state (if available in primitives) to visualize, add print \
   statements, step through the logic. Understand the root cause.

5. **Fix and improve**: Modify the approach to handle the failure case \
   while preserving behavior on seeds that already work. Write your \
   improved version to `approach.py`.

6. **Verify**: Re-run your test script on ALL seeds 0-9 to confirm the fix \
   helps the failing seed without regressing on others.

7. **Summarize**: If your fix is verified to work, write a brief summary of \
   the key improvements you made and why they work. If your fix did NOT \
   improve results or caused regressions, revert `approach.py` back to the \
   unchanged `approach_v{prev_id}.py` content and instead summarize the \
   current unsolved failure (what seed fails, what the symptoms are, what \
   you tried, and why it didn't work). This summary helps the next session \
   pick up where you left off. Do NOT leave a broken approach — either \
   improve it or keep it unchanged.

{env_description}

{interface_spec}

IMPORTANT: Your final `approach.py` must be a COMPLETE, working approach — \
not a patch or diff. It should be the full improved version.\
"""

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MultiSessionAgenticApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses multiple sequential Claude agent sessions to
    iteratively generate and improve approach code."""

    def __init__(
        self,
        action_space: Space[_ActType],
        observation_space: Space[_ObsType],
        seed: int,
        primitives: dict[str, Callable[..., Any]],
        env_description_path: str | None = None,
        model: str = "sonnet",
        num_sessions: int = 3,
        session_budgets_usd: list[float] | None = None,
        output_dir: str = ".",
        load_dir: str | None = None,
        use_docker: bool = False,
        geometry_prompt: bool = False,
        modular_code_prompt: bool = True,
        max_output_tokens: int = 16384,
        autocompact_pct: int = 80,
        start_session: int = 0,
    ) -> None:
        super().__init__(
            action_space,
            observation_space,
            seed,
            primitives,
            env_description_path,
        )
        self._model = model
        self._num_sessions = num_sessions
        # Per-session budgets: if not provided, default to $2 each.
        if session_budgets_usd is not None:
            self._session_budgets_usd = list(session_budgets_usd)
        else:
            self._session_budgets_usd = [2.0] * num_sessions
        self._output_dir = Path(output_dir)
        self._load_dir = Path(load_dir) if load_dir is not None else None
        self._use_docker = use_docker
        self._geometry_prompt = geometry_prompt
        self._modular_code_prompt = modular_code_prompt
        self._max_output_tokens = max_output_tokens
        self._autocompact_pct = autocompact_pct
        self._start_session = start_session
        self._generated: Any = None
        self.total_cost_usd: float = 0.0
        self.session_costs: list[float] = []

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_primitives_description(self) -> str:
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
            return primitives_desc
        return "`primitives` is an empty dict."

    def _build_interface_spec(self) -> str:
        python_exe = DOCKER_PYTHON if self._use_docker else sys.executable
        return _INTERFACE_SPEC.format(
            python_executable=python_exe,
            primitives_description=self._build_primitives_description(),
        )

    def _build_initial_prompt(self) -> str:
        interface_spec = self._build_interface_spec()
        modular = _MODULAR_CODE_PROMPT if self._modular_code_prompt else ""

        if self._env_description_path is not None:
            env_desc = Path(self._env_description_path).read_text(encoding="utf-8")
            geometry = _GEOMETRY_PROMPT if self._geometry_prompt else ""
            return _INITIAL_SESSION_PROMPT.format(
                env_description=env_desc,
                geometry_prompt=geometry,
                interface_spec=interface_spec,
                modular_code_prompt=modular,
            )
        return _INITIAL_SESSION_PROMPT_NO_DESC.format(
            interface_spec=interface_spec,
            modular_code_prompt=modular,
        )

    def _build_improve_prompt(self, session_id: int) -> str:
        interface_spec = self._build_interface_spec()
        env_desc = ""
        if self._env_description_path is not None:
            env_desc = Path(self._env_description_path).read_text(encoding="utf-8")

        return _IMPROVE_SESSION_PROMPT.format(
            session_id=session_id,
            prev_id=session_id - 1,
            env_description=env_desc,
            interface_spec=interface_spec,
        )

    # ------------------------------------------------------------------
    # Sandbox execution
    # ------------------------------------------------------------------

    def _make_sandbox_config(
        self,
        sandbox_dir: Path,
        prompt: str,
        system_prompt: str,
        budget_usd: float,
        init_files: dict[str, Path] | None = None,
    ) -> SandboxConfig | DockerSandboxConfig:
        kwargs: dict[str, Any] = {
            "sandbox_dir": sandbox_dir,
            "output_filename": "approach.py",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": self._model,
            "max_budget_usd": budget_usd,
            "max_output_tokens": self._max_output_tokens,
            "autocompact_pct": self._autocompact_pct,
        }
        if init_files:
            kwargs["init_files"] = init_files

        if self._use_docker:
            return DockerSandboxConfig(
                **kwargs,
                primitive_names=tuple(self._primitives),
            )
        return SandboxConfig(**kwargs)

    def _run_sandbox(self, config: SandboxConfig | DockerSandboxConfig) -> SandboxResult:
        """Run a single sandbox session with rate-limit retry."""
        while True:
            if isinstance(config, DockerSandboxConfig):
                result = run_async(lambda: run_agent_in_docker_sandbox(config))
            else:
                result = run_async(lambda: run_agent_in_sandbox(config))

            if result.rate_limit_reset is None:
                return result

            reset_hour = parse_reset_hour(result.rate_limit_reset)
            wait_secs = seconds_until_reset(reset_hour)
            hours = wait_secs / 3600
            logger.warning(
                "Rate-limited (%s). Sleeping %.1f hours until %d:05 ...",
                result.error,
                hours,
                reset_hour,
            )
            time.sleep(wait_secs)
            logger.info("Woke up after rate-limit sleep, retrying...")

    # ------------------------------------------------------------------
    # Session runners
    # ------------------------------------------------------------------

    def _run_initial_session(self, sandbox_dir: Path, budget_usd: float) -> SandboxResult:
        """Session 0: generate the initial approach from scratch."""
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        prompt = self._build_initial_prompt()
        config = self._make_sandbox_config(
            sandbox_dir, prompt, _SYSTEM_PROMPT_INITIAL, budget_usd,
        )

        sandbox_logger = self._get_sandbox_logger()
        log_path = sandbox_dir / "agent_log.txt"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        sandbox_logger.addHandler(file_handler)
        try:
            result = self._run_sandbox(config)
        finally:
            sandbox_logger.removeHandler(file_handler)
            file_handler.close()

        return result

    def _run_improve_session(
        self,
        session_id: int,
        sandbox_dir: Path,
        prev_approach: Path,
        prev_sandbox_dir: Path,
        budget_usd: float,
    ) -> SandboxResult:
        """Session x (x >= 1): load previous approach, test, debug, improve."""
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Copy the previous approach and all its helper modules into the new
        # sandbox so the agent can read and test them.
        init_files: dict[str, Path] = {}
        prev_id = session_id - 1

        # Copy the previous approach as approach_v{prev_id}.py
        versioned_name = f"approach_v{prev_id}.py"
        init_files[versioned_name] = prev_approach

        # Also copy any helper .py files from the previous sandbox (excluding
        # test scripts and the approach file itself) so the agent has access to
        # all modules the previous approach may import.
        for py_file in prev_sandbox_dir.glob("**/*.py"):
            rel = py_file.relative_to(prev_sandbox_dir)
            name = str(rel)
            if name in ("approach.py", versioned_name):
                continue
            init_files[name] = py_file

        prompt = self._build_improve_prompt(session_id)
        config = self._make_sandbox_config(
            sandbox_dir, prompt, _SYSTEM_PROMPT_IMPROVE, budget_usd, init_files,
        )

        sandbox_logger = self._get_sandbox_logger()
        log_path = sandbox_dir / "agent_log.txt"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        sandbox_logger.addHandler(file_handler)
        try:
            result = self._run_sandbox(config)
        finally:
            sandbox_logger.removeHandler(file_handler)
            file_handler.close()

        return result

    def _get_sandbox_logger(self) -> logging.Logger:
        if self._use_docker:
            return logging.getLogger("robocode.utils.docker_sandbox")
        return logging.getLogger("robocode.utils.sandbox")

    # ------------------------------------------------------------------
    # Train (main loop)
    # ------------------------------------------------------------------

    def train(self) -> None:
        # If loading from a pre-existing directory, just load the latest version.
        if self._load_dir is not None:
            approach_file = self._find_latest_approach(self._load_dir / "sandbox")
            if approach_file is None:
                raise FileNotFoundError(
                    f"No approach files found in {self._load_dir / 'sandbox'}"
                )
            self._load_generated(approach_file)
            return

        latest_approach: Path | None = None
        latest_sandbox_dir: Path | None = None

        for session_id in range(self._start_session, self._num_sessions):
            sandbox_dir = self._output_dir / f"sandbox_v{session_id}"

            # Determine budget: use per-session list if available, else last entry.
            if session_id < len(self._session_budgets_usd):
                budget = self._session_budgets_usd[session_id]
            else:
                budget = self._session_budgets_usd[-1]

            logger.info(
                "=== Starting session %d / %d (budget: $%.2f) ===",
                session_id,
                self._num_sessions - 1,
                budget,
            )

            if session_id == 0:
                result = self._run_initial_session(sandbox_dir, budget)
            else:
                assert latest_approach is not None
                assert latest_sandbox_dir is not None
                result = self._run_improve_session(
                    session_id, sandbox_dir, latest_approach, latest_sandbox_dir,
                    budget,
                )

            # Track cost
            cost = result.total_cost_usd or 0.0
            self.session_costs.append(cost)
            self.total_cost_usd += cost
            logger.info(
                "Session %d cost: $%.2f (total: $%.2f)",
                session_id,
                cost,
                self.total_cost_usd,
            )

            if result.success and result.output_file is not None:
                # Save as approach_v{session_id}.py
                versioned = sandbox_dir / f"approach_v{session_id}.py"
                shutil.copy2(result.output_file, versioned)
                latest_approach = result.output_file
                latest_sandbox_dir = sandbox_dir
                logger.info(
                    "Session %d succeeded, saved %s",
                    session_id,
                    versioned,
                )
            else:
                logger.warning(
                    "Session %d failed: %s. Keeping previous version.",
                    session_id,
                    result.error,
                )
                # If the initial session fails, we have nothing to improve.
                if latest_approach is None:
                    logger.error(
                        "Initial session failed — cannot continue iterations."
                    )
                    return

        # Load the best version we have
        if latest_approach is not None:
            self._load_generated(latest_approach)
        else:
            logger.warning("No approach was generated across all sessions.")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _find_latest_approach(sandbox_base: Path) -> Path | None:
        """Find the highest-versioned approach file under a directory tree."""
        candidates: list[tuple[int, Path]] = []
        # Check both flat (sandbox/approach_vN.py) and per-session dirs
        for p in sandbox_base.rglob("approach_v*.py"):
            m = re.search(r"approach_v(\d+)\.py$", p.name)
            if m:
                candidates.append((int(m.group(1)), p))
        # Also check plain approach.py as fallback
        plain = sandbox_base / "approach.py"
        if plain.exists() and not candidates:
            return plain
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    def _load_generated(self, path: Path) -> None:
        """Load a GeneratedApproach class from the given file."""
        sandbox_dir = str(path.parent.resolve())
        if sandbox_dir not in sys.path:
            sys.path.insert(0, sandbox_dir)
        try:
            source = path.read_text()
            namespace: dict[str, Any] = {}
            exec(  # pylint: disable=exec-used
                compile(source, str(path), "exec"), namespace
            )
        finally:
            sys.path.remove(sandbox_dir)
        cls = namespace["GeneratedApproach"]
        self._generated = cls(
            self._action_space,
            self._state_space,
            primitives=self._primitives,
        )
        logger.info("Loaded generated approach from %s", path)

    # ------------------------------------------------------------------
    # Episode interface
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
