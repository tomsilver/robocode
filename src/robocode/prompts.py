"""Centralized prompt text and composition for the LLM-agent approaches.

Single source of truth for the instructions fed to the coding-agent backends
(``AgenticApproach``, ``AgenticCDLApproach``) and the non-agentic
``LLMGenPlanApproach``. Source-free, like ``primitive_specs``: it imports only
stdlib plus the already-centralized backend and MCP suffix constants, never the
env or primitives packages, so it stays safe to import on the host during
``train()``. Dynamic strings (the primitives description, env-description text,
the python-executable path) are passed in as arguments.

Two axes of variation are kept orthogonal and are NEVER expressed as near-copies:
- approach (agentic / CDL / genplan): legitimately different structural content,
  kept as separate fragments.
- blackbox vs non-blackbox: a small named delta selected by a ``blackbox`` bool.

Shared text lives in exactly one fragment; only the genuine difference gets its
own piece. The four agentic/CDL system prompts, for instance, share their
file-discipline and subagent boilerplate verbatim and differ only in a short
identity sentence and a blackbox-discovery clause, so they are composed from
fragments here rather than stored as four full strings.
"""

from robocode.mcp import (
    MCP_TOOLS_SYSTEM_PROMPT_SUFFIX,
    MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX,
    mcp_tool_descriptions,
)
from robocode.utils.backends import CLAUDE_PROMPT_SUFFIX, OPENCODE_PROMPT_SUFFIX

# ---------------------------------------------------------------------------
# System prompt fragments (shared by AgenticApproach and AgenticCDLApproach)
# ---------------------------------------------------------------------------

# Identity sentence: the one part of the intro that genuinely differs by approach.
_AGENTIC_IDENTITY = "You are an expert at writing policies for gymnasium environments. "
_CDL_IDENTITY = (
    "You are an expert at writing purely imperative, feedforward policies for "
    "gymnasium environments. You decompose tasks into a fixed sequence of "
    "BEHAVIORS, where each behavior is a small, self-contained module with an "
    "explicit precondition, subgoal, and a deterministic policy body. "
)

# Blackbox-discovery clause, shared by both approaches' blackbox intros. No
# trailing punctuation: the per-approach tail supplies it.
_BLACKBOX_DISCOVERY = (
    "The environment is a black box: its source code is not available, so "
    "you will discover the dynamics, reward structure, and termination "
    "conditions empirically by interacting with a live environment instance"
)

# How the agent learns the dynamics and what it produces. Shared by both
# approaches; only the blackbox vs non-blackbox wording differs.
_LEARN_SOURCE = (
    "You will read environment source code, understand the dynamics, "
    "and write an optimal approach class. "
)
_LEARN_BLACKBOX = _BLACKBOX_DISCOVERY + ", and write an optimal approach class. "

# File-writing + version-control discipline. Byte-identical across all four
# agentic/CDL system prompts.
SYSTEM_FILE_DISCIPLINE = (
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
)

# Subagent guidance tails (identical across agentic and CDL). No trailing space:
# the optional extra tail (CDL token budget) or backend suffix begins with one.
SYSTEM_SUBAGENTS = (
    "Use subagents to explore source code in parallel, e.g. spawn "
    "subagents to read environment dynamics, reward functions, and object "
    "types simultaneously rather than sequentially."
)
SYSTEM_SUBAGENTS_BLACKBOX = (
    "Use subagents to run exploration experiments in parallel, e.g. spawn "
    "subagents to probe action effects, reward structure, and termination "
    "conditions simultaneously rather than sequentially; each test script "
    "gets its own fresh environment instance."
)

# Token-budget guidance appended to every system prompt. Leading space joins it
# to the subagent sentence above.
SYSTEM_TOKEN_BUDGET = (
    " TOKEN BUDGET: You have a limited output-token budget per turn. Be concise. "
    "Do NOT write lengthy prose reasoning about geometry or arithmetic — put "
    "all numerical calculations in code (Python scripts or inline print "
    "statements) and read the results. Your text should be SHORT: state what "
    "you will do, then immediately write code. Never narrate step-by-step "
    "arithmetic in text."
)

# Composed intros (single source of truth): per-approach identity + the shared
# learning clause; blackbox swaps only the learning clause.
AGENTIC_INTRO = _AGENTIC_IDENTITY + _LEARN_SOURCE
AGENTIC_INTRO_BLACKBOX = _AGENTIC_IDENTITY + _LEARN_BLACKBOX
CDL_INTRO = _CDL_IDENTITY + _LEARN_SOURCE
CDL_INTRO_BLACKBOX = _CDL_IDENTITY + _LEARN_BLACKBOX

# ---------------------------------------------------------------------------
# Shared body fragments (used by both agentic approaches)
# ---------------------------------------------------------------------------

# Appended to the interface spec in non-blackbox mode. Takes {python_executable}.
INSPECT_SOURCE_SUFFIX = """

You can also inspect the source code of any imported module to understand \
the environment's dynamics in detail (reward function, transition logic, \
termination conditions, etc.). To locate a module's source file:
```bash
{python_executable} -c "import some_module; print(some_module.__file__)"
```
Then read the source to inform your approach.\
"""

# How to interact with the live env server in blackbox mode. Takes
# {set_state_note}: empty for the monolithic approach, the behavior-precondition
# note for CDL (the only difference between the two blackbox interaction specs).
BLACKBOX_INTERACTION_SPEC = """\
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
independent environment instance.{set_state_note}

Start by exploring systematically: reset with several seeds, apply \
controlled actions, and study how the observation vector changes to \
identify what each dimension means and how actions affect the state.

CRITICAL: `approach.py` itself must NOT import `env_client`. It will be \
evaluated against the real environment by a separate harness that calls \
`reset(state, info)` and `get_action(state)` directly. Use `env_client` \
ONLY in test and exploration scripts.\
"""

# CDL-only {set_state_note} value for BLACKBOX_INTERACTION_SPEC.
BLACKBOX_SET_STATE_NOTE = (
    " Use `set_state` to put the environment "
    "into the state a behavior's precondition requires when testing it in "
    "isolation."
)

# Prefix for an optional env description in blackbox mode.
BLACKBOX_DESCRIPTION_PREFIX = """\

The environment is described below.

"""

# ---------------------------------------------------------------------------
# AgenticApproach (monolithic policy) fragments
# ---------------------------------------------------------------------------

AGENTIC_INTERFACE_SPEC = """\
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

Write the best approach you can — ideally one that solves the environment \
optimally. Your `approach.py` should only use packages available in the \
current environment. Write test scripts that use the real environment to \
verify your approach works.

IMPORTANT: Use `{python_executable}` to run your test scripts, since that \
interpreter has all required packages installed. For example:
```bash
{python_executable} test_approach.py
```\
"""

# Geometric-reasoning prompt, shared by both approaches.
GEOMETRY_PROMPT = """\

BEFORE writing any code, briefly describe (in 5-10 bullet points) the key \
geometric relationships in this environment:
- What shapes are involved and how they interact spatially.
- What motions/transformations are needed (translate, rotate, extend arm).
- What collision constraints exist (clearances, boundaries).

Keep this analysis SHORT and QUALITATIVE — no numbers, no arithmetic. \
If you need to compute specific positions or offsets, write a Python \
script to do it and print the results. NEVER do arithmetic in text.
"""

MODULAR_CODE_PROMPT = """\

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

_AGENTIC_WITH_DESCRIPTION = """\
You are writing an approach for the environment described below.

Your approach should be general enough to solve any instance of this environment (env.reset()), \
but it does NOT need to be adaptable to different other environments.

{env_description}
{geometry_prompt}
{interface_spec}
{modular_code_prompt}\
"""

_AGENTIC_WITH_SOURCE = """\
Read the environment source files in this directory to understand the state \
type, action space, and dynamics.
{interface_spec}
{modular_code_prompt}\
"""

_AGENTIC_BLACKBOX = """\
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

# ---------------------------------------------------------------------------
# AgenticCDLApproach (behavior-decomposed policy) fragments
# ---------------------------------------------------------------------------

CDL_DECOMPOSITION_PROMPT = """\

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

CDL_BEHAVIOR_IMPLEMENTATION_PROMPT = """\

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

CDL_OBS_INSPECTION_NOTE = """\
BEFORE writing this file, you MUST run: \
``feats = env.unwrapped.observation_space.devectorize(obs)`` \
to inspect the observation structure. This returns a dictionary mapping \
feature names to their values, so you can see exactly what each part of \
the observation vector represents. Use this to determine the correct \
indices, feature names, and semantics — do NOT guess the observation layout. \
"""

CDL_OBS_INSPECTION_NOTE_BLACKBOX = """\
BEFORE writing this file, you MUST map the observation layout empirically \
with env_client: reset with several seeds, perturb the state with \
``set_state``, and observe which observation entries change as you act. \
Do NOT guess the observation layout. \
"""

CDL_INTERFACE_SPEC = """\
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

CDL_INITIAL_HELPERS_PROMPT = """\

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

# Filled into the behavior-impl prompt's {obs_helpers_note}/{act_helpers_note}
# slots when env-specific helper files are provided.
CDL_HELPERS_PROVIDED_NOTE = (
    "This file is ALREADY PROVIDED in your working directory with "
    "correct parsing logic. Read it first, then extend it with any "
    "additional helpers you need. Do NOT rewrite it from scratch."
)

_CDL_WITH_DESCRIPTION = """\
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

_CDL_WITH_SOURCE = """\
Read the environment source files in this directory to understand the state \
type, action space, and dynamics.
{initial_helpers_prompt}
{cdl_decomposition_prompt}
{interface_spec}
{behavior_implementation_prompt}\
"""

_CDL_BLACKBOX = """\
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

# ---------------------------------------------------------------------------
# LLMGenPlanApproach (non-agentic baseline) fragments
# ---------------------------------------------------------------------------
# Kept faithful to the upstream llm-genplan method; textually independent from
# the agentic prompts above (single code block, no test scripts).

GENPLAN_INTERFACE_SPEC = """\
Implement the strategy as a Python class named `GeneratedApproach` in a single \
code block:

```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        \"\"\"action_space and observation_space are the gym spaces above.\"\"\"
        ...

    def reset(self, state, info):
        \"\"\"Called at the start of each episode with the initial observation.\"\"\"
        ...

    def get_action(self, state):
        \"\"\"Return a valid action (matching action_space) for this state.\"\"\"
        ...
```

`state` is a numpy observation matching the observation space. `get_action` is \
called every step and must return an action inside the action space. The class \
may keep internal state between calls (e.g. a precomputed plan). Return ONLY \
the code block; do not write tests or explanations."""

GENPLAN_SUMMARY_PROMPT = "Write a short summary of this environment in words."

GENPLAN_STRATEGY_PROMPT = (
    "There is a simple strategy for solving all instances of this environment "
    "without using search. What is that strategy?"
)


# ---------------------------------------------------------------------------
# Composition helpers (pure: strings/bools in, string out)
# ---------------------------------------------------------------------------


def build_system_prompt(
    *,
    intro: str,
    blackbox: bool,
    backend_name: str,
    mcp_tools: tuple[str, ...] = (),
) -> str:
    """Compose a coding-agent system prompt from shared fragments.

    ``intro`` is one of the composed intro constants (AGENTIC_INTRO[_BLACKBOX],
    CDL_INTRO[_BLACKBOX]). The shared file-discipline block, blackbox-aware
    subagent guidance, token-budget guidance, the backend-specific suffix, and
    the optional MCP-tools suffix follow.
    """
    subagents = SYSTEM_SUBAGENTS_BLACKBOX if blackbox else SYSTEM_SUBAGENTS
    system_prompt = intro + SYSTEM_FILE_DISCIPLINE + subagents + SYSTEM_TOKEN_BUDGET
    if backend_name == "opencode":
        system_prompt += OPENCODE_PROMPT_SUFFIX
    else:
        system_prompt += CLAUDE_PROMPT_SUFFIX
    if mcp_tools:
        system_prompt += (
            MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX
            if blackbox
            else MCP_TOOLS_SYSTEM_PROMPT_SUFFIX
        )
    return system_prompt


def build_interface_spec(
    *,
    template: str,
    python_executable: str,
    primitives_description: str,
    blackbox: bool,
) -> str:
    """Fill an interface-spec template; append the inspect-source suffix when the agent
    can read env source (non-blackbox)."""
    interface_spec = template.format(
        python_executable=python_executable,
        primitives_description=primitives_description,
    )
    if not blackbox:
        interface_spec += INSPECT_SOURCE_SUFFIX.format(
            python_executable=python_executable
        )
    return interface_spec


def build_mcp_tool_lines(
    *,
    mcp_tools: tuple[str, ...],
    backend_name: str,
    blackbox: bool,
) -> str:
    """Return the MCP-tools section appended to the primitives description, or "" when
    no MCP tools are configured."""
    if not mcp_tools:
        return ""
    tool_descs = mcp_tool_descriptions(backend_name, blackbox=blackbox)
    lines = [
        "\n\nYou also have MCP tools for visual debugging (they do NOT "
        "affect your test scripts):\n",
    ]
    for name in mcp_tools:
        if name in tool_descs:
            lines.append(f"- {tool_descs[name]}")
    return "\n".join(lines)


def _env_description_section(blackbox_description: str | None) -> str:
    if blackbox_description is None:
        return ""
    return BLACKBOX_DESCRIPTION_PREFIX + blackbox_description + "\n"


def build_agentic_prompt(
    *,
    blackbox: bool,
    interface_spec: str,
    geometry: bool,
    modular_code: bool,
    env_description: str | None,
) -> str:
    """Compose the monolithic-approach task prompt."""
    geometry_prompt = GEOMETRY_PROMPT if geometry else ""
    modular = MODULAR_CODE_PROMPT if modular_code else ""
    if blackbox:
        return _AGENTIC_BLACKBOX.format(
            env_description_section=_env_description_section(env_description),
            blackbox_interaction_spec=BLACKBOX_INTERACTION_SPEC.format(
                set_state_note=""
            ),
            geometry_prompt=geometry_prompt,
            interface_spec=interface_spec,
            modular_code_prompt=modular,
        )
    if env_description is not None:
        return _AGENTIC_WITH_DESCRIPTION.format(
            env_description=env_description,
            geometry_prompt=geometry_prompt,
            modular_code_prompt=modular,
            interface_spec=interface_spec,
        )
    return _AGENTIC_WITH_SOURCE.format(
        modular_code_prompt=modular,
        interface_spec=interface_spec,
    )


def build_cdl_prompt(
    *,
    blackbox: bool,
    interface_spec: str,
    geometry: bool,
    env_description: str | None,
    has_initial_helpers: bool,
) -> str:
    """Compose the behavior-decomposed-approach task prompt."""
    geometry_prompt = GEOMETRY_PROMPT if geometry else ""
    initial_helpers = CDL_INITIAL_HELPERS_PROMPT if has_initial_helpers else ""
    helpers_note = CDL_HELPERS_PROVIDED_NOTE if has_initial_helpers else ""
    behavior_impl_prompt = CDL_BEHAVIOR_IMPLEMENTATION_PROMPT.format(
        obs_inspection_note=(
            CDL_OBS_INSPECTION_NOTE_BLACKBOX if blackbox else CDL_OBS_INSPECTION_NOTE
        ),
        obs_helpers_note=helpers_note,
        act_helpers_note=helpers_note,
    )
    if blackbox:
        return _CDL_BLACKBOX.format(
            env_description_section=_env_description_section(env_description),
            blackbox_interaction_spec=BLACKBOX_INTERACTION_SPEC.format(
                set_state_note=BLACKBOX_SET_STATE_NOTE
            ),
            initial_helpers_prompt=initial_helpers,
            geometry_prompt=geometry_prompt,
            cdl_decomposition_prompt=CDL_DECOMPOSITION_PROMPT,
            interface_spec=interface_spec,
            behavior_implementation_prompt=behavior_impl_prompt,
        )
    if env_description is not None:
        return _CDL_WITH_DESCRIPTION.format(
            env_description=env_description,
            geometry_prompt=geometry_prompt,
            cdl_decomposition_prompt=CDL_DECOMPOSITION_PROMPT,
            interface_spec=interface_spec,
            behavior_implementation_prompt=behavior_impl_prompt,
            initial_helpers_prompt=initial_helpers,
        )
    return _CDL_WITH_SOURCE.format(
        initial_helpers_prompt=initial_helpers,
        cdl_decomposition_prompt=CDL_DECOMPOSITION_PROMPT,
        interface_spec=interface_spec,
        behavior_implementation_prompt=behavior_impl_prompt,
    )
