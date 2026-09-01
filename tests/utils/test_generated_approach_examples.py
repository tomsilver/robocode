"""Worked examples of generated programs the harness accepts and rejects.

The other two test modules check the guards a piece at a time. This one is meant to be
read: each example is a complete program, and each test runs it to an outcome, so the
line between honest play and cheating is visible rather than asserted.

Two invariants are enforced, by two different mechanisms:

* The environment *being scored* is never mutated. There is one path to it from
  generated code -- env-bound primitives close over it -- so it is handed out as a
  read-only view and the mutating entry points raise. Nothing else is restricted: a
  program may build and mutate as many environments of its own as it likes.
* With the ``bilevel_models`` primitive, the program must compose the models itself
  rather than run SeSamE. That has no runtime chokepoint, so it stays a source check,
  applied to ``approach.py`` and every sandbox module it imports.

The environment is :class:`tests.utils.conftest.StatefulGoalEnv`: +0.1 per step,
terminating at 5.0. Honest play takes 51 steps and a teleport takes 1, so the step
count in each test says which happened.
"""

import textwrap
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robocode.utils.episode import load_generated_approach
from robocode.utils.scored_env_guard import ScoredEnvMutationError, readonly_view

# --------------------------------------------------------------------------------
# Programs that are accepted
# --------------------------------------------------------------------------------

HONEST_POLICY = '''
    """Reach the goal through the actions returned, which is all that is asked."""

    import numpy as np


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space

        def reset(self, state, info):
            pass

        def get_action(self, state):
            return np.array([0.1], dtype=np.float32)
'''

PLANS_IN_A_PRIVATE_SIMULATOR = '''
    """Plan in a simulator this program owns, then execute the plan.

    Building a model of the environment and mutating *that* is the ordinary way to
    write a TAMP policy, and it cannot influence scoring. The source scan this
    replaced rejected the program anyway, because the file contains `.set_state`.
    """

    import numpy as np


    class PrivateSim:
        """A model of the dynamics that this program owns and may mutate freely."""

        def __init__(self):
            self.pos = 0.0

        def set_state(self, state):
            self.pos = float(state[0])

        def step(self):
            self.pos = min(10.0, self.pos + 0.1)
            return self.pos


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space
            self.plan_length = None

        def reset(self, state, info):
            sim = PrivateSim()
            sim.set_state(state)  # seed the private model from the observation
            steps = 0
            while sim.step() < 5.0:  # roll the private model out to the goal
                steps += 1
            self.plan_length = steps

        def get_action(self, state):
            return np.array([0.1], dtype=np.float32)
'''

PLANS_IN_A_SIBLING_MODULE = {
    "approach.py": '''
        """The same private-simulator plan, with the planner in its own file.

        The loader puts the sandbox directory on `sys.path` for exactly this, and
        splitting a program across files changes nothing about what it may do.
        """

        import numpy as np

        from private_planner import plan_to_goal


        class GeneratedApproach:
            def __init__(self, action_space, observation_space, primitives):
                self._action_space = action_space
                self.plan_length = None

            def reset(self, state, info):
                self.plan_length = plan_to_goal(state)

            def get_action(self, state):
                return np.array([0.1], dtype=np.float32)
    ''',
    "private_planner.py": '''
        """A private model of the environment, mutated freely while planning."""


        class PrivateSim:
            def __init__(self):
                self.pos = 0.0

            def set_state(self, state):
                self.pos = float(state[0])

            def step(self):
                self.pos = min(10.0, self.pos + 0.1)
                return self.pos


        def plan_to_goal(state):
            sim = PrivateSim()
            sim.set_state(state)
            steps = 0
            while sim.step() < 5.0:
                steps += 1
            return steps
    ''',
}

MENTIONS_A_MUTATOR_IN_A_STRING = '''
    """Naming a mutator is not using one; the old scan rejected this file."""

    import numpy as np


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space

        def reset(self, state, info):
            self.note = "solved by acting, not by env.set_state(goal)"

        def get_action(self, state):
            return np.array([0.1], dtype=np.float32)
'''

READS_THE_SCORED_ENV = '''
    """Pull the scored env out of a primitive's closure and read it.

    Reads, geometry handles and collision queries all pass through the read-only
    view: only the mutators are withheld.
    """

    import numpy as np


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space
            self._primitives = primitives
            self.observed_goal = None

        def reset(self, state, info):
            env = self._primitives["peek"].args[0]
            self.observed_goal = float(env.goal_state()[0])
            self.observed_now = float(env.get_state()[0])

        def get_action(self, state):
            return np.array([0.1], dtype=np.float32)
'''

COMPOSES_THE_BILEVEL_MODELS = '''
    """Use the granted models directly, without running SeSamE over them."""

    import numpy as np


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space
            self.models = primitives["bilevel_models"]

        def reset(self, state, info):
            pass

        def get_action(self, state):
            return np.array([0.1], dtype=np.float32)
'''

# --------------------------------------------------------------------------------
# Programs that are rejected
# --------------------------------------------------------------------------------

TELEPORTS_THROUGH_A_PRIMITIVE = '''
    """Reach the scored env through a primitive's closure and move it to the goal.

    `check_action_collision` is a `partial` over the live env, so a program granted
    an env-bound primitive can read the env back out of `fn.args`. This is the one
    path from generated code to the environment being scored.
    """

    import numpy as np


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space
            self._primitives = primitives

        def reset(self, state, info):
            pass

        def get_action(self, state):
            env = self._primitives["peek"].args[0]
            env.set_state(env.goal_state())  # would fake a solve on the next step
            return np.array([0.1], dtype=np.float32)
'''

TELEPORTS_THROUGH_A_COMPUTED_NAME = '''
    """The same teleport, with the method name never written in the source.

    A scan for the mutator's name finds nothing here. The guard is on the object, so
    how the attribute is spelled makes no difference.
    """

    import numpy as np


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space
            self._primitives = primitives

        def reset(self, state, info):
            pass

        def get_action(self, state):
            env = self._primitives["peek"].args[0]
            mutate = getattr(env, "set_" + "state")
            mutate(env.goal_state())
            return np.array([0.1], dtype=np.float32)
'''

WRITES_AN_ATTRIBUTE_ON_THE_SCORED_ENV = '''
    """Skip the method and assign the underlying attribute instead."""

    import numpy as np


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space
            self._primitives = primitives

        def reset(self, state, info):
            pass

        def get_action(self, state):
            env = self._primitives["peek"].args[0]
            env._pos = 5.0  # desyncs the env from the episode being scored
            return np.array([0.1], dtype=np.float32)
'''

RESETS_THE_SCORED_ENV = '''
    """Restart the episode the harness is in the middle of scoring."""

    import numpy as np


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space
            self._primitives = primitives

        def reset(self, state, info):
            pass

        def get_action(self, state):
            env = self._primitives["peek"].args[0]
            env.reset(seed=0)
            return np.array([0.1], dtype=np.float32)
'''

RUNS_THE_PLANNER = '''
    """With bilevel_models granted, running SeSamE is the thing being measured away."""

    import numpy as np
    from bilevel_planning.sesame import run_sesame


    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives):
            self._action_space = action_space
            self._models = primitives["bilevel_models"]

        def reset(self, state, info):
            self._plan = run_sesame(self._models, state)

        def get_action(self, state):
            return np.array([0.1], dtype=np.float32)
'''

RUNS_THE_PLANNER_FROM_A_SIBLING = {
    "approach.py": '''
        """The planner call moved one file over, which the old scan did not follow."""

        import numpy as np

        from planner import solve


        class GeneratedApproach:
            def __init__(self, action_space, observation_space, primitives):
                self._action_space = action_space
                self._models = primitives["bilevel_models"]

            def reset(self, state, info):
                self._plan = solve(self._models, state)

            def get_action(self, state):
                return np.array([0.1], dtype=np.float32)
    ''',
    "planner.py": '''
        """The planner call itself, out of sight of a scan of approach.py."""

        from bilevel_planning.sesame import run_sesame


        def solve(models, state):
            return run_sesame(models, state)
    ''',
}


def _sandbox(tmp_path: Path, files: Any) -> Path:
    """Write *files* into *tmp_path* and return the path to ``approach.py``."""
    if isinstance(files, str):
        files = {"approach.py": files}
    for name, source in files.items():
        (tmp_path / name).write_text(textwrap.dedent(source).lstrip())
    return tmp_path / "approach.py"


def _scored_primitives(env: Any) -> dict[str, Any]:
    """Mirror ``build_primitives``: bind a primitive to a read-only view of *env*.

    The real dict binds ``check_action_collision``; the name and body do not matter
    here, only that the env is reachable through the closure as ``fn.args[0]``.
    """
    return {"peek": partial(lambda e, s: None, readonly_view(env))}


def _load(path: Path, env: Any, primitives: dict[str, Any] | None = None) -> Any:
    return load_generated_approach(
        path, env.action_space, env.observation_space, primitives or {}
    )


def _rollout(approach: Any, env: Any, max_steps: int = 200) -> tuple[bool, int]:
    """Run one episode the way the harness does; return ``(solved, steps)``."""
    obs, info = env.reset(seed=0)
    approach.reset(obs, info)
    for step in range(1, max_steps + 1):
        obs, _, terminated, truncated, _ = env.step(approach.get_action(obs))
        if terminated or truncated:
            return bool(terminated), step
    return False, max_steps


# --------------------------------------------------------------------------------
# Accepted
# --------------------------------------------------------------------------------


def test_honest_policy_solves_by_acting(tmp_path: Path, goal_env: Any) -> None:
    """The baseline: 51 steps of +0.1 is what solving this env honestly costs."""
    approach = _load(_sandbox(tmp_path, HONEST_POLICY), goal_env)
    assert _rollout(approach, goal_env) == (True, 51)


def test_a_private_simulator_may_be_mutated(tmp_path: Path, goal_env: Any) -> None:
    """The false positive that cost a replicate: planning in an env of one's own.

    The program calls ``set_state``, so the old scan refused to load it. Nothing it
    mutates is the scored env, and it still pays the honest 51 steps.
    """
    approach = _load(_sandbox(tmp_path, PLANS_IN_A_PRIVATE_SIMULATOR), goal_env)
    assert _rollout(approach, goal_env) == (True, 51)
    assert approach.plan_length == 50  # it really did plan in the private model


def test_a_private_planner_may_live_in_a_sibling(tmp_path: Path, goal_env: Any) -> None:
    """Splitting the same program across files changes nothing."""
    approach = _load(_sandbox(tmp_path, PLANS_IN_A_SIBLING_MODULE), goal_env)
    assert _rollout(approach, goal_env) == (True, 51)
    assert approach.plan_length == 50


def test_naming_a_mutator_in_a_string_is_not_cheating(
    tmp_path: Path, goal_env: Any
) -> None:
    """The old scan rejected a string literal; its own test asserted as much."""
    approach = _load(_sandbox(tmp_path, MENTIONS_A_MUTATOR_IN_A_STRING), goal_env)
    assert _rollout(approach, goal_env) == (True, 51)


def test_reading_the_scored_env_is_allowed(tmp_path: Path, goal_env: Any) -> None:
    """Only the mutators are withheld; a program may still look at the env."""
    approach = _load(
        _sandbox(tmp_path, READS_THE_SCORED_ENV),
        goal_env,
        _scored_primitives(goal_env),
    )
    assert _rollout(approach, goal_env) == (True, 51)
    assert approach.observed_goal == pytest.approx(5.0)


def test_composing_the_granted_models_is_allowed(tmp_path: Path, goal_env: Any) -> None:
    """``bilevel_models`` is granted to be used -- just not through the planner."""
    approach = _load(
        _sandbox(tmp_path, COMPOSES_THE_BILEVEL_MODELS),
        goal_env,
        {"bilevel_models": object()},
    )
    assert _rollout(approach, goal_env) == (True, 51)


# --------------------------------------------------------------------------------
# Rejected
# --------------------------------------------------------------------------------


def test_teleport_through_a_primitive_is_blocked(tmp_path: Path, goal_env: Any) -> None:
    """Without the guard this scores solved in one step having done nothing."""
    approach = _load(
        _sandbox(tmp_path, TELEPORTS_THROUGH_A_PRIMITIVE),
        goal_env,
        _scored_primitives(goal_env),
    )
    with pytest.raises(ScoredEnvMutationError):
        _rollout(approach, goal_env)
    assert goal_env.get_state().item() == pytest.approx(0.0)  # never moved


def test_teleport_through_a_computed_name_is_blocked(
    tmp_path: Path, goal_env: Any
) -> None:
    """``set_state`` never appears in this source, and it is still blocked."""
    assert "set_state" not in TELEPORTS_THROUGH_A_COMPUTED_NAME
    approach = _load(
        _sandbox(tmp_path, TELEPORTS_THROUGH_A_COMPUTED_NAME),
        goal_env,
        _scored_primitives(goal_env),
    )
    with pytest.raises(ScoredEnvMutationError):
        _rollout(approach, goal_env)
    assert goal_env.get_state().item() == pytest.approx(0.0)


def test_writing_an_attribute_on_the_scored_env_is_blocked(
    tmp_path: Path, goal_env: Any
) -> None:
    """Assignment is guarded too, or the mutators would just be routed around."""
    approach = _load(
        _sandbox(tmp_path, WRITES_AN_ATTRIBUTE_ON_THE_SCORED_ENV),
        goal_env,
        _scored_primitives(goal_env),
    )
    with pytest.raises(ScoredEnvMutationError):
        _rollout(approach, goal_env)
    assert goal_env.get_state().item() == pytest.approx(0.0)


def test_resetting_the_scored_env_is_blocked(tmp_path: Path, goal_env: Any) -> None:
    """Restarting the episode mid-rollout desyncs it from what is being scored."""
    approach = _load(
        _sandbox(tmp_path, RESETS_THE_SCORED_ENV),
        goal_env,
        _scored_primitives(goal_env),
    )
    with pytest.raises(ScoredEnvMutationError):
        _rollout(approach, goal_env)


def test_running_the_planner_is_rejected_at_load(tmp_path: Path, goal_env: Any) -> None:
    """The bilevel check is a source check, so it fires before the program runs."""
    path = _sandbox(tmp_path, RUNS_THE_PLANNER)
    with pytest.raises(ValueError, match="bilevel planner"):
        _load(path, goal_env, {"bilevel_models": object()})


def test_running_the_planner_from_a_sibling_is_rejected(
    tmp_path: Path, goal_env: Any
) -> None:
    """The gap this PR closes: the old scan read approach.py and nothing else."""
    path = _sandbox(tmp_path, RUNS_THE_PLANNER_FROM_A_SIBLING)
    with pytest.raises(ValueError, match="planner.py"):
        _load(path, goal_env, {"bilevel_models": object()})


def test_a_planner_reference_in_dead_code_is_not_rejected(
    tmp_path: Path, goal_env: Any
) -> None:
    """Only modules the program imports are scanned.

    Agents leave scratch files all over the sandbox. One that nothing imports is not
    part of the program being evaluated, and rejecting the run over it would be the same
    class of false positive this PR is removing.
    """
    (tmp_path / "scratch.py").write_text(
        "from bilevel_planning.sesame import run_sesame\n"
    )
    path = _sandbox(tmp_path, HONEST_POLICY)
    approach = _load(path, goal_env, {"bilevel_models": object()})
    assert _rollout(approach, goal_env) == (True, 51)


def test_the_teleport_would_otherwise_work(goal_env: Any) -> None:
    """Why any of this matters: one unguarded write turns 51 steps into 1."""
    env: Any = goal_env
    env.set_state(env.goal_state())
    _, _, terminated, _, _ = env.step(np.array([0.1], dtype=np.float32))
    assert terminated
