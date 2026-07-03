"""The `bilevel_models` primitive: bilevel planning models as building blocks.

Hands the coding agent the *models* the SeSamE planner is built from -- the
symbolic layer (predicates, lifted operators, goal) plus the parameterized skills
-- WITHOUT the planner itself (`run_sesame`). The agent composes these into one
generalized program instead of running per-instance search. It is env-bound (like
`check_action_collision`): built from the environment's `bilevel_env_name` /
`bilevel_env_model_kwargs` mapping.

Design note: `run_skill` samples one set of continuous parameters and rolls the
skill forward in the ground-truth transition simulator -- convenient, but it
inherits the planner's refinement cost and low per-sample success. The intended
fast path is for the agent to sequence skills by reading the abstract state and
to obtain reliable continuous control elsewhere (e.g. the `crv_motion_planning`
primitive), executing closed-loop against real observations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robocode.utils.bilevel import build_sesame_models

# Cap on the number of low-level actions a single skill rollout may produce
# (matches the planner's default max_skill_horizon).
_MAX_SKILL_HORIZON = 100


class BilevelModels:
    """Bilevel planning models for one environment, exposed as agent building blocks.

    Methods take the raw observation vector (`obs`) the agent already holds and
    devectorize it internally, so the agent never needs the object-centric
    conversion. Object arguments are `relational_structs.Object` handles as
    returned by `get_objects`.
    """

    def __init__(self, env: Any) -> None:
        # Models are built lazily so constructing this for a non-bilevel env (the
        # factory builds every primitive up front) is free until it is used.
        self._env = env
        self._models: Any | None = None

    @property
    def models(self) -> Any:
        """The underlying `SesameModels` bundle (built once, cached)."""
        if self._models is None:
            self._models = build_sesame_models(self._env)
        return self._models

    def _to_state(self, obs: np.ndarray) -> Any:
        return self.models.observation_to_state(obs)

    # -- symbolic layer -----------------------------------------------------

    def get_abstract_state(self, obs: np.ndarray) -> set[Any]:
        """The set of ground atoms (e.g. `Holding(robot, block)`) true in `obs`."""
        return set(self.models.state_abstractor(self._to_state(obs)).atoms)

    def get_goal_atoms(self, obs: np.ndarray) -> set[Any]:
        """The set of ground atoms the task requires (the goal)."""
        return set(self.models.goal_deriver(self._to_state(obs)).atoms)

    @property
    def predicates(self) -> set[Any]:
        """The predicates (classifiers) of the domain."""
        return set(self.models.predicates)

    @property
    def types(self) -> set[Any]:
        """The object types of the domain."""
        return set(self.models.types)

    @property
    def operators(self) -> list[Any]:
        """The lifted operators (name, parameters, preconditions, effects).

        Inspect these to know each skill's name, the object parameter order to
        pass to `run_skill`, and the abstract effect the skill is meant to achieve.
        """
        return [skill.operator for skill in self.models.skills]

    @property
    def skill_names(self) -> list[str]:
        """Sorted names of the available skills (== operator names)."""
        return sorted(skill.operator.name for skill in self.models.skills)

    def get_objects(self, obs: np.ndarray, type_name: str | None = None) -> list[Any]:
        """Objects present in `obs`, optionally filtered to a type (e.g. `"circle"`)."""
        objects = list(self._to_state(obs))
        if type_name is not None:
            objects = [o for o in objects if o.type.name == type_name]
        return objects

    # -- skills -------------------------------------------------------------

    def run_skill(
        self,
        obs: np.ndarray,
        skill_name: str,
        objects: list[Any],
        rng: np.random.Generator | None = None,
    ) -> list[np.ndarray]:
        """Low-level actions for one skill grounded on `objects`.

        Grounds the named skill on `objects` (order must match the operator's
        parameters), samples one set of parameters, and rolls the controller
        forward in the ground-truth transition simulator, returning the action
        sequence. Simulator-backed and single-sample: it may not achieve the
        skill's effect on a given draw. Vary `rng` (or reason about parameters
        directly) for reliability.
        """
        if rng is None:
            rng = np.random.default_rng()
        models = self.models
        state = self._to_state(obs)
        skill = next(s for s in models.skills if s.operator.name == skill_name)
        controller = skill.ground(tuple(objects)).controller
        controller.reset(state, controller.sample_parameters(state, rng))
        actions: list[np.ndarray] = []
        while not controller.terminated() and len(actions) < _MAX_SKILL_HORIZON:
            action = controller.step()
            actions.append(action)
            state = models.transition_fn(state, action)
            controller.observe(state)
        return actions
