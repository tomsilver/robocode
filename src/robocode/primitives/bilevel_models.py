"""The `bilevel_models` primitive: the SeSamE planning models for an environment.

Exposes the symbolic layer (predicates, lifted operators, goal) and the
parameterized skills the planner is built from. The query methods take a raw
observation vector and devectorize it internally. The full `SesameModels` bundle is
available as `.models`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robocode.utils.bilevel import build_sesame_models


class BilevelModels:
    """Bilevel planning models for one environment, exposed as building blocks.

    Query methods take a raw observation vector `obs` and devectorize it
    internally. Object arguments are `relational_structs.Object` handles from
    `get_objects`. The full `SesameModels` bundle is available as `.models`.
    """

    def __init__(self, env: Any) -> None:
        # Built lazily: the factory constructs every primitive up front, so this
        # must stay cheap until the models are actually used.
        self._env = env
        self._models: Any | None = None

    @property
    def models(self) -> Any:
        """The `SesameModels` bundle (built once, cached).

        Includes `predicates`, `types`, `operators`, `skills` (each with a
        controller via `.ground(objects).controller`), `transition_fn(state,
        action)`, `state_abstractor`, `goal_deriver`, and the observation/state
        converters `observation_to_state` and `observation_space.vectorize`.
        """
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
        """The lifted operators, one per skill.

        Each has `.name`, `.parameters`, `.preconditions`, `.add_effects`, and
        `.delete_effects`.
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
