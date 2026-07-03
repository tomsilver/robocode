"""Base class for approaches."""

import abc
import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, SupportsFloat, TypeVar

import numpy as np
from gymnasium.spaces import Space

_StateType = TypeVar("_StateType")
_ActType = TypeVar("_ActType")


@dataclass
class InstanceResult:
    """Outcome of a per-instance approach solving a single eval seed.

    Reward/step metrics are nullable because a crashed or failed attempt (no
    program produced, or a scoring crash) has no episode metrics. ``cost_usd``
    is always populated so the runner can charge it against the global budget
    even when the attempt failed.

    ``extras`` carries approach-specific per-instance metrics (e.g. a planner's
    ``planning_time``). Numeric extras are averaged across scored episodes by
    ``run_per_instance_eval`` and surfaced as ``mean_<key>`` in the results.
    """

    solved: bool
    total_reward: float | None
    num_steps: int | None
    cost_usd: float
    crashed: bool = False
    frames: list[Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class BaseApproach(Generic[_StateType, _ActType], abc.ABC):
    """Base class for a sequential decision-making agent."""

    # Per-instance approaches spend eval-time agent budget per seed via
    # ``solve_instance`` instead of training one generalized policy. The runner
    # branches on this flag to choose the eval lifecycle; the two lifecycles are
    # deliberately kept separate (a generalized approach trains once and is then
    # rolled out for free, which is a different experiment).
    per_instance: bool = False

    def __init__(  # pylint: disable=unused-argument
        self,
        action_space: Space[_ActType],
        observation_space: Space[_StateType],
        seed: int,
        primitives: dict[str, Callable[..., Any]],
        env_description_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._state_space = copy.deepcopy(observation_space)
        self._action_space = copy.deepcopy(action_space)
        self._action_space.seed(seed)
        self._env_description_path = env_description_path
        self._primitives = primitives
        self._last_state: _StateType | None = None
        self._last_action: _ActType | None = None
        self._last_reward: SupportsFloat | None = None
        self._last_done: bool | None = None
        self._last_info: dict[str, Any] | None = None

    @abc.abstractmethod
    def _get_action(self) -> _ActType:
        """Produce an action to execute now."""

    def reset(
        self,
        state: _StateType,
        info: dict[str, Any],
    ) -> None:
        """Start a new episode."""
        self._last_state = state
        self._last_info = info

    def step(self) -> _ActType:
        """Get the next action to take."""
        self._last_action = self._get_action()
        return self._last_action

    def update(
        self, state: _StateType, reward: float, done: bool, info: dict[str, Any]
    ) -> None:
        """Record the reward and next state following an action."""
        self._last_state = state
        self._last_info = info
        self._last_reward = reward
        self._last_done = done

    def train(self) -> None:
        """Train the approach.

        Override to implement learning.
        """

    def solve_instance(
        self,
        *,
        env: Any,
        seed: int,
        budget_usd: float,
        output_subdir: Path,
        render: bool = False,
    ) -> InstanceResult:
        """Spend up to ``budget_usd`` of eval-time agent budget on one seed.

        Only implemented by per-instance approaches (``per_instance = True``).
        Generalized approaches train once and are rolled out for free, so they
        leave this unimplemented. ``render`` requests RGB frames on the scored
        episode (for video saving).
        """
        raise NotImplementedError(
            "solve_instance is only implemented by per-instance approaches"
        )

    def seed(self, seed: int) -> None:
        """Reset the random number generator."""
        self._rng = np.random.default_rng(seed)
