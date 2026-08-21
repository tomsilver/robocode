"""The contract an environment implements to be evaluated across object counts.

Generalized-planning experiments sweep the object count at evaluation, reporting a
solve-rate-vs-count curve and separating counts the approach was designed against
(``design_counts``) from held-out ones (``eval_counts`` beyond them). The runner
needs four things from an environment to do that, and they are gathered here so the
lifecycle is not tied to one environment family.

Implementations must also accept ``options={"object_count": k}`` in ``reset`` to pin
an instance's count, and report the realized count as ``info["object_count"]`` so a
per-episode record carries it.
"""

from __future__ import annotations

import abc
from typing import TypeVar

from robocode.environments.base_env import BaseEnv

_StateType = TypeVar("_StateType")
_ActType = TypeVar("_ActType")


class VariableCountEnv(BaseEnv[_StateType, _ActType], abc.ABC):
    """An environment whose object count varies across resets."""

    @property
    @abc.abstractmethod
    def design_counts(self) -> list[int]:
        """The object counts an unpinned reset samples from."""

    @property
    @abc.abstractmethod
    def eval_counts(self) -> list[int]:
        """The object counts swept at evaluation, held-out ones included."""

    @property
    @abc.abstractmethod
    def current_count(self) -> int:
        """The object count of the current instance."""

    @abc.abstractmethod
    def max_steps_for_count(self, count: int) -> int:
        """Evaluation step budget for an instance of this object count.

        Larger instances need more steps, so the horizon scales with the count;
        otherwise a big instance would be scored as failed just for hitting a fixed cap
        before it could reasonably finish.
        """
