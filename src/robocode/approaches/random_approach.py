"""An approach that takes random actions."""

from typing import TypeVar

from gymnasium.spaces import Space
from prpl_utils.gym_agent import Agent

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class RandomApproach(Agent[_ObsType, _ActType]):
    """An approach that takes random actions."""

    def __init__(
        self,
        action_space: Space[_ActType],
        seed: int,
    ) -> None:
        super().__init__(seed)
        self._action_space = action_space

    def _get_action(self) -> _ActType:
        return self._action_space.sample()
