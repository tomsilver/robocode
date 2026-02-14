"""An approach that takes random actions."""

from typing import TypeVar

from robocode.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class RandomApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that takes random actions."""

    def _get_action(self) -> _ActType:
        return self._action_space.sample()
