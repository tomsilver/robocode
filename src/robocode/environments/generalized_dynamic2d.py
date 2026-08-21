"""Adapters for count-parameterized Dynamic2D environment families."""

from typing import Any

from kinder.envs.dynamic2d.dyn_scooppour2d import DynScoopPour2DEnv


class CountParameterizedDynScoopPour2DEnv(DynScoopPour2DEnv):
    """Expose ScoopPour's total small-object count as one constructor argument."""

    def __init__(self, num_small_objects: int = 10, **kwargs: Any) -> None:
        num_small_circles = num_small_objects // 2
        num_small_squares = num_small_objects - num_small_circles
        super().__init__(
            num_small_circles=num_small_circles,
            num_small_squares=num_small_squares,
            **kwargs,
        )
