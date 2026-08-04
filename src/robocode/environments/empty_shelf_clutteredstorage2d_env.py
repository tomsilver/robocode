"""ClutteredStorage2D variant with every block initially outside the shelf."""

from kinder.envs.kinematic2d.base_env import ObjectCentricKinematic2DRobotEnv
from kinder.envs.kinematic2d.clutteredstorage2d import (
    ClutteredStorage2DEnv,
    ObjectCentricClutteredStorage2DEnv,
)

from robocode.environments.variable_object_count_env import VariableObjectCountEnv


class EmptyShelfObjectCentricClutteredStorage2DEnv(
    ObjectCentricClutteredStorage2DEnv
):
    """Initialize all blocks outside while retaining the standard shelf geometry."""

    def __init__(self, num_blocks: int = 3, **kwargs) -> None:
        super().__init__(num_blocks=num_blocks, **kwargs)
        self._num_init_shelf_blocks = 0
        self._num_init_outside_blocks = num_blocks


class EmptyShelfClutteredStorage2DEnv(ClutteredStorage2DEnv):
    """Constant-object wrapper for the empty-shelf intervention."""

    def __init__(self, num_blocks: int = 3, **kwargs) -> None:
        super().__init__(num_blocks=num_blocks, **kwargs)
        # The parent wrapper uses these only in its natural-language variant
        # description. Keep that description consistent with the object-centric
        # environment created below.
        self._num_init_shelf_blocks = 0
        self._num_init_outside_blocks = num_blocks

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic2DRobotEnv:
        return EmptyShelfObjectCentricClutteredStorage2DEnv(*args, **kwargs)

    def _create_variant_specific_description(self) -> str:
        total = self._num_init_shelf_blocks + self._num_init_outside_blocks
        return f"This variant has {total} blocks, all initially outside the shelf."


class EmptyShelfVariableObjectCountEnv(VariableObjectCountEnv):
    """Variable-count wrapper that makes the empty initial shelf explicit."""

    def _generalization_section(self) -> str:
        return (
            super()._generalization_section()
            + "\n\nIn every configured instance, all blocks begin outside the shelf; "
            "no block initially satisfies the goal."
        )
