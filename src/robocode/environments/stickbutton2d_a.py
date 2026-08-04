"""StickButton2D with the stick spawn range pinned to x in [3.435, 3.450].

Reproduces the sampler of kindergarden branch ``exp/sb-reset-a`` (f5788362) on top
of this checkout's own kinder install: the two samplers differ only in this one
config field. Evaluation and replay can therefore run both spawn ranges in one
process. Synthesis runs must instead use that branch directly, so the kinder source
the sandboxed agent reads matches the sampler it observes.
"""

from dataclasses import replace

from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnv as _Base
from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnvConfig
from kinder.envs.kinematic2d.structs import SE2Pose

_STANDARD = StickButton2DEnvConfig()
_CONFIG = replace(
    _STANDARD,
    stick_init_pose_bounds=(
        SE2Pose(3.435, _STANDARD.table_pose.y - _STANDARD.stick_shape[1] / 2, 0),
        _STANDARD.stick_init_pose_bounds[1],
    ),
)


class StickButton2DEnv(_Base):
    """StickButton2DEnv with the stick spawn range fixed at construction."""

    def __init__(self, num_buttons: int = 3, **kwargs) -> None:
        super().__init__(num_buttons=num_buttons, config=_CONFIG, **kwargs)
