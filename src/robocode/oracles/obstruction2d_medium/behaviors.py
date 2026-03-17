"""Oracle behaviors for Obstruction2D-o2 (medium, 2 obstructions).

Three sequential behaviors that solve the task:
  RemoveObstruction  -> GoalRegionClear
  PickTargetBlock    -> HoldingTarget
  PlaceTargetBlock   -> GoalAchieved

Observation layout (49 features):
  Robot          [0:9]   x y theta base_r arm_j arm_l vac grip_h grip_w
  Target surface [9:19]  x y theta static cr cg cb z w h
  Target block   [19:29] x y theta static cr cg cb z w h
  Obstruction 0  [29:39] x y theta static cr cg cb z w h
  Obstruction 1  [39:49] x y theta static cr cg cb z w h

Position convention: (x, y) is the bottom-left corner of each rectangle.
"""

from __future__ import annotations

from typing import Callable
from numpy.typing import NDArray

from robocode.utils.structs import Behavior
from robocode.oracles.obstruction2d_medium.obs_helpers import *

# ---------------------------------------------------------------------------
# PickTargetBlock
# ---------------------------------------------------------------------------

class PickTargetBlock(Behavior[NDArray, NDArray]):
    """Pick up the target block.

    Subgoal  (HoldingTarget): vacuum on and block lifted off the table.
    Precond: goal region is clear and robot is not holding any block.
    """

    def __init__(self, num_obstructions: int = 2) -> None:
        self._num_obs = num_obstructions
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step


    def reset(self, x: NDArray) -> None:
        """Reset the waypoints for the policy to follow."""


    def _generate_waypoints(self, x: NDArray) -> None:
        """Generate waypoints for the current target block."""


    def initializable(self, x: NDArray) -> bool:
        """Check that the goal region is clear and robot is not already holding any block."""


    def terminated(self, x: NDArray) -> bool:
        """Check if the robot is holding the target block."""


    def step(self, x: NDArray) -> NDArray:
        """Return the next action to execute."""
