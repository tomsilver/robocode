"""Oracle behaviors for PushPullHook2D.

GraspRotate: approach the hook from its closest side, grasp the midpoint
of that side, then rotate the hook to theta=-π (vertical with short side
at bottom).

Observation layout (38 features):
  Robot            [0:9]   x y theta base_r arm_j arm_l vac grip_h grip_w
  Hook             [9:20]  x y theta static cr cg cb z w l1 l2
  Movable button   [20:29] x y theta static cr cg cb z radius
  Target button    [29:38] x y theta static cr cg cb z radius

Position convention:
  Robot/buttons (x, y) = centre.  Hook (x, y) = top-right vertex at theta=0.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from kinder.envs.geom2d.structs import SE2Pose

from robocode.oracles.pushpullhook2d.act_helpers import (
    DX_LIM,
    DY_LIM,
    DTH_LIM,
    connecting_waypoints,
    waypoints_to_actions,
)
from robocode.oracles.pushpullhook2d.obs_helpers import (
    TABLE_Y,
    WORLD_WIDTH,
    RobotPose,
    extract_hook,
    extract_robot,
    hook_grasped_and_rotated,
)
from robocode.primitives.behavior import Behavior


def _current_pose(robot: RobotPose) -> RobotPose:
    """Return a copy of *robot* as-is (used as the first waypoint)."""
    return RobotPose(
        x=robot.x,
        y=robot.y,
        theta=robot.theta,
        base_radius=robot.base_radius,
        arm_joint=robot.arm_joint,
        arm_length=robot.arm_length,
        vacuum=robot.vacuum,
        gripper_height=robot.gripper_height,
        gripper_width=robot.gripper_width,
    )


# ---------------------------------------------------------------------------
# GraspRotate
# ---------------------------------------------------------------------------


class GraspRotate(Behavior[NDArray, NDArray]):
    """Grasp the hook from its closest side and rotate it to vertical.

    Steps:
      1. Identify the closest side (left or right edge) of the hook.
      2. Move to grasp the middle point of that side.
      3. Move to a safe y (near the table edge for rotation clearance).
      4. Rotate the hook to theta=-π (vertical with short side at bottom).

    Subgoal:  hook held and theta ≈ -π.
    Precond:  hook is NOT already grasped-and-rotated.
    """

    def __init__(self) -> None:
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step
        self._actions: deque[NDArray] = deque()

    def reset(self, x: NDArray) -> None:
        self._generate_waypoints(x)

    def _generate_waypoints(self, x: NDArray) -> None:
        robot = extract_robot(x)
        hook = extract_hook(x)

        robot_pose = SE2Pose(robot.x, robot.y, robot.theta)
        hook_pose = SE2Pose(hook.x, hook.y, hook.theta)

        hook2robot = hook_pose.inverse * robot_pose
        hook2desired: SE2Pose | None = None
        rotate_theta: float | None = None
        if hook2robot.y > 0:
            # Approach from left side of hook.
            hook2desired = SE2Pose(
                x=-hook.length_side1 / 2,
                y=robot.arm_length - robot.gripper_width,
                theta=-np.pi / 2,
            )
            rotate_theta = np.pi
        else:
            # Approach from right side of hook.
            hook2desired = SE2Pose(
                x=-hook.length_side1 / 2,
                y=-robot.arm_length + robot.gripper_width - hook.width,
                theta=np.pi / 2,
            )
            rotate_theta = 0.0

        assert hook2desired is not None  # mypy
        assert rotate_theta is not None  # mypy
        desired_pose = hook_pose * hook2desired
        safe_pose = SE2Pose(
            x=WORLD_WIDTH / 2,
            y=TABLE_Y * 0.6,  # safe y for rotation clearance
            theta=rotate_theta,
        )

        # ---- 4. Build waypoint sequence ----------------------------------
        def wp(
            px: float,
            py: float,
            theta: float,
            arm_joint: float,
            vacuum: float,
        ) -> RobotPose:
            return RobotPose(
                x=px,
                y=py,
                theta=theta,
                base_radius=robot.base_radius,
                arm_joint=arm_joint,
                arm_length=robot.arm_length,
                vacuum=vacuum,
                gripper_height=robot.gripper_height,
                gripper_width=robot.gripper_width,
            )

        current = _current_pose(robot)

        travel_waypoints = [
            current,
            wp(desired_pose.x, desired_pose.y, desired_pose.theta, robot.arm_joint, robot.vacuum)
        ]


        key_waypoints = travel_waypoints + [
            # Extend arm to reach hook
            wp(desired_pose.x, desired_pose.y, desired_pose.theta, robot.arm_length, 0.0),
            # Vacuum ON — grasp hook
            wp(desired_pose.x, desired_pose.y, desired_pose.theta, robot.arm_length, 1.0),
            # Retract arm (pull hook close)
            wp(safe_pose.x, safe_pose.y, desired_pose.theta, robot.arm_length, 1.0),
            # Move to rotation height
            wp(safe_pose.x, safe_pose.y, desired_pose.theta, robot.base_radius, 1.0),
            # Rotate to target theta (hook becomes vertical)
            wp(safe_pose.x, safe_pose.y, safe_pose.theta, robot.base_radius, 1.0),
        ]

        # Use fine arm steps (0.01) so the arm can gradually extend to
        # the collision boundary instead of overshooting in a single step.
        dense = connecting_waypoints(
            key_waypoints,
            action_limits=(DX_LIM, DY_LIM, DTH_LIM, 0.01),
        )
        self._actions = waypoints_to_actions(dense)

    def initializable(self, x: NDArray) -> bool:
        """True when the hook is NOT already grasped and at target theta."""
        return not hook_grasped_and_rotated(x)

    def terminated(self, x: NDArray) -> bool:
        """True when the hook is held and at theta ≈ -π."""
        return hook_grasped_and_rotated(x)

    def step(self, x: NDArray) -> NDArray:
        """Pop next action; re-plan if exhausted but not done."""
        if not self._actions:
            self._generate_waypoints(x)
        return self._actions.popleft()
