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
    buttons_vertically_aligned,
    extract_hook,
    extract_robot,
    get_feature,
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


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


class Sweep(Behavior[NDArray, NDArray]):
    """Sweep the movable button to vertically align it with the target button.

    The robot holds the hook vertically and sweeps horizontally to push the
    movable button to the same x-position as the target button.

    If the current grasp is too close to the hook bottom for the hook to
    reach the button level, the robot re-grasps at a lower point first.

    Subgoal:  movable button x ≈ target button x.
    Precond:  hook is grasped and at target theta (from GraspRotate).
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

        mov_x = get_feature(x, "movable_button", "x")
        mov_y = get_feature(x, "movable_button", "y")
        mov_r = get_feature(x, "movable_button", "radius")
        tgt_x = get_feature(x, "target_button", "x")

        margin = 0.02
        min_x = robot.base_radius + margin
        max_x = WORLD_WIDTH - robot.base_radius - margin
        sweep_y = TABLE_Y - robot.base_radius - margin

        hook_pose = SE2Pose(hook.x, hook.y, hook.theta)
        robot_pose = SE2Pose(robot.x, robot.y, robot.theta)
        hook2robot = hook_pose.inverse * robot_pose

        # Distance from robot to the arm tip (the sweeping end).
        # In hook local frame the arm tip is at x = -l1, robot is at h2r.x.
        # current_dist = l1 + h2r.x  (positive when robot is between origin and tip).
        current_dist = hook.length_side1 + hook2robot.x

        # Minimum distance for the hook end to effectively sweep the button.
        min_dist = mov_y - TABLE_Y + 2 * mov_r + robot.base_radius

        # Sweep direction: push button toward target.
        sweep_dir = 1.0 if tgt_x > mov_x else -1.0

        # The hook arm is offset from the robot in x.  At theta ≈ -π/2 the
        # arm's pushing edge (left for rightward sweep, right for leftward)
        # is at:  robot_x + pushing_edge_offset.
        # hook_right_x = robot_x - h2r.y;  hook_left_x = hook_right_x - w.
        pushing_edge_offset = (
            -hook2robot.y - hook.width if sweep_dir > 0 else -hook2robot.y
        )

        # Pre-sweep: position pushing edge just behind the button.
        pre_sweep_x = (
            mov_x - sweep_dir * (mov_r + margin) - pushing_edge_offset
        )
        pre_sweep_x = max(min_x, min(pre_sweep_x, max_x))

        # Sweep end: push the button past the target x.
        sweep_end_x = (
            tgt_x + sweep_dir * (mov_r + margin) - pushing_edge_offset
        )
        sweep_end_x = max(min_x, min(sweep_end_x, max_x))

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

        if current_dist >= min_dist:
            # Good distance — move to pre-sweep then sweep.
            key_waypoints = [
                current,
                # Move horizontally to pre-sweep x (keep current y / theta).
                wp(pre_sweep_x, robot.y, robot.theta, robot.arm_joint, 1.0),
                # Move up to sweep y.
                wp(pre_sweep_x, sweep_y, robot.theta, robot.arm_joint, 1.0),
                # Sweep toward target x.
                wp(sweep_end_x, sweep_y, robot.theta, robot.arm_joint, 1.0),
            ]
        else:
            # Need to re-grasp closer to the hook origin so more of the
            # arm extends above the robot toward the button.
            # Target: l1 + new_h2r_x = min_dist  →  new_h2r_x = min_dist - l1.
            regrasp_h2r_x = min_dist - hook.length_side1 - margin

            # Compute re-grasp pose in hook frame.
            hook2regrasp = SE2Pose(
                x=regrasp_h2r_x,
                y=hook2robot.y,
                theta=hook2robot.theta,
            )
            regrasp_world = hook_pose * hook2regrasp

            key_waypoints = [
                current,
                # 1) Move horizontally to pre-sweep x (still holding hook).
                wp(pre_sweep_x, robot.y, robot.theta, robot.arm_joint, 1.0),
                # 2) Release vacuum.
                wp(pre_sweep_x, robot.y, robot.theta, robot.arm_joint, 0.0),
                # 3) Move to re-grasp position.
                wp(
                    regrasp_world.x,
                    regrasp_world.y,
                    regrasp_world.theta,
                    robot.base_radius,
                    0.0,
                ),
                # 4) Extend arm to hook.
                wp(
                    regrasp_world.x,
                    regrasp_world.y,
                    regrasp_world.theta,
                    robot.arm_length,
                    0.0,
                ),
                # 5) Vacuum ON — re-grasp.
                wp(
                    regrasp_world.x,
                    regrasp_world.y,
                    regrasp_world.theta,
                    robot.arm_length,
                    1.0,
                ),
                # 6) Retract arm.
                wp(
                    regrasp_world.x,
                    regrasp_world.y,
                    regrasp_world.theta,
                    robot.base_radius,
                    1.0,
                ),
                # 7) Move vertically to sweep y.
                wp(pre_sweep_x, sweep_y, robot.theta, robot.base_radius, 1.0),
                # 8) Sweep toward target x.
                wp(sweep_end_x, sweep_y, robot.theta, robot.base_radius, 1.0),
            ]

        dense = connecting_waypoints(
            key_waypoints,
            action_limits=(DX_LIM, DY_LIM, DTH_LIM, 0.01),
        )
        self._actions = waypoints_to_actions(dense)

    def initializable(self, x: NDArray) -> bool:
        """True when the hook is grasped and at target theta."""
        return hook_grasped_and_rotated(x)

    def terminated(self, x: NDArray) -> bool:
        """True when the movable button is vertically aligned with target."""
        return buttons_vertically_aligned(x)

    def step(self, x: NDArray) -> NDArray:
        """Pop next action; re-plan if exhausted but not done."""
        if not self._actions:
            self._generate_waypoints(x)
        return self._actions.popleft()
