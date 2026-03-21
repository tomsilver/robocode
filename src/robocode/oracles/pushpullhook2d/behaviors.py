"""Oracle behaviors for PushPullHook2D.

GraspRotate: approach the hook from its closest side, grasp it,
rotate it to horizontal (theta=0), and position it at the centre
of the bottom half of the world.

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

from robocode.oracles.pushpullhook2d.act_helpers import (
    connecting_waypoints,
    waypoints_to_actions,
)
from robocode.oracles.pushpullhook2d.obs_helpers import (
    HOOK_TARGET_CX,
    HOOK_TARGET_CY,
    TABLE_Y,
    WORLD_WIDTH,
    HookPose,
    RobotPose,
    extract_hook,
    extract_robot,
    hook_grasped_and_horizontal,
    hook_long_arm_center,
    hook_x_extent_at_y,
)
from robocode.oracles.pushpullhook2d.obs_helpers import hook_bbox as _hook_bbox
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


def _hook_origin_after_rotate(
    approach_x: float,
    rotate_y: float,
    approach_theta: float,
    target_theta: float,
    robot: RobotPose,
    hook: HookPose,
    approach_y: float,
) -> tuple[float, float]:
    """Predict the hook origin position after grasping then rotating in place.

    The robot grasps at (approach_x, approach_y) with arm fully extended, then
    retracts the arm, moves to rotate_y, and rotates to target_theta.  This
    function returns the hook origin (x, y) after that rotation.
    """
    # Gripper position at the moment of grasping (arm extended).
    gx = approach_x + robot.arm_length * math.cos(approach_theta)
    gy = approach_y + robot.arm_length * math.sin(approach_theta)

    # Hook-origin offset from the gripper in world frame at grasp time.
    dhx = hook.x - gx
    dhy = hook.y - gy

    # After rotation by (target_theta - approach_theta) the offset in world
    # frame becomes  R(target_theta - approach_theta) @ (dhx, dhy).
    # Since target_theta = approach_theta - hook.theta, the net rotation of the
    # offset is R(-hook.theta).
    cos_h = math.cos(hook.theta)
    sin_h = math.sin(hook.theta)
    rot_dhx = dhx * cos_h + dhy * sin_h
    rot_dhy = -dhx * sin_h + dhy * cos_h

    # After retracting the arm to base_radius and rotating:
    new_gx = approach_x + robot.base_radius * math.cos(target_theta)
    new_gy = rotate_y + robot.base_radius * math.sin(target_theta)

    return (new_gx + rot_dhx, new_gy + rot_dhy)


# ---------------------------------------------------------------------------
# GraspRotate
# ---------------------------------------------------------------------------


class GraspRotate(Behavior[NDArray, NDArray]):
    """Grasp the hook from its closest side and rotate it to horizontal.

    The robot chooses the closer side of the hook (left or right edge),
    navigates there, extends its arm to grasp the hook, retracts the arm,
    moves to a safe height near the table edge, rotates the hook to theta=0
    (horizontal), and finally translates so the hook is centred in the
    bottom half of the world.

    Subgoal  (HookGraspedAndHorizontal): hook held, theta~0, centred.
    Precond: hook is NOT already grasped-and-horizontal.
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

        margin = 0.02
        min_x = robot.base_radius + margin
        max_x = WORLD_WIDTH - robot.base_radius - margin
        min_y = robot.base_radius + margin
        max_y = TABLE_Y - robot.base_radius - margin
        arm_reach = robot.arm_length + 1.5 * robot.gripper_width

        # ---- 1. Determine safe_y and approach side/height -------------------
        # Compute a safe lateral-movement height below the hook so the robot
        # body does not collide with the hook while travelling horizontally.
        _, hook_min_y, _, _ = _hook_bbox(hook)
        safe_y = max(min_y, hook_min_y - robot.base_radius - margin)

        # Use the centre of the long arm as the primary target.  Then query
        # the actual hook polygon at the (clamped) approach y to find the
        # real left/right edge — the bounding box is misleading for rotated
        # L-shapes.
        arm_cx, arm_cy = hook_long_arm_center(hook)

        # Search several candidate y values and pick the approach that is
        # closest to the robot, feasible, and body-collision-free.
        candidate_ys = sorted(
            {
                max(min_y, min(arm_cy, max_y)),
                max_y,
                min_y,
                max(min_y, min((min_y + max_y) / 2, max_y)),
            }
        )

        best_dist = float("inf")
        approach_x = arm_cx - arm_reach
        approach_y = max(min_y, min(arm_cy, max_y))
        approach_theta = 0.0

        for test_y in candidate_ys:
            extent = hook_x_extent_at_y(hook, test_y)
            if extent is None:
                continue
            left_edge, right_edge = extent

            # Left approach (theta=0, arm points right).
            lax = left_edge - arm_reach
            if lax >= min_x and self._body_clears_hook(
                lax, safe_y, test_y, robot, hook, margin
            ):
                d = (robot.x - lax) ** 2 + (robot.y - test_y) ** 2
                if d < best_dist:
                    best_dist = d
                    approach_x, approach_y, approach_theta = lax, test_y, 0.0

            # Right approach (theta=pi, arm points left).
            rax = right_edge + arm_reach
            if rax <= max_x and self._body_clears_hook(
                rax, safe_y, test_y, robot, hook, margin
            ):
                d = (robot.x - rax) ** 2 + (robot.y - test_y) ** 2
                if d < best_dist:
                    best_dist = d
                    approach_x, approach_y, approach_theta = (
                        rax,
                        test_y,
                        float(np.pi),
                    )

        # Clamp for safety (shouldn't be needed but just in case).
        approach_x = max(min_x, min(approach_x, max_x))
        approach_y = max(min_y, min(approach_y, max_y))

        # ---- 2. Target robot theta so hook theta becomes 0 ---------------
        target_theta = approach_theta - hook.theta
        # Normalise to [-pi, pi].
        target_theta = math.remainder(target_theta, 2 * math.pi)

        # ---- 3. Rotation height ------------------------------------------
        # Move near the table edge before rotating so the hook (which may
        # extend ~l1 downward) has room.  The table is FLOOR and the hook is
        # SURFACE, so the hook passes through the table — only walls matter.
        rotate_y = max_y  # as high as possible in bottom half

        # ---- 4. Final robot position after rotation -----------------------
        hook_after_x, hook_after_y = _hook_origin_after_rotate(
            approach_x, rotate_y, approach_theta, target_theta,
            robot, hook, approach_y,
        )

        # Desired hook origin when theta=0, bounding-box centred at target:
        #   centre_x = origin_x - l1/2   =>  origin_x = centre_x + l1/2
        #   centre_y = origin_y - l2/2   =>  origin_y = centre_y + l2/2
        hook_desired_x = HOOK_TARGET_CX + hook.length_side1 / 2
        hook_desired_y = HOOK_TARGET_CY + hook.length_side2 / 2

        final_x = approach_x + (hook_desired_x - hook_after_x)
        final_y = rotate_y + (hook_desired_y - hook_after_y)

        # Clamp to valid region.
        final_x = max(min_x, min(final_x, max_x))
        final_y = max(min_y, min(final_y, max_y))

        # ---- 5. Build waypoint sequence ----------------------------------
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

        key_waypoints = [
            # (0) Current state
            current,
            # (1) Retract arm, go to safe lateral height, set approach theta
            wp(robot.x, safe_y, approach_theta, robot.base_radius, 0.0),
            # (2) Move to approach x
            wp(approach_x, safe_y, approach_theta, robot.base_radius, 0.0),
            # (3) Move to approach y (near hook centre)
            wp(approach_x, approach_y, approach_theta, robot.base_radius, 0.0),
            # (4) Extend arm to reach hook
            wp(approach_x, approach_y, approach_theta, robot.arm_length, 0.0),
            # (5) Vacuum ON — grasp hook
            wp(approach_x, approach_y, approach_theta, robot.arm_length, 1.0),
            # (6) Retract arm (pull hook close to robot)
            wp(approach_x, approach_y, approach_theta, robot.base_radius, 1.0),
            # (7) Move up near table edge for safe rotation
            wp(approach_x, rotate_y, approach_theta, robot.base_radius, 1.0),
            # (8) Rotate to target theta (hook becomes horizontal)
            wp(approach_x, rotate_y, target_theta, robot.base_radius, 1.0),
            # (9) Translate to final position (hook centred in bottom half)
            wp(final_x, final_y, target_theta, robot.base_radius, 1.0),
        ]

        dense = connecting_waypoints(key_waypoints)
        self._actions = waypoints_to_actions(dense)

    @staticmethod
    def _body_clears_hook(
        ax: float,
        safe_y: float,
        approach_y: float,
        robot: RobotPose,
        hook: HookPose,
        margin: float,
    ) -> bool:
        """True if the robot body at *ax* clears the hook along the vertical
        path from *safe_y* to *approach_y*."""
        y_lo = min(safe_y, approach_y)
        y_hi = max(safe_y, approach_y)
        body_left = ax - robot.base_radius - margin
        body_right = ax + robot.base_radius + margin
        for y in np.linspace(y_lo, y_hi, 10):
            extent = hook_x_extent_at_y(hook, float(y))
            if extent is None:
                continue
            if body_left < extent[1] and body_right > extent[0]:
                return False
        return True

    def initializable(self, x: NDArray) -> bool:
        """True when the hook is NOT already grasped, horizontal, and centred."""
        return not hook_grasped_and_horizontal(x)

    def terminated(self, x: NDArray) -> bool:
        """True when the hook is held, horizontal, and roughly centred."""
        return hook_grasped_and_horizontal(x)

    def step(self, x: NDArray) -> NDArray:
        """Pop next action; re-plan if exhausted but not done."""
        if not self._actions:
            self._generate_waypoints(x)
        return self._actions.popleft()
