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
from kinder.envs.geom2d.structs import SE2Pose
from numpy.typing import NDArray

from robocode.oracles.pushpullhook2d.act_helpers import (
    DTH_LIM,
    DX_LIM,
    DY_LIM,
    connecting_waypoints,
    waypoints_to_actions,
)
from robocode.oracles.pushpullhook2d.obs_helpers import (
    TABLE_Y,
    WORLD_WIDTH,
    RobotPose,
    both_buttons_pressed,
    buttons_vertically_aligned,
    extract_hook,
    extract_robot,
    get_feature,
    hook_grasped_and_rotated,
    hook_ready_for_pushpull,
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
            wp(
                desired_pose.x,
                desired_pose.y,
                desired_pose.theta,
                robot.arm_joint,
                robot.vacuum,
            ),
        ]

        key_waypoints = travel_waypoints + [
            # Extend arm to reach hook
            wp(
                desired_pose.x,
                desired_pose.y,
                desired_pose.theta,
                robot.arm_length,
                0.0,
            ),
            # Vacuum ON — grasp hook
            wp(
                desired_pose.x,
                desired_pose.y,
                desired_pose.theta,
                robot.arm_length,
                1.0,
            ),
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
        mov_r = get_feature(x, "movable_button", "radius")
        tgt_x = get_feature(x, "target_button", "x")

        margin = 0.02
        min_x = robot.base_radius + margin
        max_x = WORLD_WIDTH - robot.base_radius - margin
        sweep_y = TABLE_Y - robot.base_radius - margin

        hook_pose = SE2Pose(hook.x, hook.y, hook.theta)
        robot_pose = SE2Pose(robot.x, robot.y, robot.theta)
        hook2robot = hook_pose.inverse * robot_pose

        # First regrasp the bottom
        if hook2robot.y > 0:
            regrasp_h2r = SE2Pose(
                x=-robot.base_radius,
                y=robot.arm_length,
                theta=-np.pi / 2,
            )
        else:
            regrasp_h2r = SE2Pose(
                x=-robot.base_radius - hook.width - margin,
                y=-robot.arm_length - hook.width,
                theta=np.pi / 2,
            )
        regrasp_world = hook_pose * regrasp_h2r
        middle_pose_1 = SE2Pose(
            x=regrasp_world.x,
            y=robot_pose.y,
            theta=robot_pose.theta,
        )

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
            current,
            # Move horizontally to middle x (keep current y / theta).
            wp(robot_pose.x, robot_pose.y, robot_pose.theta, robot.arm_joint, 0.0),
            wp(
                middle_pose_1.x,
                middle_pose_1.y,
                middle_pose_1.theta,
                robot.arm_joint,
                0.0,
            ),
            wp(
                regrasp_world.x,
                regrasp_world.y,
                regrasp_world.theta,
                robot.arm_joint,
                0.0,
            ),
            wp(
                regrasp_world.x,
                regrasp_world.y,
                regrasp_world.theta,
                robot.arm_length,
                0.0,
            ),
            wp(
                regrasp_world.x,
                regrasp_world.y,
                regrasp_world.theta,
                robot.arm_length,
                1.0,
            ),
        ]

        # Sweep direction: push button toward target.
        sweep_dir = 1.0 if tgt_x > mov_x else -1.0
        pre_sweep_hook_x = (
            mov_x - (mov_r + margin)
            if tgt_x > mov_x
            else mov_x + (mov_r + margin + hook.width)
        )
        pre_sweep_hook_pose = SE2Pose(pre_sweep_hook_x, hook.y, hook.theta)
        hook2robot = hook_pose.inverse * robot_pose
        pre_sweep_robot_pose = pre_sweep_hook_pose * regrasp_h2r

        key_waypoints.append(
            wp(
                pre_sweep_robot_pose.x,
                pre_sweep_robot_pose.y,
                pre_sweep_robot_pose.theta,
                robot.arm_length,
                1.0,
            )
        )
        key_waypoints.append(
            wp(
                pre_sweep_robot_pose.x,
                sweep_y,
                pre_sweep_robot_pose.theta,
                robot.arm_length,
                1.0,
            )
        )
        key_waypoints.append(
            wp(
                min_x if sweep_dir < 0 else max_x,
                sweep_y,
                pre_sweep_robot_pose.theta,
                robot.arm_length,
                1.0,
            )
        )

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


# ---------------------------------------------------------------------------
# PushPull
# ---------------------------------------------------------------------------

PUSHPULL_HOOK_THETA = math.pi / 2


class PrePushPull(Behavior[NDArray, NDArray]):
    """Push or pull the movable button to overlap with the target button.

    Assumes the buttons are already vertically aligned (from Sweep).

    Steps:
      1. Move vertically down to safe_y, then horizontally to safe_x.
      2. Rotate counterclockwise until hook theta ≈ π/2.
      3. Move to pre-push or pre-pull pose.
      4. Sweep vertically (toward y_min or y_max) to push/pull the button.

    Subgoal:  both buttons pressed (green).
    Precond:  buttons vertically aligned.
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

        hook_min_y = 0.05

        hook_pose = SE2Pose(hook.x, hook.y, hook.theta)
        robot_pose = SE2Pose(robot.x, robot.y, robot.theta)
        hook2robot = hook_pose.inverse * robot_pose
        hook_down_pose = SE2Pose(hook.x, hook_min_y, hook.theta)
        robot_down_pose_calculate = hook_down_pose * hook2robot
        robot_down_pose = SE2Pose(
            x=robot_down_pose_calculate.x,
            y=robot_down_pose_calculate.y,
            theta=robot.theta,
        )

        # First regrasp the bottom
        if hook2robot.y > 0:
            regrasp_h2r = SE2Pose(
                x=-hook.length_side1 / 2,
                y=robot.arm_length,
                theta=-np.pi / 2,
            )
        else:
            regrasp_h2r = SE2Pose(
                x=-hook.length_side1 / 2 - hook.width,
                y=-robot.arm_length - hook.width,
                theta=np.pi / 2,
            )
        regrasp_world = hook_down_pose * regrasp_h2r
        middle_pose_1 = robot_down_pose * SE2Pose(
            x=-robot.arm_length,
            y=0,
            theta=0,
        )

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
        target_theta = regrasp_world.theta - math.pi  # CCW by π → hook to π/2

        key_waypoints = [
            current,
            # Move down, still holding.
            wp(
                robot_down_pose.x,
                robot_down_pose.y,
                robot_down_pose.theta,
                robot.arm_joint,
                1.0,
            ),
            # Release hook.
            wp(
                robot_down_pose.x,
                robot_down_pose.y,
                robot_down_pose.theta,
                robot.arm_joint,
                0.0,
            ),
            # Move to intermediate position.
            wp(
                middle_pose_1.x,
                robot_down_pose.y,
                middle_pose_1.theta,
                robot.base_radius,
                0.0,
            ),
            wp(
                middle_pose_1.x,
                middle_pose_1.y,
                middle_pose_1.theta,
                robot.base_radius,
                0.0,
            ),
            wp(
                middle_pose_1.x,
                regrasp_world.y,
                regrasp_world.theta,
                robot.base_radius,
                0.0,
            ),
            # Extend arm and regrasp.
            wp(
                regrasp_world.x,
                regrasp_world.y,
                regrasp_world.theta,
                robot.arm_length,
                0.0,
            ),
            wp(
                regrasp_world.x,
                regrasp_world.y,
                regrasp_world.theta,
                robot.arm_length,
                1.0,
            ),
            # Retract arm.
            wp(
                regrasp_world.x,
                regrasp_world.y,
                regrasp_world.theta,
                robot.base_radius,
                1.0,
            ),
            # Move to safe position and rotate hook to π/2.
            wp(regrasp_world.x, regrasp_world.y, target_theta, robot.base_radius, 1.0),
        ]

        dense = connecting_waypoints(
            key_waypoints,
            action_limits=(DX_LIM, DY_LIM, DTH_LIM, 0.01),
            rotation_direction="counterclockwise",
        )
        self._actions = waypoints_to_actions(dense)

    def initializable(self, x: NDArray) -> bool:
        """True when the buttons are vertically aligned."""
        return buttons_vertically_aligned(x)

    def terminated(self, x: NDArray) -> bool:
        """True when hook is held and at θ≈π/2."""
        return hook_ready_for_pushpull(x)

    def step(self, x: NDArray) -> NDArray:
        """Pop next action; re-plan if exhausted but not done."""
        if not self._actions:
            self._generate_waypoints(x)
        return self._actions.popleft()


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------


class Pull(Behavior[NDArray, NDArray]):
    """Pull the movable button down to overlap with the target button.

    Uses the hook's L-shape at θ=π/2 to catch the movable button from
    above and pull it downward toward the target.

    Steps:
      1. Re-grasp bottom of hook's long arm.
      2. Move left  (robot_x = tgt_x − L2 − arm_length).
      3. Move up    (hook_y ≥ mov_y + mov_r + hook_width).
      4. Move right (long arm just left of button as a wall).
      5. Move down  (short arm pushes button down to target).

    Subgoal:  both buttons pressed (green).
    Precond:  buttons vertically aligned, mov_y > tgt_y, hook at θ≈π/2.
    """

    def __init__(self) -> None:
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step
        self._actions: deque[NDArray] = deque()
        self._regrasp_done: bool = False

    def reset(self, x: NDArray) -> None:
        self._regrasp_done = False
        self._generate_regrasp_waypoints(x)

    # ------------------------------------------------------------------
    # phase 1 — re-grasp the bottom of the long arm
    # ------------------------------------------------------------------

    def _generate_regrasp_waypoints(self, x: NDArray) -> None:
        robot = extract_robot(x)
        hook = extract_hook(x)

        margin = 0.05
        min_y = robot.base_radius + margin

        # Target: robot at (hook_x − arm_length, min_y, 0) with arm
        # extended right so the gripper touches the arm near its bottom.
        regrasp_x = hook.x - robot.arm_length
        regrasp_y = min_y

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
            current,
            # Release vacuum.
            wp(robot.x, robot.y, robot.theta, robot.arm_joint, 0.0),
            # Retract arm.
            wp(robot.x, robot.y, robot.theta, robot.base_radius, 0.0),
            # Move down to min_y.
            wp(robot.x, min_y, robot.theta, robot.base_radius, 0.0),
            # Move to regrasp x (theta → 0).
            wp(regrasp_x, min_y, 0.0, robot.base_radius, 0.0),
            # Extend arm to reach hook.
            wp(regrasp_x, regrasp_y, 0.0, robot.arm_length, 0.0),
            # Vacuum on — grasp.
            wp(regrasp_x, regrasp_y, 0.0, robot.arm_length, 1.0),
        ]

        dense = connecting_waypoints(
            key_waypoints,
            action_limits=(DX_LIM, DY_LIM, DTH_LIM, 0.01),
        )
        self._actions = waypoints_to_actions(dense)

    # ------------------------------------------------------------------
    # phase 2 — left, up, right, down sweep
    # ------------------------------------------------------------------

    def _generate_sweep_waypoints(self, x: NDArray) -> None:
        robot = extract_robot(x)
        hook = extract_hook(x)

        mov_x = get_feature(x, "movable_button", "x")
        mov_y = get_feature(x, "movable_button", "y")
        mov_r = get_feature(x, "movable_button", "radius")
        tgt_y = get_feature(x, "target_button", "y")
        tgt_x = get_feature(x, "target_button", "x")

        margin = 0.02

        # Rigid hook–robot offsets (constant while vacuum is on).
        hook_dx = hook.x - robot.x  # hook_x = robot_x + hook_dx
        hook_dy = hook.y - robot.y  # hook_y = robot_y + hook_dy

        # Step 2: move left — far enough that the short arm (hook_x + L2)
        # does NOT reach the button during the upward sweep.
        left_x = (mov_x - mov_r - margin) - hook.length_side2 - hook_dx

        # Step 3: move up until hook clears button top + width.
        up_hook_y = mov_y + mov_r + hook.width
        up_y = up_hook_y - hook_dy

        # Step 4: move right (long-arm right edge just left of button).
        right_hook_x = tgt_x - mov_r - margin - hook.width
        right_x = right_hook_x - hook_dx

        # Step 5: move down (short arm pushes button to target).
        down_hook_y = tgt_y
        down_y = down_hook_y - hook_dy

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
            current,
            # Step 2 — left.
            wp(left_x, robot.y, robot.theta, robot.arm_joint, 1.0),
            # Step 3 — up.
            wp(left_x, up_y, robot.theta, robot.arm_joint, 1.0),
            # Step 4 — right.
            wp(right_x, up_y, robot.theta, robot.arm_joint, 1.0),
            # Step 5 — down.
            wp(right_x, down_y, robot.theta, robot.arm_joint, 1.0),
        ]

        dense = connecting_waypoints(
            key_waypoints,
            action_limits=(DX_LIM, DY_LIM, DTH_LIM, 0.01),
        )
        self._actions = waypoints_to_actions(dense)

    # ------------------------------------------------------------------

    def initializable(self, x: NDArray) -> bool:
        """True when buttons are aligned, mov_y > tgt_y, hook at π/2."""
        mov_y = get_feature(x, "movable_button", "y")
        tgt_y = get_feature(x, "target_button", "y")
        return (
            buttons_vertically_aligned(x)
            and mov_y > tgt_y
            and hook_ready_for_pushpull(x)
        )

    def terminated(self, x: NDArray) -> bool:
        """True when both buttons are pressed (green)."""
        return both_buttons_pressed(x)

    def step(self, x: NDArray) -> NDArray:
        """Pop next action; switch to sweep or re-plan if exhausted."""
        if not self._actions:
            if not self._regrasp_done:
                self._regrasp_done = True
                self._generate_sweep_waypoints(x)
            else:
                self._generate_sweep_waypoints(x)
        return self._actions.popleft()


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------


class Push(Behavior[NDArray, NDArray]):
    """Push the movable button up to overlap with the target button.

    Uses the hook's short arm (at θ=π/2) to push the button upward
    from below.  The long arm acts as a left wall.

    Steps:
      1. Re-grasp bottom of hook's long arm.
      2. Move right (long arm just left of button).
      3. Move up   (short arm pushes button up to target).

    Subgoal:  both buttons pressed (green).
    Precond:  buttons vertically aligned, mov_y < tgt_y, hook at θ≈π/2.
    """

    def __init__(self) -> None:
        self.subgoal: Callable[[NDArray], bool] = self.terminated
        self.precondition: Callable[[NDArray], bool] = self.initializable
        self.policy: Callable[[NDArray], NDArray] = self.step
        self._actions: deque[NDArray] = deque()
        self._regrasp_done: bool = False

    def reset(self, x: NDArray) -> None:
        self._regrasp_done = False
        self._generate_regrasp_waypoints(x)

    # ------------------------------------------------------------------
    # phase 1 — re-grasp the bottom of the long arm
    # ------------------------------------------------------------------

    def _generate_regrasp_waypoints(self, x: NDArray) -> None:
        robot = extract_robot(x)
        hook = extract_hook(x)

        margin = 0.05
        min_y = robot.base_radius + margin

        regrasp_x = hook.x - robot.arm_length
        regrasp_y = min_y

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
            current,
            # Release vacuum.
            wp(robot.x, robot.y, robot.theta, robot.arm_joint, 0.0),
            # Retract arm.
            wp(robot.x, robot.y, robot.theta, robot.base_radius, 0.0),
            # Move down to min_y.
            wp(robot.x, min_y, robot.theta, robot.base_radius, 0.0),
            # Move to regrasp x (theta → 0).
            wp(regrasp_x, min_y, 0.0, robot.base_radius, 0.0),
            # Extend arm to reach hook.
            wp(regrasp_x, regrasp_y, 0.0, robot.arm_length, 0.0),
            # Vacuum on — grasp.
            wp(regrasp_x, regrasp_y, 0.0, robot.arm_length, 1.0),
        ]

        dense = connecting_waypoints(
            key_waypoints,
            action_limits=(DX_LIM, DY_LIM, DTH_LIM, 0.01),
        )
        self._actions = waypoints_to_actions(dense)

    # ------------------------------------------------------------------
    # phase 2 — right, up sweep
    # ------------------------------------------------------------------

    def _generate_sweep_waypoints(self, x: NDArray) -> None:
        robot = extract_robot(x)
        hook = extract_hook(x)

        mov_x = get_feature(x, "movable_button", "x")
        mov_r = get_feature(x, "movable_button", "radius")
        tgt_y = get_feature(x, "target_button", "y")

        margin = 0.02

        # Rigid hook–robot offsets (constant while vacuum is on).
        hook_dx = hook.x - robot.x
        hook_dy = hook.y - robot.y

        # Step 2: move right (long-arm right edge just left of button).
        right_hook_x = mov_x - mov_r - margin
        right_x = right_hook_x - hook_dx

        # Step 3: move up (short arm pushes button to target).
        up_hook_y = tgt_y
        up_y = up_hook_y - hook_dy

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
            current,
            # Step 2 — right.
            wp(right_x, robot.y, robot.theta, robot.arm_joint, 1.0),
            # Step 3 — up.
            wp(right_x, up_y, robot.theta, robot.arm_joint, 1.0),
        ]

        dense = connecting_waypoints(
            key_waypoints,
            action_limits=(DX_LIM, DY_LIM, DTH_LIM, 0.01),
        )
        self._actions = waypoints_to_actions(dense)

    # ------------------------------------------------------------------

    def initializable(self, x: NDArray) -> bool:
        """True when mov_y < tgt_y and hook at π/2."""
        mov_y = get_feature(x, "movable_button", "y")
        tgt_y = get_feature(x, "target_button", "y")
        return mov_y < tgt_y and hook_ready_for_pushpull(x)

    def terminated(self, x: NDArray) -> bool:
        """True when both buttons are pressed (green)."""
        return both_buttons_pressed(x)

    def step(self, x: NDArray) -> NDArray:
        """Pop next action; switch to sweep or re-plan if exhausted."""
        if not self._actions:
            if not self._regrasp_done:
                self._regrasp_done = True
                self._generate_sweep_waypoints(x)
            else:
                self._generate_sweep_waypoints(x)
        return self._actions.popleft()
