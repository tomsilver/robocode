"""Parameterized push skill for the PushPullHook2D environment.

Assumes the hook is already grasped by the robot (vacuum on). Moves the
robot+hook to a pre-push configuration and then pushes the movable button
towards the target button using the bottom edge of the hook's side2 arm.
"""

from typing import Optional, Sequence, cast

import numpy as np
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace
from kinder.envs.utils import state_2d_has_collision
from relational_structs import (
    Object,
    ObjectCentricState,
)

from kinder_models.kinematic2d.utils import Kinematic2dRobotController

from robocode.skills.utils import (
    TrajectorySamplingFailure,
    run_motion_planning_for_crv_robot,
)


class GroundPushController(Kinematic2dRobotController):
    """Controller for pushing the movable button with the grasped hook."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._hook = objects[1]
        self._movable_button = objects[2]
        self._target_button = objects[3]
        self._action_space = action_space

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float]:
        """Sample (width_rt, pre_push_dist).

        width_rt: where along the hook's short side (width) the contact is
                  [0.0, 1.0].
        pre_push_dist: standoff distance from the movable button to the
                       hook's pushing edge, in [0, width].
        """
        width_rt = rng.uniform(0.0, 1.0)
        hook_width = x.get(self._hook, "width")
        pre_push_dist = rng.uniform(0.0, 1.0) * hook_width
        return (width_rt, pre_push_dist)

    def _get_vacuum_actions(self) -> tuple[float, float]:
        # Vacuum stays on throughout (hook is grasped).
        return 1.0, 1.0

    def _compute_push_direction(
        self, state: ObjectCentricState
    ) -> tuple[float, float, float]:
        """Return (push_dir_x, push_dir_y, distance) from movable to target."""
        mx = state.get(self._movable_button, "x")
        my = state.get(self._movable_button, "y")
        tx = state.get(self._target_button, "x")
        ty = state.get(self._target_button, "y")
        dx = tx - mx
        dy = ty - my
        dist = float(np.sqrt(dx**2 + dy**2))
        if dist < 1e-8:
            return 1.0, 0.0, dist
        return dx / dist, dy / dist, dist

    def _compute_desired_hook_pose(
        self,
        state: ObjectCentricState,
        width_rt: float,
        pre_push_dist: float,
        push_dir_x: float,
        push_dir_y: float,
    ) -> SE2Pose:
        """Compute the desired hook pose for the pre-push configuration.

        The pushing surface is the outer face of side2 (at x=0 in local
        frame).  Its outward normal points in the local +x direction.
        Side1 (where the robot grasps) extends in -x, keeping the robot
        behind the push surface and within the workspace.

        We orient the hook so that local +x aligns with push_dir, and
        position it so the contact point on the face sits pre_push_dist
        behind the movable button.
        """
        hook_width = state.get(self._hook, "width")
        mx = state.get(self._movable_button, "x")
        my = state.get(self._movable_button, "y")

        # Hook orientation: local +x should align with push_dir.
        # local +x in world = (cos(theta), sin(theta))
        hook_theta = float(np.arctan2(push_dir_y, push_dir_x))

        # Contact point in hook local frame: on side2 outer face (x=0),
        # offset along y by width_rt * width from the origin.
        contact_local = SE2Pose(0.0, -width_rt * hook_width, 0.0)

        # Desired contact point in world: pre_push_dist + button_radius
        # behind the button center so the hook surface clears the button.
        button_radius = state.get(self._movable_button, "radius")
        standoff = pre_push_dist + button_radius
        desired_contact = SE2Pose(
            mx - push_dir_x * standoff,
            my - push_dir_y * standoff,
            hook_theta,
        )

        # hook_pose * contact_local = desired_contact (for position)
        # => hook_pose = desired_contact * inverse(contact_local)
        hook_pose = desired_contact * contact_local.inverse
        return hook_pose

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate waypoints: navigate to pre-push, then push."""
        params = cast(tuple[float, ...], self._current_params)
        width_rt = params[0]
        pre_push_dist = params[1]

        # Current poses.
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        arm_joint = state.get(self._robot, "arm_joint")

        hook_x = state.get(self._hook, "x")
        hook_y = state.get(self._hook, "y")
        hook_theta = state.get(self._hook, "theta")

        cur_hook_pose = SE2Pose(hook_x, hook_y, hook_theta)
        cur_robot_pose = SE2Pose(robot_x, robot_y, robot_theta)

        # Relative transform: hook_to_robot (preserved since hook is grasped).
        hook_to_robot = cur_hook_pose.inverse * cur_robot_pose

        # Push direction and distance.
        push_dir_x, push_dir_y, button_dist = self._compute_push_direction(
            state
        )

        # Desired hook pose at pre-push configuration.
        desired_hook_pose = self._compute_desired_hook_pose(
            state, width_rt, pre_push_dist, push_dir_x, push_dir_y
        )

        # Desired robot pose (preserving hook-robot relative transform).
        target_robot_pose = desired_hook_pose * hook_to_robot

        # Collision check for the pre-push configuration.
        # Must move both robot and hook (since hook is grasped).
        full_state = state.copy()
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            full_state.data.update(init_constant_state.data)

        full_state.set(self._robot, "x", target_robot_pose.x)
        full_state.set(self._robot, "y", target_robot_pose.y)
        full_state.set(self._robot, "theta", target_robot_pose.theta)
        full_state.set(self._robot, "arm_joint", arm_joint)

        full_state.set(self._hook, "x", desired_hook_pose.x)
        full_state.set(self._hook, "y", desired_hook_pose.y)
        full_state.set(self._hook, "theta", desired_hook_pose.theta)

        moving_objects = {self._robot, self._hook}
        static_objects = set(full_state) - moving_objects
        if state_2d_has_collision(full_state, moving_objects, static_objects, {}):
            raise TrajectorySamplingFailure(
                "Pre-push configuration is in collision."
            )

        # Motion plan to the pre-push robot pose.
        mp_state = state.copy()
        if init_constant_state is not None:
            mp_state.data.update(init_constant_state.data)
        collision_free_waypoints = run_motion_planning_for_crv_robot(
            mp_state, self._robot, target_robot_pose, self._action_space
        )
        if collision_free_waypoints is None:
            raise TrajectorySamplingFailure(
                "Failed to find a collision-free path to pre-push pose."
            )

        # Build waypoint list: navigate to pre-push position.
        final_waypoints: list[tuple[SE2Pose, float]] = []
        for wp in collision_free_waypoints:
            final_waypoints.append((wp, arm_joint))
        final_waypoints.append((target_robot_pose, arm_joint))

        # Push phase: move robot along push_dir to push button to target.
        button_radius = state.get(self._movable_button, "radius")
        push_distance = pre_push_dist + button_radius + button_dist + 0.05
        push_end_pose = SE2Pose(
            target_robot_pose.x + push_dir_x * push_distance,
            target_robot_pose.y + push_dir_y * push_distance,
            target_robot_pose.theta,
        )
        final_waypoints.append((push_end_pose, arm_joint))

        return final_waypoints
