"""Parameterized skills for the PushPullHook2D environment."""

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


# Controllers.
class GroundPickController(Kinematic2dRobotController):
    """Controller for moving the robot to pick the hook."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._hook = objects[1]
        self._action_space = action_space

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float, float]:
        """Sample (length_rt, rel_theta, arm_length).

        length_rt: fraction along hook's side1 to place the TCP [0.4, 0.9].
        rel_theta: approach angle, either pi/2 or -pi/2.
        arm_length: desired arm extension length.
        """
        length_rt = rng.uniform(0.4, 0.9)
        rel_theta = rng.choice([np.pi / 2, -np.pi / 2])
        max_arm_length = x.get(self._robot, "arm_length")
        min_arm_length = (
            x.get(self._robot, "base_radius")
            + x.get(self._robot, "gripper_width") / 2
            + 1e-4
        )
        arm_length = rng.uniform(min_arm_length, max_arm_length)
        return (length_rt, rel_theta, arm_length)

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 0.0, 1.0

    def _calculate_grasp_robot_pose(
        self,
        state: ObjectCentricState,
        length_rt: float,
        rel_theta: float,
        arm_length: float,
    ) -> SE2Pose:
        """Calculate robot pose for grasping the hook.

        Uses SE2Pose composition matching _solve_grasp logic:
          hook_pose * SE2Pose(-length_side1 * length_rt, 0, rel_theta) -> tcp_pose
          tcp_pose * SE2Pose(-arm_length - gripper_w - 0.01, 0, 0)    -> robot_pose
        """
        hook_x = state.get(self._hook, "x")
        hook_y = state.get(self._hook, "y")
        hook_theta = state.get(self._hook, "theta")
        hook_length_side1 = state.get(self._hook, "length_side1")
        gripper_w = state.get(self._robot, "gripper_width")

        hook_pose = SE2Pose(hook_x, hook_y, hook_theta)
        hook2tcp = SE2Pose(-hook_length_side1 * length_rt, 0.0, rel_theta)
        tcp_pose = hook_pose * hook2tcp
        tcp2robot = SE2Pose(-arm_length - gripper_w - 0.01, 0.0, 0.0)
        robot_pose = tcp_pose * tcp2robot
        return robot_pose

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate waypoints to the grasp point."""
        params = cast(tuple[float, ...], self._current_params)
        length_rt = params[0]
        rel_theta = params[1]
        desired_arm_length = params[2]
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")

        target_se2_pose = self._calculate_grasp_robot_pose(
            state, length_rt, rel_theta, desired_arm_length
        )

        full_state = state.copy()
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            full_state.data.update(init_constant_state.data)

        # Check if the target pose is collision-free with arm extended
        full_state.set(self._robot, "x", target_se2_pose.x)
        full_state.set(self._robot, "y", target_se2_pose.y)
        full_state.set(self._robot, "theta", target_se2_pose.theta)
        full_state.set(self._robot, "arm_joint", desired_arm_length)

        moving_objects = {self._robot}
        static_objects = set(full_state) - moving_objects
        if state_2d_has_collision(full_state, moving_objects, static_objects, {}):
            raise TrajectorySamplingFailure(
                "Failed to find a collision-free path to target."
            )

        # Plan collision-free waypoints with arm retracted
        mp_state = state.copy()
        mp_state.set(self._robot, "arm_joint", robot_radius)
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            mp_state.data.update(init_constant_state.data)
        collision_free_waypoints = run_motion_planning_for_crv_robot(
            mp_state, self._robot, target_se2_pose, self._action_space
        )

        # First retract arm, then follow planned path, then extend arm
        final_waypoints: list[tuple[SE2Pose, float]] = [
            (SE2Pose(robot_x, robot_y, robot_theta), robot_radius)
        ]

        if collision_free_waypoints is not None:
            for wp in collision_free_waypoints:
                final_waypoints.append((wp, robot_radius))
            final_waypoints.append((target_se2_pose, desired_arm_length))
            return final_waypoints

        raise TrajectorySamplingFailure(
            "Failed to find a collision-free path to target."
        )
