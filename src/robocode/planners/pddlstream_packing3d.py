"""PDDLStream planner for Packing3D, executed against a robocode environment.

``kinder_pddlstream_planning`` ships the Packing3D domain, its streams, and an
open-loop executor, all of which plan on and drive one ``ObjectCentricPacking3DEnv``.
Here that env is a *twin* of the evaluated environment: it is put in the evaluated
instance's state, the plan is computed on it, and every executed action is applied
to the twin and handed to the caller to apply to the evaluated environment, so the
twin's grasp bookkeeping (which the place correction reads) tracks the real rollout.
The evaluated environment alone scores the episode.

``pddlstream`` and ``kinder_pddlstream_planning`` are imported lazily: they are the
optional ``pddlstream`` extra, which compiles FastDownward on install.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from kinder.envs.kinematic3d.packing3d import ObjectCentricPacking3DEnv
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
    remove_fingers_from_extended_joints,
)
from numpy.typing import NDArray
from pybullet_helpers.geometry import multiply_poses
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.joint import get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    smoothly_follow_end_effector_path,
)

PACKING3D_ENV_PATH = "kinder.envs.kinematic3d.packing3d:Packing3DEnv"

# A move_base/pick/place plan as pddlstream returns it: (action name, arguments).
Plan = list[tuple[str, tuple[Any, ...]]]
ActionCallback = Callable[[NDArray[np.float32]], None]


class StopExecution(Exception):
    """Raised by an action callback to end plan execution early."""


class Packing3DPDDLStreamPlanner:
    """Plan with PDDLStream on a twin Packing3D sim and replay the plan through it."""

    def __init__(self, num_parts: int) -> None:
        # Default config, matching the evaluated Packing3D backend.
        self._sim = ObjectCentricPacking3DEnv(
            num_parts=num_parts, allow_state_access=True
        )

    def close(self) -> None:
        """Release the twin's PyBullet client."""
        self._sim.close()

    def state(self) -> Any:
        """The twin's current object-centric state."""
        return self._sim.get_state()

    def sync(self, state: Any) -> None:
        """Put the twin in *state* (the evaluated env's object-centric observation)."""
        self._sim.set_state(state)

    def plan(self, state: Any, max_time: float) -> Plan | None:
        """Solve the instance in *state* with PDDLStream's adaptive algorithm.

        Planning scratch-mutates the twin, so call :meth:`sync` before executing.
        """
        # pylint: disable=import-outside-toplevel
        from kinder_pddlstream_planning.packing3d.run import plan_packing3d

        self.sync(state)
        return plan_packing3d(self._sim, state, max_time=max_time)

    def execute(self, plan: Plan, on_action: ActionCallback) -> None:
        """Replay *plan* on the twin, calling *on_action* with each clipped action.

        The callback applies the action to the evaluated environment and may raise
        :class:`StopExecution` (episode over, horizon reached), which propagates.
        """
        sim = self._sim

        def step(action: Any) -> None:
            clipped = _clip_action(sim, action)
            sim.step(clipped)
            on_action(clipped)

        def step_to_joint_target(
            target_joints: Any, tol: float = 1e-3, max_attempts: int = 5
        ) -> None:
            # Re-derive the delta from the current joints on each attempt: float32
            # rounding and per-step clipping leave a residual that compounds
            # otherwise.
            for _ in range(max_attempts):
                current = remove_fingers_from_extended_joints(
                    sim.robot.arm.get_joint_positions()
                )
                delta = get_jointwise_difference(
                    _joint_infos(sim), list(target_joints[:7]), current
                )
                step([0.0, 0.0, 0.0] + delta + [0.0])
                if max(abs(d) for d in delta) < tol:
                    break

        def run_gripper(close: bool, max_steps: int = 5) -> None:
            action = [0.0] * 10 + [-1.0 if close else 1.0]
            for _ in range(max_steps):
                step(action)
                grasped = sim._grasped_object  # pylint: disable=protected-access
                if close and grasped is not None:
                    break
                if not close and grasped is None:
                    break

        for name, args in plan:
            if name == "move_base":
                _, _, base_plan = args
                for target_base in base_plan[1:]:
                    delta = target_base - sim.robot.get_base()
                    step([delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0])
            elif name == "pick":
                _, _, _, _, traj = args
                joint_plan = traj.joint_plan
                for target_joints in joint_plan[1:]:
                    step_to_joint_target(target_joints)
                run_gripper(close=True)
                for target_joints in reversed(joint_plan[:-1]):
                    step_to_joint_target(target_joints)
            elif name == "place":
                _, target_pose, _, _, traj = args
                joint_plan = list(traj.joint_plan)
                for target_joints in joint_plan[1:-1]:
                    step_to_joint_target(target_joints)
                # The correction's IK search moves the twin's arm; restore the whole
                # state afterwards so the twin stays in step with the evaluated env.
                snapshot = sim.get_state()
                final_joints = _corrected_place_conf(sim, target_pose, joint_plan[-1])
                sim.set_state(snapshot)
                step_to_joint_target(final_joints)
                run_gripper(close=False)
                for target_joints in reversed(joint_plan[:-1]):
                    step_to_joint_target(target_joints)
            else:
                raise ValueError(f"Unknown plan action {name!r}")


def _joint_infos(sim: ObjectCentricPacking3DEnv) -> Any:
    return sim.robot.arm.get_arm_joint_infos()[:7]


def _clip_action(sim: ObjectCentricPacking3DEnv, action: Any) -> NDArray[np.float32]:
    action_space = sim.action_space
    assert isinstance(action_space, Kinematic3DRobotActionSpace)
    return np.clip(
        np.asarray(action, dtype=np.float32), action_space.low, action_space.high
    )


def _corrected_place_conf(
    sim: ObjectCentricPacking3DEnv, target_pose: Any, planned_final_joints: Any
) -> Any:
    """The final place conf, corrected with the grasp transform recorded at grasp time.

    Without the correction the release misses the placement tolerance. Re-planning
    rather than a bare IK call keeps the correction on the same elbow branch, since a
    branch switch needs a jump that hits the rack. Falls back to the planned conf when
    no correction is found.
    """
    # pylint: disable=protected-access
    real_transform = sim._grasped_object_transform
    grasped_object_id = sim._grasped_object_id
    if real_transform is None:
        return planned_final_joints
    corrected_ee = multiply_poses(target_pose, real_transform.invert())
    # Motion planning wants the full extended (arm + finger) joint vector, matching
    # how joint_plan waypoints are produced upstream.
    current_joints = sim.robot.arm.get_joint_positions()
    try:
        correction_plan = smoothly_follow_end_effector_path(
            sim.robot.arm,
            [corrected_ee],
            initial_joints=current_joints,
            collision_ids={sim.table_id, sim._rack_id},
            joint_distance_fn=create_joint_distance_fn(sim.robot.arm),
            held_object=grasped_object_id,
            base_link_to_held_obj=real_transform,
            include_start=False,
        )
    except InverseKinematicsError:
        correction_plan = []
    # The search leaves the arm wherever its last IK attempt landed; the caller
    # derives its delta from the live joints, so restore them.
    sim.robot.arm.set_joints(current_joints)
    # A goal-only BiRRT query can settle for the closest tree node rather than
    # raising, so verify via forward kinematics before trusting it.
    if correction_plan and not sim.robot.arm.forward_kinematics(
        correction_plan[-1]
    ).allclose(corrected_ee, atol=1e-3):
        correction_plan = []
    return correction_plan[-1] if correction_plan else planned_final_joints
