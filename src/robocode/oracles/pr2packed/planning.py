"""Task and motion planning primitives for the PR2 ``packed`` oracle.

The structure mirrors what PDDLStream's own ``packed`` streams do -- sample a base
pose from which a grasp is reachable (inverse reachability), solve arm IK there,
then plan collision-free joint motions -- but it is written directly against the
environment instead of going through PDDL and a symbolic planner.

Every candidate configuration and every path is validated against
:func:`collides`, which reproduces the environment's own collision test. ss-pybullet's
planners are checked against a collision function configured separately from the
environment's, so a path that the planner accepts is not automatically one the
environment will execute; validating here is what makes the oracle's plans
executable rather than merely plausible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from robocode.environments.pr2_tamp_env import PR2PackedEnv
from robocode.environments.ss_pybullet import (
    Attachment,
    Euler,
    Point,
    Pose,
    WorldSaver,
    get_aabb,
    get_joint_positions,
    get_link_pose,
    get_pose,
    get_top_grasps,
    invert,
    multiply,
    pairwise_collision,
    plan_joint_motion,
    set_joint_positions,
    set_pose,
    sub_inverse_kinematics,
)

# Height above the block centre for the pre-grasp, and above the plate top for the
# release. The release height clears blocks already standing on the plate, so a
# carried block can travel over them; the environment settles it straight down.
APPROACH_Z = 0.15
RELEASE_Z = 0.20
# Plate cell pitch. Blocks are 0.07 wide and the plate is 0.27, so a 3x3 grid at this
# pitch keeps every cell's footprint inside the plate with clearance to its neighbours.
CELL_SPACING = 0.085
PATH_RESOLUTION = 0.05
IK_TOLERANCE = 0.02


class PlanFailure(RuntimeError):
    """A stage of the current block's plan failed; resample and try again."""


def plate_cells(env: PR2PackedEnv, count: int) -> list[Any]:
    """Target poses on the plate, one per block, released from above."""
    top = get_aabb(env.plate)[1][2]
    offsets = [0.0, -CELL_SPACING, CELL_SPACING]
    grid = [(x, y) for x in offsets for y in offsets]
    if count > len(grid):
        raise ValueError(
            f"plate has {len(grid)} cells but {count} blocks were asked for"
        )
    return [
        Pose(Point(x=x, y=y, z=top + RELEASE_Z), Euler(yaw=0)) for x, y in grid[:count]
    ]


def shortest(delta: NDArray[Any], circular: NDArray[Any]) -> NDArray[Any]:
    """Wrap the entries of *delta* belonging to continuous joints into [-pi, pi]."""
    wrapped = (delta + np.pi) % (2 * np.pi) - np.pi
    return np.where(circular, wrapped, delta)


def collides(env: PR2PackedEnv, held: int | None = None) -> bool:
    """The environment's own collision test, for validating a candidate config.

    Callers already hold ``env.client()``; the lock is reentrant, so the outermost
    planning call is what actually establishes the critical section.
    """
    for body in [env.robot] + ([held] if held is not None else []):
        if any(pairwise_collision(body, o) for o in env.obstacles):
            return True
        if any(
            pairwise_collision(body, b) for b in env.blocks if b not in (held, body)
        ):
            return True
    return False


def arm_ik(
    env: PR2PackedEnv,
    tool_target: Any,
    held: int | None = None,
    held_pose: Any = None,
) -> NDArray[Any] | None:
    """An arm config putting the tool at *tool_target*, or None if infeasible.

    Callers hold ``env.client()``; the lock is reentrant.
    """
    restore_arm = get_joint_positions(env.robot, env.arm_joints)
    restore_held = get_pose(held) if held is not None else None
    try:
        set_joint_positions(env.robot, env.arm_joints, env.initial_arm_conf)
        if (
            sub_inverse_kinematics(
                env.robot, env.arm_joints[0], env.tool_link, tool_target
            )
            is None
        ):
            return None
        error = np.linalg.norm(
            np.array(get_link_pose(env.robot, env.tool_link)[0])
            - np.array(tool_target[0])
        )
        if error > IK_TOLERANCE:
            return None
        if held is not None and held_pose is not None:
            set_pose(held, held_pose)
        if collides(env, held=held):
            return None
        return np.array(get_joint_positions(env.robot, env.arm_joints))
    finally:
        set_joint_positions(env.robot, env.arm_joints, restore_arm)
        if restore_held is not None:
            set_pose(held, restore_held)


def interpolate(
    start: Any, end: Any, circular: NDArray[Any], resolution: float = PATH_RESOLUTION
) -> list[NDArray[Any]]:
    """Configs from *start* to *end* no more than *resolution* apart."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = shortest(end - start, circular)
    steps = max(1, int(np.ceil(np.max(np.abs(delta)) / resolution)))
    return [start + delta * (k / steps) for k in range(steps + 1)]


def path_is_free(
    env: PR2PackedEnv,
    joints: list[int],
    path: list[NDArray[Any]],
    attachment: Attachment | None = None,
) -> bool:
    """Validate *path* against the environment's collision model.

    Callers hold ``env.client()``; the lock is reentrant.
    """
    restore = get_joint_positions(env.robot, joints)
    held = attachment.child if attachment else None
    restore_held = get_pose(held) if held is not None else None
    try:
        for conf in path:
            set_joint_positions(env.robot, joints, conf)
            if attachment is not None:
                attachment.assign()
            if collides(env, held=held):
                return False
        return True
    finally:
        set_joint_positions(env.robot, joints, restore)
        if restore_held is not None:
            set_pose(held, restore_held)


def motion(
    env: PR2PackedEnv,
    joints: list[int],
    target: Any,
    attachment: Attachment | None = None,
) -> list[NDArray[Any]] | None:
    """A validated joint path to *target*: straight line if possible, else RRT.

    Planning drives the simulator through candidate configurations, so the whole call
    runs under a WorldSaver. Without it the environment would be left wherever the
    planner stopped, and the controller's next action -- a delta from where it believes
    the robot to be -- would move it somewhere else entirely.
    """
    circular = (
        env.base_circular if list(joints) == list(env.base_joints) else env.arm_circular
    )
    with env.client(), WorldSaver():
        start = np.array(get_joint_positions(env.robot, joints))
        obstacles = list(env.obstacles) + [
            b for b in env.blocks if attachment is None or b != attachment.child
        ]
        direct = interpolate(start, target, circular)
        if path_is_free(env, joints, direct, attachment):
            return direct
        path = plan_joint_motion(
            env.robot,
            joints,
            list(target),
            obstacles=obstacles,
            self_collisions=False,
            attachments=[attachment] if attachment else [],
            resolutions=np.full(len(joints), PATH_RESOLUTION),
            restarts=4,
            iterations=80,
            smooth=40,
        )
        if path is None:
            return None
        dense = [start]
        for a, b in zip([start] + list(path), path):
            dense.extend(interpolate(a, b, circular)[1:])
        return dense if path_is_free(env, joints, dense, attachment) else None


def plan_pick(
    env: PR2PackedEnv,
    block: int,
    cell: Any,
    rng: np.random.Generator,
    tries: int = 400,
) -> dict[str, Any] | None:
    """Sample a base pose plus arm configs that pick *block* and release over *cell*.

    This is inverse reachability: rather than solving for a base pose analytically,
    draw one from a ring around the target and keep it if arm IK succeeds there for
    the grasp, the lift, and the release.
    """
    with env.client(), WorldSaver():
        block_pose = get_pose(block)
        grasps = list(get_top_grasps(block, grasp_length=0.0))
        lift_pose = (tuple(np.array(block_pose[0]) + [0, 0, APPROACH_Z]), block_pose[1])
        for _ in range(tries):
            radius, theta = rng.uniform(0.40, 0.85), rng.uniform(-np.pi, np.pi)
            bx = block_pose[0][0] + radius * np.cos(theta)
            by = block_pose[0][1] + radius * np.sin(theta)
            facing = np.arctan2(block_pose[0][1] - by, block_pose[0][0] - bx)
            base = [bx, by, facing + rng.uniform(-0.4, 0.4)]
            set_joint_positions(env.robot, env.base_joints, base)
            set_joint_positions(env.robot, env.arm_joints, env.initial_arm_conf)
            if collides(env):
                continue
            for grasp in grasps:
                grasp_q = arm_ik(env, multiply(block_pose, invert(grasp)))
                if grasp_q is None:
                    continue
                lift_q = arm_ik(
                    env,
                    multiply(lift_pose, invert(grasp)),
                    held=block,
                    held_pose=lift_pose,
                )
                if lift_q is None:
                    continue
                release_q = arm_ik(
                    env, multiply(cell, invert(grasp)), held=block, held_pose=cell
                )
                if release_q is None:
                    continue
                return {
                    "base": np.array(base),
                    "grasp": grasp_q,
                    "lift": lift_q,
                    "release": release_q,
                }
    return None
