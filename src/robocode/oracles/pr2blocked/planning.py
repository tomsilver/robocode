"""Task and motion planning specific to the PR2 ``blocked`` oracle.

The generic machinery -- inverse-reachability sampling, arm IK, path validation and
motion planning -- is shared with the other PR2 TAMP oracles and lives in
:mod:`robocode.oracles.pr2_tamp_planning`. What is here is what ``blocked`` alone
needs.

Two things differ from ``packed``. Picks use *side* grasps, because the pen walls are
as tall as the block and only a horizontal approach clears them -- which is what makes
the red blocker an obstruction rather than scenery. And there is a second kind of
target: somewhere on the near table to park that blocker, which has to be clear of the
pen it is being moved out of and of the plate the episode is decided on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robocode.environments.pr2_tamp_blocked_env import PR2BlockedEnv
from robocode.environments.pr2_tamp_scenes import (
    BLOCKED_BLOCK_WIDTH,
    BLOCKED_SPACING,
)
from robocode.environments.ss_pybullet import (
    Euler,
    Point,
    Pose,
    WorldSaver,
    get_aabb,
    get_joint_positions,
    get_pose,
    get_side_grasps,
    invert,
    multiply,
    pairwise_collision,
    set_joint_positions,
    set_point,
    stable_z,
)
from robocode.oracles.pr2_tamp_planning import (
    APPROACH_Z,
    IK_TOLERANCE,
    PATH_RESOLUTION,
    RELEASE_Z,
    PlanFailure,
    arm_ik,
    collides,
    interpolate,
    motion,
    path_is_free,
    shortest,
)

__all__ = [
    "APPROACH_Z",
    "IK_TOLERANCE",
    "PATH_RESOLUTION",
    "RELEASE_Z",
    "PlanFailure",
    "arm_ik",
    "collides",
    "interpolate",
    "motion",
    "MIN_PARK_DISTANCE",
    "park_targets",
    "path_is_free",
    "plan_transfer",
    "plan_unblock",
    "plate_target",
    "shortest",
]

# How far the blocker must end up from the penned block. The pen's gap is
# BLOCKED_SPACING centre to centre, and the gripper has to come in through it, so the
# blocker has to clear both the gap and the space the fingers need behind it.
MIN_PARK_DISTANCE = 2 * BLOCKED_SPACING
# Margin kept between a parked blocker and the plate's footprint. The plate is as wide
# as the near table, so the free area is the strip of table beside it, not a ring.
PLATE_MARGIN = 0.05
# Matches the environment's own settle epsilon, so a tested pose is the pose the
# environment will actually produce when the block is released there.
SETTLE_MARGIN = 1e-4


def plate_target(env: PR2BlockedEnv) -> Any:
    """Release pose above the middle of the plate.

    One block finishes the episode and the plate is 0.6m square, so there is no need
    for the grid of cells ``packed`` needs -- the centre is always free.
    """
    top = get_aabb(env.plate)[1][2]
    plate_point = get_pose(env.plate)[0]
    return Pose(
        Point(x=plate_point[0], y=plate_point[1], z=top + RELEASE_Z),
        Euler(yaw=0),
    )


def park_targets(
    env: PR2BlockedEnv, rng: np.random.Generator, count: int = 12
) -> list[Any]:
    """Release poses on the near table for the blocker, shortest move first.

    Rejection-sampled over the whole table rather than a ring, because the plate is as
    wide as the table: the free area is the strip beside the plate, and a ring around
    the pen mostly misses it. A candidate has to keep the block's footprint on the
    table, stay off the plate, end far enough from the penned block to stop obstructing
    it, and be collision-free where it lands. Ordering by how far the blocker moves
    keeps the relocation short, since every step costs reward.
    """
    table_aabb = get_aabb(env.near_table)
    plate_aabb = get_aabb(env.plate)
    top = table_aabb[1][2]
    penned_point = np.array(get_pose(env.penned)[0][:2])
    blocker_point = np.array(get_pose(env.blocker)[0][:2])
    half = BLOCKED_BLOCK_WIDTH / 2

    candidates: list[tuple[float, Any]] = []
    with env.client(), WorldSaver():
        for _ in range(600):
            point = np.array(
                [
                    rng.uniform(table_aabb[0][0] + half, table_aabb[1][0] - half),
                    rng.uniform(table_aabb[0][1] + half, table_aabb[1][1] - half),
                ]
            )
            if np.linalg.norm(point - penned_point) < MIN_PARK_DISTANCE:
                continue
            # Off the plate: a red block left on the goal surface is in the way of the
            # green one that has to land there.
            on_plate = all(
                plate_aabb[0][axis] - PLATE_MARGIN - half
                <= point[axis]
                <= plate_aabb[1][axis] + PLATE_MARGIN + half
                for axis in (0, 1)
            )
            if on_plate:
                continue
            set_point(
                env.blocker,
                Point(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=stable_z(env.blocker, env.near_table) + SETTLE_MARGIN,
                ),
            )
            if any(
                pairwise_collision(env.blocker, body)
                for body in [*env.walls, *env.movables, env.plate]
                if body != env.blocker
            ):
                continue
            candidates.append(
                (
                    float(np.linalg.norm(point - blocker_point)),
                    Pose(
                        Point(x=float(point[0]), y=float(point[1]), z=top + RELEASE_Z),
                        Euler(yaw=0),
                    ),
                )
            )
            if len(candidates) >= count:
                break
    return [pose for _, pose in sorted(candidates, key=lambda item: item[0])]


def plan_transfer(
    env: PR2BlockedEnv,
    block: int,
    target: Any,
    rng: np.random.Generator,
    tries: int = 240,
    place_tries: int = 40,
) -> dict[str, Any] | None:
    """Plan to side-grasp *block* at one base pose and release it at *target* from
    another.

    Inverse reachability, as in ``packed``, but over two base poses rather than one.
    ``packed`` can grasp and release without driving because its plate sits among the
    blocks on a 0.27m table; here the plate's centre is 0.6m from the penned block and
    the spares are on a table nine metres away, so no single base pose reaches both.
    Upstream's own plans for this problem are ``move_base, pick, move_base, place`` for
    the same reason.

    Returns the two base poses and the arm configurations at each, or None if no
    combination worked within the sample budget.
    """
    with env.client(), WorldSaver():
        block_pose = get_pose(block)
        grasps = list(get_side_grasps(block, grasp_length=0.0))
        lift_pose = (tuple(np.array(block_pose[0]) + [0, 0, APPROACH_Z]), block_pose[1])
        for _ in range(tries):
            pick_base = _ring_pose(block_pose[0], rng)
            set_joint_positions(env.robot, env.base_joints, pick_base)
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
                placement = _plan_place(env, block, grasp, target, rng, place_tries)
                if placement is None:
                    continue
                return {
                    "pick_base": np.array(pick_base),
                    "grasp": grasp_q,
                    "lift": lift_q,
                    **placement,
                }
    return None


def _ring_pose(point: Any, rng: np.random.Generator) -> list[float]:
    """A base pose on a ring around *point*, roughly facing it."""
    radius, theta = rng.uniform(0.45, 0.85), rng.uniform(-np.pi, np.pi)
    bx = point[0] + radius * np.cos(theta)
    by = point[1] + radius * np.sin(theta)
    facing = np.arctan2(point[1] - by, point[0] - bx)
    return [bx, by, facing + rng.uniform(-0.4, 0.4)]


def _plan_place(
    env: PR2BlockedEnv,
    block: int,
    grasp: Any,
    target: Any,
    rng: np.random.Generator,
    tries: int,
) -> dict[str, Any] | None:
    """A base pose and arm config releasing *block* at *target* under *grasp*.

    Called with the pick base pose set, and restores it before returning, so the
    caller's grasp configurations stay valid.
    """
    restore_base = get_joint_positions(env.robot, env.base_joints)
    try:
        for _ in range(tries):
            place_base = _ring_pose(target[0], rng)
            set_joint_positions(env.robot, env.base_joints, place_base)
            set_joint_positions(env.robot, env.arm_joints, env.initial_arm_conf)
            if collides(env):
                continue
            release_q = arm_ik(
                env, multiply(target, invert(grasp)), held=block, held_pose=target
            )
            if release_q is None:
                continue
            return {"place_base": np.array(place_base), "release": release_q}
        return None
    finally:
        set_joint_positions(env.robot, env.base_joints, restore_base)


def plan_unblock(
    env: PR2BlockedEnv, rng: np.random.Generator, park_attempts: int = 6
) -> dict[str, Any] | None:
    """Plan the blocker's relocation, verified to actually free the penned block.

    Moving the blocker is only worth anything if the penned block can be picked
    afterwards, and most of the near table fails that test: the walls leave one approach
    direction open, so a spot that is merely off the plate and clear of the pen can
    still stand in the gripper's way. Each candidate is therefore checked by planning
    the penned pick with the blocker where it would end up, and a candidate that does
    not free it is discarded rather than executed.

    Only the blocker's plan is returned. The penned pick is re-planned for real once the
    blocker has actually moved, because the arm lands within IK tolerance of its planned
    configuration rather than exactly on it, so the verified plan would be slightly
    stale.
    """
    target = plate_target(env)
    for park in park_targets(env, rng)[:park_attempts]:
        blocker_plan = plan_transfer(env, env.blocker, park, rng)
        if blocker_plan is None:
            continue
        with env.client(), WorldSaver():
            point = park[0]
            set_point(
                env.blocker,
                Point(
                    x=point[0],
                    y=point[1],
                    z=stable_z(env.blocker, env.near_table) + SETTLE_MARGIN,
                ),
            )
            if plan_transfer(env, env.penned, target, rng) is None:
                continue
        return blocker_plan
    return None
