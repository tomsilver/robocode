"""Task and motion planning specific to the PR2 ``packed`` oracle.

The generic machinery -- inverse-reachability sampling, arm IK, path validation and
motion planning -- is shared with the other PR2 TAMP oracles and lives in
:mod:`robocode.oracles.pr2_tamp_planning`. What is here is what ``packed`` alone
needs: the grid of target cells on its small plate, and a pick that uses *top*
grasps.

``PlanFailure``, ``PATH_RESOLUTION``, ``motion`` and ``shortest`` are re-exported so
the approach module and the PDDLStream planner keep importing them from one place.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robocode.environments.pr2_tamp_env import PR2PackedEnv
from robocode.environments.ss_pybullet import (
    Euler,
    Point,
    Pose,
    WorldSaver,
    get_aabb,
    get_pose,
    get_top_grasps,
    invert,
    multiply,
    set_joint_positions,
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
    "CELL_SPACING",
    "IK_TOLERANCE",
    "PATH_RESOLUTION",
    "RELEASE_Z",
    "PlanFailure",
    "arm_ik",
    "collides",
    "interpolate",
    "motion",
    "path_is_free",
    "plan_pick",
    "plate_cells",
    "shortest",
]

# Plate cell pitch. Blocks are 0.07 wide and the plate is 0.27, so a 3x3 grid at this
# pitch keeps every cell's footprint inside the plate with clearance to its neighbours.
CELL_SPACING = 0.085


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
