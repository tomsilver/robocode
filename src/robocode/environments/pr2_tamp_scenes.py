"""PR2 tabletop scenes for the ``packed`` TAMP benchmark.

``build_packed_scene`` is adapted from ``examples/pybullet/tamp/problems.py`` in
Caelan Garrett's PDDLStream repository (https://github.com/caelan/pddlstream,
MIT-licensed), which is where the ``packed`` benchmark used in the LLM-PDDLStream
paper is defined. Only the scene construction is carried over: the PDDL domain,
the stream samplers, and the planner are deliberately left behind, because
robocode's agents synthesize their own task and motion planning against the
environment rather than consuming PDDLStream's primitives.

The one behavioral change from the original is that block placement sampling is
split out into :func:`resample_block_placements` so that an episode reset can
re-randomize the blocks without paying to reload the PR2 URDF.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robocode.environments.ss_pybullet import (
    BLUE,
    GREEN,
    REST_LEFT_ARM,
    Point,
    Problem,
    add_data_path,
    arm_conf,
    close_arm,
    create_box,
    create_pr2,
    create_table,
    get_bodies,
    get_carry_conf,
    get_other_arm,
    load_pybullet,
    open_arm,
    pairwise_collision,
    sample_placement,
    set_arm_conf,
    set_group_conf,
    set_point,
    stable_z,
)

# Geometry of the original benchmark, kept verbatim so instances are comparable
# with the PDDLStream and LLM-PDDLStream results for `packed`.
BLOCK_WIDTH = 0.07
BLOCK_HEIGHT = 0.1
PLATE_WIDTH = 0.27
PLATE_HEIGHT = 0.001
BASE_EXTENT = 5.0
# Minimum clearance between two blocks sampled onto the table, so that an
# instance never starts with blocks wedged too close to grasp individually.
BLOCK_MIN_DISTANCE = 0.05
# The arm and grasp the benchmark fixes; `packed` is a top-grasp problem.
ARM = "left"
GRASP_TYPE = "top"

# Retries for one block's rejection sampling before the instance is abandoned.
_MAX_PLACEMENT_ATTEMPTS = 100


def resample_block_placements(
    blocks: list[int],
    surface: int,
    min_distance: float = BLOCK_MIN_DISTANCE,
) -> bool:
    """Rejection-sample a collision-free placement on *surface* for each block.

    Adapted from PDDLStream's ``sample_placements``, with the original's unbounded
    ``while True`` replaced by a bounded retry count so that a reset cannot hang.
    Returns whether every block was placed. Draws from the global numpy RNG, which
    is what ``sample_placement`` uses; callers seed it for reproducibility.
    """
    obstacles = [body for body in get_bodies() if body not in blocks]
    for block in blocks:
        for _ in range(_MAX_PLACEMENT_ATTEMPTS):
            pose = sample_placement(block, surface)
            if pose is None:
                return False
            if not any(
                pairwise_collision(block, obstacle, max_distance=min_distance)
                for obstacle in obstacles
                if obstacle not in (block, surface)
            ):
                obstacles.append(block)
                break
        else:
            return False
    return True


def build_packed_scene(num_blocks: int) -> tuple[Problem, dict[str, Any]]:
    """Build the ``packed`` scene in the current physics client.

    Returns the PDDLStream ``Problem`` (whose ``goal_on`` lists every block on the
    plate) and a dict of the handles the environment needs to drive it.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")

    base_limits = (
        -BASE_EXTENT / 2.0 * np.ones(2),
        BASE_EXTENT / 2.0 * np.ones(2),
    )
    other_arm = get_other_arm(ARM)
    initial_conf = get_carry_conf(ARM, GRASP_TYPE)

    add_data_path()
    floor = load_pybullet("plane.urdf")
    pr2 = create_pr2()
    set_arm_conf(pr2, ARM, initial_conf)
    open_arm(pr2, ARM)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)
    # Setting the base group rather than the body pose; the drake PR2 has a
    # planar base joint, so moving the body directly would desync the joints.
    set_group_conf(pr2, "base", [-1.0, 0, 0])

    table = create_table()
    plate = create_box(PLATE_WIDTH, PLATE_WIDTH, PLATE_HEIGHT, color=GREEN)
    set_point(plate, Point(z=stable_z(plate, table)))

    blocks = [
        create_box(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_HEIGHT, color=BLUE)
        for _ in range(num_blocks)
    ]
    resample_block_placements(blocks, table)

    problem = Problem(
        robot=pr2,
        movable=blocks,
        arms=[ARM],
        grasp_types=[GRASP_TYPE],
        surfaces=[table, plate],
        goal_on=[(block, plate) for block in blocks],
        base_limits=base_limits,
    )
    handles = {
        "robot": pr2,
        "floor": floor,
        "blocks": blocks,
        "table": table,
        "plate": plate,
        "initial_arm_conf": list(initial_conf),
    }
    return problem, handles
