"""PR2 tabletop scenes for the ``packed`` and ``blocked`` TAMP benchmarks.

``build_packed_scene`` and ``build_blocked_scene`` are adapted from
``examples/pybullet/tamp/problems.py`` in
Caelan Garrett's PDDLStream repository (https://github.com/caelan/pddlstream,
MIT-licensed), which is where the ``packed`` and ``blocked`` benchmarks used in the
LLM-PDDLStream paper are defined. Only the scene construction is carried over:
the PDDL domain, the stream samplers, and the planner are deliberately left
behind, because
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
    GREY,
    RED,
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
    get_point,
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
        "initial_base_conf": [-1.0, 0.0, 0.0],
        "blocks": blocks,
        "table": table,
        "plate": plate,
        "initial_arm_conf": list(initial_conf),
    }
    return problem, handles


# ------------------------------------------------------------------- blocked

# Geometry of the original `blocked` benchmark, kept verbatim. The blocks are taller
# than `packed`'s (2x the width rather than 0.10m) because the task turns on a *side*
# grasp: a three-sided pen of walls the same height as the block leaves exactly one
# approach direction open, and a red block sits in it.
BLOCKED_BLOCK_WIDTH = 0.07
BLOCKED_BLOCK_HEIGHT = 2 * BLOCKED_BLOCK_WIDTH
BLOCKED_PLATE_WIDTH = 0.6
BLOCKED_X_EXTENT = 10.0
# Spacing between the penned block and both the red blocker and the walls.
BLOCKED_SPACING = 0.15
BLOCKED_WALL_THICKNESS = 0.01
# `blocked` is a side-grasp problem, which is what makes the pen and blocker matter.
BLOCKED_GRASP_TYPE = "side"
# The direction left open by the walls, and so the one the red block occupies.
BLOCKED_OPEN_DIRECTION = (-1.0, 0.0)


def build_blocked_scene(num_spares: int) -> tuple[Problem, dict[str, Any]]:
    """Build the ``blocked`` scene in the current physics client.

    One green block sits on the near table inside a three-sided pen of walls, with a
    red block in the one gap. *num_spares* further green blocks go on the far table,
    which is where the count axis lives: at zero the penned block is the only green
    one and the blocker has to be moved, and above zero there is the alternative of
    hauling a spare across instead.

    Returns the PDDLStream ``Problem``, whose ``goal_on`` is satisfied by *any* green
    block reaching the plate, and the handles the environment needs to drive it.
    """
    if num_spares < 0:
        raise ValueError(f"num_spares must be non-negative, got {num_spares}")

    base_limits = (
        -BLOCKED_X_EXTENT / 2.0 * np.ones(2),
        BLOCKED_X_EXTENT / 2.0 * np.ones(2),
    )
    table_x = (BLOCKED_X_EXTENT - 1) / 2.0
    other_arm = get_other_arm(ARM)
    initial_conf = get_carry_conf(ARM, BLOCKED_GRASP_TYPE)

    add_data_path()
    floor = load_pybullet("plane.urdf")
    pr2 = create_pr2()
    set_arm_conf(pr2, ARM, initial_conf)
    open_arm(pr2, ARM)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)
    # Setting the base group rather than the body pose; see build_packed_scene.
    set_group_conf(pr2, "base", [BLOCKED_X_EXTENT / 4, 0, 0])

    near_table = create_table()
    set_point(near_table, Point(x=+table_x, y=0))
    far_table = create_table()
    set_point(far_table, Point(x=-table_x, y=0))

    plate = create_box(
        BLOCKED_PLATE_WIDTH, BLOCKED_PLATE_WIDTH, PLATE_HEIGHT, color=GREEN
    )
    table_point = get_point(near_table)
    plate_x, plate_y = float(table_point[0]), float(table_point[1])
    set_point(
        plate,
        Point(x=plate_x, y=plate_y - 0.3, z=stable_z(plate, near_table)),
    )

    def _new_block(color: Any) -> int:
        return create_box(
            BLOCKED_BLOCK_WIDTH, BLOCKED_BLOCK_WIDTH, BLOCKED_BLOCK_HEIGHT, color=color
        )

    penned = _new_block(BLUE)
    penned_x, penned_y = plate_x, plate_y + 0.3
    set_point(penned, Point(x=penned_x, y=penned_y, z=stable_z(penned, near_table)))

    blocker = _new_block(RED)
    set_point(
        blocker,
        Point(
            x=penned_x + BLOCKED_SPACING * BLOCKED_OPEN_DIRECTION[0],
            y=penned_y + BLOCKED_SPACING * BLOCKED_OPEN_DIRECTION[1],
            z=stable_z(blocker, near_table),
        ),
    )

    # Three walls seal the pen on +x and both y sides, leaving -x -- where the red
    # block stands -- as the only direction a side grasp can come from.
    side_wall = create_box(
        BLOCKED_WALL_THICKNESS,
        2 * BLOCKED_SPACING,
        BLOCKED_BLOCK_HEIGHT,
        color=GREY,
    )
    end_walls = [
        create_box(
            BLOCKED_SPACING,
            BLOCKED_WALL_THICKNESS,
            BLOCKED_BLOCK_HEIGHT,
            color=GREY,
        )
        for _ in range(2)
    ]
    wall_z = stable_z(side_wall, near_table)
    set_point(side_wall, Point(x=penned_x + BLOCKED_SPACING, y=penned_y, z=wall_z))
    for wall, sign in zip(end_walls, (+1.0, -1.0)):
        set_point(
            wall,
            Point(
                x=penned_x + BLOCKED_SPACING / 2,
                y=penned_y + sign * BLOCKED_SPACING,
                z=wall_z,
            ),
        )
    walls = [side_wall, *end_walls]

    spares = [_new_block(BLUE) for _ in range(num_spares)]
    if spares:
        resample_block_placements(spares, far_table)

    greens = [penned, *spares]
    problem = Problem(
        robot=pr2,
        movable=[*greens, blocker],
        arms=[ARM],
        grasp_types=[BLOCKED_GRASP_TYPE],
        surfaces=[near_table, far_table, plate],
        # Upstream writes this goal as the single existential ('?green', plate); the
        # environment scores it as "any green block on the plate", which is the same
        # condition without needing PDDL's typing machinery.
        goal_on=[(green, plate) for green in greens],
        body_types=[(green, "green") for green in greens],
        base_limits=base_limits,
        costs=True,
    )
    handles = {
        "robot": pr2,
        "floor": floor,
        "initial_base_conf": [BLOCKED_X_EXTENT / 4, 0.0, 0.0],
        "greens": greens,
        "penned": penned,
        "spares": spares,
        "blocker": blocker,
        "walls": walls,
        "near_table": near_table,
        "far_table": far_table,
        "plate": plate,
        "initial_arm_conf": list(initial_conf),
    }
    return problem, handles
