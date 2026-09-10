"""Rover scenes for PDDLStream's ``rovers`` TAMP benchmark.

``build_rovers_scene`` is adapted from ``examples/pybullet/turtlebot_rovers/problems.py``
in Caelan Garrett's PDDLStream repository (https://github.com/caelan/pddlstream,
MIT-licensed), where the ``rovers`` benchmark is defined. Only the scene construction
is carried over: the PDDL domain, the stream samplers, and the planner are deliberately
left behind, because robocode's agents synthesize their own task and motion planning
against the environment rather than consuming PDDLStream's primitives.

Two behavioural changes from the original. Object placement sampling is split out so an
episode reset can re-randomize the scene without paying to reload the robot URDFs, as
in the PR2 scenes. And the mound layout is drawn from the episode's generator rather
than the global ``random`` module, so an instance is reproducible from its seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robocode.environments.ss_pybullet import (
    BLACK,
    BLUE,
    BROWN,
    GREY,
    HUSKY_URDF,
    TAN,
    TURTLEBOT_URDF,
    HideOutput,
    Point,
    add_data_path,
    create_box,
    get_bodies,
    get_point,
    joints_from_names,
    load_model,
    load_pybullet,
    pairwise_collision,
    sample_placement,
    set_joint_positions,
    set_point,
    stable_z,
)

# Geometry of the original benchmark, kept verbatim so instances are comparable with
# PDDLStream's own `rovers` results.
BASE_EXTENT = 5.0
MOUND_WIDTH = 0.5
MOUND_HEIGHT = 0.1
OBJECTIVE_WIDTH = 0.07
OBJECTIVE_HEIGHT = 0.2
ROCK_WIDTH = 0.075
ROCK_HEIGHT = 0.01
SOIL_WIDTH = 0.1
SOIL_HEIGHT = 0.005

# The rover's base is a planar (x, y, theta) joint group, and its camera hangs off the
# Kinect frame; both names come from the turtlebot URDF the benchmark loads.
BASE_JOINT_NAMES = ["x", "y", "theta"]
KINECT_FRAME = "camera_rgb_optical_frame"

# Sensor ranges. Imaging is short-range and needs a clear cone to the objective;
# the radio reaches twice as far but still needs line of sight to the lander.
VIS_RANGE = 2.0
COM_RANGE = 2 * VIS_RANGE

# Where the two rovers start, and where the goal requires them to return.
ROVER_CONFS = [(+1.0, -1.75, np.pi), (-1.0, -1.75, 0.0)]
LANDER_POINT = (-1.9, -2.0)

# Clearances the original passes to its placement sampler.
OBJECTIVE_MIN_DISTANCE = 0.05
ROCK_MIN_DISTANCE = 0.2
# Clearance kept around each rover's home pose. The goal requires both rovers back
# there on their starting heading, so a pillar dropped just clear of a rover still
# ruins the instance: it leaves the rover able to sit at home but not to turn to
# face the right way, blocking the arc in both directions. Non-overlap is not
# enough; there has to be room to rotate.
HOME_CLEARANCE = 0.6
_MAX_PLACEMENT_ATTEMPTS = 100


def base_joints(robot: int) -> list[int]:
    """The rover's planar base joint group."""
    return list(joints_from_names(robot, BASE_JOINT_NAMES))


def resample_rover_scene(
    objectives: list[int],
    mounds: list[int],
    rocks: list[int],
    soils: list[int],
    obstacles: list[int],
    floor: int,
    rng: np.random.Generator,
    home_points: list[Any] | None = None,
) -> bool:
    """Re-randomize everything an episode varies, returning whether it succeeded.

    Objectives sit on mounds -- that is what puts them above the clutter and makes
    imaging a line-of-sight problem rather than a driving one -- while rocks, soil and
    the obstacle pillars are scattered on the floor. The mound an objective lands on is
    drawn per episode, so which objectives are occluded changes between instances.

    Bodies are placed one at a time and each is checked only against what has already
    been placed *this* episode. Checking against all of them would mean checking against
    wherever the previous episode happened to leave the ones still to move, which makes
    the layout depend on episode order rather than on the seed alone.
    """
    movable = [*objectives, *rocks, *soils, *obstacles]
    homes = [np.asarray(point, dtype=float)[:2] for point in (home_points or [])]
    placed: list[int] = []
    chosen = list(mounds)
    rng.shuffle(chosen)
    for objective, mound in zip(objectives, chosen):
        if not _place_on(
            objective, mound, OBJECTIVE_MIN_DISTANCE, movable, placed, homes
        ):
            return False
        placed.append(objective)
    for body in [*rocks, *soils, *obstacles]:
        distance = ROCK_MIN_DISTANCE if body in rocks or body in soils else 0.0
        if not _place_on(body, floor, distance, movable, placed, homes):
            return False
        placed.append(body)
    return True


def _place_on(
    body: int,
    surface: int,
    min_distance: float,
    movable: list[int],
    placed: list[int],
    homes: list[Any] | None = None,
) -> bool:
    """Rejection-sample a collision-free placement of *body* on *surface*.

    Adapted from PDDLStream's ``sample_placements``, with the original's unbounded
    ``while True`` replaced by a bounded retry count so a reset cannot hang.
    """
    fixed = [
        other
        for other in get_bodies()
        if other not in movable and other not in (body, surface)
    ]
    for _ in range(_MAX_PLACEMENT_ATTEMPTS):
        pose = sample_placement(body, surface)
        if pose is None:
            return False
        if any(pairwise_collision(body, other) for other in fixed):
            continue
        if any(
            pairwise_collision(body, other, max_distance=min_distance)
            for other in placed
        ):
            continue
        if homes:
            here = np.asarray(get_point(body), dtype=float)[:2]
            if any(
                float(np.linalg.norm(here - home)) < HOME_CLEARANCE for home in homes
            ):
                continue
        return True
    return False


def build_rovers_scene(
    num_objectives: int = 4,
    num_rocks: int = 3,
    num_soils: int = 3,
    num_obstacles: int = 8,
    num_rovers: int = 2,
) -> dict[str, Any]:
    """Build the ``rovers`` scene in the current physics client, returning its handles.

    The walls, the dividing wall and the mounds are the benchmark's fixed furniture;
    what an episode varies is where the objectives, rocks, soil and obstacle pillars
    end up, which :func:`resample_rover_scene` draws.
    """
    if num_rovers > len(ROVER_CONFS):
        raise ValueError(f"at most {len(ROVER_CONFS)} rovers, got {num_rovers}")
    if num_objectives < 1:
        raise ValueError(f"num_objectives must be positive, got {num_objectives}")

    floor = create_box(BASE_EXTENT, BASE_EXTENT, 0.001, color=TAN)
    set_point(floor, Point(z=-0.001 / 2.0))

    walls = []
    for dx, dy in ((0.0, +1.0), (0.0, -1.0)):
        wall = create_box(
            BASE_EXTENT + MOUND_HEIGHT, MOUND_HEIGHT, MOUND_HEIGHT, color=GREY
        )
        set_point(wall, Point(x=dx, y=dy * BASE_EXTENT / 2.0, z=MOUND_HEIGHT / 2.0))
        walls.append(wall)
    for dx, dy in ((+1.0, 0.0), (-1.0, 0.0)):
        wall = create_box(
            MOUND_HEIGHT, BASE_EXTENT + MOUND_HEIGHT, MOUND_HEIGHT, color=GREY
        )
        set_point(wall, Point(x=dx * BASE_EXTENT / 2.0, y=dy, z=MOUND_HEIGHT / 2.0))
        walls.append(wall)
    # The divider down the middle: it is what makes a rover drive around rather than
    # straight across, and what occludes the lander from half the arena.
    divider = create_box(MOUND_HEIGHT, BASE_EXTENT, MOUND_HEIGHT, color=GREY)
    set_point(divider, Point(z=MOUND_HEIGHT / 2.0))
    walls.append(divider)

    add_data_path()
    with HideOutput():
        lander = load_pybullet(HUSKY_URDF, scale=1)
    set_point(
        lander,
        Point(LANDER_POINT[0], LANDER_POINT[1], stable_z(lander, floor)),
    )

    mounds = []
    for x in (+2.0, -2.0, +0.5, -0.5):
        mound = create_box(MOUND_WIDTH, MOUND_WIDTH, MOUND_HEIGHT, color=GREY)
        set_point(mound, Point(x=x, y=2.0, z=MOUND_HEIGHT / 2.0))
        mounds.append(mound)

    obstacles = [
        create_box(MOUND_HEIGHT, MOUND_HEIGHT, 4 * MOUND_HEIGHT, color=GREY)
        for _ in range(num_obstacles)
    ]

    rovers = []
    for index in range(num_rovers):
        with HideOutput():
            rover = load_model(TURTLEBOT_URDF)
        set_point(rover, Point(z=stable_z(rover, floor)))
        set_joint_positions(rover, base_joints(rover), ROVER_CONFS[index])
        rovers.append(rover)

    objectives = [
        create_box(OBJECTIVE_WIDTH, OBJECTIVE_WIDTH, OBJECTIVE_HEIGHT, color=BLUE)
        for _ in range(num_objectives)
    ]
    rocks = [
        create_box(ROCK_WIDTH, ROCK_WIDTH, ROCK_HEIGHT, color=BLACK)
        for _ in range(num_rocks)
    ]
    soils = [
        create_box(SOIL_WIDTH, SOIL_WIDTH, SOIL_HEIGHT, color=BROWN)
        for _ in range(num_soils)
    ]

    return {
        "floor": floor,
        "walls": walls,
        "lander": lander,
        "mounds": mounds,
        "obstacles": obstacles,
        "rovers": rovers,
        "objectives": objectives,
        "rocks": rocks,
        "soils": soils,
        "rover_confs": [list(conf) for conf in ROVER_CONFS[:num_rovers]],
    }
