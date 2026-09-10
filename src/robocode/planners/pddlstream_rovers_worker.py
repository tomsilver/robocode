"""Plan one ``rovers`` instance with stock PDDLStream, as a subprocess.

This module is executed as a script and imports the *stock* PDDLStream tree, including
its own vendored copy of ss-pybullet. That copy is a second instance of the module
robocode imports as ``pybullet_tools`` (see :mod:`robocode.environments.ss_pybullet`),
and the two disagree about which physics client is live, so the stock tree must never be
imported into a process holding a robocode environment. Running it here, behind a JSON
pipe, is the isolation the PR2 packed planner and the differential test both use.

It reads one instance on stdin -- where the rovers, objectives, samples and obstacles
are -- restores that into a stock ``rovers1`` scene, plans, and writes the plan back as
a flat list of "drive this rover here, then apply this operator". Nothing from robocode
is imported.
"""

# mypy: disable-error-code="import-untyped"
# Every import inside `_plan` comes from the stock PDDLStream tree, which ships no
# type information and vendors an `examples` package mypy can see but not check.

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

# The plan's operators, in the vocabulary the robocode environment uses. ``move`` is a
# drive with no operator; ``drop_rock`` applies wherever the rover already stands.
_OPERATORS = {
    "move": "noop",
    "calibrate": "calibrate",
    "take_image": "image",
    "send_image": "send",
    "send_analysis": "send",
    "sample_rock": "sample",
    "drop_rock": "drop",
}
# Which argument of each action carries the base configuration it happens at.
_CONF_INDEX = {
    "move": 3,
    "calibrate": 1,
    "take_image": 1,
    "send_image": 1,
    "send_analysis": 1,
    "sample_rock": 1,
}


def _plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore the evaluated instance into a stock scene and plan on it."""
    # pylint: disable=import-outside-toplevel,import-error,no-name-in-module
    import numpy as np
    from examples.pybullet.turtlebot_rovers.problems import get_base_joints, rovers1
    from examples.pybullet.turtlebot_rovers.run import pddlstream_from_problem
    from examples.pybullet.utils.pybullet_tools.utils import (
        HideOutput,
        connect,
        get_aabb,
        set_client,
        set_joint_positions,
        set_point,
    )
    from pddlstream.algorithms.meta import solve
    from pddlstream.language.stream import StreamInfo
    from pddlstream.utils import INF

    client = connect(use_gui=False)
    set_client(client)
    np.random.seed(int(payload.get("seed", 0)) % (2**31))
    with HideOutput():
        problem = rovers1(
            n_rovers=len(payload["rovers"]),
            n_objectives=len(payload["objectives"]),
            n_rocks=len(payload["rocks"]),
            n_soil=len(payload["soils"]),
            n_stores=1,
            n_obstacles=len(payload["obstacles"]),
        )

    # Put the stock scene into the evaluated instance's layout. `rovers1` builds the
    # movable bodies in this order, so index correspondence is all that is needed.
    for rover, conf in zip(problem.rovers, payload["rovers"]):
        set_joint_positions(rover, get_base_joints(rover), conf)
    for body, point in zip(problem.objectives, payload["objectives"]):
        set_point(body, point)
    # `rovers1` keeps stones and soils in one `rocks` list, stones first.
    for body, point in zip(problem.rocks, payload["rocks"] + payload["soils"]):
        set_point(body, point)
    obstacles = [
        b
        for b in problem.fixed
        if b not in problem.objectives and b not in problem.rocks
    ]
    pillars = _scatterable(obstacles, get_aabb)
    if len(pillars) != len(payload["obstacles"]):
        return {
            "solved": False,
            "steps": [],
            "error": (
                f"matched {len(pillars)} stock pillars against "
                f"{len(payload['obstacles'])} in the instance"
            ),
        }
    for body, point in zip(pillars, payload["obstacles"]):
        set_point(body, point)

    pddlstream_problem = pddlstream_from_problem(
        problem,
        collisions=True,
        teleport=False,
        holonomic=False,
        reversible=True,
        use_aabb=True,
    )
    stream_info = {
        "test-cfree-ray-conf": StreamInfo(),
        "test-reachable": StreamInfo(p_success=1e-1),
        "obj-inv-visible": StreamInfo(),
        "com-inv-visible": StreamInfo(),
        "sample-above": StreamInfo(),
        "sample-motion": StreamInfo(overhead=10),
    }
    with HideOutput():
        plan, _cost, _evaluations = solve(
            pddlstream_problem,
            algorithm=payload.get("algorithm", "adaptive"),
            stream_info=stream_info,
            planner="ff-wastar3",
            max_planner_time=10,
            debug=False,
            unit_costs=False,
            success_cost=INF,
            max_time=float(payload["max_time"]),
            verbose=False,
            unit_efforts=True,
            effort_weight=1,
            search_sample_ratio=2,
        )
    if plan is None:
        return {"solved": False, "steps": []}

    index_of = {rover: i for i, rover in enumerate(problem.rovers)}
    steps: list[dict[str, Any]] = []
    for name, args in plan:
        if name not in _OPERATORS:
            return {"solved": False, "steps": [], "error": f"unknown action {name}"}
        rover = args[0]
        if rover not in index_of:
            continue
        waypoints: list[list[float]] = []
        if name == "move":
            # The plan's motion is a trajectory its own motion stream checked against
            # the obstacles. Keeping only its endpoint would mean driving straight at
            # that endpoint and into whatever the trajectory was routed around.
            waypoints = _waypoints(args[2])
        elif name in _CONF_INDEX:
            waypoints = [[float(v) for v in args[_CONF_INDEX[name]].values]]
        steps.append(
            {
                "rover": index_of[rover],
                "waypoints": waypoints,
                "op": _OPERATORS[name],
            }
        )
    return {"solved": True, "steps": steps}


def _waypoints(trajectory: Any) -> list[list[float]]:
    """The base configurations along one planned motion."""
    path = getattr(trajectory, "path", None)
    if path is None:
        return []
    out = []
    for conf in path:
        values = getattr(conf, "values", None)
        if values is not None:
            out.append([float(v) for v in values])
    return out


def _scatterable(bodies: list[int], get_aabb: Any) -> list[int]:
    """The obstacle pillars, told from the walls and mounds by their size.

    `rovers1` does not hand its pillars back separately, and they are the only fixed
    bodies it scatters, so they are the ones whose placement the caller has to
    overwrite. They are the tall thin ones -- a tenth of a metre square and four times
    that high -- which no wall or mound matches, so size identifies them where position
    cannot: the whole point is that they have moved. Ordering by body id keeps the
    correspondence with the caller's own list stable.
    """
    found = []
    for body in sorted(bodies):
        lower, upper = get_aabb(body)
        extent = [upper[i] - lower[i] for i in range(3)]
        if extent[0] < 0.2 and extent[1] < 0.2 and extent[2] > 0.25:
            found.append(body)
    return found


def main() -> int:
    """Read one instance on stdin and write its plan on stdout."""
    payload = json.loads(sys.stdin.read())
    try:
        result = _plan(payload)
    except Exception as error:  # pylint: disable=broad-exception-caught
        traceback.print_exc(file=sys.stderr)
        result = {
            "solved": False,
            "steps": [],
            "error": f"{type(error).__name__}: {error}",
        }
    # Stock PDDLStream and pybullet both print to stdout, so the result is framed.
    sys.stdout.write("\n__PLAN__" + json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
