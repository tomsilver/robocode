"""Plan one PR2 ``packed`` instance with stock PDDLStream, as a subprocess.

This module is executed as a script and imports the *stock* PDDLStream tree,
including its own vendored copy of ss-pybullet under
``examples.pybullet.utils.pybullet_tools``. That copy is a second instance of the
module robocode imports as ``pybullet_tools`` (see
:mod:`robocode.environments.ss_pybullet`), and the two disagree about which
physics client is live, so the stock tree must never be imported into a process
that holds a robocode environment. Running it here, behind a JSON pipe, is the
same isolation ``scripts/compare_pddlstream_rollout.py`` uses.

It reads one instance on stdin -- the evaluated episode's block poses and robot
configuration -- restores that state into a stock ``packed`` scene, plans, and
writes the plan back on stdout as joint-space waypoints. Nothing from robocode is
imported here.
"""

# mypy: disable-error-code="import-untyped"
# Every import inside `_plan` comes from the stock PDDLStream tree, which ships no
# type information and vendors an `examples` package mypy can see but not check.

from __future__ import annotations

import json
import sys
import traceback
from typing import Any


def _plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore the evaluated instance into a stock scene and plan on it."""
    # pylint: disable=import-outside-toplevel,import-error,no-name-in-module
    import numpy as np
    from examples.pybullet.pr2.run import post_process
    from examples.pybullet.tamp.problems import packed
    from examples.pybullet.tamp.run import pddlstream_from_problem
    from examples.pybullet.utils.pybullet_tools.pr2_primitives import Conf
    from examples.pybullet.utils.pybullet_tools.pr2_utils import (
        get_arm_joints,
        get_gripper_joints,
        get_group_joints,
    )
    from examples.pybullet.utils.pybullet_tools.utils import (
        HideOutput,
        connect,
        get_max_limit,
        set_client,
        set_joint_positions,
        set_pose,
    )
    from pddlstream.algorithms.meta import solve
    from pddlstream.language.function import FunctionInfo
    from pddlstream.language.stream import StreamInfo

    count = int(payload["count"])
    client = connect(use_gui=False)
    set_client(client)

    # The scene is built at the seed the differential test pins, then overwritten
    # with the evaluated instance's state; the build seed only decides the layout
    # that is about to be replaced.
    np.random.seed(0)
    with HideOutput():
        problem = packed(num=count)

    robot = problem.robot
    base_joints = list(get_group_joints(robot, "base"))
    arm_joints = list(get_arm_joints(robot, "left"))
    gripper_joints = list(get_gripper_joints(robot, "left"))

    set_joint_positions(robot, base_joints, payload["base"])
    set_joint_positions(robot, arm_joints, payload["arm"])
    # The environment stores one opening shared by every finger joint.
    opening = float(payload["gripper_opening"])
    set_joint_positions(robot, gripper_joints, [opening] * len(gripper_joints))
    gripper_max = float(get_max_limit(robot, gripper_joints[0]))
    # `packed` appends the blocks last and in order, so movable[i] is our block i.
    for body, pose in zip(problem.movable, payload["block_poses"]):
        set_pose(body, (tuple(pose[0]), tuple(pose[1])))

    stream_info = {
        "inverse-kinematics": StreamInfo(),
        "plan-base-motion": StreamInfo(overhead=1e1),
        "test-cfree-pose-pose": StreamInfo(p_success=1e-3, verbose=False),
        "test-cfree-approach-pose": StreamInfo(p_success=1e-2, verbose=False),
        "test-cfree-traj-pose": StreamInfo(p_success=1e-1, verbose=False),
        "Distance": FunctionInfo(p_success=0.99),
    }

    pddlstream_problem = pddlstream_from_problem(
        problem, collisions=True, teleport=False
    )
    with HideOutput():
        plan, _cost, _evaluations = solve(
            pddlstream_problem,
            algorithm=payload.get("algorithm", "adaptive"),
            stream_info=stream_info,
            max_time=float(payload["max_time"]),
            success_cost=float("inf"),
            verbose=False,
            debug=False,
        )
    if plan is None:
        return {"solved": False, "steps": []}

    commands = post_process(problem, plan, teleport=False)

    # Flatten the command list into joint-space waypoints the caller can servo to
    # without importing any of this. A trajectory is tagged by which group it
    # drives, since the caller's action splits base and arm into fixed slices.
    base_set, arm_set = set(base_joints), set(arm_joints)
    steps: list[dict[str, Any]] = []
    for command in commands:
        name = type(command).__name__
        if name == "Trajectory":
            for conf in command.path:
                if not isinstance(conf, Conf):
                    continue
                joints = set(conf.joints)
                if joints <= base_set:
                    group = "base"
                elif joints <= arm_set:
                    group = "arm"
                else:
                    continue
                steps.append({"kind": group, "values": [float(v) for v in conf.values]})
        elif name == "GripperCommand":
            # Closing carries one width per finger joint; opening carries the
            # joint's max limit as a scalar.
            position = command.position
            commanded = float(
                max(position) if isinstance(position, (list, tuple)) else position
            )
            # Opening commands the joint's max limit exactly; a grasp commands the
            # block's width, which is strictly less but not necessarily by much (0.43
            # against a 0.55 limit here), so anything below the limit is a close.
            steps.append({"kind": "gripper", "close": commanded < gripper_max - 1e-6})
    return {"solved": True, "steps": steps}


def main() -> int:
    """Read one instance on stdin and write its plan on stdout."""
    payload = json.loads(sys.stdin.read())
    try:
        result = _plan(payload)
    except Exception as error:  # pylint: disable=broad-exception-caught
        # The traceback goes to stderr, which the caller surfaces on failure; stdout
        # is reserved for the framed result.
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
