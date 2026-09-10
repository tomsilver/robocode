"""Stock PDDLStream planning for the ``rovers`` environment.

``RoversEnv`` re-exposes PDDLStream's ``rovers`` benchmark as a closed-loop gymnasium
environment but ships none of the planner: the PDDL domain and the stream samplers stay
upstream. This module borrows them back, so the benchmark's own solver can be scored on
the same episodes as robocode's agents.

The plan is computed by :mod:`robocode.planners.pddlstream_rovers_worker` in a
subprocess, because the stock tree vendors a second copy of ss-pybullet whose
physics-client globals collide with the one the evaluated environment holds. What
crosses the pipe is a flat list of "drive this rover here, then apply this operator",
which maps directly onto the environment's own action: the upstream plan is built from
the same operators the environment implements, so nothing has to be reinterpreted.

Plan steps are executed one at a time, with the other rover holding still. The plan is a
total order over both rovers' actions and the environment steps both at once, so running
them concurrently would need a schedule the plan does not carry.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from robocode.environments.rovers_env import (
    CALIBRATE,
    DROP,
    IMAGE,
    NOOP,
    SAMPLE,
    SEND,
    RoversEnv,
    operator_action,
)
from robocode.planners.pddlstream_pr2packed import PlanningFailure

logger = logging.getLogger(__name__)

# The same exception the PR2 upstream planner raises, so one ``except`` in the approach
# covers every family whose domain lives upstream. Re-exported here so callers of this
# module do not have to reach into a sibling planner for it.
__all__ = ["ROVERS_ENV_PATH", "PlanningFailure", "RoversPDDLStreamPlanner"]

ROVERS_ENV_PATH = "robocode.environments.rovers_env:RoversEnv"

_MAX_DELTA = 0.2
_MAX_DELTA_THETA = 0.4
_ACTION_DIM = 8
_POSITION_TOLERANCE = 0.08
_ANGLE_TOLERANCE = 0.15
# Steps allowed to reach one plan waypoint before the step is abandoned. A plan hop can
# cross the arena, which is 5m at 0.2m a step.
_MAX_STEPS_PER_STEP = 40
_RESULT_MARKER = "__PLAN__"

_OPERATORS = {
    "noop": NOOP,
    "sample": SAMPLE,
    "calibrate": CALIBRATE,
    "image": IMAGE,
    "send": SEND,
    "drop": DROP,
}


class RoversPDDLStreamPlanner:
    """Plan a ``rovers`` instance upstream and replay it as environment actions."""

    def __init__(self, env: RoversEnv) -> None:
        self._env = env

    def plan(self, *, max_time: float, seed: int) -> list[dict[str, Any]]:
        """Return the plan's steps, or raise :class:`PlanningFailure`."""
        payload = self._instance(max_time=max_time, seed=seed)
        worker = Path(__file__).with_name("pddlstream_rovers_worker.py")
        completed = subprocess.run(
            [sys.executable, str(worker)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
            # The planner has its own wall-clock budget; the subprocess is killed a
            # little after it so a wedged sampler cannot outlive the episode.
            timeout=max_time + 60.0,
        )
        result = self._parse(completed.stdout)
        if result is None:
            raise PlanningFailure(
                f"planner subprocess produced no result (exit {completed.returncode}): "
                f"{completed.stderr[-400:]}"
            )
        if not result.get("solved"):
            raise PlanningFailure(result.get("error") or "no plan within the budget")
        return list(result["steps"])

    @staticmethod
    def _parse(stdout: str) -> dict[str, Any] | None:
        """Pull the framed result out of the worker's noisy stdout."""
        for line in reversed(stdout.splitlines()):
            if line.startswith(_RESULT_MARKER):
                return dict(json.loads(line[len(_RESULT_MARKER) :]))
        return None

    def _instance(self, *, max_time: float, seed: int) -> dict[str, Any]:
        """Describe the evaluated episode's layout for the stock scene."""
        # pylint: disable=import-outside-toplevel
        from robocode.environments.ss_pybullet import get_point

        env = self._env
        with env.client():
            rovers = [
                [float(v) for v in env.rover_conf(i)] for i in range(len(env.rovers))
            ]
            objectives = [[float(v) for v in get_point(b)] for b in env.objectives]
            rocks = [[float(v) for v in get_point(b)] for b in env.rocks]
            soils = [[float(v) for v in get_point(b)] for b in env.soils]
            # The pillars are the obstacles the layout scatters; the walls, the mounds
            # and the lander are fixed furniture the stock scene already matches.
            obstacles = [
                [float(v) for v in get_point(b)] for b in env.scattered_obstacles
            ]
        return {
            "rovers": rovers,
            "objectives": objectives,
            "rocks": rocks,
            "soils": soils,
            "obstacles": obstacles,
            "max_time": max_time,
            "seed": seed,
        }

    def actions(self, steps: list[dict[str, Any]]) -> Iterator[NDArray[Any]]:
        """Yield the actions that execute a plan, one plan step at a time."""
        for step in steps:
            index = int(step["rover"])
            for waypoint in step.get("waypoints") or []:
                yield from self._drive(index, np.asarray(waypoint, dtype=float))
            operator = _OPERATORS[step["op"]]
            if operator != NOOP:
                yield self._action(index, operator=operator)

    def _drive(self, index: int, target: NDArray[Any]) -> Iterator[NDArray[Any]]:
        """Yield actions taking one rover to a plan waypoint.

        Waypoints come from the plan's own motion trajectories, so consecutive ones are
        close together and free of the obstacles the motion stream routed around;
        tracking them greedily is enough. Driving at a motion's endpoint instead would
        head straight through whatever it was routed around.
        """
        env = self._env
        for _ in range(_MAX_STEPS_PER_STEP):
            conf = env.rover_conf(index)
            error = target - conf
            error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
            if (
                float(np.linalg.norm(error[:2])) <= _POSITION_TOLERANCE
                and abs(float(error[2])) <= _ANGLE_TOLERANCE
            ):
                return
            yield self._action(index, delta=error)

    def _action(
        self, index: int, delta: Any = None, operator: int = NOOP
    ) -> NDArray[Any]:
        """One environment action driving a single rover; the other holds still."""
        action = np.zeros(_ACTION_DIM, dtype=np.float32)
        for other in range(len(self._env.rovers)):
            action[4 * other + 3] = operator_action(NOOP)
        if delta is not None:
            action[4 * index] = float(np.clip(delta[0], -_MAX_DELTA, _MAX_DELTA))
            action[4 * index + 1] = float(np.clip(delta[1], -_MAX_DELTA, _MAX_DELTA))
            action[4 * index + 2] = float(
                np.clip(delta[2], -_MAX_DELTA_THETA, _MAX_DELTA_THETA)
            )
        action[4 * index + 3] = operator_action(operator)
        return action
