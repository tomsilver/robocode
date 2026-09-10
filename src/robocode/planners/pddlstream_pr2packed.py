"""Stock PDDLStream planning for the PR2 ``packed`` environment.

``PR2PackedEnv`` re-exposes PDDLStream's ``packed`` benchmark as a closed-loop
gymnasium environment but deliberately ships none of the planner: the PDDL domain
and the stream samplers stay upstream. This module borrows them back, so the
benchmark's own solver can be scored on the same episodes as robocode's agents.

The plan is computed by :mod:`robocode.planners.pddlstream_pr2packed_worker` in a
subprocess, because the stock tree vendors a second copy of ss-pybullet whose
physics-client globals collide with the one the evaluated environment holds. What
crosses the pipe is joint-space waypoints, which :class:`PR2PackedPDDLStreamPlanner`
turns into the environment's bounded delta actions by servoing toward each in turn
-- the same tracking the hand-written oracle does, since a PDDLStream trajectory is
a sequence of configurations and the environment accepts only bounded deltas.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from robocode.environments.pr2_tamp_base import PR2TampEnv
from robocode.environments.pr2_tamp_blocked_env import PR2BlockedEnv
from robocode.oracles.pr2packed.planning import shortest

logger = logging.getLogger(__name__)

PR2PACKED_ENV_PATH = "robocode.environments.pr2_tamp_env:PR2PackedEnv"

_MAX_DELTA = 0.2
_ACTION_DIM = 11
_REACH_TOLERANCE = 0.02
# A waypoint is at most one joint-space revolution away, so this bounds the servo
# loop well above the steps any single hop needs while still ending a hop that the
# environment keeps rejecting (a rejected step leaves the robot where it was).
_MAX_STEPS_PER_WAYPOINT = 64
_RESULT_MARKER = "__PLAN__"
# How long past its planning budget the worker may run before it is killed.
_PLANNER_GRACE_SECS = 30.0
_PROBLEMS = ("packed", "blocked")


class PlanningFailure(Exception):
    """Raised when the stock planner returns no plan for an instance."""


class PR2PackedPDDLStreamPlanner:
    """Plan a ``packed`` or ``blocked`` instance upstream and replay it as actions.

    *problem* names the stock PDDLStream scene the evaluated environment mirrors.
    Its count axis differs per scene: ``packed`` counts the blocks, ``blocked``
    counts the spare green blocks on the far table.
    """

    def __init__(self, env: PR2TampEnv, problem: str = "packed") -> None:
        if problem not in _PROBLEMS:
            raise ValueError(f"problem must be one of {_PROBLEMS}, got {problem!r}")
        self._env = env
        self._problem = problem

    def plan(self, *, max_time: float, seed: int) -> list[dict[str, Any]]:
        """Return the plan's waypoints, or raise :class:`PlanningFailure`."""
        payload = self._instance(max_time=max_time, seed=seed)
        worker = Path(__file__).with_name("pddlstream_pr2packed_worker.py")
        # PDDLStream writes FastDownward's inputs and outputs below fixed paths
        # relative to the working directory, so every planning call gets its own
        # directory; concurrent planners sharing a cwd read each other's plans.
        with tempfile.TemporaryDirectory(prefix=".pddlstream-pr2packed-") as tmp:
            try:
                completed = subprocess.run(
                    [sys.executable, str(worker)],
                    input=json.dumps(payload),
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=tmp,
                    # The planner is given its own wall-clock budget; the subprocess
                    # is killed a little after it so a wedged sampler cannot outlive
                    # the episode's timeout.
                    timeout=max_time + _PLANNER_GRACE_SECS,
                )
            except subprocess.TimeoutExpired as error:
                # A planner that overran its budget found no plan within it, which
                # the caller scores as unsolved rather than as a crash.
                raise PlanningFailure(
                    f"planner subprocess exceeded its {max_time:.0f} s budget by more "
                    f"than {_PLANNER_GRACE_SECS:.0f} s"
                ) from error
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
        """Describe the evaluated episode's state for the stock scene."""
        # pylint: disable=import-outside-toplevel
        from robocode.environments.ss_pybullet import get_joint_positions, get_pose

        env = self._env
        with env.client():
            base = [float(v) for v in get_joint_positions(env.robot, env.base_joints)]
            arm = [float(v) for v in get_joint_positions(env.robot, env.arm_joints)]
            block_poses = []
            for body in env.movables:
                point, quat = get_pose(body)
                block_poses.append(
                    [[float(v) for v in point], [float(v) for v in quat]]
                )
            # The blocked pen (walls included) is re-posed per instance; the packed
            # scene has no walls to carry over.
            wall_poses = []
            if isinstance(env, PR2BlockedEnv):
                for body in env.walls:
                    point, quat = get_pose(body)
                    wall_poses.append(
                        [[float(v) for v in point], [float(v) for v in quat]]
                    )
        # The count axis is scene-specific: `packed` counts its blocks, which are
        # all of its movables; `blocked` counts the spare greens, which are its
        # movables minus the penned green and the red blocker.
        count = len(env.movables) - (2 if self._problem == "blocked" else 0)
        return {
            "problem": self._problem,
            "count": count,
            "base": base,
            "arm": arm,
            "gripper_opening": float(self._obs()[10]),
            "block_poses": block_poses,
            "wall_poses": wall_poses,
            "max_time": max_time,
            "seed": seed,
        }

    @staticmethod
    def _subsample(
        waypoints: list[NDArray[Any]], circular: NDArray[Any]
    ) -> list[NDArray[Any]]:
        """Thin one trajectory down to the environment's action resolution.

        PDDLStream returns a densely interpolated joint path -- hundreds of nodes a few
        thousandths of a radian apart -- and servoing to every node spends one action
        per node, which overruns the episode's step budget many times over while
        retracing the identical geometric path. Keeping only the nodes at least one
        action apart follows that same path at the resolution the action space can
        actually express, which is what the hand-written oracle does with a fixed stride
        over its own paths.
        """
        kept: list[NDArray[Any]] = []
        anchor: NDArray[Any] | None = None
        for waypoint in waypoints:
            if (
                anchor is None
                or np.max(np.abs(shortest(waypoint - anchor, circular))) >= _MAX_DELTA
            ):
                kept.append(waypoint)
                anchor = waypoint
        # The final node is the one the next command starts from, so it is never
        # dropped even when it falls inside the last kept node's action radius.
        if waypoints and (not kept or not np.array_equal(kept[-1], waypoints[-1])):
            kept.append(waypoints[-1])
        return kept

    def actions(self, steps: list[dict[str, Any]]) -> Iterator[NDArray[Any]]:
        """Yield the delta actions that track a plan's waypoints in order.

        Each configuration waypoint is servoed to rather than emitted as a single
        action: the environment bounds every joint delta, so a hop longer than that
        bound needs several steps, and it silently rejects a motion that collides,
        which leaves the robot short of the waypoint.
        """
        env = self._env
        for kind, waypoints in self._segments(steps):
            if kind == "gripper":
                yield self._action(gripper=-1.0 if waypoints else 1.0)
                continue
            base = kind == "base"
            joints = slice(0, 3) if base else slice(3, 10)
            circular = env.base_circular if base else env.arm_circular
            for target in self._subsample(waypoints, circular):
                for _ in range(_MAX_STEPS_PER_WAYPOINT):
                    error = shortest(target - self._obs()[joints], circular)
                    if np.max(np.abs(error)) <= _REACH_TOLERANCE:
                        break
                    yield (
                        self._action(base_target=target)
                        if base
                        else self._action(arm_target=target)
                    )

    @staticmethod
    def _segments(
        steps: list[dict[str, Any]],
    ) -> Iterator[tuple[str, list[NDArray[Any]]]]:
        """Group the flat step list into whole trajectories and gripper commands.

        Thinning is only sound over a single trajectory: consecutive commands are
        separated by a gripper action that must land at the exact configuration the
        plan grasps or releases from.
        """
        run: list[NDArray[Any]] = []
        run_kind: str | None = None
        for step in steps:
            kind = step["kind"]
            if kind == "gripper":
                if run_kind is not None:
                    yield run_kind, run
                    run, run_kind = [], None
                # A close is carried as a non-empty list, an open as an empty one.
                yield "gripper", [np.zeros(1)] if step["close"] else []
                continue
            if kind != run_kind:
                if run_kind is not None:
                    yield run_kind, run
                run, run_kind = [], kind
            run.append(np.asarray(step["values"], dtype=np.float64))
        if run_kind is not None:
            yield run_kind, run

    def _obs(self) -> NDArray[Any]:
        return np.asarray(self._env.get_state(), dtype=np.float64)

    def _action(
        self,
        base_target: Any = None,
        arm_target: Any = None,
        gripper: float = 0.0,
    ) -> NDArray[Any]:
        """Build one bounded delta action toward a configuration."""
        action = np.zeros(_ACTION_DIM, dtype=np.float32)
        obs = self._obs()
        if base_target is not None:
            delta = np.asarray(base_target) - obs[0:3]
            action[0:3] = np.clip(
                shortest(delta, self._env.base_circular), -_MAX_DELTA, _MAX_DELTA
            )
        if arm_target is not None:
            delta = np.asarray(arm_target) - obs[3:10]
            action[3:10] = np.clip(
                shortest(delta, self._env.arm_circular), -_MAX_DELTA, _MAX_DELTA
            )
        action[10] = gripper
        return action
