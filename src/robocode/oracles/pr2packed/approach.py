"""Oracle approach for the PR2 ``packed`` environment.

Places blocks one at a time. For each block it samples a base pose that can reach
both the block and its target cell on the plate, plans validated joint paths for
every transit, and then tracks those paths with the environment's bounded delta
actions. If any stage fails -- no reachable base pose, no collision-free path, or a
step the environment rejects -- it drops whatever it holds and resamples that block's
plan rather than abandoning the episode.

The oracle is a reference policy, not a scored agent: it reads the simulator directly
to do IK and motion planning. It never mutates the episode's state, though. Planning
runs under a ``WorldSaver`` and the robot only ever moves through ``step``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach
from robocode.environments.pr2_tamp_env import PR2PackedEnv
from robocode.environments.ss_pybullet import Attachment
from robocode.oracles.pr2packed.planning import (
    PATH_RESOLUTION,
    PlanFailure,
    motion,
    plan_pick,
    plate_cells,
    shortest,
)

_MAX_DELTA = 0.2
_REACH_TOLERANCE = 0.02
_ACTION_DIM = 11
_ATTEMPTS_PER_BLOCK = 6


class PR2PackedOracleApproach(BaseApproach[NDArray[Any], NDArray[Any]]):
    """Oracle that picks each block and releases it over a free plate cell."""

    def __init__(
        self,
        action_space: Space[NDArray[Any]],
        observation_space: Space[NDArray[Any]],
        seed: int = 0,
        primitives: dict[str, Callable[..., Any]] | None = None,
        env_description_path: str | None = None,
        env: PR2PackedEnv | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_space,
            observation_space,
            seed,
            primitives or {},
            env_description_path,
        )
        if env is None:
            raise ValueError("PR2PackedOracleApproach needs the environment to plan")
        self._env = env
        # For the variable-count wrapper the scene lives in a per-count backend, and
        # that is what carries the body ids and joint groups planning needs. It only
        # exists after a reset has chosen a count, so it is resolved there.
        self._backend: PR2PackedEnv | None = None
        self._plan: Iterator[NDArray[Any]] = iter(())
        self._failure: str | None = None

    @property
    def failure(self) -> str | None:
        """Why the oracle stopped early, or None if it is still working."""
        return self._failure

    def reset(self, state: Any, info: dict[str, Any]) -> None:
        super().reset(state, info)
        self._backend = (
            self._env
            if isinstance(self._env, PR2PackedEnv)
            else self._env.current_backend
        )
        self._failure = None
        self._plan = self._solve()

    def _get_action(self) -> NDArray[Any]:
        try:
            return next(self._plan)
        except StopIteration:
            # Out of plan: hold still rather than perturbing a finished scene.
            return np.zeros(_ACTION_DIM, dtype=np.float32)

    # ------------------------------------------------------------------ control

    def _obs(self) -> NDArray[Any]:
        """The current robot/scene vector.

        Read from the backend rather than from ``_last_state`` so the oracle works
        unchanged under the variable-count wrapper, whose observations are
        object-centric states rather than the flat vector planning works in.
        """
        assert self._backend is not None, "reset() must run first"
        return self._backend.get_state()

    def _action(
        self,
        base_target: Any = None,
        arm_target: Any = None,
        gripper: float = 0.0,
    ) -> NDArray[Any]:
        action = np.zeros(_ACTION_DIM, dtype=np.float32)
        obs = self._obs()
        assert self._backend is not None
        if base_target is not None:
            delta = np.array(base_target) - obs[0:3]
            action[0:3] = np.clip(
                shortest(delta, self._backend.base_circular), -_MAX_DELTA, _MAX_DELTA
            )
        if arm_target is not None:
            delta = np.array(arm_target) - obs[3:10]
            action[3:10] = np.clip(
                shortest(delta, self._backend.arm_circular), -_MAX_DELTA, _MAX_DELTA
            )
        action[10] = gripper
        return action

    def _follow(self, path: list[NDArray[Any]], base: bool) -> Iterator[NDArray[Any]]:
        """Yield one action per path node.

        The path is subsampled so consecutive nodes are at most one action apart in
        every joint, which makes each clipped action land exactly on the next node.
        Every node was validated against the environment's collision model, so a node
        the robot fails to reach means the two models have diverged; that raises rather
        than spinning in place.
        """
        stride = max(1, int(_MAX_DELTA / PATH_RESOLUTION))
        nodes = list(path[::stride])
        if not np.array_equal(nodes[-1], path[-1]):
            nodes.append(path[-1])
        joints = slice(0, 3) if base else slice(3, 10)
        assert self._backend is not None
        circular = self._backend.base_circular if base else self._backend.arm_circular
        for node in nodes[1:]:
            yield (
                self._action(base_target=node)
                if base
                else self._action(arm_target=node)
            )
            error = shortest(np.array(node) - self._obs()[joints], circular)
            if np.max(np.abs(error)) > _REACH_TOLERANCE:
                raise PlanFailure("the environment rejected a step")

    def _transit(
        self, joints: list[int], target: Any, base: bool, attachment: Any = None
    ) -> Iterator[NDArray[Any]]:
        assert self._backend is not None
        path = motion(self._backend, joints, target, attachment=attachment)
        if path is None:
            raise PlanFailure("no collision-free path")
        yield from self._follow(path, base=base)

    # ------------------------------------------------------------------ planning

    def _place_block(
        self, block: int, cell: Any, carry: NDArray[Any]
    ) -> Iterator[NDArray[Any]]:
        """Pick *block* and release it over *cell*, or raise PlanFailure."""
        env = self._backend
        assert env is not None
        plan = plan_pick(env, block, cell, self._rng)
        if plan is None:
            raise PlanFailure("no reachable base pose")
        yield from self._transit(env.arm_joints, carry, base=False)
        yield from self._transit(env.base_joints, plan["base"], base=True)
        yield from self._transit(env.arm_joints, plan["lift"], base=False)
        yield from self._transit(env.arm_joints, plan["grasp"], base=False)
        yield self._action(gripper=-1.0)

        # Plan the carry against the grasp the environment actually formed: the arm
        # lands within IK tolerance of the planned config, not exactly on it, so the
        # planned grasp transform would be slightly wrong.
        attachment: Attachment | None = env.attachment
        if attachment is None or attachment.child != block:
            raise PlanFailure("the grasp did not take")
        for target in (plan["lift"], plan["release"]):
            yield from self._transit(
                env.arm_joints, target, base=False, attachment=attachment
            )
        yield self._action(gripper=1.0)

    def _solve(self) -> Iterator[NDArray[Any]]:
        env = self._backend
        assert env is not None, "reset() must run first"
        carry = np.array(env.initial_arm_conf)
        free_cells = plate_cells(env, len(env.blocks))
        for index, block in enumerate(env.blocks):
            placed = False
            for _ in range(_ATTEMPTS_PER_BLOCK):
                for cell_index, cell in enumerate(free_cells):
                    try:
                        yield from self._place_block(block, cell, carry)
                    except PlanFailure:
                        if env.attachment is not None:
                            # Drop it and replan from wherever the robot ended up.
                            yield self._action(gripper=1.0)
                        continue
                    free_cells.pop(cell_index)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                self._failure = f"could not place block {index}"
                return
