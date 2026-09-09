"""Oracle approach for the PR2 ``blocked`` environment.

The goal is existential -- any one green block on the plate -- so the oracle picks a
route rather than a sequence. It prefers the penned block: everything that matters is
already on the near table, so the cost is one relocation of the red blocker plus one
pick and place, with no base travel to speak of. Fetching a spare instead means a
nine-metre drive each way, which at -1 per step is far more expensive even though it
skips the relocation. Spares are therefore the fallback, tried only if the penned
route fails, and with no spares that fallback does not exist.

Picks use side grasps, because the pen walls are as tall as the block. That is also
why the blocker has to move first: it stands in the one horizontal direction the walls
leave open, and the environment's own collision test rejects any grasp that reaches
through it.

The oracle is a reference policy, not a scored agent: it reads the simulator directly
to do IK and motion planning. It never mutates the episode's state, though. Planning
runs under a ``WorldSaver`` and the robot only ever moves through ``step``.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from typing import Any

import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach
from robocode.environments.pr2_tamp_blocked_env import PR2BlockedEnv
from robocode.environments.ss_pybullet import Attachment
from robocode.oracles.pr2blocked.planning import (
    PATH_RESOLUTION,
    PlanFailure,
    motion,
    plan_transfer,
    plan_unblock,
    plate_target,
    shortest,
)

_MAX_DELTA = 0.2
_REACH_TOLERANCE = 0.02
_ACTION_DIM = 11
# Attempts at one block before giving up on it and trying the next route.
_ATTEMPTS_PER_BLOCK = 4
# Parking spots tried per attempt at the penned block, nearest the pen first.
_PARK_ATTEMPTS = 6


class PR2BlockedOracleApproach(BaseApproach[NDArray[Any], NDArray[Any]]):
    """Oracle that gets one green block onto the plate, moving the blocker if needed."""

    def __init__(
        self,
        action_space: Space[NDArray[Any]],
        observation_space: Space[NDArray[Any]],
        seed: int = 0,
        primitives: dict[str, Callable[..., Any]] | None = None,
        env_description_path: str | None = None,
        env: PR2BlockedEnv | None = None,
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
            raise ValueError("PR2BlockedOracleApproach needs the environment to plan")
        self._env = env
        # For the variable-count wrapper the scene lives in a per-count backend, and
        # that is what carries the body ids and joint groups planning needs. It only
        # exists after a reset has chosen a count, so it is resolved there.
        self._backend: PR2BlockedEnv | None = None
        self._rng = np.random.default_rng(seed)
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
            if isinstance(self._env, PR2BlockedEnv)
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

    def _move_block(
        self, block: int, target: Any, carry: NDArray[Any]
    ) -> Iterator[NDArray[Any]]:
        """Pick *block* and release it over *target*, or raise PlanFailure.

        Two base poses: one to grasp from and one to release at. The plate is 0.6m
        from the penned block and nine metres from the spares, so unlike ``packed``
        there is a drive in the middle of every transfer, carried out holding the
        block.
        """
        env = self._backend
        assert env is not None
        plan = plan_transfer(env, block, target, self._rng)
        if plan is None:
            raise PlanFailure("no reachable base pose")
        yield from self._execute_transfer(block, plan, carry)

    def _execute_transfer(
        self, block: int, plan: dict[str, Any], carry: NDArray[Any]
    ) -> Iterator[NDArray[Any]]:
        """Drive an already-planned transfer, or raise PlanFailure.

        Kept separate from planning so a plan that was verified before being committed
        to -- the blocker's, whose worth depends on what it frees -- is the plan that
        actually runs, rather than a fresh sample of it.
        """
        env = self._backend
        assert env is not None
        yield from self._transit(env.arm_joints, carry, base=False)
        yield from self._transit(env.base_joints, plan["pick_base"], base=True)
        yield from self._transit(env.arm_joints, plan["lift"], base=False)
        yield from self._transit(env.arm_joints, plan["grasp"], base=False)
        yield self._action(gripper=-1.0)

        # Plan the carry against the grasp the environment actually formed: the arm
        # lands within IK tolerance of the planned config, not exactly on it, so the
        # planned grasp transform would be slightly wrong.
        attachment: Attachment | None = env.attachment
        if attachment is None or attachment.child != block:
            raise PlanFailure("the grasp did not take")
        yield from self._transit(
            env.arm_joints, plan["lift"], base=False, attachment=attachment
        )
        yield from self._transit(
            env.base_joints, plan["place_base"], base=True, attachment=attachment
        )
        yield from self._transit(
            env.arm_joints, plan["release"], base=False, attachment=attachment
        )
        yield self._action(gripper=1.0)

    def _drop_anything_held(self) -> Iterator[NDArray[Any]]:
        """Release whatever is held so the next attempt starts empty-handed."""
        env = self._backend
        assert env is not None
        if env.attachment is not None:
            yield self._action(gripper=1.0)

    def _solve(self) -> Iterator[NDArray[Any]]:
        env = self._backend
        assert env is not None, "reset() must run first"
        carry = np.array(env.initial_arm_conf)

        # The penned block first: it and the plate are on the same table, so the whole
        # route is local, where a spare costs a nine-metre drive each way.
        for _ in range(_ATTEMPTS_PER_BLOCK):
            plan = plan_unblock(env, self._rng)
            if plan is None:
                break
            try:
                yield from self._execute_transfer(env.blocker, plan, carry)
            except PlanFailure:
                yield from self._drop_anything_held()
                continue
            if (yield from self._place(env.penned, carry)):
                return

        for spare in env.spares:
            if (yield from self._place(spare, carry)):
                return

        self._failure = (
            "could not place the penned block after moving the blocker, "
            f"nor any of the {len(env.spares)} spare block(s)"
        )

    def _place(
        self, block: int, carry: NDArray[Any]
    ) -> Generator[NDArray[Any], None, bool]:
        """Try to put *block* on the plate; the generator returns whether it worked.

        The boolean is the generator's *return* value, so callers read it with
        ``if (yield from self._place(...)):`` -- the actions come out as it runs.
        """
        env = self._backend
        assert env is not None
        target = plate_target(env)
        for _ in range(_ATTEMPTS_PER_BLOCK):
            try:
                yield from self._move_block(block, target, carry)
            except PlanFailure:
                yield from self._drop_anything_held()
                continue
            return True
        return False
