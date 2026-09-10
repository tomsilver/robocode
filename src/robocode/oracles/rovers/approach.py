"""Oracle approach for the PDDLStream ``rovers`` environment.

A wall down the middle of the arena splits its floor into two disconnected halves with
one rover parked in each, so neither can do the mission alone: whatever is on the left
is the left rover's errand and whatever is on the right is the right rover's. The wall
is only ankle-high, though, and the cameras sit above it, so *sight* crosses it freely
-- either rover can radio the lander wherever it stands. Both facts come straight from
the benchmark's own scene, and together they are what makes this a two-robot task.

The oracle therefore assigns each errand to a rover that can actually reach a place to
do it from, builds one plan per rover, and runs both at once, since every step carries
an action for each. Each plan is a sequence of "drive here, then apply this operator";
driving is a grid search over the floor and the places to stand are rejection-sampled
against the environment's own visibility test.

The oracle is a reference policy, not a scored agent: it reads the simulator directly
to plan. It never mutates the episode's state -- planning runs under a ``WorldSaver``
and the rovers only ever move through ``step``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray

from robocode.approaches.base_approach import BaseApproach
from robocode.environments.rovers_env import (
    CALIBRATE,
    DROP,
    IMAGE,
    NOOP,
    SAMPLE,
    SAMPLE_RADIUS,
    SEND,
    RoversEnv,
    operator_action,
)
from robocode.environments.ss_pybullet import get_point
from robocode.oracles.rovers.planning import (
    PlanFailure,
    com_viewpoint,
    free_cells,
    image_viewpoint,
    route,
    standable,
)

_MAX_DELTA = 0.2
_MAX_DELTA_THETA = 0.4
_ACTION_DIM = 8
_POSITION_TOLERANCE = 0.08
_ANGLE_TOLERANCE = 0.15
# A waypoint is at most one grid step away, so this bounds the servo loop well above
# what any hop needs while still giving up on one the environment keeps rejecting.
_MAX_STEPS_PER_WAYPOINT = 24
# Consecutive rejected motions before the route is abandoned and replanned.
_STUCK_LIMIT = 4


class RoversOracleApproach(BaseApproach[NDArray[Any], NDArray[Any]]):
    """Oracle that splits the mission between the two rovers and runs both at once."""

    def __init__(
        self,
        action_space: Space[NDArray[Any]],
        observation_space: Space[NDArray[Any]],
        seed: int = 0,
        primitives: dict[str, Callable[..., Any]] | None = None,
        env_description_path: str | None = None,
        env: RoversEnv | None = None,
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
            raise ValueError("RoversOracleApproach needs the environment to plan")
        self._env = env
        self._backend: RoversEnv | None = None
        self._rng = np.random.default_rng(seed)
        self._plans: list[Iterator[NDArray[Any]]] = []
        self._failure: str | None = None

    @property
    def failure(self) -> str | None:
        """Why the oracle stopped early, or None if it is still working."""
        return self._failure

    def reset(self, state: Any, info: dict[str, Any]) -> None:
        super().reset(state, info)
        self._backend = (
            self._env if isinstance(self._env, RoversEnv) else self._env.current_backend
        )
        self._failure = None
        self._plans = self._build_plans()

    def _get_action(self) -> NDArray[Any]:
        """Advance both rovers' plans by one step.

        A rover whose plan has run out holds still rather than perturbing a scene the
        other rover is still working in.
        """
        action = np.zeros(_ACTION_DIM, dtype=np.float32)
        for index, plan in enumerate(self._plans):
            try:
                action[4 * index : 4 * index + 4] = next(plan)
            except StopIteration:
                continue
        return action

    # ------------------------------------------------------------------ control

    def _conf(self, index: int) -> NDArray[Any]:
        assert self._backend is not None
        return self._backend.rover_conf(index)

    @staticmethod
    def _chunk(delta: Any = None, operator: int = NOOP) -> NDArray[Any]:
        """One rover's slice of an action: a bounded base delta plus an operator."""
        chunk = np.zeros(4, dtype=np.float32)
        if delta is not None:
            chunk[0] = float(np.clip(delta[0], -_MAX_DELTA, _MAX_DELTA))
            chunk[1] = float(np.clip(delta[1], -_MAX_DELTA, _MAX_DELTA))
            chunk[2] = float(np.clip(delta[2], -_MAX_DELTA_THETA, _MAX_DELTA_THETA))
        # The centre of the operator's band, so rounding cannot select a neighbour.
        chunk[3] = operator_action(operator)
        return chunk

    def _drive(
        self, index: int, waypoints: list[NDArray[Any]]
    ) -> Iterator[NDArray[Any]]:
        """Yield actions following a sequence of base positions.

        A rejected motion leaves the rover exactly where it was, so commanding the same
        delta again achieves nothing: a rover wedged against a pillar would otherwise
        push into it for the rest of the episode. Repeated no-progress is detected, and
        answered first by sliding sideways to clear whatever is in the way and then, if
        that fails, by abandoning the route so the caller can plan a new one from
        wherever the rover ended up.
        """
        for point in waypoints:
            stuck = 0
            for _ in range(_MAX_STEPS_PER_WAYPOINT):
                conf = self._conf(index)
                error = np.asarray(point)[:2] - conf[:2]
                if float(np.linalg.norm(error)) <= _POSITION_TOLERANCE:
                    break
                if stuck >= _STUCK_LIMIT:
                    return
                if stuck:
                    # Perpendicular to the blocked direction, alternating sides.
                    side = 1.0 if stuck % 2 else -1.0
                    nudge = side * np.array([-error[1], error[0]])
                    scale = _MAX_DELTA / max(float(np.linalg.norm(nudge)), 1e-6)
                    yield self._chunk([nudge[0] * scale, nudge[1] * scale, 0.0])
                else:
                    yield self._chunk([error[0], error[1], 0.0])
                moved = float(np.linalg.norm(self._conf(index)[:2] - conf[:2]))
                stuck = 0 if moved > 1e-4 else stuck + 1

    def _turn(self, index: int, heading: float) -> Iterator[NDArray[Any]]:
        """Yield actions rotating in place to *heading*."""
        for _ in range(_MAX_STEPS_PER_WAYPOINT):
            conf = self._conf(index)
            error = (heading - conf[2] + np.pi) % (2 * np.pi) - np.pi
            if abs(float(error)) <= _ANGLE_TOLERANCE:
                return
            yield self._chunk([0.0, 0.0, error])

    def _go(
        self,
        index: int,
        grid: dict[tuple[int, int], bool],
        target: NDArray[Any],
        heading: float | None = None,
        predicate: Callable[[], bool] | None = None,
        rounds: int = 4,
    ) -> Iterator[NDArray[Any]]:
        """Drive to a base position, optionally on a heading and until a test passes.

        The route is recomputed from wherever the rover actually is, a few times over,
        rather than followed once and hoped for. A rover that gets wedged against a
        pillar has its motions rejected and would otherwise burn through the remaining
        waypoints on the spot and arrive nowhere; re-routing from the stuck pose plans
        around whatever stopped it.

        With a *predicate* -- the environment's own test for whatever this pose was
        chosen for -- arrival means the test passing, not merely being within driving
        tolerance. Visibility is sharp, and a few centimetres is enough to put a pillar
        back in the way.
        """
        assert self._backend is not None
        for _ in range(rounds):
            here = self._conf(index)
            if float(np.linalg.norm(np.asarray(target)[:2] - here[:2])) <= (
                _POSITION_TOLERANCE
            ):
                break
            waypoints = route(grid, here[:2], target[:2])
            yield from self._drive(index, waypoints)
        if heading is not None:
            yield from self._turn(index, heading)
        if predicate is None:
            return
        for _ in range(_MAX_STEPS_PER_WAYPOINT):
            if predicate():
                return
            conf = self._conf(index)
            error = np.asarray(target)[:3] - conf
            error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
            if float(np.max(np.abs(error))) < 1e-3:
                return
            yield self._chunk(error)

    # ------------------------------------------------------------------ planning

    def _build_plans(self) -> list[Iterator[NDArray[Any]]]:
        """Assign every errand to a rover that can reach it, and plan both rovers."""
        env = self._backend
        assert env is not None, "reset() must run first"
        # Planning reads body poses directly, and pybullet_tools addresses PyBullet
        # through a module-global client. Under the variable-count wrapper there is one
        # client per count, so the whole of planning has to hold this backend's.
        with env.client():
            return self._plan_all(env)

    def _plan_all(self, env: RoversEnv) -> list[Iterator[NDArray[Any]]]:
        """Assign the errands and build both rovers' plans, client already held."""
        grids = [free_cells(env, index) for index in range(len(env.rovers))]
        viewpoints = [com_viewpoint(env, i, grids[i], self._rng) for i in (0, 1)]

        # Objectives go to whichever rover can find somewhere to photograph them from.
        assigned: list[list[tuple[str, Any]]] = [[], []]
        for objective in env.objectives:
            for index in (0, 1):
                spot = image_viewpoint(env, index, objective, grids[index], self._rng)
                if spot is not None:
                    assigned[index].append(("image", np.append(spot, float(objective))))
                    break
            else:
                self._failure = f"no rover can photograph objective {objective}"

        # One stone and one soil, each taken by whichever rover can drive onto it.
        for kind, bodies in (("stone", env.rocks), ("soil", env.soils)):
            for body, index in self._reachable_samples(env, grids, bodies):
                assigned[index].append(("sample", np.array(get_point(body))[:2]))
                break
            else:
                self._failure = f"no rover can reach any {kind} sample"

        return [
            self._rover_plan(index, grids[index], viewpoints[index], assigned[index])
            for index in (0, 1)
        ]

    def _reachable_samples(
        self,
        env: RoversEnv,
        grids: list[dict[tuple[int, int], bool]],
        bodies: list[int],
    ) -> list[tuple[int, int]]:
        """(sample, rover) pairs a rover can actually drive onto, nearest first."""
        options: list[tuple[float, int, int]] = []
        for body in bodies:
            point = np.array(get_point(body))[:2]
            for index in range(len(env.rovers)):
                here = self._conf(index)[:2]
                # Standing on the sample is the operator's precondition, so a sample
                # wedged against a pillar is unusable however routable its
                # neighbourhood is. Routing alone would pick it and then stall a
                # quarter-metre short, with the operator refused every step.
                if not standable(env, index, point):
                    continue
                try:
                    route(grids[index], here, point)
                except PlanFailure:
                    continue
                options.append((float(np.linalg.norm(point - here)), body, index))
        options.sort(key=lambda item: item[0])
        return [(body, index) for _, body, index in options]

    def _rover_plan(
        self,
        index: int,
        grid: dict[tuple[int, int], bool],
        com: NDArray[Any] | None,
        errands: list[tuple[str, Any]],
    ) -> Iterator[NDArray[Any]]:
        """One rover's whole plan, as a generator of its action slices."""
        env = self._backend
        assert env is not None
        for kind, target in errands:
            try:
                if kind == "image":
                    objective = int(target[3])
                    yield from self._reach(
                        index,
                        grid,
                        target,
                        _visible_test(env, index, objective),
                        _image_sampler(env, index, objective, grid, self._rng),
                    )
                    # The operator shoots the nearest objective this rover still
                    # needs, which is not always the one this viewpoint was chosen
                    # for. Repeat until the intended one is held: each shot takes a
                    # different objective off the "still needed" list, so the target
                    # comes up within a few.
                    for _ in range(len(env.objectives)):
                        if env.has_image(index, objective):
                            break
                        yield self._chunk(operator=CALIBRATE)
                        yield self._chunk(operator=IMAGE)
                    if not env.has_image(index, objective):
                        self._failure = (
                            f"rover {index} could not photograph {objective}"
                        )
                else:
                    yield from self._go(index, grid, np.asarray(target))
                    # Close the last few centimetres: the grid lands within a cell of
                    # the sample and the operator needs the base on top of it.
                    yield from self._creep(index, np.asarray(target))
                    yield self._chunk(operator=SAMPLE)
                if com is None:
                    self._failure = f"rover {index} cannot see the lander"
                    return
                yield from self._reach(
                    index,
                    grid,
                    com,
                    _com_test(env, index),
                    _com_sampler(env, index, grid, self._rng),
                )
                yield self._chunk(operator=SEND)
                if kind == "sample":
                    yield self._chunk(operator=DROP)
            except PlanFailure as error:
                self._failure = f"rover {index}: {error}"
                return
        home = env.home_confs[index]
        try:
            # Being home is a goal condition with its own tolerance, so it is servoed
            # to like the viewpoints rather than merely driven at: stopping inside
            # driving tolerance is not the same as satisfying it.
            # More re-routing rounds than a viewpoint gets: home is a fixed pose that
            # cannot be re-chosen when the way there is awkward, so the only recourse
            # is to keep planning around whatever blocked the last attempt.
            yield from self._go(
                index,
                grid,
                home,
                heading=float(home[2]),
                predicate=_home_test(env, index),
                rounds=10,
            )
        except PlanFailure as error:
            self._failure = f"rover {index} could not get home: {error}"

    def _reach(
        self,
        index: int,
        grid: dict[tuple[int, int], bool],
        first: NDArray[Any],
        predicate: Callable[[], bool],
        resample: Callable[[], NDArray[Any] | None],
        tries: int = 6,
    ) -> Iterator[NDArray[Any]]:
        """Get to somewhere *predicate* holds, re-choosing the spot if it does not.

        A viewpoint is chosen against the scene as it stands when the plan is built,
        and the other rover then drives through it. Rovers occlude each other's rays --
        upstream models that too, with its ``Blocked`` fluent over rays and rover
        configurations -- so a radio spot that was clear at planning time can be blocked
        by the time this rover arrives. Re-sampling against the live scene fixes it,
        where servoing harder into a blocked pose never would.
        """
        target: NDArray[Any] | None = first
        for attempt in range(tries):
            if target is None:
                self._failure = f"rover {index} ran out of viewpoints"
                return
            yield from self._go(
                index, grid, target, heading=float(target[2]), predicate=predicate
            )
            if predicate():
                return
            if attempt < tries - 1:
                target = resample()
        self._failure = (
            f"rover {index} never reached a pose satisfying its viewpoint test"
        )

    def _creep(self, index: int, target: NDArray[Any]) -> Iterator[NDArray[Any]]:
        """Close on a sample until the environment would accept the operator."""
        for _ in range(_MAX_STEPS_PER_WAYPOINT):
            conf = self._conf(index)
            error = target[:2] - conf[:2]
            if float(np.linalg.norm(error)) <= SAMPLE_RADIUS * 0.6:
                return
            yield self._chunk([error[0], error[1], 0.0])


def _with_target(spot: NDArray[Any] | None, objective: int) -> NDArray[Any] | None:
    """Tag a viewpoint with the objective it was chosen for."""
    return None if spot is None else np.append(spot, float(objective))


def _visible_test(env: RoversEnv, index: int, objective: int) -> Callable[[], bool]:
    """Whether this rover can currently photograph this objective."""
    return lambda: env.image_visible(index, objective)


def _home_test(env: RoversEnv, index: int) -> Callable[[], bool]:
    """Whether this rover is back where the goal wants it."""
    return lambda: env.at_home(index)


def _com_test(env: RoversEnv, index: int) -> Callable[[], bool]:
    """Whether this rover can currently reach the lander."""
    return lambda: env.com_visible(index)


def _image_sampler(
    env: RoversEnv,
    index: int,
    objective: int,
    grid: dict[tuple[int, int], bool],
    rng: np.random.Generator,
) -> Callable[[], NDArray[Any] | None]:
    """Draw another place to photograph this objective from."""
    return lambda: _with_target(
        image_viewpoint(env, index, objective, grid, rng), objective
    )


def _com_sampler(
    env: RoversEnv,
    index: int,
    grid: dict[tuple[int, int], bool],
    rng: np.random.Generator,
) -> Callable[[], NDArray[Any] | None]:
    """Draw another place to radio the lander from."""
    return lambda: com_viewpoint(env, index, grid, rng)
