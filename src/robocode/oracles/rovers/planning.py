"""Navigation and viewpoint selection for the ``rovers`` oracle.

The task is not a manipulation problem, so none of the PR2 machinery applies. What a
rover needs is somewhere to stand -- over a sample, in sight of an objective, in sight
of the lander -- and a way to drive there without hitting anything. Both are solved
here directly against the environment: viewpoints are rejection-sampled and checked
with the environment's own visibility test, and paths come from a grid search over the
arena floor using the environment's own collision test, so a plan the oracle commits to
is one the environment will actually execute.
"""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robocode.environments.rovers_env import COM_RANGE, VIS_RANGE, RoversEnv
from robocode.environments.rovers_scenes import BASE_EXTENT
from robocode.environments.ss_pybullet import (
    WorldSaver,
    get_joint_positions,
    get_point,
    pairwise_collision,
    set_joint_positions,
)

# Grid pitch for the navigation search. The arena is 5m across and a rover moves at
# most 0.2m a step, so this is fine enough that a grid path is close to what the rover
# can actually drive and coarse enough to search quickly.
GRID_PITCH = 0.2
# Keep planned waypoints this far inside the boundary walls.
MARGIN = 0.45


class PlanFailure(RuntimeError):
    """No viewpoint or route was found; the caller should try something else."""


def _in_collision(env: RoversEnv, index: int, conf: Any) -> bool:
    """Whether a rover placed at *conf* collides with anything it must avoid."""
    rover, joints = env.rovers[index], env.base_joint_groups[index]
    restore = get_joint_positions(rover, joints)
    try:
        set_joint_positions(rover, joints, conf)
        others = [r for r in env.rovers if r != rover]
        return any(pairwise_collision(rover, b) for b in env.obstacles + others)
    finally:
        set_joint_positions(rover, joints, restore)


def free_cells(env: RoversEnv, index: int) -> dict[tuple[int, int], bool]:
    """Cells this rover can actually get to, as a grid of base positions.

    Collision-free is not the same as reachable here. A wall runs the full width of the
    arena, so the floor is two disconnected halves with one rover in each, and a cell in
    the far half is perfectly free while being unreachable forever. The grid is
    therefore flooded from the rover's own position, so a viewpoint drawn from it is
    somewhere this rover can be.
    """
    limit = BASE_EXTENT / 2.0 - MARGIN
    steps = int(np.floor(limit / GRID_PITCH))
    clear: set[tuple[int, int]] = set()
    with env.client(), WorldSaver():
        start = _cell(
            get_joint_positions(env.rovers[index], env.base_joint_groups[index])
        )
        for i in range(-steps, steps + 1):
            for j in range(-steps, steps + 1):
                if not _in_collision(env, index, [i * GRID_PITCH, j * GRID_PITCH, 0.0]):
                    clear.add((i, j))
    reachable = _flood(clear, start)
    return {cell: cell in reachable for cell in clear}


def _flood(clear: set[tuple[int, int]], start: tuple[int, int]) -> set[tuple[int, int]]:
    """The clear cells connected to *start*, eight-connected."""
    if start not in clear:
        near = min(
            clear,
            key=lambda c: (c[0] - start[0]) ** 2 + (c[1] - start[1]) ** 2,
            default=None,
        )
        if near is None:
            return set()
        start = near
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nxt = (cell[0] + di, cell[1] + dj)
                if nxt in clear and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return seen


def _cell(point: Any) -> tuple[int, int]:
    return (int(round(point[0] / GRID_PITCH)), int(round(point[1] / GRID_PITCH)))


def route(
    grid: dict[tuple[int, int], bool],
    start: Any,
    goal: Any,
) -> list[NDArray[Any]]:
    """A collision-free sequence of base positions from *start* to *goal*.

    Dijkstra over the free cells, eight-connected. The rover holds whatever heading it
    has while driving; heading only matters where an operator is applied, and the caller
    turns on the spot there.
    """
    origin, target = _cell(start), _cell(goal)
    if origin == target:
        return [np.array([goal[0], goal[1]])]
    if not grid.get(target, False):
        nearest = _nearest_free(grid, target)
        if nearest is None:
            raise PlanFailure("no free cell near the goal")
        target = nearest
    if not grid.get(origin, False):
        origin_free = _nearest_free(grid, origin)
        if origin_free is None:
            raise PlanFailure("the rover is not on a free cell")
        origin = origin_free

    dist = {origin: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, origin)]
    while queue:
        cost, cell = heapq.heappop(queue)
        if cell == target:
            break
        if cost > dist.get(cell, float("inf")):
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0:
                    continue
                nxt = (cell[0] + di, cell[1] + dj)
                if not grid.get(nxt, False):
                    continue
                step = float(np.hypot(di, dj)) * GRID_PITCH
                if cost + step < dist.get(nxt, float("inf")):
                    dist[nxt] = cost + step
                    previous[nxt] = cell
                    heapq.heappush(queue, (cost + step, nxt))
    if target not in dist:
        raise PlanFailure("no route")
    path = [target]
    while path[-1] != origin:
        path.append(previous[path[-1]])
    path.reverse()
    points = [np.array([c[0] * GRID_PITCH, c[1] * GRID_PITCH]) for c in path[1:]]
    points.append(np.array([goal[0], goal[1]]))
    return points


def _nearest_free(
    grid: dict[tuple[int, int], bool], cell: tuple[int, int], radius: int = 6
) -> tuple[int, int] | None:
    """The closest free cell to *cell*, or None."""
    best, best_distance = None, float("inf")
    for (i, j), free in grid.items():
        if not free:
            continue
        distance = float(np.hypot(i - cell[0], j - cell[1]))
        if distance <= radius and distance < best_distance:
            best, best_distance = (i, j), distance
    return best


def viewpoint(
    env: RoversEnv,
    index: int,
    target: int,
    grid: dict[tuple[int, int], bool],
    rng: np.random.Generator,
    max_range: float,
    attempts: int = 400,
) -> NDArray[Any] | None:
    """A base configuration from which *target* is visible, or None.

    Inverse visibility, the same idea as the PR2 oracles' inverse reachability: rather
    than solving for a standing spot analytically, draw one from a ring around the
    target, point the rover at it, and keep it if the environment's own visibility test
    passes there. Candidates are ordered by how close they are to the rover, so the
    drive is short.
    """
    point = np.array(get_point(target))[:2]
    rover, joints = env.rovers[index], env.base_joint_groups[index]
    limit = BASE_EXTENT / 2.0 - MARGIN
    with env.client(), WorldSaver():
        here = np.array(get_joint_positions(rover, joints))[:2]
        candidates = []
        for _ in range(attempts):
            radius = rng.uniform(0.6, max_range * 0.9)
            theta = rng.uniform(-np.pi, np.pi)
            base = point + radius * np.array([np.cos(theta), np.sin(theta)])
            if np.max(np.abs(base)) > limit:
                continue
            if not grid.get(_cell(base), False):
                continue
            candidates.append((float(np.linalg.norm(base - here)), base))
        candidates.sort(key=lambda item: item[0])
        for _, base in candidates:
            # A free cell is not the same as a drivable pose: the grid is coarse, and a
            # candidate wedged between two pillars is one the rover will stall short
            # of, leaving the operator refused at a pose that tested fine when it was
            # sampled. Requiring a route makes "visible from here" mean "visible from
            # somewhere this rover can actually stand".
            try:
                route(grid, here, base)
            except PlanFailure:
                continue
            heading = float(np.arctan2(point[1] - base[1], point[0] - base[0]))
            conf = np.array([base[0], base[1], heading])
            if _in_collision(env, index, conf):
                continue
            set_joint_positions(rover, joints, conf)
            if env.visible(index, target, max_range):
                return conf
    return None


def image_viewpoint(
    env: RoversEnv,
    index: int,
    objective: int,
    grid: dict[tuple[int, int], bool],
    rng: np.random.Generator,
) -> NDArray[Any] | None:
    """Somewhere the objective can be photographed from."""
    return viewpoint(env, index, objective, grid, rng, VIS_RANGE)


def com_viewpoint(
    env: RoversEnv,
    index: int,
    grid: dict[tuple[int, int], bool],
    rng: np.random.Generator,
) -> NDArray[Any] | None:
    """Somewhere the lander can be radioed from."""
    return viewpoint(env, index, env.lander, grid, rng, COM_RANGE)


def standable(env: RoversEnv, index: int, point: Any, headings: int = 8) -> bool:
    """Whether a rover can sit on *point* at some heading without colliding.

    Sampling requires the base to be over the sample, so a sample that no heading fits
    is one the operator will always refuse -- worth knowing before driving to it.
    """
    with env.client(), WorldSaver():
        for theta in np.linspace(-np.pi, np.pi, headings, endpoint=False):
            if not _in_collision(env, index, [point[0], point[1], float(theta)]):
                return True
    return False
