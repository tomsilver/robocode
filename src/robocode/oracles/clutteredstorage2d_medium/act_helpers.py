"""Action helpers for ClutteredStorage2D-b3 oracle behaviors."""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from robocode.oracles.clutteredstorage2d_medium.obs_helpers import RobotPose

DX_LIM = 0.05
DY_LIM = 0.05
DTH_LIM = np.pi / 16
DARM_LIM = 0.1
GRID_STEP = DX_LIM
PATH_CLEARANCE_MARGIN = 0.04
SEGMENT_CHECK_STEP = 0.025

Point2D = tuple[float, float]
ObstacleCircle = tuple[float, float, float]


def wrap_angle(theta: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def shortest_angle_diff(target: float, source: float) -> float:
    """Return the shortest signed angle from *source* to *target*."""
    return wrap_angle(target - source)


def path_length(points: list[Point2D]) -> float:
    """Return the Euclidean length of a 2D polyline."""
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(x1 - x0, y1 - y0)
            for (x0, y0), (x1, y1) in zip(points[:-1], points[1:])
        )
    )


def inflate_block_radius(
    width: float,
    height: float,
    robot_base_radius: float,
    margin: float = PATH_CLEARANCE_MARGIN,
) -> float:
    """Return a conservative circular obstacle radius for a rectangular block."""
    return 0.5 * math.hypot(width, height) + robot_base_radius + margin


def _snap(value: float, step: float = GRID_STEP) -> float:
    return round(value / step) * step


def _point_collision_free(
    point: Point2D,
    obstacles: list[ObstacleCircle],
    bounds: tuple[float, float, float, float],
) -> bool:
    x, y = point
    min_x, max_x, min_y, max_y = bounds
    if not (min_x <= x <= max_x and min_y <= y <= max_y):
        return False
    for cx, cy, radius in obstacles:
        if math.hypot(x - cx, y - cy) < radius:
            return False
    return True


def segment_collision_free(
    start: Point2D,
    end: Point2D,
    obstacles: list[ObstacleCircle],
    bounds: tuple[float, float, float, float],
    step: float = SEGMENT_CHECK_STEP,
) -> bool:
    """Check whether a line segment stays inside bounds and outside obstacles."""
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    checks = max(1, math.ceil(distance / step))
    for idx in range(checks + 1):
        t = idx / checks
        point = (
            start[0] + t * (end[0] - start[0]),
            start[1] + t * (end[1] - start[1]),
        )
        if not _point_collision_free(point, obstacles, bounds):
            return False
    return True


def _astar_neighbors(node: Point2D, step: float) -> list[Point2D]:
    neighbors: list[Point2D] = []
    for dx in (-step, 0.0, step):
        for dy in (-step, 0.0, step):
            if dx == 0.0 and dy == 0.0:
                continue
            neighbors.append((round(node[0] + dx, 6), round(node[1] + dy, 6)))
    return neighbors


def plan_base_path(
    start: Point2D,
    goal: Point2D,
    obstacles: list[ObstacleCircle],
    bounds: tuple[float, float, float, float],
    step: float = GRID_STEP,
) -> list[Point2D] | None:
    """Plan a conservative 2D base path with a coarse grid A* search."""
    # The robot can begin very close to a clutter block. If we keep those
    # overlapping inflated circles, the planner incorrectly decides that the
    # current state is already in collision and refuses to move at all.
    active_obstacles = [
        (cx, cy, radius)
        for cx, cy, radius in obstacles
        if math.hypot(start[0] - cx, start[1] - cy) >= radius - 1e-6
    ]

    if segment_collision_free(start, goal, active_obstacles, bounds):
        return [start, goal]

    start_node = (round(_snap(start[0], step), 6), round(_snap(start[1], step), 6))
    goal_node = (round(_snap(goal[0], step), 6), round(_snap(goal[1], step), 6))
    if not _point_collision_free(goal_node, active_obstacles, bounds):
        return None
    if not segment_collision_free(start, start_node, active_obstacles, bounds):
        return None

    frontier: list[tuple[float, float, Point2D]] = []
    heapq.heappush(frontier, (math.hypot(goal_node[0] - start_node[0], goal_node[1] - start_node[1]), 0.0, start_node))
    came_from: dict[Point2D, Point2D | None] = {start_node: None}
    costs: dict[Point2D, float] = {start_node: 0.0}

    while frontier:
        _, cost_so_far, current = heapq.heappop(frontier)
        if current == goal_node:
            break
        if cost_so_far > costs[current] + 1e-9:
            continue
        for neighbor in _astar_neighbors(current, step):
            if not _point_collision_free(neighbor, active_obstacles, bounds):
                continue
            if not segment_collision_free(current, neighbor, active_obstacles, bounds):
                continue
            new_cost = cost_so_far + math.hypot(
                neighbor[0] - current[0], neighbor[1] - current[1]
            )
            if new_cost + 1e-9 < costs.get(neighbor, float("inf")):
                costs[neighbor] = new_cost
                came_from[neighbor] = current
                heuristic = math.hypot(goal_node[0] - neighbor[0], goal_node[1] - neighbor[1])
                heapq.heappush(frontier, (new_cost + heuristic, new_cost, neighbor))

    if goal_node not in came_from:
        return None

    node_path: list[Point2D] = []
    current: Point2D | None = goal_node
    while current is not None:
        node_path.append(current)
        current = came_from[current]
    node_path.reverse()

    path: list[Point2D] = [start]
    for point in node_path[1:]:
        path.append(point)
    if not segment_collision_free(path[-1], goal, active_obstacles, bounds):
        return None
    if path[-1] != goal:
        path.append(goal)
    return path


def connecting_waypoints(
    waypoints: list[RobotPose],
    action_limits: tuple[float, float, float, float] = (
        DX_LIM,
        DY_LIM,
        DTH_LIM,
        DARM_LIM,
    ),
) -> list[RobotPose]:
    """Linearly interpolate between key waypoints."""
    dx_lim, dy_lim, dth_lim, darm_lim = action_limits
    dense: list[RobotPose] = [waypoints[0]]

    for start, end in zip(waypoints[:-1], waypoints[1:]):
        dtheta = shortest_angle_diff(end.theta, start.theta)
        steps = max(
            1,
            math.ceil(abs(end.x - start.x) / dx_lim),
            math.ceil(abs(end.y - start.y) / dy_lim),
            math.ceil(abs(dtheta) / dth_lim) if dth_lim > 0 else 1,
            math.ceil(abs(end.arm_joint - start.arm_joint) / darm_lim),
        )
        for step in range(1, steps + 1):
            t = step / steps
            dense.append(
                RobotPose(
                    x=start.x + t * (end.x - start.x),
                    y=start.y + t * (end.y - start.y),
                    theta=wrap_angle(start.theta + t * dtheta),
                    base_radius=start.base_radius,
                    arm_joint=start.arm_joint + t * (end.arm_joint - start.arm_joint),
                    arm_length=start.arm_length,
                    vacuum=end.vacuum,
                    gripper_height=start.gripper_height,
                    gripper_width=start.gripper_width,
                )
            )

    return dense


def waypoints_to_actions(waypoints: list[RobotPose]) -> deque[NDArray]:
    """Convert dense waypoints into delta actions."""
    actions: deque[NDArray] = deque()
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        actions.append(
            np.array(
                [
                    end.x - start.x,
                    end.y - start.y,
                    shortest_angle_diff(end.theta, start.theta),
                    end.arm_joint - start.arm_joint,
                    end.vacuum,
                ],
                dtype=np.float32,
            )
        )
    return actions
