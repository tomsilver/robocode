"""Observation parsing and planning helpers for ClutteredRetrieval2D-o10.

The medium variant is backed by ``kinder/ClutteredRetrieval2D-o10-v0``.
The flat observation layout is:

  Robot          [0:9]
  Target block   [9:19]
  Target region  [19:29]
  Obstruction i  [29 + 10*i : 39 + 10*i], for i in [0, 9]
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

ROBOT_FEATURES = [
    "x",
    "y",
    "theta",
    "base_radius",
    "arm_joint",
    "arm_length",
    "vacuum",
    "gripper_height",
    "gripper_width",
]

RECT_FEATURES = [
    "x",
    "y",
    "theta",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
    "width",
    "height",
]

NUM_OBSTRUCTIONS = 10
WORLD_MIN_X = 0.0
WORLD_MAX_X = 2.5
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 2.5
DX_LIM = 0.05
DY_LIM = 0.05
DTH_LIM = math.pi / 16
DARM_LIM = 0.1

LAYOUT: dict[str, tuple[int, list[str]]] = {
    "robot": (0, ROBOT_FEATURES),
    "target_block": (9, RECT_FEATURES),
    "target_region": (19, RECT_FEATURES),
}
for _i in range(NUM_OBSTRUCTIONS):
    LAYOUT[f"obstruction{_i}"] = (29 + 10 * _i, RECT_FEATURES)


@dataclass(frozen=True)
class RobotPose:
    """Robot configuration extracted from the observation vector."""

    x: float
    y: float
    theta: float
    base_radius: float
    arm_joint: float
    arm_length: float
    vacuum: float
    gripper_height: float
    gripper_width: float


@dataclass(frozen=True)
class RectPose:
    """Rectangle pose extracted from the observation vector."""

    x: float
    y: float
    theta: float
    width: float
    height: float

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class RobotConfig:
    """Planning state used by the oracle motion planner."""

    x: float
    y: float
    theta: float
    arm_joint: float
    vacuum: float


@dataclass(frozen=True)
class AttemptedTransition:
    """A single failed planning edge used to infer a blocking obstruction."""

    start: RobotConfig
    end: RobotConfig


PlanningMode = str


@dataclass(frozen=True)
class RelativeObjectTransform:
    """Approximate transform from tool-tip frame to a held object frame."""

    dx: float
    dy: float
    dtheta: float


def _base_and_features(name: str) -> tuple[int, list[str]]:
    return LAYOUT[name]


def extract_robot(obs: NDArray) -> RobotPose:
    """Extract the robot pose from the observation."""
    base, _ = _base_and_features("robot")
    return RobotPose(
        x=float(obs[base + 0]),
        y=float(obs[base + 1]),
        theta=float(obs[base + 2]),
        base_radius=float(obs[base + 3]),
        arm_joint=float(obs[base + 4]),
        arm_length=float(obs[base + 5]),
        vacuum=float(obs[base + 6]),
        gripper_height=float(obs[base + 7]),
        gripper_width=float(obs[base + 8]),
    )


def extract_rect(obs: NDArray, name: str) -> RectPose:
    """Extract a named rectangle from the observation."""
    base, features = _base_and_features(name)
    return RectPose(
        x=float(obs[base + features.index("x")]),
        y=float(obs[base + features.index("y")]),
        theta=float(obs[base + features.index("theta")]),
        width=float(obs[base + features.index("width")]),
        height=float(obs[base + features.index("height")]),
    )


def iter_obstruction_names() -> list[str]:
    """Return all obstruction names in observation order."""
    return [f"obstruction{i}" for i in range(NUM_OBSTRUCTIONS)]


def current_config(obs: NDArray) -> RobotConfig:
    """Convert the robot state into a planning config."""
    robot = extract_robot(obs)
    return RobotConfig(
        x=robot.x,
        y=robot.y,
        theta=robot.theta,
        arm_joint=robot.arm_joint,
        vacuum=robot.vacuum,
    )


def with_config_applied(
    obs: NDArray,
    config: RobotConfig,
    *,
    held_name: str | None = None,
    held_transform: RelativeObjectTransform | None = None,
) -> NDArray:
    """Create a synthetic observation with the robot moved to ``config``.

    If ``held_name`` is provided, the held object is moved rigidly with the robot using
    ``held_transform`` from the initial held state, expressed in the tool-tip frame.
    """
    out = np.array(obs, dtype=np.float32, copy=True)
    out[0:5] = np.array(
        [config.x, config.y, config.theta, out[3], config.arm_joint], dtype=np.float32
    )
    out[6] = np.float32(config.vacuum)
    if held_name is not None and held_transform is not None:
        base, features = _base_and_features(held_name)
        robot = extract_robot(obs)
        tip_x, tip_y = tool_tip_position(config, robot)
        dx, dy = _rotate_into_world_frame(
            held_transform.dx, held_transform.dy, config.theta
        )
        center_x = tip_x + dx
        center_y = tip_y + dy
        width = float(out[base + features.index("width")])
        height = float(out[base + features.index("height")])
        out[base + features.index("x")] = np.float32(center_x - width / 2)
        out[base + features.index("y")] = np.float32(center_y - height / 2)
        out[base + features.index("theta")] = np.float32(
            wrap_angle(config.theta + held_transform.dtheta)
        )
    return out


def config_to_pose(config: RobotConfig, robot: RobotPose) -> RobotPose:
    """Lift a planning config back into a full robot pose."""
    return RobotPose(
        x=config.x,
        y=config.y,
        theta=config.theta,
        base_radius=robot.base_radius,
        arm_joint=config.arm_joint,
        arm_length=robot.arm_length,
        vacuum=config.vacuum,
        gripper_height=robot.gripper_height,
        gripper_width=robot.gripper_width,
    )


def wrap_angle(theta: float) -> float:
    """Wrap an angle into [-pi, pi]."""
    return float((theta + math.pi) % (2 * math.pi) - math.pi)


def config_distance(q1: RobotConfig, q2: RobotConfig) -> float:
    """Weighted distance over the planning configuration."""
    return float(
        np.hypot(q1.x - q2.x, q1.y - q2.y)
        + 0.2 * abs(wrap_angle(q1.theta - q2.theta))
        + 0.5 * abs(q1.arm_joint - q2.arm_joint)
    )


def interpolate_configs(q1: RobotConfig, q2: RobotConfig) -> list[RobotConfig]:
    """Linearly interpolate between two robot configs."""
    steps = max(
        1,
        math.ceil(abs(q2.x - q1.x) / DX_LIM),
        math.ceil(abs(q2.y - q1.y) / DY_LIM),
        math.ceil(abs(wrap_angle(q2.theta - q1.theta)) / DTH_LIM),
        math.ceil(abs(q2.arm_joint - q1.arm_joint) / DARM_LIM),
    )
    out: list[RobotConfig] = []
    angle_delta = wrap_angle(q2.theta - q1.theta)
    for i in range(1, steps + 1):
        t = i / steps
        out.append(
            RobotConfig(
                x=q1.x + t * (q2.x - q1.x),
                y=q1.y + t * (q2.y - q1.y),
                theta=wrap_angle(q1.theta + t * angle_delta),
                arm_joint=q1.arm_joint + t * (q2.arm_joint - q1.arm_joint),
                vacuum=q2.vacuum,
            )
        )
    return out


def action_from_config_transition(q1: RobotConfig, q2: RobotConfig) -> NDArray:
    """Convert a config transition into an env action."""
    return np.array(
        [
            q2.x - q1.x,
            q2.y - q1.y,
            wrap_angle(q2.theta - q1.theta),
            q2.arm_joint - q1.arm_joint,
            q2.vacuum,
        ],
        dtype=np.float32,
    )


def tool_tip_position(config: RobotConfig, robot: RobotPose) -> tuple[float, float]:
    """Approximate the tool-tip centre at the front edge of the gripper."""
    tip_x = config.x + math.cos(config.theta) * (config.arm_joint + robot.gripper_width / 2)
    tip_y = config.y + math.sin(config.theta) * (config.arm_joint + robot.gripper_width / 2)
    return (tip_x, tip_y)


def suction_center(config: RobotConfig, robot: RobotPose) -> tuple[float, float]:
    """Approximate suction patch centre when the vacuum is on."""
    cx = config.x + math.cos(config.theta) * (config.arm_joint + robot.gripper_width)
    cy = config.y + math.sin(config.theta) * (config.arm_joint + robot.gripper_width)
    return (cx, cy)


def rect_contains_point(rect: RectPose, x: float, y: float, tol: float = 1e-3) -> bool:
    """Approximate containment for axis-aligned goal checking."""
    return (
        rect.x - tol <= x <= rect.right + tol
        and rect.y - tol <= y <= rect.top + tol
    )


def target_inside_region(obs: NDArray) -> bool:
    """True when the target block is fully inside the target region."""
    target = extract_rect(obs, "target_block")
    region = extract_rect(obs, "target_region")
    vertices = [
        (target.x, target.y),
        (target.right, target.y),
        (target.x, target.top),
        (target.right, target.top),
    ]
    return all(rect_contains_point(region, x, y) for x, y in vertices)


def _rect_circle_radius(rect: RectPose) -> float:
    return 0.5 * math.hypot(rect.width, rect.height)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    vx, vy = ex - sx, ey - sy
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        return _distance(point, start)
    t = max(0.0, min(1.0, ((px - sx) * vx + (py - sy) * vy) / seg_len_sq))
    proj = (sx + t * vx, sy + t * vy)
    return _distance(point, proj)


def _rotate_into_robot_frame(dx: float, dy: float, theta: float) -> tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return (c * dx + s * dy, -s * dx + c * dy)


def _rotate_into_world_frame(dx: float, dy: float, theta: float) -> tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return (c * dx - s * dy, s * dx + c * dy)


def _object_close_to_suction(obs: NDArray, name: str) -> bool:
    robot = extract_robot(obs)
    if robot.vacuum <= 0.5:
        return False
    config = current_config(obs)
    suction = suction_center(config, robot)
    rect = extract_rect(obs, name)
    target_distance = _distance((rect.cx, rect.cy), suction)
    # Require the candidate object to be both geometrically close to the suction
    # patch and the closest movable object around it; otherwise dense clutter can
    # make the target look "held" even when an obstruction is the actual contact.
    threshold = _rect_circle_radius(rect) + 0.75 * robot.gripper_width + 0.02
    if target_distance > threshold:
        return False

    target = extract_rect(obs, "target_block")
    closest_name = "target_block"
    closest_distance = _distance((target.cx, target.cy), suction)
    for obstruction_name in iter_obstruction_names():
        obstruction = extract_rect(obs, obstruction_name)
        obstruction_distance = _distance((obstruction.cx, obstruction.cy), suction)
        if obstruction_distance < closest_distance:
            closest_distance = obstruction_distance
            closest_name = obstruction_name
    return closest_name == name


def holding_target_block(obs: NDArray) -> bool:
    """Heuristic check for whether the robot is holding the target block."""
    return _object_close_to_suction(obs, "target_block")


def holding_obstruction_named(obs: NDArray, name: str) -> bool:
    """Heuristic check for whether the robot is holding a named obstruction."""
    return _object_close_to_suction(obs, name)


def held_object_transform(obs: NDArray, name: str) -> RelativeObjectTransform:
    """Approximate the current transform from tool-tip frame to the held object."""
    config = current_config(obs)
    rect = extract_rect(obs, name)
    robot = extract_robot(obs)
    tip_x, tip_y = tool_tip_position(config, robot)
    dx_world = rect.cx - tip_x
    dy_world = rect.cy - tip_y
    dx_robot, dy_robot = _rotate_into_robot_frame(dx_world, dy_world, config.theta)
    return RelativeObjectTransform(
        dx=dx_robot,
        dy=dy_robot,
        dtheta=wrap_angle(rect.theta - config.theta),
    )


def candidate_pick_poses(obs: NDArray, obj_name: str = "target_block") -> list[RobotConfig]:
    """Return a small set of candidate grasp configurations for an object."""
    robot = extract_robot(obs)
    current = current_config(obs)
    obj = extract_rect(obs, obj_name)
    arm_joint = robot.arm_length
    offset = max(obj.width, obj.height) / 2 + robot.gripper_width
    directions = [0.0, math.pi / 2, math.pi, -math.pi / 2]
    configs: list[RobotConfig] = []
    for theta in directions:
        base_x = obj.cx - math.cos(theta) * (arm_joint + offset)
        base_y = obj.cy - math.sin(theta) * (arm_joint + offset)
        if not (
            WORLD_MIN_X + robot.base_radius <= base_x <= WORLD_MAX_X - robot.base_radius
            and WORLD_MIN_Y + robot.base_radius <= base_y <= WORLD_MAX_Y - robot.base_radius
        ):
            continue
        configs.append(
            RobotConfig(
                x=base_x,
                y=base_y,
                theta=theta,
                arm_joint=arm_joint,
                vacuum=0.0,
            )
        )
    configs.sort(key=lambda config: config_distance(current, config))
    return configs


def _grasp_clearance_objects(obs: NDArray, obj_name: str) -> list[tuple[str, RectPose]]:
    names = ["target_block", *iter_obstruction_names()]
    return [
        (name, extract_rect(obs, name))
        for name in names
        if name != obj_name
    ]


def quick_grasp_pose_feasible(
    obs: NDArray,
    obj_name: str,
    grasp_cfg: RobotConfig,
    *,
    pregrasp_backoff: float = 0.05,
) -> bool:
    """Fast geometric rejection for obviously bad grasp poses."""
    robot = extract_robot(obs)
    obj = extract_rect(obs, obj_name)
    pregrasp = (
        grasp_cfg.x - math.cos(grasp_cfg.theta) * pregrasp_backoff,
        grasp_cfg.y - math.sin(grasp_cfg.theta) * pregrasp_backoff,
    )
    grasp = (grasp_cfg.x, grasp_cfg.y)
    for x, y in (pregrasp, grasp):
        if not (
            WORLD_MIN_X + robot.base_radius <= x <= WORLD_MAX_X - robot.base_radius
            and WORLD_MIN_Y + robot.base_radius <= y <= WORLD_MAX_Y - robot.base_radius
        ):
            return False
    # The grasp approach should end near the object centerline.
    tip_x, tip_y = suction_center(grasp_cfg, robot)
    if _distance((tip_x, tip_y), (obj.cx, obj.cy)) > max(obj.width, obj.height) * 0.8:
        return False
    corridor_radius = robot.base_radius + 0.08
    suction_clearance = 0.11
    for _, other in _grasp_clearance_objects(obs, obj_name):
        if _distance((other.cx, other.cy), grasp) < corridor_radius + _rect_circle_radius(other):
            return False
        if (
            _point_segment_distance((other.cx, other.cy), pregrasp, grasp)
            < corridor_radius + _rect_circle_radius(other)
        ):
            return False
        if _distance((other.cx, other.cy), (tip_x, tip_y)) < suction_clearance + _rect_circle_radius(other):
            return False
    return True


def filter_feasible_grasp_poses(
    obs: NDArray,
    obj_name: str,
    candidates: list[RobotConfig],
) -> list[RobotConfig]:
    """Fast-filter grasp candidates before expensive planning."""
    return [cfg for cfg in candidates if quick_grasp_pose_feasible(obs, obj_name, cfg)]


def rank_blockers_for_removal(
    obs: NDArray,
    *,
    excluded: set[str] | None = None,
) -> list[str]:
    """Rank obstruction names by likely usefulness for target access."""
    excluded = excluded or set()
    target = extract_rect(obs, "target_block")
    ranked: list[tuple[float, str]] = []
    for name in iter_obstruction_names():
        if name in excluded:
            continue
        obstruction = extract_rect(obs, name)
        score = _distance((target.cx, target.cy), (obstruction.cx, obstruction.cy))
        ranked.append((score, name))
    ranked.sort()
    return [name for _, name in ranked]


def candidate_place_poses(obs: NDArray) -> list[RobotConfig]:
    """Return candidate place configurations for the currently held target block."""
    robot = extract_robot(obs)
    region = extract_rect(obs, "target_region")
    target = extract_rect(obs, "target_block")
    rel = held_object_transform(obs, "target_block")
    margin_x = target.width / 2 + 0.01
    margin_y = target.height / 2 + 0.01
    desired_centers = [
        (region.cx, region.cy),
        (region.x + margin_x, region.cy),
        (region.right - margin_x, region.cy),
        (region.cx, region.y + margin_y),
        (region.cx, region.top - margin_y),
    ]
    configs: list[RobotConfig] = []
    for cx, cy in desired_centers:
        desired_theta = extract_rect(obs, "target_block").theta
        robot_theta = wrap_angle(desired_theta - rel.dtheta)
        off_x, off_y = _rotate_into_world_frame(rel.dx, rel.dy, robot_theta)
        tip_offset = current_config(obs).arm_joint + robot.gripper_width / 2
        base_x = cx - math.cos(robot_theta) * tip_offset - off_x
        base_y = cy - math.sin(robot_theta) * tip_offset - off_y
        if not (
            WORLD_MIN_X + robot.base_radius <= base_x <= WORLD_MAX_X - robot.base_radius
            and WORLD_MIN_Y + robot.base_radius <= base_y <= WORLD_MAX_Y - robot.base_radius
        ):
            continue
        configs.append(
            RobotConfig(
                x=base_x,
                y=base_y,
                theta=robot_theta,
                arm_joint=current_config(obs).arm_joint,
                vacuum=1.0,
            )
        )
    return configs


def candidate_staging_poses(obs: NDArray) -> list[tuple[float, float]]:
    """Return candidate staging centres for moved obstructions."""
    target = extract_rect(obs, "target_block")
    region = extract_rect(obs, "target_region")
    candidates = [
        (0.35, 0.35),
        (2.15, 0.35),
        (0.35, 2.15),
        (2.15, 2.15),
        (0.35, 1.25),
        (2.15, 1.25),
        (1.25, 0.35),
        (1.25, 2.15),
    ]
    filtered: list[tuple[float, float]] = []
    for cx, cy in candidates:
        if _distance((cx, cy), (target.cx, target.cy)) < 0.8:
            continue
        if _distance((cx, cy), (region.cx, region.cy)) < 0.65:
            continue
        filtered.append((cx, cy))
    return filtered


def staging_robot_configs(obs: NDArray, obj_name: str) -> list[RobotConfig]:
    """Return candidate robot configs that would place a held object at staging sites."""
    robot = extract_robot(obs)
    obj = extract_rect(obs, obj_name)
    rel = held_object_transform(obs, obj_name)
    current = current_config(obs)
    configs: list[RobotConfig] = []
    robot_thetas = [
        current.theta,
        0.0,
        math.pi / 2,
        math.pi,
        -math.pi / 2,
    ]
    for cx, cy in candidate_staging_poses(obs):
        if _distance((cx, cy), (obj.cx, obj.cy)) < 0.18:
            continue
        for robot_theta in robot_thetas:
            off_x, off_y = _rotate_into_world_frame(rel.dx, rel.dy, robot_theta)
            tip_offset = current.arm_joint + robot.gripper_width / 2
            base_x = cx - math.cos(robot_theta) * tip_offset - off_x
            base_y = cy - math.sin(robot_theta) * tip_offset - off_y
            if not (
                WORLD_MIN_X + robot.base_radius <= base_x <= WORLD_MAX_X - robot.base_radius
                and WORLD_MIN_Y + robot.base_radius <= base_y <= WORLD_MAX_Y - robot.base_radius
            ):
                continue
            config = RobotConfig(
                x=base_x,
                y=base_y,
                theta=robot_theta,
                arm_joint=current.arm_joint,
                vacuum=1.0,
            )
            if config_distance(current, config) < 0.08:
                continue
            configs.append(config)
    return configs


def infer_blocking_obstruction(
    obs: NDArray,
    attempted_transition: AttemptedTransition,
    planning_mode: PlanningMode,
    held_name: str | None = None,
    held_transform: RelativeObjectTransform | None = None,
) -> str | None:
    """Infer which obstruction most likely blocked a failed motion edge."""
    robot = extract_robot(obs)
    end_tip = tool_tip_position(attempted_transition.end, robot)
    end_base = (attempted_transition.end.x, attempted_transition.end.y)
    payload_ref: tuple[float, float] | None = None
    if planning_mode != "robot_only" and held_name is not None and held_transform is not None:
        dx, dy = _rotate_into_world_frame(
            held_transform.dx, held_transform.dy, attempted_transition.end.theta
        )
        payload_ref = (
            attempted_transition.end.x + dx,
            attempted_transition.end.y + dy,
        )
    best_name: str | None = None
    best_score = float("inf")
    for name in iter_obstruction_names():
        if name == held_name:
            continue
        rect = extract_rect(obs, name)
        score = min(_distance((rect.cx, rect.cy), end_tip), _distance((rect.cx, rect.cy), end_base))
        if payload_ref is not None:
            score = min(score, 0.8 * _distance((rect.cx, rect.cy), payload_ref))
        score -= 0.1 * _rect_circle_radius(rect)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def robot_config_changed(prev_obs: NDArray, next_obs: NDArray, tol: float = 1e-3) -> bool:
    """Return whether the robot configuration changed meaningfully."""
    r1 = extract_robot(prev_obs)
    r2 = extract_robot(next_obs)
    return bool(
        abs(r1.x - r2.x) > tol
        or abs(r1.y - r2.y) > tol
        or abs(wrap_angle(r1.theta - r2.theta)) > tol
        or abs(r1.arm_joint - r2.arm_joint) > tol
        or abs(r1.vacuum - r2.vacuum) > tol
    )


def held_object_changed(
    prev_obs: NDArray,
    next_obs: NDArray,
    held_name: str,
    tol: float = 1e-3,
) -> bool:
    """Return whether a held object changed pose meaningfully."""
    o1 = extract_rect(prev_obs, held_name)
    o2 = extract_rect(next_obs, held_name)
    return bool(
        abs(o1.x - o2.x) > tol
        or abs(o1.y - o2.y) > tol
        or abs(wrap_angle(o1.theta - o2.theta)) > tol
    )


def is_blocker_cleared_from_pick_corridor(
    obs: NDArray,
    blocker_name: str,
    margin: float = 0.75,
) -> bool:
    """Return whether a blocker is no longer close to the target block."""
    target = extract_rect(obs, "target_block")
    blocker = extract_rect(obs, blocker_name)
    return _distance((target.cx, target.cy), (blocker.cx, blocker.cy)) > margin


def is_blocker_cleared_from_place_corridor(
    obs: NDArray,
    blocker_name: str,
    margin: float = 0.30,
) -> bool:
    """Return whether a blocker is no longer close to the target region."""
    region = extract_rect(obs, "target_region")
    blocker = extract_rect(obs, blocker_name)
    return _distance((region.cx, region.cy), (blocker.cx, blocker.cy)) > margin
