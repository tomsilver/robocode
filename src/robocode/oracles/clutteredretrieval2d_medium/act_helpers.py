"""Action helpers for the ClutteredRetrieval2D medium oracle."""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from robocode.oracles.clutteredretrieval2d_medium.obs_helpers import (
    RobotConfig,
    action_from_config_transition,
    interpolate_configs,
)


def config_path_to_actions(path: list[RobotConfig]) -> deque[NDArray]:
    """Convert a dense config path into env actions."""
    actions: deque[NDArray] = deque()
    for q1, q2 in zip(path[:-1], path[1:]):
        actions.append(action_from_config_transition(q1, q2))
    return actions


def append_grasp_action(actions: deque[NDArray], final_config: RobotConfig) -> None:
    """Append an in-place vacuum-on action."""
    del final_config
    actions.append(
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )


def append_arm_extension_actions(
    actions: deque[NDArray],
    start_config: RobotConfig,
    target_arm_joint: float,
    *,
    step_size: float = 0.03,
) -> RobotConfig:
    """Append short in-place arm-extension actions and return the final config."""
    current = start_config
    remaining = target_arm_joint - start_config.arm_joint
    while abs(remaining) > 1e-6:
        delta = float(np.clip(remaining, -step_size, step_size))
        current = RobotConfig(
            x=current.x,
            y=current.y,
            theta=current.theta,
            arm_joint=current.arm_joint + delta,
            vacuum=current.vacuum,
        )
        actions.append(
            np.array([0.0, 0.0, 0.0, delta, current.vacuum], dtype=np.float32)
        )
        remaining = target_arm_joint - current.arm_joint
    return current


def append_micro_approach_actions(
    actions: deque[NDArray],
    start_config: RobotConfig,
    goal_config: RobotConfig,
) -> RobotConfig:
    """Append a short dense local approach from pre-grasp to grasp config."""
    current = start_config
    for nxt in interpolate_configs(start_config, goal_config):
        actions.append(action_from_config_transition(current, nxt))
        current = nxt
    return current


def append_release_action(actions: deque[NDArray], final_config: RobotConfig) -> None:
    """Append an in-place vacuum-off action."""
    del final_config
    actions.append(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
