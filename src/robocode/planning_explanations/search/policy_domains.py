"""Closed-loop domains for milestone 8: a tiny disc world and KinDER.

A domain declares environment semantics only: resets, steps, the robot's position and
which action components are planar motion. It knows nothing about any policy's
internals.
"""

import copy
import json
import math
import random
from pathlib import Path

POLICIES = Path(__file__).with_name("policies")

# Development and validation episodes are (obstacle count, seed). Validation
# uses new seeds, including counts above the policy author's design counts.
DEVELOPMENT = ((1, 0), (3, 1), (5, 2))
VALIDATION = ((3, 100), (15, 101), (20, 102))


class DiscWorld:
    """A point robot among disc obstacles in a 2.5 x 2.5 room; pure Python."""

    name = "disc-world"
    action_space = None
    observation_space = None
    radius = 0.05

    def __init__(self) -> None:
        self.state: dict = {}

    def describe(self) -> dict:
        """Manifest entry; the world is fully defined by this code and a case."""
        return {"name": self.name, "room": 2.5, "robot_radius": self.radius}

    def reset(self, case: tuple) -> tuple[dict, dict]:
        """Seeded layout; obstacles keep clear of the start and the goal."""
        count, seed = case
        rng = random.Random(f"disc-world:{seed}")
        robot = [0.25 + 0.2 * rng.random(), 0.25 + 0.2 * rng.random()]
        goal = [2.05 + 0.2 * rng.random(), 2.05 + 0.2 * rng.random()]
        obstacles: list[list[float]] = []
        while len(obstacles) < count:
            x, y = 0.5 + 1.5 * rng.random(), 0.5 + 1.5 * rng.random()
            r = 0.08 + 0.07 * rng.random()
            if all(math.hypot(x - p[0], y - p[1]) > r + 0.25 for p in (robot, goal)):
                obstacles.append([x, y, r])
        self.state = {"robot": robot, "goal": goal, "obstacles": obstacles}
        return copy.deepcopy(self.state), {"object_count": count}

    def _free(self, x: float, y: float) -> bool:
        low, high = self.radius, 2.5 - self.radius
        if not low <= x <= high or not low <= y <= high:
            return False
        return all(
            math.hypot(x - ox, y - oy) > r + self.radius
            for ox, oy, r in self.state["obstacles"]
        )

    def step(self, action) -> tuple[dict, bool, bool]:
        """Move up to 0.035 per axis unless the new position collides."""
        dx = max(-0.035, min(0.035, float(action[0])))
        dy = max(-0.035, min(0.035, float(action[1])))
        x, y = self.state["robot"]
        if self._free(x + dx, y + dy):
            self.state["robot"] = [x + dx, y + dy]
        x, y = self.state["robot"]
        gx, gy = self.state["goal"]
        return copy.deepcopy(self.state), math.hypot(x - gx, y - gy) < 0.03, False

    @staticmethod
    def max_steps(case: tuple) -> int:
        """A generous horizon that grows with clutter."""
        return 300 + 40 * case[0]

    @staticmethod
    def copy_state(state):
        """Observations are plain dictionaries."""
        return copy.deepcopy(state)

    @staticmethod
    def robot_xy(state) -> tuple[float, float]:
        """Declared robot position."""
        return float(state["robot"][0]), float(state["robot"][1])

    @staticmethod
    def planar(action) -> tuple[float, float]:
        """Both action components are planar motion."""
        return float(action[0]), float(action[1])

    @staticmethod
    def shift_world(state, offset: float):
        """Move every obstacle along x; the robot and goal stay."""
        for obstacle in state["obstacles"]:
            obstacle[0] += offset
        return state


class KinderDomain:
    """KinDER ClutteredRetrieval2D through robocode's variable-count wrapper."""

    name = "kinder-clutteredretrieval2d"

    def __init__(self, config: dict | None = None) -> None:
        # Imported lazily: the disc-world suite needs no robotics dependencies.
        # pylint: disable-next=import-outside-toplevel
        from robocode.environments.variable_object_count_env import (
            VariableObjectCountEnv,
        )

        self.config = dict(
            config
            or json.loads(
                (POLICIES / "clutteredretrieval2d_env_config.json").read_text()
            )
        )
        kwargs = {k: v for k, v in self.config.items() if k != "_target_"}
        self.env = VariableObjectCountEnv(**kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.robot_type = self.observation_space.get_type("crv_robot")

    def describe(self) -> dict:
        """Manifest entry: the exact environment configuration."""
        return {"name": self.name, "config": self.config}

    def reset(self, case: tuple):
        """Pin the obstacle count and the seed."""
        count, seed = case
        return self.env.reset(seed=seed, options={"object_count": count})

    def step(self, action):
        """One simulator step."""
        state, _, terminated, truncated, _ = self.env.step(action)
        return state, terminated, truncated

    def max_steps(self, case: tuple) -> int:
        """The evaluation harness's horizon for this count."""
        return self.env.max_steps_for_count(case[0])

    @staticmethod
    def copy_state(state):
        """ObjectCentricState's own copy."""
        return state.copy()

    def robot_xy(self, state) -> tuple[float, float]:
        """Declared robot base position."""
        robot = list(state.get_objects(self.robot_type))[0]
        return float(state.get(robot, "x")), float(state.get(robot, "y"))

    @staticmethod
    def planar(action) -> tuple[float, float]:
        """Action components 0 and 1 are the base's planar motion."""
        return float(action[0]), float(action[1])

    @staticmethod
    def shift_world(state, offset: float):
        """Move every obstruction along x; robot, target and region stay."""
        for name in state.get_object_names():
            if str(name).startswith("obstruction"):
                obj = state.get_object_from_name(name)
                state.set(obj, "x", float(state.get(obj, "x")) + offset)
        return state
