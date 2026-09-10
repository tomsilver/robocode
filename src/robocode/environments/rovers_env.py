"""A rover exploration environment backed by PDDLStream's ``rovers`` benchmark.

The scene and goal come from PDDLStream's ``rovers`` problem (see
:mod:`robocode.environments.rovers_scenes`): two turtlebot rovers must acquire one
rock and one soil sample, photograph every objective, and radio all of it back to a
Husky lander, finishing with both rovers home and both sample stores empty. Mounds,
pillars and a dividing wall block both driving and line of sight, so what makes this
hard is *where you have to stand* -- to see an objective, to reach the lander, and to
get back.

Unlike the PR2 families this is not a manipulation problem, so the action space pairs
continuous base motion with a small discrete repertoire: sample, calibrate, image,
send, drop. Those are the benchmark's own operators, and each is refused unless its
precondition holds, exactly as the PDDL domain refuses them.

Dynamics are kinematic: base joints are set rather than driven, and a motion that
would put a rover in collision is rejected and leaves the state unchanged. Visibility
is a ray cast from the rover's Kinect frame to the target, which is a cheaper stand-in
for the swept detection cone the upstream streams build; the range limits are the
benchmark's.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, SupportsFloat, cast

import numpy as np
import pybullet as p
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box
from numpy.typing import NDArray

from robocode.environments.base_env import BaseEnv
from robocode.environments.rovers_scenes import (
    BASE_EXTENT,
    COM_RANGE,
    KINECT_FRAME,
    VIS_RANGE,
    base_joints,
    build_rovers_scene,
    resample_rover_scene,
)
from robocode.environments.ss_pybullet import (
    HideOutput,
    connect,
    disconnect,
    get_aabb,
    get_client,
    get_joint_positions,
    get_link_pose,
    get_point,
    get_pose,
    link_from_name,
    pairwise_collision,
    set_client,
    set_joint_positions,
    set_pose,
)

_CLIENT_LOCK = threading.RLock()

# Bounds on one step's base motion, matching the delta-action convention the other
# robocode environments use.
_MAX_DELTA = 0.2
_MAX_DELTA_THETA = 0.4

# The discrete repertoire, selected by the last entry of each rover's action, which
# splits [-1, 1] into six equal bands.
NOOP, SAMPLE, CALIBRATE, IMAGE, SEND, DROP = range(6)
# Which operator each band selects. NOOP sits on the band containing zero so that an
# all-zero action does nothing: the neutral action a policy reaches for when it has
# nothing to do -- and the one a half-written policy emits by accident -- must not fire
# the camera. Anything else makes idling an operator.
OPERATOR_BANDS = (SAMPLE, CALIBRATE, IMAGE, NOOP, SEND, DROP)
_NUM_DISCRETE = len(OPERATOR_BANDS)


def operator_action(operator: int) -> float:
    """The selector value that lands squarely in *operator*'s band."""
    band = OPERATOR_BANDS.index(operator)
    return (band + 0.5) / _NUM_DISCRETE * 2.0 - 1.0


# How close a rover's base has to be to a sample to be "above" it. Upstream places the
# base exactly on the rock's (x, y); a radius is the same condition with the tolerance
# that arriving by bounded delta actions needs.
SAMPLE_RADIUS = 0.25
# How close to its starting pose a rover counts as home, likewise.
HOME_RADIUS = 0.25
HOME_ANGLE = 0.4

_ROVER_FEATURES = 6
_LANDER_FEATURES = 3
_OBJECTIVE_FEATURES = 6
_ROCK_FEATURES = 7
_OBSTACLE_FEATURES = 6

_RENDER_WIDTH = 640
_RENDER_HEIGHT = 480
_CAMERA_EYE = [0.0, -6.5, 5.5]
_CAMERA_TARGET = [0.0, 0.0, 0.0]
_CAMERA_FOV_DEGREES = 55.0
_CAMERA_NEAR = 0.05
_CAMERA_FAR = 20.0


class RoversEnv(BaseEnv[NDArray[Any], NDArray[Any]]):
    """Two rovers must sample, photograph and radio results to the lander."""

    def __init__(
        self,
        num_objectives: int = 4,
        num_rocks: int = 3,
        num_soils: int = 3,
        num_obstacles: int = 8,
        max_placement_resets: int = 10,
    ) -> None:
        self._num_objectives = int(num_objectives)
        self._num_rocks = int(num_rocks)
        self._num_soils = int(num_soils)
        self._num_obstacles = int(num_obstacles)
        self._max_placement_resets = int(max_placement_resets)
        self._current_obs: NDArray[Any] | None = None

        with _CLIENT_LOCK:
            self._client = connect(use_gui=False)
            set_client(self._client)
            entropy = np.random.get_state()
            try:
                with HideOutput():
                    handles = build_rovers_scene(
                        num_objectives=self._num_objectives,
                        num_rocks=self._num_rocks,
                        num_soils=self._num_soils,
                        num_obstacles=self._num_obstacles,
                    )
            finally:
                np.random.set_state(entropy)
            self._floor: int = handles["floor"]
            self._walls: list[int] = handles["walls"]
            self._lander: int = handles["lander"]
            self._mounds: list[int] = handles["mounds"]
            self._obstacle_bodies: list[int] = handles["obstacles"]
            self._rovers: list[int] = handles["rovers"]
            self._objectives: list[int] = handles["objectives"]
            self._rocks: list[int] = handles["rocks"]
            self._soils: list[int] = handles["soils"]
            self._home_confs = [np.array(c) for c in handles["rover_confs"]]
            self._base_joints = [base_joints(rover) for rover in self._rovers]
            self._camera_links = [
                link_from_name(rover, KINECT_FRAME) for rover in self._rovers
            ]
            # Everything a rover can crash into or see through. The floor is excluded
            # because the wheels rest on it, so it collides in every state.
            self._fixed = [
                *self._walls,
                *self._mounds,
                *self._obstacle_bodies,
                self._lander,
            ]
            self._samples = [*self._rocks, *self._soils]
            self._extents = {
                body: self._half_extent(body)
                for body in (*self._mounds, *self._obstacle_bodies)
            }

        self._reset_task_state()
        self._obs_dim = (
            _ROVER_FEATURES * len(self._rovers)
            + _LANDER_FEATURES
            + _OBJECTIVE_FEATURES * len(self._objectives)
            + _ROCK_FEATURES * len(self._samples)
            + _OBSTACLE_FEATURES * (len(self._mounds) + len(self._obstacle_bodies))
        )
        self.observation_space = Box(
            -np.inf, np.inf, (self._obs_dim,), dtype=np.float32
        )
        limit = np.array(
            [_MAX_DELTA, _MAX_DELTA, _MAX_DELTA_THETA, 1.0] * len(self._rovers),
            dtype=np.float32,
        )
        self.action_space = Box(-limit, limit, dtype=np.float32)
        super().__init__()

    def _reset_task_state(self) -> None:
        """Clear every fluent the benchmark's operators set."""
        # pylint: disable=attribute-defined-outside-init
        n_rovers = len(self._rovers)
        self._store_full = [False] * n_rovers
        self._calibrated = [False] * n_rovers
        # Analysed and imaged are per (rover, target): only the rover holding a result
        # can radio it, exactly as the domain's send_* operators require.
        self._analyzed: list[set[int]] = [set() for _ in range(n_rovers)]
        self._have_image: list[set[int]] = [set() for _ in range(n_rovers)]
        self._received_analysis: set[int] = set()
        self._received_image: set[int] = set()

    @contextmanager
    def client(self) -> Iterator[int]:
        """Hold this environment's physics client for the duration of the block."""
        with _CLIENT_LOCK:
            previous = get_client()
            set_client(self._client)
            try:
                yield self._client
            finally:
                set_client(previous)

    def _half_extent(self, body: int) -> list[float]:
        lower, upper = get_aabb(body)
        return [float(v) for v in (np.array(upper) - np.array(lower)) / 2.0]

    # ------------------------------------------------------------- accessors

    # Read-only handles onto the scene, for the same reason the PR2 environments
    # expose theirs: a planner needs body ids and joint groups to reason at all.
    # None of them let a policy change the state.

    @property
    def client_id(self) -> int:
        """PyBullet client this environment's scene lives in."""
        return self._client

    @property
    def rovers(self) -> list[int]:
        """The turtlebot rovers."""
        return list(self._rovers)

    @property
    def lander(self) -> int:
        """The Husky lander every result has to be radioed to."""
        return self._lander

    @property
    def objectives(self) -> list[int]:
        """The blue objectives that have to be photographed."""
        return list(self._objectives)

    @property
    def rocks(self) -> list[int]:
        """The stone samples."""
        return list(self._rocks)

    @property
    def soils(self) -> list[int]:
        """The soil samples."""
        return list(self._soils)

    @property
    def samples(self) -> list[int]:
        """Rocks then soils, in observation order."""
        return list(self._samples)

    @property
    def obstacles(self) -> list[int]:
        """Everything a rover can collide with or have its view blocked by."""
        return list(self._fixed)

    @property
    def base_joint_groups(self) -> list[list[int]]:
        """Each rover's planar base joint group."""
        return [list(joints) for joints in self._base_joints]

    @property
    def camera_links(self) -> list[int]:
        """Each rover's Kinect link, which visibility is measured from."""
        return list(self._camera_links)

    @property
    def home_confs(self) -> list[NDArray[Any]]:
        """The base configurations the goal requires the rovers to return to."""
        return [conf.copy() for conf in self._home_confs]

    def has_image(self, index: int, objective: int) -> bool:
        """Whether rover *index* is holding a photograph of *objective*.

        The image operator carries no target -- the action is a single number -- so it
        photographs the nearest objective the rover still needs. A planner aiming at a
        particular one therefore has to check what it actually got, since standing in
        the right place is not enough when something nearer is also in frame.
        """
        return objective in self._have_image[index]

    def rover_conf(self, index: int) -> NDArray[Any]:
        """The current base configuration of rover *index*."""
        with self.client():
            return np.array(
                get_joint_positions(self._rovers[index], self._base_joints[index])
            )

    # ------------------------------------------------------------ visibility

    def _camera_point(self, index: int) -> NDArray[Any]:
        return np.array(
            get_link_pose(self._rovers[index], self._camera_links[index])[0]
        )

    def visible(self, index: int, target: int, max_range: float) -> bool:
        """Whether rover *index* has clear line of sight to *target* within range.

        A ray from the rover's Kinect frame to the target's centre, blocked by anything
        that is not the target itself. Upstream sweeps a detection cone for imaging and
        a thin cylinder for the radio; a single ray is the same test at the cheaper
        resolution, and it is what the environment's own operators are defined against
        so a policy can reproduce it exactly.
        """
        with self.client():
            source = self._camera_point(index)
            goal = np.array(get_point(target))
            if float(np.linalg.norm(goal - source)) > max_range:
                return False
            hits = p.rayTest(source, goal, physicsClientId=self._client)
        if not hits:
            return False
        hit_body = hits[0][0]
        # -1 is "nothing hit"; the rover's own body does not count as an occluder.
        return hit_body in (-1, target, self._rovers[index])

    def image_visible(self, index: int, objective: int) -> bool:
        """Whether an objective is close enough and clear enough to photograph."""
        return self.visible(index, objective, VIS_RANGE)

    def com_visible(self, index: int) -> bool:
        """Whether the lander is in radio range and line of sight."""
        return self.visible(index, self._lander, COM_RANGE)

    def sample_under(self, index: int) -> int | None:
        """The sample the rover is standing over, if any."""
        with self.client():
            conf = get_joint_positions(self._rovers[index], self._base_joints[index])
            base = np.array(conf[:2])
            best, best_distance = None, SAMPLE_RADIUS
            for body in self._samples:
                distance = float(np.linalg.norm(np.array(get_point(body))[:2] - base))
                if distance <= best_distance:
                    best, best_distance = body, distance
        return best

    def at_home(self, index: int) -> bool:
        """Whether the rover is back where it started, within tolerance."""
        with self.client():
            conf = np.array(
                get_joint_positions(self._rovers[index], self._base_joints[index])
            )
        home = self._home_confs[index]
        if float(np.linalg.norm(conf[:2] - home[:2])) > HOME_RADIUS:
            return False
        error = (conf[2] - home[2] + np.pi) % (2 * np.pi) - np.pi
        return abs(float(error)) <= HOME_ANGLE

    # --------------------------------------------------------------- gym API

    def _apply_operator(self, index: int, operator: int) -> None:
        """Run one discrete operator for a rover, or refuse it silently.

        Each mirrors the benchmark's PDDL action of the same name, and each is a no-op
        unless its precondition holds -- there is no error, the state simply does not
        change, so a policy has to check the world rather than the return value.
        """
        if operator == SAMPLE:
            # sample_rock: the rover must be over a sample with its store free.
            if self._store_full[index]:
                return
            body = self.sample_under(index)
            if body is None:
                return
            self._store_full[index] = True
            self._analyzed[index].add(body)
        elif operator == CALIBRATE:
            # calibrate: needs some objective in view.
            if any(self.image_visible(index, o) for o in self._objectives):
                self._calibrated[index] = True
        elif operator == IMAGE:
            # take_image: needs a calibrated camera and a visible objective, and
            # spends the calibration. Exactly one objective is photographed, as
            # upstream's operator does, so N objectives cost N calibrate/image pairs
            # rather than one sweep of whatever happens to be in frame. The action
            # carries no target, so the choice is the nearest objective still needed,
            # falling back to the nearest in view once this rover holds them all.
            if not self._calibrated[index]:
                return
            visible = [o for o in self._objectives if self.image_visible(index, o)]
            if not visible:
                return
            wanted = [o for o in visible if o not in self._have_image[index]] or visible
            camera = self._camera_point(index)
            target = min(
                wanted,
                key=lambda o: float(np.linalg.norm(np.array(get_point(o)) - camera)),
            )
            self._have_image[index].add(target)
            self._calibrated[index] = False
        elif operator == SEND:
            # send_image / send_analysis: needs the lander in view, and radios
            # everything this rover is currently holding.
            if not self.com_visible(index):
                return
            self._received_image.update(self._have_image[index])
            self._received_analysis.update(self._analyzed[index])
        elif operator == DROP:
            # drop_rock: empties the store. The analysis is already recorded, so a
            # rover may drop before or after radioing it.
            self._store_full[index] = False

    def _advance(self, index: int, delta: NDArray[Any]) -> None:
        """Move one rover's base, rejecting a motion that ends in collision."""
        rover, joints = self._rovers[index], self._base_joints[index]
        conf = np.array(get_joint_positions(rover, joints))
        target = conf + delta
        target[2] = (target[2] + np.pi) % (2 * np.pi) - np.pi
        set_joint_positions(rover, joints, target)
        others = [r for r in self._rovers if r != rover]
        if any(pairwise_collision(rover, body) for body in self._fixed + others):
            set_joint_positions(rover, joints, conf)

    @staticmethod
    def _operator(value: float) -> int:
        """Map an action's last entry onto the discrete repertoire."""
        bucket = int((float(np.clip(value, -1.0, 1.0)) + 1.0) / 2.0 * _NUM_DISCRETE)
        return OPERATOR_BANDS[min(bucket, _NUM_DISCRETE - 1)]

    def step(
        self, action: NDArray[Any]
    ) -> tuple[NDArray[Any], SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        with self.client():
            for index in range(len(self._rovers)):
                chunk = action[4 * index : 4 * index + 4]
                delta = np.clip(
                    chunk[:3],
                    [-_MAX_DELTA, -_MAX_DELTA, -_MAX_DELTA_THETA],
                    [_MAX_DELTA, _MAX_DELTA, _MAX_DELTA_THETA],
                )
                self._advance(index, delta)
                self._apply_operator(index, self._operator(chunk[3]))
            terminated = self._goal_reached()
            self._current_obs = self._get_obs()
        return self._current_obs, -1.0, terminated, False, {}

    def _goal_reached(self) -> bool:
        """The benchmark's goal, in full.

        One stone and one soil analysis received, every objective's image received,
        every rover home, and every store empty. The two sample clauses are separate
        because the goal names the *types*: two rocks are not a substitute for a rock
        and a soil.
        """
        if not any(rock in self._received_analysis for rock in self._rocks):
            return False
        if not any(soil in self._received_analysis for soil in self._soils):
            return False
        if not all(o in self._received_image for o in self._objectives):
            return False
        if any(self._store_full):
            return False
        return all(self.at_home(i) for i in range(len(self._rovers)))

    def reset(self, *args: Any, **kwargs: Any) -> tuple[NDArray[Any], dict[str, Any]]:
        super().reset(*args, **kwargs)
        with self.client():
            self._reset_task_state()
            for index, rover in enumerate(self._rovers):
                set_joint_positions(
                    rover, self._base_joints[index], self._home_confs[index]
                )
            # sample_placement draws from the process-global legacy numpy RNG, so it is
            # seeded from the episode's generator and put back afterwards: resetting an
            # environment should not silently reseed its caller.
            entropy = np.random.get_state()
            try:
                for _ in range(self._max_placement_resets):
                    np.random.seed(int(self.np_random.integers(0, 2**31 - 1)))
                    if resample_rover_scene(
                        self._objectives,
                        self._mounds,
                        self._rocks,
                        self._soils,
                        self._obstacle_bodies,
                        self._floor,
                        self.np_random,
                        home_points=[conf[:2] for conf in self._home_confs],
                    ):
                        break
                else:
                    raise RuntimeError(
                        "Could not lay out the rover scene after "
                        f"{self._max_placement_resets} attempts"
                    )
            finally:
                np.random.set_state(entropy)
            self._current_obs = self._get_obs()
        return self._current_obs, {}

    def _get_obs(self) -> NDArray[Any]:
        features: list[float] = []
        for index, rover in enumerate(self._rovers):
            conf = get_joint_positions(rover, self._base_joints[index])
            features.extend(float(v) for v in conf)
            features.append(1.0 if self._store_full[index] else 0.0)
            features.append(1.0 if self._calibrated[index] else 0.0)
            features.append(1.0 if self.at_home(index) else 0.0)
        features.extend(float(v) for v in get_point(self._lander))
        for objective in self._objectives:
            features.extend(float(v) for v in get_point(objective))
            for held in self._have_image:
                features.append(1.0 if objective in held else 0.0)
            features.append(1.0 if objective in self._received_image else 0.0)
        for body in self._samples:
            features.extend(float(v) for v in get_point(body))
            features.append(1.0 if body in self._soils else 0.0)
            for held in self._analyzed:
                features.append(1.0 if body in held else 0.0)
            features.append(1.0 if body in self._received_analysis else 0.0)
        for body in (*self._mounds, *self._obstacle_bodies):
            features.extend(float(v) for v in get_point(body))
            features.extend(self._extents[body])
        return np.array(features, dtype=np.float32)

    def get_state(self) -> NDArray[Any]:
        assert self._current_obs is not None, "Must call reset()"
        return self._current_obs.copy()

    def set_state(self, state: NDArray[Any]) -> None:
        # pylint: disable=attribute-defined-outside-init
        state = np.asarray(state, dtype=np.float32)
        with self.client():
            offset = 0
            for index, rover in enumerate(self._rovers):
                set_joint_positions(
                    rover, self._base_joints[index], state[offset : offset + 3]
                )
                self._store_full[index] = bool(state[offset + 3] > 0.5)
                self._calibrated[index] = bool(state[offset + 4] > 0.5)
                offset += _ROVER_FEATURES
            offset += _LANDER_FEATURES  # the lander never moves
            self._have_image = [set() for _ in self._rovers]
            self._received_image = set()
            for objective in self._objectives:
                _set_point(objective, state[offset : offset + 3])
                for index in range(len(self._rovers)):
                    if state[offset + 3 + index] > 0.5:
                        self._have_image[index].add(objective)
                if state[offset + 3 + len(self._rovers)] > 0.5:
                    self._received_image.add(objective)
                offset += _OBJECTIVE_FEATURES
            self._analyzed = [set() for _ in self._rovers]
            self._received_analysis = set()
            for body in self._samples:
                _set_point(body, state[offset : offset + 3])
                for index in range(len(self._rovers)):
                    if state[offset + 4 + index] > 0.5:
                        self._analyzed[index].add(body)
                if state[offset + 4 + len(self._rovers)] > 0.5:
                    self._received_analysis.add(body)
                offset += _ROCK_FEATURES
            for body in (*self._mounds, *self._obstacle_bodies):
                _set_point(body, state[offset : offset + 3])
                offset += _OBSTACLE_FEATURES
            self._current_obs = self._get_obs()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """An overhead view of the whole arena.

        Both rovers, the lander and every objective have to stay in frame for a rollout
        to be readable, and the arena is five metres square, so the camera looks down on
        all of it rather than following a robot.
        """
        with self.client():
            view = p.computeViewMatrix(
                _CAMERA_EYE, _CAMERA_TARGET, [0, 0, 1], physicsClientId=self._client
            )
            projection = p.computeProjectionMatrixFOV(
                _CAMERA_FOV_DEGREES,
                _RENDER_WIDTH / _RENDER_HEIGHT,
                _CAMERA_NEAR,
                _CAMERA_FAR,
                physicsClientId=self._client,
            )
            _, _, rgba, _, _ = p.getCameraImage(
                _RENDER_WIDTH,
                _RENDER_HEIGHT,
                viewMatrix=view,
                projectionMatrix=projection,
                renderer=p.ER_TINY_RENDERER,
                shadow=False,
                physicsClientId=self._client,
            )
        frame = np.asarray(rgba, dtype=np.uint8).reshape(
            _RENDER_HEIGHT, _RENDER_WIDTH, 4
        )[:, :, :3]
        return cast(RenderFrame, frame)

    def close(self) -> None:
        with _CLIENT_LOCK:
            set_client(self._client)
            disconnect()

    # ------------------------------------------------------------- agent card

    @property
    def env_description(self) -> str:
        """Markdown description of this environment for an agent."""
        return self.describe(include_access=True)

    @property
    def env_description_blackbox(self) -> str:
        """Description for blackbox mode, without source pointers."""
        return self.describe(include_access=False)

    def describe(self, include_access: bool, count_invariant: bool = False) -> str:
        """Render the environment card.

        With *count_invariant*, this instance's objective count is left out, for the
        same reason as in the PR2 families: the variable-count wrapper builds one card
        from one backend, and naming a count would leak the evaluation sweep.
        """
        variant = (
            "A variable number of objectives"
            if count_invariant
            else f"{self._num_objectives} objective"
            f"{'' if self._num_objectives == 1 else 's'}"
        )
        observation_section = (
            ""
            if count_invariant
            else (
                f"## Observation Space\n\n"
                f"`Box(-inf, inf, ({self._obs_dim},), float32)`, laid out as "
                f"{len(self._rovers)} rovers, the lander, "
                f"{self._num_objectives} objectives, {len(self._samples)} samples "
                f"(stone first, then soil), then the mounds and pillars.\n\n"
            )
        )
        title = (
            "# Rovers\n\n"
            if count_invariant
            else f"# Rovers-o{self._num_objectives}\n\n"
        )
        return (
            title
            + (
                f"Two turtlebot rovers explore a {BASE_EXTENT:.0f}m square arena. They "
                f"must acquire **one stone sample and one soil sample**, **photograph "
                f"every objective**, and **radio all of it to the Husky lander**, "
                f"finishing with **both rovers back where they started** and **both "
                f"sample stores empty**. This is PDDLStream's `rovers` benchmark.\n\n"
                f"Mounds, pillars and a wall down the middle of the arena block both "
                f"driving and line of sight, so the difficulty is *where you have to "
                f"stand*: within {VIS_RANGE:.0f}m of an objective with nothing in the "
                f"way to photograph it, and within {COM_RANGE:.0f}m of the lander with "
                f"nothing in the way to radio anything at all.\n\n"
                f"## Variant\n\n"
                f"{variant}, standing on mounds; {self._num_rocks} stone samples and "
                f"{self._num_soils} soil samples scattered on the floor; "
                f"{self._num_obstacles} pillars. Objectives are reassigned to mounds "
                f"every episode, so which ones are occluded changes.\n\n"
                f"{observation_section}"
                f"## Action Space\n\n"
                f"`Box((8,), float32)` -- four entries per rover, rover0 "
                f"then rover1:\n\n"
                f"| **Index** | **Description** |\n"
                f"| --- | --- |\n"
                f"| 0, 4 | delta base x (+/-{_MAX_DELTA}) |\n"
                f"| 1, 5 | delta base y (+/-{_MAX_DELTA}) |\n"
                f"| 2, 6 | delta base heading (+/-{_MAX_DELTA_THETA}) |\n"
                f"| 3, 7 | operator selector (+/-1) |\n\n"
                f"The selector splits [-1, 1] into six equal bands. Band *k* "
                f"spans `-1 + k/3` to `-1 + (k+1)/3`, so its centre is "
                f"`(k + 0.5) / 3 - 1`, and the bands select, in order: "
                f"**sample**, **calibrate**, **image**, **noop**, **send**, "
                f"**drop**. Noop is the band containing zero, so an all-zero "
                f"action does nothing.\n\n"
                f"Each operator is refused silently -- the state simply "
                f"does not change -- "
                f"unless its precondition holds:\n\n"
                f"- **sample**: the rover's base is within {SAMPLE_RADIUS}m of a sample "
                f"and its store is empty. Fills the store and records the analysis.\n"
                f"- **calibrate**: some objective is visible. Cameras "
                f"must be calibrated "
                f"immediately before each photograph.\n"
                f"- **image**: the camera is calibrated and an objective is visible. "
                f"Photographs **one** objective -- the nearest one this "
                f"rover still needs "
                f"-- and spends the calibration, so N objectives cost N "
                f"calibrate/image pairs.\n"
                f"- **send**: the lander is visible. Radios everything this rover is "
                f"holding, images and analyses alike. Only the rover that took a result "
                f"can send it.\n"
                f"- **drop**: empties the store. The analysis is already recorded, so "
                f"dropping before or after sending both work.\n\n"
                f"Both rovers act on every step. Dynamics are kinematic: a motion that "
                f"would put a rover in collision with a wall, a mound, a pillar, the "
                f"lander or the other rover is rejected, leaving it where it was, and "
                f"the operator on that same step still applies.\n\n"
                f"## Reward\n\n"
                f"-1 per step, so the return is the negated number of "
                f"steps and a better "
                f"policy is a shorter one.\n\n"
            )
            + (
                ""
                if not include_access
                else (
                    "## Example Usage\n\n"
                    "```python\n"
                    "import numpy as np\n"
                    "from robocode.environments.rovers_env import RoversEnv\n\n"
                    + (
                        "env = RoversEnv(num_objectives=<count of your choice>)\n"
                        if count_invariant
                        else f"env = RoversEnv(num_objectives={self._num_objectives})\n"
                    )
                    + "obs, info = env.reset(seed=0)\n"
                    "action = env.action_space.sample()\n"
                    "next_obs, reward, terminated, truncated, info = env.step(action)\n"
                    "```\n\n"
                    "## Source Code\n\n"
                    "- `robocode/environments/rovers_env.py` — the operators, the "
                    "visibility tests, and the goal check\n"
                    "- `robocode/environments/rovers_scenes.py` — the arena, and the "
                    "per-episode layout sampler\n"
                    "- The kinematics and collision helpers live in `pybullet_tools`, "
                    "re-exported by `robocode/environments/ss_pybullet.py`\n\n"
                    "`pybullet_tools` addresses PyBullet through a module-global client "
                    "handle, so call it against `env.rovers` and the other "
                    "handles inside "
                    "`with env.client():`.\n"
                )
            )
        )


def _set_point(body: int, point: Any) -> None:
    """Move a body to *point*, keeping its orientation."""
    set_pose(
        body,
        ((float(point[0]), float(point[1]), float(point[2])), get_pose(body)[1]),
    )
