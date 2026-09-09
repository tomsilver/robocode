"""Shared machinery for the PR2 tabletop TAMP environments.

PDDLStream's ``packed`` and ``blocked`` problems put the same robot, the same
kinematic dynamics, and the same action space around different scenes and goals:
``packed`` wants every block on a plate and is a top-grasp problem, while
``blocked`` wants any one green block on a plate and is a side-grasp problem whose
target starts penned in behind a blocker. Everything that does not depend on which
of those is being played -- the physics client guard, the delta-action stepping and
its collision rejection, the release-and-settle rule, the grasp test, and the flat
observation layout -- lives here, and each scene supplies the rest.

Subclasses build their scene in ``_build_scene`` and are then expected to have set
the attributes ``_surfaces``, ``_movables`` and ``_movable_half_extents`` (which
together fix the observation layout), the grasp geometry, and the render camera.
They implement ``_goal_reached``, ``_reset_scene``, ``describe`` and
``_observation_table``.

Dynamics are kinematic, not dynamic: joints are set rather than servoed, a grasp
attaches a body to the tool frame rigidly, and a motion that puts the robot or a
held body in collision is rejected and leaves the state unchanged. That matches
the kinder 3D envs, and it matches how PDDLStream itself evaluates these problems
-- its samplers check placements and collisions in exactly this kinematic sense.
"""

from __future__ import annotations

import threading
from abc import abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, SupportsFloat, cast

import numpy as np
import pybullet as p
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box
from numpy.typing import NDArray

from robocode.environments.base_env import BaseEnv
from robocode.environments.pr2_tamp_scenes import ARM
from robocode.environments.ss_pybullet import (
    PR2_TOOL_FRAMES,
    Attachment,
    Point,
    aabb2d_from_aabb,
    aabb_contains_point,
    close_arm,
    connect,
    create_attachment,
    disconnect,
    get_aabb,
    get_arm_joints,
    get_client,
    get_gripper_joints,
    get_group_joints,
    get_joint_limits,
    get_joint_positions,
    get_link_pose,
    get_pose,
    invert,
    is_circular,
    link_from_name,
    multiply,
    open_arm,
    pairwise_collision,
    set_client,
    set_joint_positions,
    set_point,
    set_pose,
    stable_z,
)

# PyBullet addresses clients through a module-global in pybullet_tools, so two
# environments in one process have to take turns; the lock is reentrant so a
# nested guard on the same environment does not deadlock.
_CLIENT_LOCK = threading.RLock()

_MAX_DELTA = 0.2
_GRIPPER_THRESHOLD = 0.5

# ss-pybullet's is_placement admits a body resting up to 1cm above a surface but
# nothing at all below it (below_epsilon=0.0), and stable_z lands exactly on the
# boundary -- where float error puts the body an ULP under the surface about half
# the time, reading as "not placed". Settling a hair above keeps a released body
# visually flush while staying robustly inside that tolerance.
_SETTLE_EPSILON = 1e-4

# Slack on the grasp test, absorbing the error of arriving by bounded delta actions
# rather than landing exactly on a planned configuration.
GRASP_APPROACH_TOLERANCE = 0.01

# Widths of the fixed-size observation blocks, documented in ``describe``.
ROBOT_FEATURES = 19
SURFACE_FEATURES = 10
BODY_FEATURES = 11

NUM_ARM_JOINTS = 7

# Index of the yaw entry within the base group (x, y, yaw).
BASE_YAW_INDEX = 2

_RENDER_WIDTH = 640
_RENDER_HEIGHT = 480
_CAMERA_FOV_DEGREES = 55.0
_CAMERA_NEAR = 0.05
_CAMERA_FAR = 10.0


class PR2TampEnv(BaseEnv[NDArray[Any], NDArray[Any]]):
    """A PR2 driven by bounded delta actions over a PDDLStream tabletop scene."""

    # Set by ``_bind_scene``; together these fix the observation layout. Surfaces are
    # reported before movables, and both in the order the scene registers them.
    _surfaces: list[int]
    _movables: list[int]
    _movable_half_extents: dict[int, list[float]]

    def __init__(self, grasp_radius: float = 0.08) -> None:
        self._grasp_radius = float(grasp_radius)
        self._attachment: Attachment | None = None
        self._current_obs: NDArray[Any] | None = None

        with _CLIENT_LOCK:
            self._client = connect(use_gui=False)
            set_client(self._client)
            # Building the scene samples placements from the process-global legacy
            # RNG too; put it back for the same reason reset() does.
            entropy = np.random.get_state()
            try:
                self._problem, handles = self._build_scene()
            finally:
                np.random.set_state(entropy)
            self._robot: int = handles["robot"]
            self._initial_arm_conf: list[float] = handles["initial_arm_conf"]
            self._initial_base_conf: list[float] = list(handles["initial_base_conf"])
            self._base_joints = list(get_group_joints(self._robot, "base"))
            self._arm_joints = list(get_arm_joints(self._robot, ARM))
            self._gripper_joints = list(get_gripper_joints(self._robot, ARM))
            self._tool_link = link_from_name(self._robot, PR2_TOOL_FRAMES[ARM])
            self._gripper_open = float(
                max(get_joint_positions(self._robot, self._gripper_joints))
            )
            # The floor is excluded: the PR2's wheels rest on it, so it registers a
            # collision in every state and would reject every action.
            self._obstacles = [
                body for body in self._problem.fixed if body != handles["floor"]
            ]
            self._bind_scene(handles)
            self._surface_extents = {
                surface: self._half_extent(surface) for surface in self._surfaces
            }
            # The two forearm/wrist roll joints are continuous. Clipping them at the
            # +/-pi that get_joint_limits reports would put an artificial hard stop in
            # the middle of their range and make a band of wrist orientations
            # unreachable, so they wrap instead.
            self._base_circular = np.array(
                [is_circular(self._robot, joint) for joint in self._base_joints]
            )
            # The planar base's rotation joint carries explicit +/-pi limits in the
            # URDF, but a mobile base spins freely. Clipping there strands the robot
            # facing backwards and unable to turn through pi to reach the table.
            self._base_circular[BASE_YAW_INDEX] = True
            self._arm_circular = np.array(
                [is_circular(self._robot, joint) for joint in self._arm_joints]
            )
            self._base_limits = self._joint_limits(self._base_joints)
            self._arm_limits = self._joint_limits(self._arm_joints)

        self._obs_dim = (
            ROBOT_FEATURES
            + SURFACE_FEATURES * len(self._surfaces)
            + BODY_FEATURES * len(self._movables)
        )
        self.observation_space = Box(
            -np.inf, np.inf, (self._obs_dim,), dtype=np.float32
        )
        self.action_space = Box(
            -np.array([_MAX_DELTA] * (3 + NUM_ARM_JOINTS) + [1.0], dtype=np.float32),
            np.array([_MAX_DELTA] * (3 + NUM_ARM_JOINTS) + [1.0], dtype=np.float32),
            dtype=np.float32,
        )
        super().__init__()

    # ------------------------------------------------------------ scene hooks

    @abstractmethod
    def _build_scene(self) -> tuple[Any, dict[str, Any]]:
        """Build the scene in the live client, returning its Problem and handles."""

    @abstractmethod
    def _bind_scene(self, handles: dict[str, Any]) -> None:
        """Store scene handles, and set ``_surfaces``/``_movables`` and their extents.

        Called with the live client held, before the observation layout is fixed.
        """

    @abstractmethod
    def _reset_scene(self) -> None:
        """Re-randomize the movable bodies for a new episode, client already held."""

    @abstractmethod
    def _goal_reached(self) -> bool:
        """Whether this scene's goal holds in the current state."""

    @abstractmethod
    def describe(self, include_access: bool, count_invariant: bool = False) -> str:
        """Render this environment's card for an agent."""

    @property
    @abstractmethod
    def _camera(self) -> tuple[list[float], list[float]]:
        """Render camera as (eye, target)."""

    @property
    @abstractmethod
    def _max_grasp_approach(self) -> float:
        """How far along the tool's approach axis a graspable body's centre may sit.

        A grasp requires the fingers to close around the body rather than merely be near
        it. +x of the tool frame is the approach direction, so the limit is the body's
        half-extent along that axis: half its height for a top grasp, half its width for
        a side grasp. Beyond it the fingers cannot reach past the far face.
        """

    @property
    @abstractmethod
    def _grasp_lateral_limit(self) -> tuple[float, float]:
        """Half-extents the body's centre must lie within, across the approach axis."""

    @property
    def movables(self) -> list[int]:
        """Every body a policy can move, in observation order.

        A planner needs these to reproduce the environment's collision test, which
        treats each of them as an obstacle except the one currently held.
        """
        return list(self._movables)

    @property
    def surfaces(self) -> list[int]:
        """Every surface a released body can settle onto, in observation order."""
        return list(self._surfaces)

    def _graspable(self) -> list[int]:
        """Bodies a closing gripper may pick up.

        Defaults to every movable.
        """
        return list(self._movables)

    # ------------------------------------------- shared machinery

    @contextmanager
    def client(self) -> Iterator[int]:
        """Hold this environment's physics client for the duration of the block.

        pybullet_tools addresses PyBullet through a module-global ``CLIENT``, so any
        code calling it against the handles below -- the oracle, or a program an agent
        writes -- must hold this, or a second environment in the same process can
        rebind that global underneath it mid-computation. The environment's own
        methods take it internally; this is the same lock, and it is reentrant.

        The previous binding is restored on exit. The lock is reentrant and global to
        the process, so one thread can legitimately be inside two environments'
        guards at once; without restoring, the outer one would silently continue
        against the inner one's simulation after the inner block ended.
        """
        with _CLIENT_LOCK:
            previous = get_client()
            set_client(self._client)
            try:
                yield self._client
            finally:
                set_client(previous)

    @property
    def client_id(self) -> int:
        """PyBullet client this environment's scene lives in."""
        return self._client

    @property
    def robot(self) -> int:
        """PyBullet body id of the PR2."""
        return self._robot

    @property
    def obstacles(self) -> list[int]:
        """Fixed bodies that reject a motion on contact (the floor is excluded)."""
        return list(self._obstacles)

    @property
    def base_joints(self) -> list[int]:
        """Joint indices of the planar base group (x, y, yaw)."""
        return list(self._base_joints)

    @property
    def arm_joints(self) -> list[int]:
        """Joint indices of the controllable arm."""
        return list(self._arm_joints)

    @property
    def tool_link(self) -> int:
        """Link index of the gripper tool frame that grasps are measured from."""
        return self._tool_link

    @property
    def initial_arm_conf(self) -> list[float]:
        """The carry configuration the arm starts each episode in."""
        return list(self._initial_arm_conf)

    @property
    def base_circular(self) -> NDArray[Any]:
        """Per-base-joint mask of which joints wrap rather than clip."""
        return self._base_circular.copy()

    @property
    def arm_circular(self) -> NDArray[Any]:
        """Per-arm-joint mask of which joints wrap rather than clip."""
        return self._arm_circular.copy()

    @property
    def attachment(self) -> Attachment | None:
        """The grasp the environment currently holds, or None."""
        return self._attachment

    @property
    def env_description(self) -> str:
        """Markdown description of this environment for an agent."""
        return self.describe(include_access=True)

    @property
    def env_description_blackbox(self) -> str:
        """Description for blackbox mode.

        Omits the direct-import example and the source-code pointers; a blackbox agent
        has access to neither and reaches the environment only through env_client.
        """
        return self.describe(include_access=False)

    def _half_extent(self, body: int) -> NDArray[np.float64]:
        lower, upper = get_aabb(body)
        return (np.array(upper) - np.array(lower)) / 2.0

    def _joint_limits(self, joints: list[int]) -> tuple[NDArray[Any], NDArray[Any]]:
        limits = [get_joint_limits(self._robot, joint) for joint in joints]
        return (
            np.array([low for low, _ in limits]),
            np.array([high for _, high in limits]),
        )

    @staticmethod
    def _advance(
        conf: NDArray[Any],
        delta: NDArray[Any],
        limits: tuple[NDArray[Any], NDArray[Any]],
        circular: NDArray[Any],
    ) -> NDArray[Any]:
        """Apply a joint delta, wrapping continuous joints and clipping bounded ones."""
        target = conf + delta
        wrapped = (target + np.pi) % (2 * np.pi) - np.pi
        return np.where(circular, wrapped, np.clip(target, limits[0], limits[1]))

    def _in_collision(self) -> bool:
        held = None if self._attachment is None else self._attachment.child
        moving = [self._robot] + ([held] if held is not None else [])
        for body in moving:
            for obstacle in self._obstacles:
                if pairwise_collision(body, obstacle):
                    return True
            for block in self._movables:
                if block in (held, body):
                    continue
                if pairwise_collision(body, block):
                    return True
        return False

    def _settle(self, block: int) -> bool:
        """Drop *block* onto the highest surface it is over, else onto the floor.

        Returns whether the drop was accepted. A settled pose that interpenetrates an
        already-placed block, or a fixed body, is refused and the block is left where
        it was: releasing is how a block reaches the plate, so an unchecked drop is a
        free way to stack blocks inside one another.
        """
        original = get_pose(block)
        if not self._try_settle(block):
            set_pose(block, original)
            return False
        if self._settled_pose_blocked(block):
            set_pose(block, original)
            return False
        return True

    def _settled_pose_blocked(self, block: int) -> bool:
        """True if *block* at its current pose overlaps another body."""
        for other in self._obstacles:
            if pairwise_collision(block, other):
                return True
        return any(
            pairwise_collision(block, other)
            for other in self._movables
            if other != block
        )

    def _try_settle(self, block: int) -> bool:
        """Move *block* down onto whatever it is over.

        Always succeeds today.
        """
        point, quat = get_pose(block)
        # The released body's own height, since the scenes do not agree on one.
        height = 2 * self._movable_half_extents[block][2]
        candidates = sorted(
            self._surfaces,
            key=lambda surface: get_aabb(surface)[1][2],
            reverse=True,
        )
        for surface in candidates:
            aabb = get_aabb(surface)
            if aabb_contains_point(point[:2], aabb2d_from_aabb(aabb)) and (
                aabb[1][2] <= point[2] + height
            ):
                set_pose(block, (point, quat))
                set_point(
                    block,
                    Point(
                        x=point[0],
                        y=point[1],
                        z=stable_z(block, surface) + _SETTLE_EPSILON,
                    ),
                )
                return True
        set_point(block, Point(x=point[0], y=point[1], z=height / 2 + _SETTLE_EPSILON))
        return True

    def step(
        self, action: NDArray[Any]
    ) -> tuple[NDArray[Any], SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        with self.client():
            base_conf = np.array(get_joint_positions(self._robot, self._base_joints))
            arm_conf = np.array(get_joint_positions(self._robot, self._arm_joints))
            held_pose = (
                None if self._attachment is None else get_pose(self._attachment.child)
            )

            new_base = self._advance(
                base_conf, action[:3], self._base_limits, self._base_circular
            )
            new_arm = self._advance(
                arm_conf, action[3:10], self._arm_limits, self._arm_circular
            )
            set_joint_positions(self._robot, self._base_joints, new_base)
            set_joint_positions(self._robot, self._arm_joints, new_arm)
            if self._attachment is not None:
                self._attachment.assign()

            if self._in_collision():
                set_joint_positions(self._robot, self._base_joints, base_conf)
                set_joint_positions(self._robot, self._arm_joints, arm_conf)
                if self._attachment is not None and held_pose is not None:
                    set_pose(self._attachment.child, held_pose)

            gripper = float(action[10])
            if gripper < -_GRIPPER_THRESHOLD:
                close_arm(self._robot, ARM)
                if self._attachment is None:
                    candidate = self._grasp_candidate()
                    if candidate is not None:
                        self._attachment = create_attachment(
                            self._robot, self._tool_link, candidate
                        )
            elif gripper > _GRIPPER_THRESHOLD:
                if self._attachment is not None:
                    released = self._attachment.child
                    if self._settle(released):
                        self._attachment = None
                        open_arm(self._robot, ARM)
                else:
                    open_arm(self._robot, ARM)

            terminated = self._goal_reached()
            self._current_obs = self._get_obs()
        return self._current_obs, -1.0, terminated, False, {}

    def get_state(self) -> NDArray[Any]:
        assert self._current_obs is not None, "Must call reset()"
        return self._current_obs.copy()

    def close(self) -> None:
        # Not routed through client(): disconnect invalidates the id, so there is
        # nothing coherent to restore it to afterwards.
        with _CLIENT_LOCK:
            set_client(self._client)
            disconnect()

    # ------------------------------------------- observation, state, rendering

    def _get_obs(self) -> NDArray[Any]:
        features: list[float] = []
        features.extend(get_joint_positions(self._robot, self._base_joints))
        features.extend(get_joint_positions(self._robot, self._arm_joints))
        features.append(
            float(max(get_joint_positions(self._robot, self._gripper_joints)))
        )
        held = None if self._attachment is None else self._attachment.child
        features.append(1.0 if held is not None else 0.0)
        if self._attachment is None:
            features.extend([0.0] * 7)
        else:
            point, quat = self._attachment.grasp_pose
            features.extend(list(point) + list(quat))
        for surface in self._surfaces:
            point, quat = get_pose(surface)
            features.extend(list(point) + list(quat))
            features.extend(self._surface_extents[surface])
        for body in self._movables:
            point, quat = get_pose(body)
            features.extend(list(point) + list(quat))
            features.append(1.0 if body == held else 0.0)
            features.extend(self._movable_half_extents[body])
        return np.array(features, dtype=np.float32)

    def set_state(self, state: NDArray[Any]) -> None:
        state = np.asarray(state, dtype=np.float32)
        with self.client():
            set_joint_positions(self._robot, self._base_joints, state[0:3])
            set_joint_positions(self._robot, self._arm_joints, state[3:10])
            set_joint_positions(
                self._robot,
                self._gripper_joints,
                [state[10]] * len(self._gripper_joints),
            )
            offset = ROBOT_FEATURES + SURFACE_FEATURES * len(self._surfaces)
            held: int | None = None
            for i, body in enumerate(self._movables):
                base = offset + i * BODY_FEATURES
                set_pose(
                    body,
                    (tuple(state[base : base + 3]), tuple(state[base + 3 : base + 7])),
                )
                if state[base + 7] > 0.5:
                    held = body
            if held is None or state[11] <= 0.5:
                self._attachment = None
            else:
                self._attachment = Attachment(
                    self._robot,
                    self._tool_link,
                    (tuple(state[12:15]), tuple(state[15:19])),
                    held,
                )
            self._current_obs = self._get_obs()

    def _grasp_candidate(self) -> int | None:
        """The body the gripper is closed around, if any.

        Distance to the tool frame alone is not enough: a gripper hovering short of a
        body is within any usable radius of it while its fingers cannot reach around
        it. The body's centre is therefore required to lie between the fingers across
        the approach axis, and no further along that axis than its own half-extent,
        which is where the fingers stop.
        """
        tool = get_link_pose(self._robot, self._tool_link)
        lateral_limit = np.array(self._grasp_lateral_limit)
        best_body: int | None = None
        best_distance = self._grasp_radius
        for body in self._graspable():
            offset = np.array(multiply(invert(tool), get_pose(body))[0])
            approach, lateral = offset[0], offset[1:]
            if not -GRASP_APPROACH_TOLERANCE <= approach <= self._max_grasp_approach:
                continue
            if np.any(np.abs(lateral) > lateral_limit):
                continue
            distance = float(np.linalg.norm(offset))
            if distance <= best_distance:
                best_distance = distance
                best_body = body
        return best_body

    def reset(self, *args: Any, **kwargs: Any) -> tuple[NDArray[Any], dict[str, Any]]:
        super().reset(*args, **kwargs)
        with self.client():
            self._attachment = None
            set_joint_positions(self._robot, self._base_joints, self._initial_base_conf)
            set_joint_positions(self._robot, self._arm_joints, self._initial_arm_conf)
            open_arm(self._robot, ARM)
            # sample_placement draws from the process-global legacy numpy RNG, so it
            # has to be seeded from the episode's generator to keep instances
            # reproducible per eval seed. That global is shared with approaches and
            # anything else in this process, so its state is put back afterwards:
            # resetting an environment should not silently reseed its caller.
            entropy = np.random.get_state()
            try:
                self._reset_scene()
            finally:
                np.random.set_state(entropy)
            self._current_obs = self._get_obs()
        return self._current_obs, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        # Driving PyBullet's camera directly rather than through pybullet_tools'
        # get_image: that helper forwards its `vertical_fov` to get_projection_matrix,
        # which documents the argument as RADIANS and calls math.degrees() on it, while
        # get_image defaults it to 60.0. The projection is then built from
        # degrees(60) = 3438 degrees, which renders a fisheye horizon with the scene a
        # few pixels across and no error raised.
        eye, target = self._camera
        with self.client():
            view = p.computeViewMatrix(
                eye, target, [0, 0, 1], physicsClientId=self._client
            )
            projection = p.computeProjectionMatrixFOV(
                _CAMERA_FOV_DEGREES,
                _RENDER_WIDTH / _RENDER_HEIGHT,
                _CAMERA_NEAR,
                _CAMERA_FAR,
                physicsClientId=self._client,
            )
            # DIRECT mode has no OpenGL context, so the hardware renderer is not an
            # option here.
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
        # gymnasium declares RenderFrame as an unbound TypeVar, so a concrete frame
        # type never unifies with it.
        return cast(RenderFrame, frame)
