"""A PR2 tabletop TAMP environment backed by the ``packed`` benchmark scene.

The scene and goal come from PDDLStream's ``packed`` problem (see
:mod:`robocode.environments.pr2_tamp_scenes`): a PR2 must place every block from
the table onto a green plate. PDDLStream solves that by search over its own
streams; here the scene is instead exposed as a closed-loop gymnasium
environment, so a robocode approach has to synthesize its own task and motion
planning to drive the robot there.

The action and observation layout deliberately mirrors kinder's 3D
mobile-manipulation envs (delta base pose, delta arm joints, gripper open/close,
against a flat ``Box`` of object features), so approaches and prompts written
against those transfer without a new interface to learn.

Dynamics are kinematic, not dynamic: joints are set rather than servoed, a grasp
attaches the block to the tool frame rigidly, and a motion that puts the robot or
a held block in collision is rejected and leaves the state unchanged. That
matches the kinder envs, and it matches how PDDLStream itself evaluates ``packed``
-- its samplers check placements and collisions in exactly this kinematic sense.
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
from robocode.environments.pr2_tamp_scenes import (
    ARM,
    BLOCK_HEIGHT,
    BLOCK_WIDTH,
    PLATE_WIDTH,
    build_packed_scene,
    resample_block_placements,
)
from robocode.environments.ss_pybullet import (
    PR2_TOOL_FRAMES,
    Attachment,
    HideOutput,
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
    is_placement,
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

# pybullet_tools keeps the live physics client in a module-level ``CLIENT`` global
# that every helper reads, so two envs on two threads would otherwise drive each
# other's simulation. env_server hands each agent connection its own env on its own
# ThreadingTCPServer thread, so every entry point below takes this lock and rebinds
# the global to its own client for the duration of the call.
_CLIENT_LOCK = threading.RLock()

# Per-step bounds, matched to kinder's 3D mobile-manipulation action space so that
# the two families are interchangeable from an approach's point of view.
_MAX_DELTA = 0.2
_GRIPPER_THRESHOLD = 0.5

# Widths of the fixed-size observation blocks, documented in ``env_description``.
# ss-pybullet's is_placement admits a body resting up to 1cm above a surface but
# nothing at all below it (below_epsilon=0.0), and stable_z lands exactly on the
# boundary -- where float error puts the block an ULP under the surface about half
# the time, reading as "not placed". Settling a hair above keeps a released block
# visually flush while staying robustly inside that tolerance.
_SETTLE_EPSILON = 1e-4

# A grasp requires the fingers to close around the block rather than merely near it.
# +x of the tool frame is the approach direction, so a top grasp puts the block's
# centre one half-height along it; beyond that the fingers cannot reach past the
# block's top face. The tolerance absorbs the slack of arriving by delta actions.
_GRASP_APPROACH_TOLERANCE = 0.01
_MAX_GRASP_APPROACH = BLOCK_HEIGHT / 2 + _GRASP_APPROACH_TOLERANCE

_ROBOT_FEATURES = 19
_SURFACE_FEATURES = 10
_BLOCK_FEATURES = 11

_NUM_ARM_JOINTS = 7

# Index of the yaw entry within the base group (x, y, yaw).
_BASE_YAW_INDEX = 2

# Rendering camera: an elevated three-quarter view. The robot circles the table over
# an episode, so the camera sits far enough back that it stays in frame throughout,
# and high enough to look down onto the plate, which is where the task is decided.
_RENDER_WIDTH = 640
_RENDER_HEIGHT = 480
_CAMERA_EYE = [1.9, -1.9, 2.4]
_CAMERA_TARGET = [-0.2, 0.0, 0.8]
_CAMERA_FOV_DEGREES = 55.0
_CAMERA_NEAR = 0.05
_CAMERA_FAR = 10.0


class PR2PackedEnv(BaseEnv[NDArray[Any], NDArray[Any]]):
    """PR2 must place every block onto the plate (PDDLStream's ``packed``)."""

    def __init__(
        self,
        num_blocks: int = 3,
        grasp_radius: float = 0.08,
        max_placement_resets: int = 10,
    ) -> None:
        self._num_blocks = int(num_blocks)
        self._grasp_radius = float(grasp_radius)
        self._max_placement_resets = int(max_placement_resets)
        self._attachment: Attachment | None = None
        self._current_obs: NDArray[Any] | None = None

        with _CLIENT_LOCK:
            self._client = connect(use_gui=False)
            set_client(self._client)
            # Building the scene samples placements from the process-global legacy
            # RNG too; put it back for the same reason reset() does.
            entropy = np.random.get_state()
            try:
                with HideOutput():
                    self._problem, handles = build_packed_scene(self._num_blocks)
            finally:
                np.random.set_state(entropy)
            self._robot: int = handles["robot"]
            self._blocks: list[int] = handles["blocks"]
            self._table: int = handles["table"]
            self._plate: int = handles["plate"]
            self._initial_arm_conf: list[float] = handles["initial_arm_conf"]
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
            self._surface_extents = {
                surface: self._half_extent(surface)
                for surface in (self._table, self._plate)
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
            self._base_circular[_BASE_YAW_INDEX] = True
            self._arm_circular = np.array(
                [is_circular(self._robot, joint) for joint in self._arm_joints]
            )
            self._base_limits = self._joint_limits(self._base_joints)
            self._arm_limits = self._joint_limits(self._arm_joints)

        self._obs_dim = (
            _ROBOT_FEATURES + 2 * _SURFACE_FEATURES + _BLOCK_FEATURES * self._num_blocks
        )
        self.observation_space = Box(
            -np.inf, np.inf, (self._obs_dim,), dtype=np.float32
        )
        self.action_space = Box(
            -np.array([_MAX_DELTA] * (3 + _NUM_ARM_JOINTS) + [1.0], dtype=np.float32),
            np.array([_MAX_DELTA] * (3 + _NUM_ARM_JOINTS) + [1.0], dtype=np.float32),
            dtype=np.float32,
        )
        super().__init__()

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

    # -------------------------------------------------------- scene accessors

    # Read-only handles onto the scene. A planner (the oracle, or a program the agent
    # writes with whitebox access) needs the body ids and joint groups to do IK and
    # motion planning at all; exposing them here keeps that out of the private
    # attributes. None of them let a policy change the state -- the environment is
    # still driven through step().

    @property
    def client_id(self) -> int:
        """PyBullet client this environment's scene lives in."""
        return self._client

    @property
    def robot(self) -> int:
        """PyBullet body id of the PR2."""
        return self._robot

    @property
    def blocks(self) -> list[int]:
        """Body ids of the movable blocks."""
        return list(self._blocks)

    @property
    def table(self) -> int:
        """Body id of the table the blocks start on."""
        return self._table

    @property
    def plate(self) -> int:
        """Body id of the plate every block must end up on."""
        return self._plate

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

    # ------------------------------------------------------------- descriptions

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

    def _observation_table(self) -> str:
        rows = [
            "| **Index** | **Object** | **Feature** |",
            "| --- | --- | --- |",
        ]
        names = ["base_x", "base_y", "base_rot"]
        names += [f"joint_{i}" for i in range(1, _NUM_ARM_JOINTS + 1)]
        names += ["gripper_opening", "grasp_active"]
        names += [f"grasp_tf_{f}" for f in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        index = 0
        for name in names:
            rows.append(f"| {index} | robot | {name} |")
            index += 1
        pose = [
            "pose_x",
            "pose_y",
            "pose_z",
            "pose_qx",
            "pose_qy",
            "pose_qz",
            "pose_qw",
        ]
        extent = ["half_extent_x", "half_extent_y", "half_extent_z"]
        for surface in ("table", "plate"):
            for name in pose + extent:
                rows.append(f"| {index} | {surface} | {name} |")
                index += 1
        for block in range(self._num_blocks):
            for name in pose + ["grasp_active"] + extent:
                rows.append(f"| {index} | block{block} | {name} |")
                index += 1
        return "\n".join(rows)

    def describe(self, include_access: bool, count_invariant: bool = False) -> str:
        """Render the environment card.

        With *count_invariant*, every mention of this instance's block count is left
        out. The variable-count wrapper builds its card from one backend, and naming
        that backend's count -- or any count -- would tell an agent about the evaluation
        sweep, which is the experimenter's to know, not the agent's.
        """
        variant = (
            f"A variable number of blocks, each {BLOCK_WIDTH:.2f}m square and "
            f"{BLOCK_HEIGHT:.2f}m tall"
            if count_invariant
            else f"{self._num_blocks} blocks, each {BLOCK_WIDTH:.2f}m square and "
            f"{BLOCK_HEIGHT:.2f}m tall"
        )
        observation_section = (
            ""
            if count_invariant
            else (
                f"## Observation Space\n\n"
                f"`Box(-inf, inf, ({self._obs_dim},), float32)`. The "
                f"entries correspond to the following object features:\n\n"
                f"{self._observation_table()}\n\n"
            )
        )
        title = (
            "# PR2Packed\n\n"
            if count_invariant
            else (f"# PR2Packed-b{self._num_blocks}\n\n")
        )
        description = title + (
            f"A PR2 robot must pick up every block from the table and place it on "
            f"the green plate. This is the `packed` task and motion planning "
            f"benchmark: the blocks start scattered on the table at random "
            f"collision-free poses, the plate is small enough that they have to be "
            f"packed together, and reaching a block may require driving the base as "
            f"well as moving the arm.\n\n"
            f"## Variant\n\n"
            f"{variant}, on a plate {PLATE_WIDTH:.2f}m square. Only "
            f"the left arm is controllable; the right arm is tucked.\n\n"
            f"{observation_section}"
            f"`grasp_active` on the robot is 1.0 while a block is held, and the "
            f"block's own `grasp_active` marks which one. `grasp_tf` is the pose of "
            f"the held block in the gripper's tool frame, and is all zeros when "
            f"nothing is held. Poses are `(x, y, z)` position followed by an "
            f"`(qx, qy, qz, qw)` quaternion.\n\n"
            f"## Action Space\n\n"
            f"`Box(-0.2, 0.2, (11,), float32)` except for index 10, which is bounded "
            f"by +/-1. Actions are bounded relative base position, rotation, and arm "
            f"joint positions, plus gripper open/close.\n\n"
            f"| **Index** | **Description** |\n"
            f"| --- | --- |\n"
            f"| 0 | delta base x |\n"
            f"| 1 | delta base y |\n"
            f"| 2 | delta base rotation |\n"
            f"| 3 | delta joint 1 |\n"
            f"| 4 | delta joint 2 |\n"
            f"| 5 | delta joint 3 |\n"
            f"| 6 | delta joint 4 |\n"
            f"| 7 | delta joint 5 |\n"
            f"| 8 | delta joint 6 |\n"
            f"| 9 | delta joint 7 |\n"
            f"| 10 | gripper open/close |\n\n"
            f"The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise "
            f"no change.\n\n"
            f"**Closing** grasps a block only if the gripper is actually around one: "
            f"the block's centre must be between the fingers (within "
            f"{BLOCK_WIDTH / 2:.3f}m of the tool frame laterally) and no further than "
            f"{_MAX_GRASP_APPROACH:.2f}m along the approach axis, which is where the "
            f"fingers stop. Hovering above a block does not grasp it, however close "
            f"the tool frame is -- you have to bring the fingers down around it. The "
            f"nearest qualifying block is taken, and held rigidly until the gripper "
            f"opens. Check `grasp_active` to see whether a grasp took.\n\n"
            f"**Opening** drops the held block straight down onto whichever surface "
            f"it is above (the plate if it is over the plate, otherwise the table, "
            f"otherwise the floor). The drop is REFUSED if the block would land "
            f"overlapping another block or a fixed body: the block stays held and the "
            f"gripper stays closed, so `grasp_active` remains 1. Move somewhere clear "
            f"and open again.\n\n"
            f"Base and arm targets are clipped to the robot's joint limits, except "
            f"for base rotation (index 2) and the two continuous arm roll joints "
            f"(indices 7 and 9), which wrap around at +/-pi. Dynamics "
            f"are kinematic: a motion that would put the robot or the block it is "
            f"holding in collision with the table, the plate, or another block is "
            f"rejected, leaving the robot where it was. The gripper command on that "
            f"same step still applies.\n\n"
            f"## Reward\n\n"
            f"-1 per step. The episode terminates when every block rests on the plate "
            f"AND no two blocks overlap -- the plate is small enough that they have to "
            f"be packed, so dropping them all at one spot does not count. The return "
            f"is the negated number of steps taken, so a better policy is a shorter "
            f"one.\n\n"
        )
        if not include_access:
            return description
        return description + (
            "## Example Usage\n\n"
            "```python\n"
            "import numpy as np\n"
            "from robocode.environments.pr2_tamp_env import PR2PackedEnv\n\n"
            + (
                "env = PR2PackedEnv(num_blocks=<count of your choice>)\n"
                if count_invariant
                else f"env = PR2PackedEnv(num_blocks={self._num_blocks})\n"
            )
            + (
                "obs, info = env.reset(seed=0)\n"
                "# Take a random action\n"
                "action = env.action_space.sample()\n"
                "next_obs, reward, terminated, truncated, info = env.step(action)"
                "\n\n"
                "# Save and restore state\n"
                "saved = env.get_state()\n"
                "env.step(env.action_space.sample())\n"
                "env.set_state(saved)  # restores to the saved state\n"
                "```\n\n"
                "`obs` and `action` are numpy arrays matching the tables above.\n\n"
                "## Source Code\n\n"
                "- `robocode/environments/pr2_tamp_env.py` \u2014 `step()` transition "
                "dynamics, grasping, collision handling, and the goal check\n"
                "- `robocode/environments/pr2_tamp_scenes.py` \u2014 the scene: block, "
                "plate, and table geometry, and the initial-placement sampler\n"
                "- The underlying kinematics and collision helpers live in the "
                "`pybullet_tools` package, re-exported by "
                "`robocode/environments/ss_pybullet.py`\n\n"
                "`pybullet_tools` addresses PyBullet through a module-global client "
                "handle. If you call it directly against `env.robot`, `env.blocks` "
                "and the other handles above, do it inside `with env.client():` -- "
                "otherwise another environment in the same process can rebind that "
                "global underneath you mid-computation:\n\n"
                "```python\n"
                "with env.client():\n"
                "    ...  # pybullet_tools calls against env.robot, env.blocks, ...\n"
                "```\n"
            )
        )

    # ------------------------------------------------------------------ helpers

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
        for surface in (self._table, self._plate):
            point, quat = get_pose(surface)
            features.extend(list(point) + list(quat))
            features.extend(self._surface_extents[surface])
        for block in self._blocks:
            point, quat = get_pose(block)
            features.extend(list(point) + list(quat))
            features.append(1.0 if block == held else 0.0)
            features.extend([BLOCK_WIDTH / 2, BLOCK_WIDTH / 2, BLOCK_HEIGHT / 2])
        return np.array(features, dtype=np.float32)

    def _goal_reached(self) -> bool:
        """Every block resting on the plate, and no two blocks overlapping.

        The non-overlap clause is what makes this the ``packed`` task. ``is_placement``
        is a per-block centre-in-AABB-plus-height test, so without it a policy that
        released every block over one point would end with N interpenetrating blocks
        that each satisfy it, and score a solve for defeating the premise of the
        benchmark -- the plate being small enough to force a packing.
        """
        if not all(
            is_placement(block, surface) for block, surface in self._problem.goal_on
        ):
            return False
        return not self._blocks_overlap()

    def _blocks_overlap(self) -> bool:
        """True if any two blocks interpenetrate."""
        return any(
            pairwise_collision(a, b)
            for i, a in enumerate(self._blocks)
            for b in self._blocks[i + 1 :]
        )

    def _in_collision(self) -> bool:
        held = None if self._attachment is None else self._attachment.child
        moving = [self._robot] + ([held] if held is not None else [])
        for body in moving:
            for obstacle in self._obstacles:
                if pairwise_collision(body, obstacle):
                    return True
            for block in self._blocks:
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
            pairwise_collision(block, other) for other in self._blocks if other != block
        )

    def _try_settle(self, block: int) -> bool:
        """Move *block* down onto whatever it is over.

        Always succeeds today.
        """
        point, quat = get_pose(block)
        candidates = sorted(
            (self._plate, self._table),
            key=lambda surface: get_aabb(surface)[1][2],
            reverse=True,
        )
        for surface in candidates:
            aabb = get_aabb(surface)
            if aabb_contains_point(point[:2], aabb2d_from_aabb(aabb)) and (
                aabb[1][2] <= point[2] + BLOCK_HEIGHT
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
        set_point(
            block, Point(x=point[0], y=point[1], z=BLOCK_HEIGHT / 2 + _SETTLE_EPSILON)
        )
        return True

    def _grasp_candidate(self) -> int | None:
        """The block the gripper is closed around, if any.

        Distance to the tool frame alone is not enough: a gripper hovering above a
        block is within any usable radius of it while its fingers cannot reach past
        the block's top face. The block's centre is therefore required to lie between
        the fingers laterally, and no further along the approach axis than the block's
        own half-height, which is where the fingers stop.
        """
        tool = get_link_pose(self._robot, self._tool_link)
        best_block: int | None = None
        best_distance = self._grasp_radius
        for block in self._blocks:
            offset = np.array(multiply(invert(tool), get_pose(block))[0])
            approach, lateral = offset[0], offset[1:]
            if not -_GRASP_APPROACH_TOLERANCE <= approach <= _MAX_GRASP_APPROACH:
                continue
            if np.max(np.abs(lateral)) > BLOCK_WIDTH / 2:
                continue
            distance = float(np.linalg.norm(offset))
            if distance <= best_distance:
                best_distance = distance
                best_block = block
        return best_block

    # ------------------------------------------------------------------- gym API

    def reset(self, *args: Any, **kwargs: Any) -> tuple[NDArray[Any], dict[str, Any]]:
        super().reset(*args, **kwargs)
        with self.client():
            self._attachment = None
            set_joint_positions(self._robot, self._base_joints, [-1.0, 0.0, 0.0])
            set_joint_positions(self._robot, self._arm_joints, self._initial_arm_conf)
            open_arm(self._robot, ARM)
            # sample_placement draws from the process-global legacy numpy RNG, so it
            # has to be seeded from the episode's generator to keep instances
            # reproducible per eval seed. That global is shared with approaches and
            # anything else in this process, so its state is put back afterwards:
            # resetting an environment should not silently reseed its caller.
            entropy = np.random.get_state()
            try:
                for _ in range(self._max_placement_resets):
                    np.random.seed(int(self.np_random.integers(0, 2**31 - 1)))
                    if resample_block_placements(self._blocks, self._table):
                        break
                else:
                    raise RuntimeError(
                        f"Could not place {self._num_blocks} blocks on the table "
                        f"after {self._max_placement_resets} attempts"
                    )
            finally:
                np.random.set_state(entropy)
            self._current_obs = self._get_obs()
        return self._current_obs, {}

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
            offset = _ROBOT_FEATURES + 2 * _SURFACE_FEATURES
            held: int | None = None
            for i, block in enumerate(self._blocks):
                base = offset + i * _BLOCK_FEATURES
                set_pose(
                    block,
                    (tuple(state[base : base + 3]), tuple(state[base + 3 : base + 7])),
                )
                if state[base + 7] > 0.5:
                    held = block
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

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        # Driving PyBullet's camera directly rather than through pybullet_tools'
        # get_image: that helper forwards its `vertical_fov` to get_projection_matrix,
        # which documents the argument as RADIANS and calls math.degrees() on it, while
        # get_image defaults it to 60.0. The projection is then built from
        # degrees(60) = 3438 degrees, which renders a fisheye horizon with the scene a
        # few pixels across and no error raised.
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

    def close(self) -> None:
        # Not routed through client(): disconnect invalidates the id, so there is
        # nothing coherent to restore it to afterwards.
        with _CLIENT_LOCK:
            set_client(self._client)
            disconnect()
