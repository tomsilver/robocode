"""A PR2 tabletop TAMP environment backed by the ``blocked`` benchmark scene.

The scene and goal come from PDDLStream's ``blocked`` problem (see
:mod:`robocode.environments.pr2_tamp_scenes`): a green block sits on the near table
inside a three-sided pen of walls with a red block standing in the one gap, and the
goal is to get *any* green block onto the plate. Spare green blocks, if the instance
has them, wait on a second table nine metres away.

What makes this a different problem from ``packed`` rather than a re-skin of it is
the grasp. ``blocked`` is a *side*-grasp task: the pen walls are as tall as the
block, so the only approach that clears them is horizontal, and the red block
occupies the one horizontal direction left open. So the robot either moves the
blocker aside and takes the penned block, or drives to the far table and hauls a
spare back. With no spares, only the first is available.

The robot, the kinematic dynamics, the delta action space and the flat observation
layout are shared with the other PR2 TAMP scenes and live in
:mod:`robocode.environments.pr2_tamp_base`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robocode.environments.pr2_tamp_base import (
    GRASP_APPROACH_TOLERANCE,
    NUM_ARM_JOINTS,
    PR2TampEnv,
)
from robocode.environments.pr2_tamp_scenes import (
    BLOCKED_BLOCK_HEIGHT,
    BLOCKED_BLOCK_WIDTH,
    BLOCKED_PLATE_WIDTH,
    BLOCKED_SPACING,
    build_blocked_scene,
    resample_block_placements,
    resample_blocked_layout,
)
from robocode.environments.ss_pybullet import (
    HideOutput,
    get_pose,
    is_placement,
    set_pose,
)

# Rendering camera: the two tables are nine metres apart, so a view framing both
# would show neither usefully. The camera watches the near table, where the pen, the
# blocker and the plate are, and so where the task is decided; a robot that has
# driven off to the far table leaves the frame, which reads correctly as "it went to
# fetch a spare".
# A side grasp approaches horizontally (0 degrees) and a top grasp straight down
# (90). The limit sits well clear of both, leaving room for the tracking error of
# arriving by clipped delta actions while still refusing a reach over the pen.
_MAX_GRASP_PITCH_DEGREES = 25.0

_CAMERA_EYE = [6.6, -1.9, 2.4]
_CAMERA_TARGET = [4.5, 0.0, 0.85]


class PR2BlockedEnv(PR2TampEnv):
    """PR2 must place any green block onto the plate (PDDLStream's ``blocked``)."""

    def __init__(
        self,
        num_spares: int = 1,
        grasp_radius: float = 0.08,
        max_placement_resets: int = 10,
    ) -> None:
        self._num_spares = int(num_spares)
        self._max_placement_resets = int(max_placement_resets)
        super().__init__(grasp_radius=grasp_radius)

    # ------------------------------------------------------------ scene hooks

    def _build_scene(self) -> tuple[Any, dict[str, Any]]:
        with HideOutput():
            return build_blocked_scene(self._num_spares)

    def _bind_scene(self, handles: dict[str, Any]) -> None:
        self._greens: list[int] = handles["greens"]
        self._penned: int = handles["penned"]
        self._spares: list[int] = handles["spares"]
        self._blocker: int = handles["blocker"]
        self._walls: list[int] = handles["walls"]
        self._near_table: int = handles["near_table"]
        self._far_table: int = handles["far_table"]
        self._plate: int = handles["plate"]
        self._surfaces = [self._near_table, self._far_table, self._plate]
        # The blocker is movable and graspable -- relocating it is the whole task when
        # there are no spares -- but it is not a goal object.
        self._movables = [*self._greens, self._blocker]
        half = [
            BLOCKED_BLOCK_WIDTH / 2,
            BLOCKED_BLOCK_WIDTH / 2,
            BLOCKED_BLOCK_HEIGHT / 2,
        ]
        self._movable_half_extents = {body: half for body in self._movables}
        # The penned block, the blocker and the walls are the benchmark's fixed
        # layout rather than something sampled, so their initial poses are recorded
        # here and restored on every reset. Without that a solved episode would leave
        # a green block on the plate and the next reset would start already solved.
        self._initial_poses = {body: get_pose(body) for body in self._movables}
        # The walls move with the pen, so their build poses are recorded too, and the
        # pen's centre is the penned block's -- the point the assembly rotates about.
        self._initial_wall_poses = {body: get_pose(body) for body in self._walls}
        self._pen_centre = get_pose(self._penned)[0][:2]

    def _reset_scene(self) -> None:
        """Re-pose the pen assembly, then re-randomize the spares on the far table.

        The pen -- the penned block, the blocker and the walls -- moves rigidly, so its
        internal geometry stays exactly the benchmark's and the blocker always stands in
        the only gap; what varies is where the assembly sits and which way that gap
        faces. That is what makes an instance an instance: the goal is existential and
        the far-table spares never interact with it, so a fixed near table would leave
        the whole family solvable by one hard-coded trajectory.

        The bodies are restored to their build poses first, since the previous episode
        moved them and the resampling is defined relative to the benchmark's layout.
        """
        for body, pose in self._initial_poses.items():
            set_pose(body, pose)
        for body, pose in self._initial_wall_poses.items():
            set_pose(body, pose)

        obstacles = [self._near_table, self._plate, *self._spares]
        if not resample_blocked_layout(
            [self._penned, self._blocker, *self._walls],
            self._pen_centre,
            self.np_random,
            obstacles,
        ):
            raise RuntimeError("Could not place the pen assembly on the near table")

        if not self._spares:
            return
        for _ in range(self._max_placement_resets):
            np.random.seed(int(self.np_random.integers(0, 2**31 - 1)))
            if resample_block_placements(self._spares, self._far_table):
                return
        raise RuntimeError(
            f"Could not place {self._num_spares} spare blocks on the far table "
            f"after {self._max_placement_resets} attempts"
        )

    def _goal_reached(self) -> bool:
        """Whether any green block rests on the plate.

        Upstream writes the goal existentially, as one green block reaching the
        plate, so a single placement ends the episode however many spares exist. The
        red blocker is not a green block: parking *it* on the plate proves nothing.
        """
        return any(is_placement(green, self._plate) for green in self._greens)

    @property
    def _camera(self) -> tuple[list[float], list[float]]:
        return _CAMERA_EYE, _CAMERA_TARGET

    @property
    def _max_grasp_approach(self) -> float:
        # A side grasp approaches horizontally, so the block's centre sits one half
        # *width* along the approach axis rather than one half height.
        return BLOCKED_BLOCK_WIDTH / 2 + GRASP_APPROACH_TOLERANCE

    @property
    def _grasp_max_pitch_deg(self) -> float | None:
        """``blocked`` is a side-grasp task, so the approach must be near horizontal.

        Without this the pen is decorative. The position window alone is satisfied by
        reaching in diagonally over the blocker -- a synthesized policy found exactly
        that, taking the penned block at a 40 degree pitch without ever moving the
        blocker -- because the window only says where the block's centre sits relative
        to the tool, not which way the tool points. ss-pybullet's side grasps come in
        at 0 degrees and its top grasps at 90, so this admits the former with slack for
        arriving by bounded delta actions, and rejects anything reaching over the pen.
        """
        return _MAX_GRASP_PITCH_DEGREES

    @property
    def _grasp_lateral_limit(self) -> tuple[float, float]:
        # Across the approach axis a side grasp spans the block's other horizontal
        # half-width and its half-height; ss-pybullet's side grasps grip below the top
        # face, which puts the centre a few centimetres off along the tool's z.
        return BLOCKED_BLOCK_WIDTH / 2, BLOCKED_BLOCK_HEIGHT / 2

    # -------------------------------------------------------- scene accessors

    # Read-only handles onto the scene, for the same reason PR2PackedEnv exposes
    # its own: a planner needs body ids and joint groups to do IK and motion planning
    # at all. None of them let a policy change the state.

    @property
    def greens(self) -> list[int]:
        """The green blocks, any one of which on the plate ends the episode."""
        return list(self._greens)

    @property
    def penned(self) -> int:
        """The green block walled in on the near table, behind the blocker."""
        return self._penned

    @property
    def spares(self) -> list[int]:
        """The green blocks on the far table; empty when the count is zero."""
        return list(self._spares)

    @property
    def blocker(self) -> int:
        """The red block standing in the pen's one gap.

        Movable, but not a goal.
        """
        return self._blocker

    @property
    def walls(self) -> list[int]:
        """The three fixed walls penning the target block in."""
        return list(self._walls)

    @property
    def near_table(self) -> int:
        """The table holding the pen, the blocker and the plate."""
        return self._near_table

    @property
    def far_table(self) -> int:
        """The table holding the spare green blocks."""
        return self._far_table

    @property
    def plate(self) -> int:
        """The goal surface."""
        return self._plate

    # -------------------------------------------------------------- agent card

    def _observation_table(self) -> str:
        rows = [
            "| **Index** | **Object** | **Feature** |",
            "| --- | --- | --- |",
        ]
        names = ["base_x", "base_y", "base_rot"]
        names += [f"joint_{i}" for i in range(1, NUM_ARM_JOINTS + 1)]
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
        for surface in ("near_table", "far_table", "plate"):
            for name in pose + extent:
                rows.append(f"| {index} | {surface} | {name} |")
                index += 1
        labels = ["green0 (penned)"]
        labels += [f"green{i + 1} (spare)" for i in range(len(self._spares))]
        labels += ["blocker (red)"]
        for label in labels:
            for name in pose + ["grasp_active"] + extent:
                rows.append(f"| {index} | {label} | {name} |")
                index += 1
        return "\n".join(rows)

    def describe(self, include_access: bool, count_invariant: bool = False) -> str:
        """Render the environment card.

        With *count_invariant*, every mention of this instance's spare count is left
        out, for the same reason as in ``PR2PackedEnv``: the variable-count wrapper
        builds one card from one backend, and naming any count would leak the
        evaluation sweep, which is the experimenter's to know and not the agent's.
        """
        spares = (
            "There may be any number of spare green blocks on the far table, "
            "including none"
            if count_invariant
            else (
                f"There {'is' if self._num_spares == 1 else 'are'} "
                f"{self._num_spares} spare green "
                f"block{'' if self._num_spares == 1 else 's'} on the far table"
                if self._num_spares
                else "There are no spare green blocks: the penned block is the only "
                "one, so the blocker has to be moved"
            )
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
            "# PR2Blocked\n\n"
            if count_invariant
            else f"# PR2Blocked-s{self._num_spares}\n\n"
        )
        description = title + (
            f"A PR2 robot must get **any one green block** onto the green plate. This "
            f"is the `blocked` task and motion planning benchmark. One green block "
            f"sits on the near table inside a pen of three fixed walls, and a red "
            f"block stands in the pen's one gap. The walls are as tall as the block, "
            f"so the only grasp that clears them is a horizontal one -- and the red "
            f"block is in the way of it. Either move the red block aside and take the "
            f"penned block, or fetch a spare green block from the far table.\n\n"
            f"The red block is movable and graspable, but it is not a green block: "
            f"putting it on the plate does not finish the episode.\n\n"
            f"## Variant\n\n"
            f"Every block is {BLOCKED_BLOCK_WIDTH:.2f}m square and "
            f"{BLOCKED_BLOCK_HEIGHT:.2f}m tall, on a plate "
            f"{BLOCKED_PLATE_WIDTH:.2f}m square. The pen leaves a "
            f"{BLOCKED_SPACING:.2f}m gap, measured centre to centre, between the "
            f"penned block and the blocker. {spares}. The two tables are nine metres "
            f"apart, so fetching a spare is a long base drive. Only the left arm is "
            f"controllable; the right arm is tucked.\n\n"
            f"{observation_section}"
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
            f"the block's centre must be within "
            f"{BLOCKED_BLOCK_WIDTH / 2:.3f}m of the tool frame across the approach "
            f"axis and {BLOCKED_BLOCK_HEIGHT / 2:.3f}m along the gripper's vertical, "
            f"and no further than {self._max_grasp_approach:.3f}m along the approach "
            f"axis itself, which is where the fingers stop. That is half a block "
            f"*width*, not half its height: this is a side grasp, so you have to "
            f"bring the fingers in from the side rather than down from above. The "
            f"nearest qualifying block is taken, and held rigidly until the gripper "
            f"opens. Check `grasp_active` to see whether a grasp took.\n\n"
            f"**Opening** drops the held block straight down onto whichever surface "
            f"it is above (the plate if it is over the plate, otherwise whichever "
            f"table, otherwise the floor). The drop is REFUSED if the block would "
            f"land overlapping another block or a fixed body: the block stays held "
            f"and the gripper stays closed, so `grasp_active` remains 1. Move "
            f"somewhere clear and open again.\n\n"
            f"Base and arm targets are clipped to the robot's joint limits, except "
            f"for base rotation (index 2) and the two continuous arm roll joints "
            f"(indices 7 and 9), which wrap around at +/-pi. Dynamics "
            f"are kinematic: a motion that would put the robot or the block it is "
            f"holding in collision with a table, the plate, a wall, or another block "
            f"is rejected, leaving the robot where it was. The gripper command on "
            f"that same step still applies.\n\n"
            f"## Reward\n\n"
            f"-1 per step. The episode terminates as soon as any green block rests on "
            f"the plate. The return is the negated number of steps taken, so a better "
            f"policy is a shorter one -- which is the choice the task poses: moving "
            f"the blocker is a short manipulation, and fetching a spare is a long "
            f"drive.\n\n"
        )
        if not include_access:
            return description
        return description + (
            "## Example Usage\n\n"
            "```python\n"
            "import numpy as np\n"
            "from robocode.environments.pr2_tamp_blocked_env import PR2BlockedEnv\n\n"
            + (
                "env = PR2BlockedEnv(num_spares=<count of your choice>)\n"
                if count_invariant
                else f"env = PR2BlockedEnv(num_spares={self._num_spares})\n"
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
                "- `robocode/environments/pr2_tamp_blocked_env.py` — the scene "
                "handles, the goal check, and the side-grasp geometry\n"
                "- `robocode/environments/pr2_tamp_base.py` — `step()` transition "
                "dynamics, grasping, collision handling, and the release rule\n"
                "- `robocode/environments/pr2_tamp_scenes.py` — the scene: block, "
                "wall, plate and table geometry, and the spare-placement sampler\n"
                "- The underlying kinematics and collision helpers live in the "
                "`pybullet_tools` package, re-exported by "
                "`robocode/environments/ss_pybullet.py`\n\n"
                "`pybullet_tools` addresses PyBullet through a module-global client "
                "handle. If you call it directly against `env.robot`, `env.greens` "
                "and the other handles above, do it inside `with env.client():` -- "
                "otherwise another environment in the same process can rebind that "
                "global underneath you mid-computation:\n\n"
                "```python\n"
                "with env.client():\n"
                "    ...  # pybullet_tools calls against env's handles\n"
                "```\n"
            )
        )
