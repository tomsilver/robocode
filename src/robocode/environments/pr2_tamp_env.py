"""A PR2 tabletop TAMP environment backed by the ``packed`` benchmark scene.

The scene and goal come from PDDLStream's ``packed`` problem (see
:mod:`robocode.environments.pr2_tamp_scenes`): a PR2 must place every block from
the table onto a green plate. PDDLStream solves that by search over its own
streams; here the scene is instead exposed as a closed-loop gymnasium
environment, so a robocode approach has to synthesize its own task and motion
planning to drive the robot there.

The robot, the kinematic dynamics, the delta action space and the flat
observation layout are shared with the other PR2 TAMP scenes and live in
:mod:`robocode.environments.pr2_tamp_base`; what is here is the ``packed`` scene,
its goal, and its card. ``packed`` is a top-grasp problem, so a graspable block's
centre sits within half its height along the tool's approach axis.
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
    BLOCK_HEIGHT,
    BLOCK_WIDTH,
    PLATE_WIDTH,
    build_packed_scene,
    resample_block_placements,
)
from robocode.environments.ss_pybullet import (
    HideOutput,
    is_placement,
    pairwise_collision,
)

# Rendering camera: an elevated three-quarter view. The robot circles the table over
# an episode, so the camera sits far enough back that it stays in frame throughout,
# and high enough to look down onto the plate, which is where the task is decided.
_CAMERA_EYE = [1.9, -1.9, 2.4]
_CAMERA_TARGET = [-0.2, 0.0, 0.8]


class PR2PackedEnv(PR2TampEnv):
    """PR2 must place every block onto the plate (PDDLStream's ``packed``)."""

    def __init__(
        self,
        num_blocks: int = 3,
        grasp_radius: float = 0.08,
        max_placement_resets: int = 10,
    ) -> None:
        self._num_blocks = int(num_blocks)
        self._max_placement_resets = int(max_placement_resets)
        super().__init__(grasp_radius=grasp_radius)

    # ------------------------------------------------------------ scene hooks

    def _build_scene(self) -> tuple[Any, dict[str, Any]]:
        with HideOutput():
            return build_packed_scene(self._num_blocks)

    def _bind_scene(self, handles: dict[str, Any]) -> None:
        self._blocks: list[int] = handles["blocks"]
        self._table: int = handles["table"]
        self._plate: int = handles["plate"]
        self._surfaces = [self._table, self._plate]
        self._movables = list(self._blocks)
        half = [BLOCK_WIDTH / 2, BLOCK_WIDTH / 2, BLOCK_HEIGHT / 2]
        self._movable_half_extents = {block: half for block in self._blocks}

    def _reset_scene(self) -> None:
        for _ in range(self._max_placement_resets):
            np.random.seed(int(self.np_random.integers(0, 2**31 - 1)))
            if resample_block_placements(self._blocks, self._table):
                return
        raise RuntimeError(
            f"Could not place {self._num_blocks} blocks on the table "
            f"after {self._max_placement_resets} attempts"
        )

    @property
    def _camera(self) -> tuple[list[float], list[float]]:
        return _CAMERA_EYE, _CAMERA_TARGET

    @property
    def _max_grasp_approach(self) -> float:
        # A top grasp puts the block's centre one half-height along the approach.
        return BLOCK_HEIGHT / 2 + GRASP_APPROACH_TOLERANCE

    @property
    def _grasp_lateral_limit(self) -> tuple[float, float]:
        return BLOCK_WIDTH / 2, BLOCK_WIDTH / 2

    # -------------------------------------------------------- scene accessors

    # Read-only handles onto the scene. A planner (the oracle, or a program the agent
    # writes with whitebox access) needs the body ids and joint groups to do IK and
    # motion planning at all; exposing them here keeps that out of the private
    # attributes. None of them let a policy change the state -- the environment is
    # still driven through step().

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
            f"{self._max_grasp_approach:.2f}m along the approach axis, which is "
            f"where the fingers stop. Hovering above a block does not grasp it, "
            f"however close "
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
