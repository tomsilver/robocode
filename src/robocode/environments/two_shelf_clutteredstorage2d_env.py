"""ClutteredStorage2D variant with two shelves and per-block target shelves.

Each shelf is a fixed unit at the top of the world: a 0.32-wide opening with a
front area and, above it, one or two pocket levels formed by static posts.
Every block carries a ``target_shelf`` feature (0 or 1) and the task
terminates when each block is fully inside its target shelf's opening.

Geometry knobs (per instance):
- ``front_height``: height of the front area below the pockets.
- ``pocket_levels``: 1 (single 0.18 x 0.10 pocket) or 2 (0.18 and 0.10 wide
  levels stacked, with a matching third block size).
- ``stored_at_top``: pre-store the long block flush with the top of the front
  area rather than centered in it.
"""

import numpy as np
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict
from tomsgeoms2d.structs import Rectangle

from kinder.core import ConstantObjectKinDEREnv
from kinder.envs.kinematic2d.base_env import ObjectCentricKinematic2DRobotEnv
from kinder.envs.kinematic2d.clutteredstorage2d import (
    ObjectCentricClutteredStorage2DEnv,
    ShelfType,
)
from kinder.envs.kinematic2d.object_types import (
    CRVRobotType,
    Kinematic2DRobotEnvTypeFeatures,
    RectangleType,
)
from kinder.envs.kinematic2d.structs import SE2Pose, ZOrder
from kinder.envs.kinematic2d.utils import is_inside_shelf
from kinder.envs.utils import BLACK, PURPLE, sample_se2_pose, state_2d_has_collision

# Blocks with a target-shelf assignment feature.
StorageBlockType = Type("storage_block", parent=RectangleType)
Kinematic2DRobotEnvTypeFeatures[StorageBlockType] = list(
    Kinematic2DRobotEnvTypeFeatures[RectangleType]
) + ["target_shelf"]

# Shared geometry (world is 5.0 x 3.0; shelves sit flush with the top wall).
SHELF_OUTER_WIDTH = 1.0
SHELF_INNER_WIDTH = 0.32
SHELF_INNER_CENTERS_X = (1.4, 3.6)
WORLD_MAX_Y = 3.0

# Defaults for the single-pocket configuration (kept for existing scripts).
FRONT_HEIGHT = 0.06
POST_HEIGHT = 0.10
SHELF_HEIGHT = FRONT_HEIGHT + POST_HEIGHT
SHELF_Y = WORLD_MAX_Y - SHELF_HEIGHT

# Per-block shapes (width, height) by pocket_levels.
BLOCK_SHAPES = ((0.28, 0.04), (0.14, 0.04), (0.28, 0.04))
BLOCK_SHAPES_TWO_LEVELS = ((0.28, 0.04), (0.14, 0.04), (0.08, 0.04))

# Pocket levels as (channel_width, height), bottom-up, by pocket_levels.
POCKET_LEVELS = {1: ((0.18, 0.10),), 2: ((0.18, 0.06), (0.10, 0.06))}

# In-shelf initialization jitter for pre-stored blocks.
STORED_BLOCK_X_JITTER = 0.01
STORED_BLOCK_THETA_JITTER = 0.02


class TwoShelfObjectCentricClutteredStorage2DEnv(ObjectCentricClutteredStorage2DEnv):
    """Object-centric two-shelf variant.

    ``target_pattern`` assigns a target shelf to each block relative to the
    shelf holding the pre-stored long block: entry 0 means that same shelf,
    entry 1 means the other one.
    """

    def __init__(
        self,
        num_blocks: int = 3,
        target_pattern: tuple[int, int, int] = (0, 0, 1),
        front_height: float = FRONT_HEIGHT,
        pocket_levels: int = 1,
        stored_at_top: bool = False,
        **kwargs,
    ) -> None:
        assert pocket_levels in POCKET_LEVELS
        self._levels = POCKET_LEVELS[pocket_levels]
        self._block_shapes = (
            BLOCK_SHAPES if pocket_levels == 1 else BLOCK_SHAPES_TWO_LEVELS
        )
        assert num_blocks == len(self._block_shapes) == len(target_pattern)
        super().__init__(num_blocks=num_blocks, **kwargs)
        self._target_pattern = tuple(target_pattern)
        self._front_height = front_height
        self._stored_at_top = stored_at_top
        # Blocks 0..(pocket_levels-1) start stored: block0 in the front area
        # and, with two levels, block1 in the lower pocket.
        self._num_stored = pocket_levels
        self._shelf_height = front_height + sum(h for _, h in self._levels)
        self._shelf_y = WORLD_MAX_Y - self._shelf_height

    def _stored_poses(self, stored_shelf: int) -> list[SE2Pose]:
        cx = SHELF_INNER_CENTERS_X[stored_shelf]
        poses = []
        h0 = self._block_shapes[0][1]
        if self._stored_at_top:
            y0 = self._shelf_y + self._front_height - h0 / 2 - 0.005
            th0 = 0.0
        else:
            y0 = self._shelf_y + self._front_height / 2
            th0 = self.np_random.uniform(
                -STORED_BLOCK_THETA_JITTER, STORED_BLOCK_THETA_JITTER
            )
        poses.append(
            SE2Pose(
                cx + self.np_random.uniform(-STORED_BLOCK_X_JITTER, STORED_BLOCK_X_JITTER),
                y0,
                th0,
            )
        )
        if self._num_stored > 1:
            level_y = self._shelf_y + self._front_height + self._levels[0][1] / 2
            poses.append(
                SE2Pose(cx + self.np_random.uniform(-0.005, 0.005), level_y, 0.0)
            )
        return poses

    def _sample_initial_state(self) -> ObjectCentricState:
        static_objects = set(self.initial_constant_state)
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        stored_shelf = int(self.np_random.integers(2))
        targets = tuple(stored_shelf ^ p for p in self._target_pattern)
        stored_poses = self._stored_poses(stored_shelf)
        outside_poses: list[SE2Pose] = []
        num_outside = len(self._block_shapes) - self._num_stored
        for _ in range(num_outside):
            for _ in range(self.config.max_init_sampling_attempts):
                pose = sample_se2_pose(
                    self.config.target_block_out_of_shelf_pose_bounds, self.np_random
                )
                state = self._create_two_shelf_state(
                    robot_pose, stored_poses, outside_poses + [pose], targets
                )
                obj_name_to_obj = {o.name: o for o in state}
                new_idx = self._num_stored + len(outside_poses)
                new_block = obj_name_to_obj[f"block{new_idx}"]
                full_state = state.copy()
                full_state.data.update(self.initial_constant_state.data)
                if not state_2d_has_collision(
                    full_state, {new_block}, set(full_state), {}
                ):
                    break
            else:
                raise RuntimeError("Failed to sample block pose.")
            outside_poses.append(pose)
        state = self._create_two_shelf_state(
            robot_pose, stored_poses, outside_poses, targets
        )
        robot = state.get_objects(CRVRobotType)[0]
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        assert not state_2d_has_collision(full_state, {robot}, static_objects, {})
        return state

    def _create_two_shelf_state(
        self,
        robot_pose: SE2Pose,
        stored_poses: list[SE2Pose],
        outside_poses: list[SE2Pose],
        targets: tuple[int, ...],
    ) -> ObjectCentricState:
        init_state_dict: dict[Object, dict[str, float]] = {}

        robot = Object("robot", CRVRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": robot_pose.theta,
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,
            "arm_length": self.config.robot_arm_length,
            "vacuum": 0.0,
            "gripper_height": self.config.robot_gripper_height,
            "gripper_width": self.config.robot_gripper_width,
        }

        post_num = 0
        for shelf_idx, inner_center_x in enumerate(SHELF_INNER_CENTERS_X):
            shelf = Object(f"shelf{shelf_idx}", ShelfType)
            init_state_dict[shelf] = {
                "x1": inner_center_x - SHELF_INNER_WIDTH / 2,
                "y1": self._shelf_y,
                "theta1": 0.0,
                "width1": SHELF_INNER_WIDTH,
                "height1": self._shelf_height,
                "static": True,
                "color_r1": PURPLE[0],
                "color_g1": PURPLE[1],
                "color_b1": PURPLE[2],
                "z_order1": ZOrder.NONE.value,
                "x": inner_center_x - SHELF_OUTER_WIDTH / 2,
                "y": self._shelf_y,
                "theta": 0.0,
                "width": SHELF_OUTER_WIDTH,
                "height": self._shelf_height,
                "color_r": BLACK[0],
                "color_g": BLACK[1],
                "color_b": BLACK[2],
                "z_order": ZOrder.ALL.value,
            }
            level_bottom = self._shelf_y + self._front_height
            prev_width = SHELF_INNER_WIDTH
            for channel_width, level_height in self._levels:
                post_width = (prev_width - channel_width) / 2
                for post_x in (
                    inner_center_x - prev_width / 2,
                    inner_center_x + prev_width / 2 - post_width,
                ):
                    post = Object(f"post{post_num}", RectangleType)
                    post_num += 1
                    init_state_dict[post] = {
                        "x": post_x,
                        "y": level_bottom,
                        "theta": 0.0,
                        "width": post_width,
                        "height": level_height,
                        "static": True,
                        "color_r": 0.35,
                        "color_g": 0.35,
                        "color_b": 0.35,
                        "z_order": ZOrder.ALL.value,
                    }
                level_bottom += level_height
                prev_width = channel_width

        block_poses = []
        for pose, shape in zip(stored_poses, self._block_shapes, strict=False):
            rect = Rectangle.from_center(pose.x, pose.y, shape[0], shape[1], pose.theta)
            block_poses.append(SE2Pose(rect.x, rect.y, rect.theta))
        block_poses += list(outside_poses)
        num_present = len(block_poses)
        for block_idx, (pose, shape, target) in enumerate(
            zip(
                block_poses,
                self._block_shapes[:num_present],
                targets[:num_present],
                strict=True,
            )
        ):
            block = Object(f"block{block_idx}", StorageBlockType)
            init_state_dict[block] = {
                "x": pose.x,
                "y": pose.y,
                "theta": pose.theta,
                "width": shape[0],
                "height": shape[1],
                "static": False,
                "color_r": self.config.target_block_rgb[0],
                "color_g": self.config.target_block_rgb[1],
                "color_b": self.config.target_block_rgb[2],
                "z_order": ZOrder.SURFACE.value,
                "target_shelf": float(target),
            }

        return create_state_from_dict(init_state_dict, Kinematic2DRobotEnvTypeFeatures)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        assert self._current_state is not None
        shelves = sorted(
            self._current_state.get_objects(ShelfType), key=lambda o: o.name
        )
        blocks = sorted(
            self._current_state.get_objects(StorageBlockType), key=lambda o: o.name
        )
        terminated = all(
            is_inside_shelf(
                self._current_state,
                block,
                shelves[int(self._current_state.get(block, "target_shelf"))],
                self._static_object_body_cache,
            )
            for block in blocks
        )
        return -1.0, terminated


class TwoShelfClutteredStorage2DEnv(ConstantObjectKinDEREnv):
    """Constant-object wrapper for the two-shelf variant."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic2DRobotEnv:
        return TwoShelfObjectCentricClutteredStorage2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "shelf0", "shelf1"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith(("post", "block")):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 2D environment with two shelves where the goal is to put every block inside its target shelf. Each block's target_shelf feature gives the index of the shelf (shelf0 or shelf1) that it must end up inside.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.
"""

    def _create_variant_markdown_description(self) -> str:
        return "Variants differ in how target shelves are assigned to blocks."

    def _create_variant_specific_description(self) -> str:
        return "This variant has 3 blocks; at least one block starts inside a shelf."

    def _create_reward_markdown_description(self) -> str:
        return "A penalty of -1.0 is given at every time step until termination, which occurs when every block is inside its target shelf.\n"  # pylint: disable=line-too-long

    def _create_references_markdown_description(self) -> str:
        return "Similar environments have been considered by many others, especially in the task and motion planning literature.\n"  # pylint: disable=line-too-long
