"""ClutteredStorage2D variant with two shelves and per-block target shelves.

Each shelf is a fixed unit at the top of the world: a 0.32-wide opening whose
upper region is narrowed by two posts, leaving a 0.32 x 0.06 front area and a
0.18 x 0.10 pocket above it. Every block carries a ``target_shelf`` feature
(0 or 1) and the task terminates when each block is fully inside its target
shelf's opening.
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

# Shelf unit geometry (world is 5.0 x 3.0; shelves sit flush with the top wall).
SHELF_OUTER_WIDTH = 1.0
SHELF_INNER_WIDTH = 0.32
SHELF_HEIGHT = 0.16
SHELF_Y = 3.0 - SHELF_HEIGHT
SHELF_INNER_CENTERS_X = (1.4, 3.6)
POST_WIDTH = 0.07
POST_HEIGHT = 0.10
FRONT_HEIGHT = SHELF_HEIGHT - POST_HEIGHT

# Per-block shapes (width, height): block1 fits the pocket, blocks 0 and 2 do not.
BLOCK_SHAPES = ((0.28, 0.04), (0.14, 0.04), (0.28, 0.04))

# In-shelf initialization jitter for the pre-stored block.
STORED_BLOCK_X_JITTER = 0.01
STORED_BLOCK_THETA_JITTER = 0.02


class TwoShelfObjectCentricClutteredStorage2DEnv(ObjectCentricClutteredStorage2DEnv):
    """Object-centric two-shelf variant.

    ``target_pattern`` assigns a target shelf to each block relative to the
    shelf that holds the pre-stored block: entry 0 means that same shelf,
    entry 1 means the other one.
    """

    def __init__(
        self,
        num_blocks: int = 3,
        target_pattern: tuple[int, int, int] = (0, 0, 1),
        **kwargs,
    ) -> None:
        assert num_blocks == len(BLOCK_SHAPES) == len(target_pattern)
        super().__init__(num_blocks=num_blocks, **kwargs)
        self._target_pattern = tuple(target_pattern)

    def _sample_initial_state(self) -> ObjectCentricState:
        static_objects = set(self.initial_constant_state)
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        stored_shelf = int(self.np_random.integers(2))
        targets = tuple(stored_shelf ^ p for p in self._target_pattern)
        stored_pose = SE2Pose(
            SHELF_INNER_CENTERS_X[stored_shelf]
            + self.np_random.uniform(-STORED_BLOCK_X_JITTER, STORED_BLOCK_X_JITTER),
            SHELF_Y + FRONT_HEIGHT / 2,
            self.np_random.uniform(
                -STORED_BLOCK_THETA_JITTER, STORED_BLOCK_THETA_JITTER
            ),
        )
        outside_poses: list[SE2Pose] = []
        for _ in range(len(BLOCK_SHAPES) - 1):
            for _ in range(self.config.max_init_sampling_attempts):
                pose = sample_se2_pose(
                    self.config.target_block_out_of_shelf_pose_bounds, self.np_random
                )
                state = self._create_two_shelf_state(
                    robot_pose, stored_pose, outside_poses + [pose], targets
                )
                obj_name_to_obj = {o.name: o for o in state}
                new_block = obj_name_to_obj[f"block{len(outside_poses) + 1}"]
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
            robot_pose, stored_pose, outside_poses, targets
        )
        robot = state.get_objects(CRVRobotType)[0]
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        assert not state_2d_has_collision(full_state, {robot}, static_objects, {})
        return state

    def _create_two_shelf_state(
        self,
        robot_pose: SE2Pose,
        stored_pose: SE2Pose,
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

        for shelf_idx, inner_center_x in enumerate(SHELF_INNER_CENTERS_X):
            shelf = Object(f"shelf{shelf_idx}", ShelfType)
            init_state_dict[shelf] = {
                "x1": inner_center_x - SHELF_INNER_WIDTH / 2,
                "y1": SHELF_Y,
                "theta1": 0.0,
                "width1": SHELF_INNER_WIDTH,
                "height1": SHELF_HEIGHT,
                "static": True,
                "color_r1": PURPLE[0],
                "color_g1": PURPLE[1],
                "color_b1": PURPLE[2],
                "z_order1": ZOrder.NONE.value,
                "x": inner_center_x - SHELF_OUTER_WIDTH / 2,
                "y": SHELF_Y,
                "theta": 0.0,
                "width": SHELF_OUTER_WIDTH,
                "height": SHELF_HEIGHT,
                "color_r": BLACK[0],
                "color_g": BLACK[1],
                "color_b": BLACK[2],
                "z_order": ZOrder.ALL.value,
            }
            for side_idx, post_x in enumerate(
                (
                    inner_center_x - SHELF_INNER_WIDTH / 2,
                    inner_center_x + SHELF_INNER_WIDTH / 2 - POST_WIDTH,
                )
            ):
                post = Object(f"post{2 * shelf_idx + side_idx}", RectangleType)
                init_state_dict[post] = {
                    "x": post_x,
                    "y": SHELF_Y + FRONT_HEIGHT,
                    "theta": 0.0,
                    "width": POST_WIDTH,
                    "height": POST_HEIGHT,
                    "static": True,
                    "color_r": BLACK[0],
                    "color_g": BLACK[1],
                    "color_b": BLACK[2],
                    "z_order": ZOrder.ALL.value,
                }

        stored_rect = Rectangle.from_center(
            stored_pose.x,
            stored_pose.y,
            BLOCK_SHAPES[0][0],
            BLOCK_SHAPES[0][1],
            stored_pose.theta,
        )
        block_poses = [
            SE2Pose(stored_rect.x, stored_rect.y, stored_rect.theta)
        ] + list(outside_poses)
        num_present = len(block_poses)
        for block_idx, (pose, shape, target) in enumerate(
            zip(block_poses, BLOCK_SHAPES[:num_present], targets[:num_present], strict=True)
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
        return "This variant has 3 blocks; one block starts inside a shelf."

    def _create_reward_markdown_description(self) -> str:
        return "A penalty of -1.0 is given at every time step until termination, which occurs when every block is inside its target shelf.\n"  # pylint: disable=line-too-long

    def _create_references_markdown_description(self) -> str:
        return "Similar environments have been considered by many others, especially in the task and motion planning literature.\n"  # pylint: disable=line-too-long
