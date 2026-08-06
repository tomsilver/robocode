"""ClutteredStorage2D variant with two shelves and per-block target shelves."""

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

StorageBlockType = Type("storage_block", parent=RectangleType)
Kinematic2DRobotEnvTypeFeatures[StorageBlockType] = list(
    Kinematic2DRobotEnvTypeFeatures[RectangleType]
) + ["target_shelf"]

SHELF_OUTER_WIDTH = 1.0
SHELF_INNER_CENTERS_X = (1.4, 3.6)
WORLD_MAX_Y = 3.0
INIT_X_JITTER = 0.01
INIT_THETA_JITTER = 0.02


class TwoShelfObjectCentricClutteredStorage2DEnv(ObjectCentricClutteredStorage2DEnv):
    """Object-centric two-shelf variant.

    ``shelf_profile`` gives each shelf's interior as (width, height) sections
    from the bottom edge up. ``target_pattern`` is XOR'd with the index of the
    shelf containing block0 at reset to produce per-block target shelves.
    """

    def __init__(
        self,
        num_blocks: int = 3,
        target_pattern: tuple[int, ...] = (1, 0, 0),
        shelf_profile: tuple[tuple[float, float], ...] = ((0.32, 0.06), (0.18, 0.10)),
        block_shapes: tuple[tuple[float, float], ...] = (
            (0.28, 0.04),
            (0.14, 0.04),
            (0.28, 0.04),
        ),
        num_init_shelf_blocks: int = 1,
        init_shelf_y_frac: float = 0.5,
        **kwargs,
    ) -> None:
        assert num_blocks == len(block_shapes) == len(target_pattern)
        assert all(
            shelf_profile[i][0] >= shelf_profile[i + 1][0]
            for i in range(len(shelf_profile) - 1)
        )
        assert 0 < num_init_shelf_blocks < len(shelf_profile) + 1
        super().__init__(num_blocks=num_blocks, **kwargs)
        self._target_pattern = tuple(target_pattern)
        self._profile = tuple(shelf_profile)
        self._block_shapes = tuple(block_shapes)
        self._num_init_shelf_blocks = num_init_shelf_blocks
        self._init_shelf_y_frac = init_shelf_y_frac
        self._shelf_height = sum(h for _, h in self._profile)
        self._shelf_y = WORLD_MAX_Y - self._shelf_height

    def _init_shelf_poses(self, init_shelf: int) -> list[SE2Pose]:
        cx = SHELF_INNER_CENTERS_X[init_shelf]
        poses = []
        section_bottom = self._shelf_y
        for i in range(self._num_init_shelf_blocks):
            height = self._profile[i][1]
            if i == 0:
                y = section_bottom + height * self._init_shelf_y_frac
                x_jitter, theta_jitter = INIT_X_JITTER, INIT_THETA_JITTER
            else:
                y = section_bottom + height / 2
                x_jitter, theta_jitter = 0.005, 0.0
            poses.append(
                SE2Pose(
                    cx + self.np_random.uniform(-x_jitter, x_jitter),
                    y,
                    self.np_random.uniform(-theta_jitter, theta_jitter)
                    if theta_jitter
                    else 0.0,
                )
            )
            section_bottom += height
        return poses

    def _sample_initial_state(self) -> ObjectCentricState:
        static_objects = set(self.initial_constant_state)
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        init_shelf = int(self.np_random.integers(2))
        targets = tuple(init_shelf ^ p for p in self._target_pattern)
        shelf_poses = self._init_shelf_poses(init_shelf)
        outside_poses: list[SE2Pose] = []
        num_outside = len(self._block_shapes) - self._num_init_shelf_blocks
        for _ in range(num_outside):
            for _ in range(self.config.max_init_sampling_attempts):
                pose = sample_se2_pose(
                    self.config.target_block_out_of_shelf_pose_bounds, self.np_random
                )
                state = self._create_two_shelf_state(
                    robot_pose, shelf_poses, outside_poses + [pose], targets
                )
                obj_name_to_obj = {o.name: o for o in state}
                new_idx = self._num_init_shelf_blocks + len(outside_poses)
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
            robot_pose, shelf_poses, outside_poses, targets
        )
        robot = state.get_objects(CRVRobotType)[0]
        in_shelf = {
            o
            for o in state
            if o.name in {f"block{i}" for i in range(self._num_init_shelf_blocks)}
        }
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        assert not state_2d_has_collision(full_state, {robot}, static_objects, {})
        assert not state_2d_has_collision(
            full_state, in_shelf, set(full_state) - in_shelf, {}
        )
        return state

    def _create_two_shelf_state(
        self,
        robot_pose: SE2Pose,
        shelf_poses: list[SE2Pose],
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

        inner_width = self._profile[0][0]
        post_num = 0
        for shelf_idx, inner_center_x in enumerate(SHELF_INNER_CENTERS_X):
            shelf = Object(f"shelf{shelf_idx}", ShelfType)
            init_state_dict[shelf] = {
                "x1": inner_center_x - inner_width / 2,
                "y1": self._shelf_y,
                "theta1": 0.0,
                "width1": inner_width,
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
            y0 = self._shelf_y
            for (w_prev, h_prev), (w, h) in zip(self._profile, self._profile[1:]):
                y0 += h_prev
                side_width = (w_prev - w) / 2
                for post_x in (
                    inner_center_x - w_prev / 2,
                    inner_center_x + w_prev / 2 - side_width,
                ):
                    post = Object(f"post{post_num}", RectangleType)
                    post_num += 1
                    init_state_dict[post] = {
                        "x": post_x,
                        "y": y0,
                        "theta": 0.0,
                        "width": side_width,
                        "height": h,
                        "static": True,
                        "color_r": 0.35,
                        "color_g": 0.35,
                        "color_b": 0.35,
                        "z_order": ZOrder.ALL.value,
                    }

        block_poses = []
        for pose, shape in zip(shelf_poses, self._block_shapes, strict=False):
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
        return "The assignment of target shelves to blocks differs between environment variants."  # pylint: disable=line-too-long

    def _create_variant_specific_description(self) -> str:
        return "This variant has 3 blocks."

    def _create_reward_markdown_description(self) -> str:
        return "A penalty of -1.0 is given at every time step until termination, which occurs when every block is inside its target shelf.\n"  # pylint: disable=line-too-long

    def _create_references_markdown_description(self) -> str:
        return "Similar environments have been considered by many others, especially in the task and motion planning literature.\n"  # pylint: disable=line-too-long
