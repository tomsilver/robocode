"""PR2 ``packed`` with an object count that varies across resets.

:class:`~robocode.environments.pr2_tamp_env.PR2PackedEnv` fixes its block count at
construction and freezes it into a fixed-length ``Box`` observation, so one generated
program cannot span instances of different sizes. This wrapper keeps a backend per
configured count and returns the observation as an ``ObjectCentricState`` instead, so
a single frozen program can run on any of them.

That mirrors what ``VariableObjectCountEnv`` does for the kinder families, and it is
what makes ``packed -n 3/4/5`` -- the sweep in the LLM-PDDLStream paper -- a
generalization axis rather than three separate tasks.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.spaces import ObjectCentricStateSpace

from robocode.environments.pr2_tamp_env import (
    _BLOCK_FEATURES,
    _NUM_ARM_JOINTS,
    _ROBOT_FEATURES,
    _SURFACE_FEATURES,
    PR2PackedEnv,
)
from robocode.environments.variable_count import VariableCountEnv

# Object-centric schema. The feature names match the Box layout PR2PackedEnv
# documents, so a program written against either view reads the same quantities.
RobotType = Type("robot")
SurfaceType = Type("surface")
BlockType = Type("block")

_POSE_FEATURES = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "pose_qw",
]
_EXTENT_FEATURES = ["half_extent_x", "half_extent_y", "half_extent_z"]
_ROBOT_FEATURE_NAMES = (
    ["base_x", "base_y", "base_rot"]
    + [f"joint_{i}" for i in range(1, _NUM_ARM_JOINTS + 1)]
    + ["gripper_opening", "grasp_active"]
    + [f"grasp_tf_{f}" for f in ("x", "y", "z", "qx", "qy", "qz", "qw")]
)
_SURFACE_FEATURE_NAMES = _POSE_FEATURES + _EXTENT_FEATURES
_BLOCK_FEATURE_NAMES = _POSE_FEATURES + ["grasp_active"] + _EXTENT_FEATURES

TYPE_FEATURES: dict[Type, list[str]] = {
    RobotType: _ROBOT_FEATURE_NAMES,
    SurfaceType: _SURFACE_FEATURE_NAMES,
    BlockType: _BLOCK_FEATURE_NAMES,
}

# Object names are stable across counts so a program can address them by name. Blocks
# are the count-defining objects and carry the shared prefix the count is read from.
ROBOT_NAME = "robot"
SURFACE_NAMES = ("table", "plate")
BLOCK_PREFIX = "block"


class PR2PackedVariableCountEnv(VariableCountEnv[ObjectCentricState, NDArray[Any]]):
    """PR2 ``packed`` whose block count varies per reset (object-centric obs)."""

    def __init__(
        self,
        design_counts: list[int],
        eval_counts: list[int],
        grasp_radius: float = 0.08,
        base_steps: int = 120,
        steps_per_object: int = 90,
    ) -> None:
        self._design_counts = [int(c) for c in design_counts]
        self._eval_counts = [int(c) for c in eval_counts]
        if not self._design_counts:
            raise ValueError("design_counts must be non-empty")
        if not self._eval_counts:
            raise ValueError("eval_counts must be non-empty")
        self._grasp_radius = grasp_radius
        self._base_steps = int(base_steps)
        self._steps_per_object = int(steps_per_object)
        self._backends: dict[int, PR2PackedEnv] = {}
        self._current: PR2PackedEnv | None = None
        self._current_count: int | None = None

        # Build the largest design instance up front so a bad config fails at
        # construction rather than midway through an evaluation sweep.
        reference = self._backend_for(max(self._design_counts))
        self.observation_space = ObjectCentricStateSpace(set(TYPE_FEATURES))
        # The bare space carries no feature names; attach them (and the name lookup the
        # blackbox client mirror offers) so the object-centric serializer and a program
        # developed against the client both see the full schema.
        setattr(self.observation_space, "type_features", TYPE_FEATURES)
        setattr(
            self.observation_space,
            "get_type",
            {t.name: t for t in TYPE_FEATURES}.__getitem__,
        )
        self.action_space = reference.action_space
        super().__init__()

    # -- backends & counts ---------------------------------------------------

    def _backend_for(self, count: int) -> PR2PackedEnv:
        """Return (building and caching on first use) the backend for a given count.

        Each backend holds a live PyBullet client with its own PR2 loaded, and they
        are kept until :meth:`close`, so a full sweep costs one client and one PR2 URDF
        per configured count -- five for the default config. They are cached rather
        than rebuilt because loading the PR2 dominates a reset, and evaluation
        interleaves counts episode by episode.
        """
        backend = self._backends.get(count)
        if backend is None:
            if count > 9:
                raise ValueError(
                    f"the plate holds at most 9 blocks, got a count of {count}"
                )
            backend = PR2PackedEnv(num_blocks=count, grasp_radius=self._grasp_radius)
            self._backends[count] = backend
        return backend

    def _count_for_seed(self, seed: int | None) -> int:
        """A design-range count for an unpinned reset.

        Any other count reaches the env only through an explicit
        ``options={"object_count": k}``.
        """
        return int(np.random.default_rng(seed).choice(self._design_counts))

    @property
    def design_counts(self) -> list[int]:
        return list(self._design_counts)

    @property
    def eval_counts(self) -> list[int]:
        return list(self._eval_counts)

    @property
    def current_count(self) -> int:
        assert self._current_count is not None, "Must call reset()"
        return self._current_count

    def max_steps_for_count(self, count: int) -> int:
        return self._base_steps + self._steps_per_object * int(count)

    @property
    def current_backend(self) -> PR2PackedEnv:
        """The fixed-count environment backing the current instance."""
        assert self._current is not None, "Must call reset()"
        return self._current

    # -- observation conversion ----------------------------------------------

    @staticmethod
    def _to_object_centric(obs: NDArray[Any], count: int) -> ObjectCentricState:
        """Split the backend's flat Box observation into per-object feature rows."""
        data: dict[Object, NDArray[Any]] = {
            Object(ROBOT_NAME, RobotType): np.asarray(
                obs[:_ROBOT_FEATURES], dtype=np.float32
            )
        }
        offset = _ROBOT_FEATURES
        for name in SURFACE_NAMES:
            data[Object(name, SurfaceType)] = np.asarray(
                obs[offset : offset + _SURFACE_FEATURES], dtype=np.float32
            )
            offset += _SURFACE_FEATURES
        for index in range(count):
            data[Object(f"{BLOCK_PREFIX}{index}", BlockType)] = np.asarray(
                obs[offset : offset + _BLOCK_FEATURES], dtype=np.float32
            )
            offset += _BLOCK_FEATURES
        return ObjectCentricState(data, TYPE_FEATURES)

    @staticmethod
    def _to_box(state: ObjectCentricState, count: int) -> NDArray[Any]:
        """Flatten an object-centric state back into the backend's Box layout."""
        names = (
            [ROBOT_NAME]
            + list(SURFACE_NAMES)
            + [f"{BLOCK_PREFIX}{i}" for i in range(count)]
        )
        by_name = {obj.name: obj for obj in state}
        rows = []
        for name in names:
            try:
                obj = by_name[name]
            except KeyError as exc:
                raise ValueError(f"state is missing object {name!r}") from exc
            features = state.type_features[obj.type]
            rows.append([state.get(obj, f) for f in features])
        return np.concatenate(rows).astype(np.float32)

    def _count_from_state(self, state: ObjectCentricState) -> int:
        count = sum(
            1 for name in state.get_object_names() if name.startswith(BLOCK_PREFIX)
        )
        if count == 0:
            raise ValueError(
                f"cannot infer object count: no objects with prefix {BLOCK_PREFIX!r}"
            )
        return count

    # -- gym API -------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObjectCentricState, dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and "object_count" in options:
            count = int(options["object_count"])
            backend = self._backend_for(count)
            obs, info = backend.reset(seed=seed)
        elif options is not None and "init_state" in options:
            state = options["init_state"]
            count = self._count_from_state(state)
            backend = self._backend_for(count)
            backend.set_state(self._to_box(state, count))
            obs, info = backend.get_state(), {}
        else:
            count = self._count_for_seed(seed)
            backend = self._backend_for(count)
            obs, info = backend.reset(seed=seed)
        self._current, self._current_count = backend, count
        return self._to_object_centric(obs, count), {**info, "object_count": count}

    def step(
        self, action: NDArray[Any]
    ) -> tuple[ObjectCentricState, SupportsFloat, bool, bool, dict[str, Any]]:
        backend, count = self.current_backend, self.current_count
        obs, reward, terminated, truncated, info = backend.step(action)
        return (
            self._to_object_centric(obs, count),
            reward,
            terminated,
            truncated,
            {**info, "object_count": count},
        )

    def get_state(self) -> ObjectCentricState:
        return self._to_object_centric(
            self.current_backend.get_state(), self.current_count
        )

    def set_state(self, state: ObjectCentricState) -> None:
        count = self._count_from_state(state)
        backend = self._backend_for(count)
        backend.set_state(self._to_box(state, count))
        self._current, self._current_count = backend, count

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.current_backend.render()

    def close(self) -> None:
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
        self._current, self._current_count = None, None

    @property
    def env_description(self) -> str:
        return self._describe(include_access=True)

    @property
    def env_description_blackbox(self) -> str:
        return self._describe(include_access=False)

    def _describe(self, include_access: bool) -> str:
        """Render a card that reads the same whatever counts are configured.

        Neither the design counts nor the evaluation sweep appear anywhere: which
        counts an approach is scored on, and how far past the design range they go,
        is the experimenter's to know. The backend card is therefore requested in
        count-invariant form, and nothing here interpolates a count.
        """
        # pylint: disable=protected-access
        reference = self._backend_for(max(self._design_counts))
        base = reference.describe(include_access=include_access, count_invariant=True)
        schema_lines = [
            f"- `{typ.name}`: {', '.join(features)}"
            for typ, features in TYPE_FEATURES.items()
        ]
        return (
            f"{base}\n"
            f"## Variable Object Count\n\n"
            f"The number of blocks changes between episodes, so observations are "
            f"object-centric rather than a fixed-length vector: each is an "
            f"`ObjectCentricState` holding one `robot`, the `table` and `plate` "
            f"surfaces, and `block0`..`blockN-1`.\n\n"
            f"Feature names per type:\n\n"
            + "\n".join(schema_lines)
            + "\n\nIterate the state's objects rather than indexing fixed offsets, so "
            "one program handles any number of blocks.\n"
        )
