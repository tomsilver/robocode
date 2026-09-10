"""PDDLStream ``rovers`` with an objective count that varies across resets.

:class:`~robocode.environments.rovers_env.RoversEnv` fixes its objective count at
construction and freezes it into a fixed-length ``Box`` observation, so one generated
program cannot span instances of different sizes. This wrapper keeps a backend per
configured count and returns the observation as an ``ObjectCentricState`` instead, the
same arrangement the PR2 families use.

Objectives are the count here because each one is a separate errand: it has to be
photographed from somewhere with a clear line to it, and the picture has to be radioed
from somewhere with a clear line to the lander. The sample half of the goal -- one
stone and one soil -- does not grow with the count, so a bigger instance is more of the
same imaging work rather than a different task.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.spaces import ObjectCentricStateSpace

from robocode.environments.rovers_env import RoversEnv
from robocode.environments.variable_count import VariableCountEnv

RoverType = Type("rover")
LanderType = Type("lander")
ObjectiveType = Type("objective")
SampleType = Type("sample")
ObstacleType = Type("obstacle")

_ROVER_FEATURE_NAMES = ["x", "y", "theta", "store_full", "calibrated", "at_home"]
_LANDER_FEATURE_NAMES = ["x", "y", "z"]
_OBJECTIVE_FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "have_image_rover0",
    "have_image_rover1",
    "received_image",
]
_SAMPLE_FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "is_soil",
    "analyzed_rover0",
    "analyzed_rover1",
    "received_analysis",
]
_OBSTACLE_FEATURE_NAMES = ["x", "y", "z", "half_x", "half_y", "half_z"]

TYPE_FEATURES: dict[Type, list[str]] = {
    RoverType: _ROVER_FEATURE_NAMES,
    LanderType: _LANDER_FEATURE_NAMES,
    ObjectiveType: _OBJECTIVE_FEATURE_NAMES,
    SampleType: _SAMPLE_FEATURE_NAMES,
    ObstacleType: _OBSTACLE_FEATURE_NAMES,
}

ROVER_PREFIX = "rover"
LANDER_NAME = "lander"
OBJECTIVE_PREFIX = "objective"
SAMPLE_PREFIX = "sample"
OBSTACLE_PREFIX = "obstacle"


class RoversVariableCountEnv(VariableCountEnv[ObjectCentricState, NDArray[Any]]):
    """``rovers`` whose objective count varies per reset (object-centric obs)."""

    def __init__(
        self,
        design_counts: list[int],
        eval_counts: list[int],
        num_rocks: int = 3,
        num_soils: int = 3,
        num_obstacles: int = 8,
        base_steps: int = 250,
        steps_per_object: int = 60,
    ) -> None:
        self._design_counts = [int(c) for c in design_counts]
        self._eval_counts = [int(c) for c in eval_counts]
        if not self._design_counts:
            raise ValueError("design_counts must be non-empty")
        if not self._eval_counts:
            raise ValueError("eval_counts must be non-empty")
        if min(self._design_counts + self._eval_counts) < 1:
            raise ValueError("every instance needs at least one objective")
        self._num_rocks = int(num_rocks)
        self._num_soils = int(num_soils)
        self._num_obstacles = int(num_obstacles)
        self._base_steps = int(base_steps)
        self._steps_per_object = int(steps_per_object)
        self._backends: dict[int, RoversEnv] = {}
        self._current: RoversEnv | None = None
        self._current_count: int | None = None

        reference = self._backend_for(max(self._design_counts))
        self.observation_space = ObjectCentricStateSpace(set(TYPE_FEATURES))
        setattr(self.observation_space, "type_features", TYPE_FEATURES)
        setattr(
            self.observation_space,
            "get_type",
            {t.name: t for t in TYPE_FEATURES}.__getitem__,
        )
        self.action_space = reference.action_space
        super().__init__()

    # -- backends & counts ---------------------------------------------------

    def _backend_for(self, count: int) -> RoversEnv:
        """Return (building and caching on first use) the backend for a count.

        The mounds objectives stand on are the benchmark's four, so a count above that
        would put two objectives on one mound and change what occlusion means; the limit
        is checked here rather than left to fail during a sweep.
        """
        backend = self._backends.get(count)
        if backend is None:
            if count > 4:
                raise ValueError(
                    f"the scene has 4 mounds to stand objectives on, got {count}"
                )
            backend = RoversEnv(
                num_objectives=count,
                num_rocks=self._num_rocks,
                num_soils=self._num_soils,
                num_obstacles=self._num_obstacles,
            )
            self._backends[count] = backend
        return backend

    def _count_for_seed(self, seed: int | None) -> int:
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
        """Step budget for an instance.

        Each objective costs a drive out to it, a calibrate and an image, and a drive
        back to somewhere the lander is visible, so the per-object term is large next to
        the fixed cost of the two samples and the trip home. The budget leaves roughly
        two to three times what the reference oracle needs -- enough that a less direct
        route still finishes, not so much that wandering does.
        """
        return self._base_steps + self._steps_per_object * int(count)

    @property
    def current_backend(self) -> RoversEnv:
        """The fixed-count environment backing the current instance."""
        assert self._current is not None, "Must call reset()"
        return self._current

    # -- observation conversion ----------------------------------------------

    def _names(self, backend: RoversEnv, count: int) -> list[tuple[str, Type, int]]:
        """Object names, types and feature widths in the backend's Box order."""
        rows: list[tuple[str, Type, int]] = [
            (f"{ROVER_PREFIX}{i}", RoverType, len(_ROVER_FEATURE_NAMES))
            for i in range(len(backend.rovers))
        ]
        rows.append((LANDER_NAME, LanderType, len(_LANDER_FEATURE_NAMES)))
        rows += [
            (f"{OBJECTIVE_PREFIX}{i}", ObjectiveType, len(_OBJECTIVE_FEATURE_NAMES))
            for i in range(count)
        ]
        rows += [
            (f"{SAMPLE_PREFIX}{i}", SampleType, len(_SAMPLE_FEATURE_NAMES))
            for i in range(len(backend.samples))
        ]
        rows += [
            (f"{OBSTACLE_PREFIX}{i}", ObstacleType, len(_OBSTACLE_FEATURE_NAMES))
            for i in range(self._num_obstacles + 4)
        ]
        return rows

    def _to_object_centric(
        self, obs: NDArray[Any], count: int, backend: RoversEnv
    ) -> ObjectCentricState:
        data: dict[Object, NDArray[Any]] = {}
        offset = 0
        for name, typ, width in self._names(backend, count):
            data[Object(name, typ)] = np.asarray(
                obs[offset : offset + width], dtype=np.float32
            )
            offset += width
        return ObjectCentricState(data, TYPE_FEATURES)

    def _to_box(
        self, state: ObjectCentricState, count: int, backend: RoversEnv
    ) -> NDArray[Any]:
        by_name = {obj.name: obj for obj in state}
        rows = []
        for name, _typ, _width in self._names(backend, count):
            try:
                obj = by_name[name]
            except KeyError as exc:
                raise ValueError(f"state is missing object {name!r}") from exc
            features = state.type_features[obj.type]
            rows.append([state.get(obj, f) for f in features])
        return np.concatenate(rows).astype(np.float32)

    def _count_from_state(self, state: ObjectCentricState) -> int:
        count = sum(
            1 for name in state.get_object_names() if name.startswith(OBJECTIVE_PREFIX)
        )
        if count == 0:
            raise ValueError(
                f"cannot infer count: no objects with prefix {OBJECTIVE_PREFIX!r}"
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
            backend.set_state(self._to_box(state, count, backend))
            obs, info = backend.get_state(), {}
        else:
            count = self._count_for_seed(seed)
            backend = self._backend_for(count)
            obs, info = backend.reset(seed=seed)
        self._current, self._current_count = backend, count
        return (
            self._to_object_centric(obs, count, backend),
            {**info, "object_count": count},
        )

    def step(
        self, action: NDArray[Any]
    ) -> tuple[ObjectCentricState, SupportsFloat, bool, bool, dict[str, Any]]:
        backend, count = self.current_backend, self.current_count
        obs, reward, terminated, truncated, info = backend.step(action)
        return (
            self._to_object_centric(obs, count, backend),
            reward,
            terminated,
            truncated,
            {**info, "object_count": count},
        )

    def get_state(self) -> ObjectCentricState:
        backend = self.current_backend
        return self._to_object_centric(backend.get_state(), self.current_count, backend)

    def set_state(self, state: ObjectCentricState) -> None:
        count = self._count_from_state(state)
        backend = self._backend_for(count)
        backend.set_state(self._to_box(state, count, backend))
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
        """Render a card that reads the same whatever counts are configured."""
        reference = self._backend_for(max(self._design_counts))
        base = reference.describe(include_access=include_access, count_invariant=True)
        schema_lines = [
            f"- `{typ.name}`: {', '.join(features)}"
            for typ, features in TYPE_FEATURES.items()
        ]
        return (
            f"{base}\n"
            f"## Variable Object Count\n\n"
            f"The number of objectives changes between episodes, so observations are "
            f"object-centric rather than a fixed-length vector: each is an "
            f"`ObjectCentricState` holding `rover0` and `rover1`, the `lander`, "
            f"`objective0`..`objectiveN-1`, `sample0`.. (the stone samples first, "
            f"then the soil ones, told apart by `is_soil`), and `obstacle0`.. for the "
            f"mounds and pillars.\n\n"
            f"Only the imaging half of the goal grows with the count: one stone and "
            f"one soil analysis are required whatever it is.\n\n"
            f"Feature names per type:\n\n"
            + "\n".join(schema_lines)
            + "\n\nIterate the state's objects rather than indexing fixed offsets, so "
            "one program handles any number of objectives.\n"
        )
