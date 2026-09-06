"""Shared helper for building bilevel planning models from a robocode env.

Both the SeSamE baseline (`BilevelPlanningApproach`) and the `bilevel_models`
primitive need the same `SesameModels` bundle for an environment, built from the
`bilevel_env_name` / `bilevel_env_model_kwargs` mapping carried on the env (see
`KinderGeom2DEnv`). This module owns that one construction so the two callers do
not duplicate the mapping read, and it owns the table of kinder families that
have models in `kinder_bilevel_planning`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BilevelFamily:
    """A kinder env family that has SeSamE models in `kinder_bilevel_planning`."""

    # Family token of the kinder gym id: "Obstruction2D" in "kinder/Obstruction2D-o2-v0".
    id_family: str
    # "module:Class" of the ConstantObjectKinDEREnv, as VariableObjectCountEnv takes it.
    env_path: str
    # File name of the model module under kinder_bilevel_planning/env_models/. The
    # loader resolves it on disk, so it is case-sensitive and not derivable from the id.
    bilevel_env_name: str
    # Object-count kwarg of that module's create_bilevel_planning_models. It can
    # differ from the env's own count kwarg (Transport3D takes num_cubes, its model
    # num_objects).
    count_kwarg: str


_FAMILIES: tuple[BilevelFamily, ...] = (
    BilevelFamily(
        "Obstruction2D",
        "kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",
        "obstruction2d",
        "num_obstructions",
    ),
    BilevelFamily(
        "StickButton2D",
        "kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",
        "stickbutton2d",
        "num_buttons",
    ),
    BilevelFamily(
        "ClutteredStorage2D",
        "kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv",
        "clutteredstorage2d",
        "num_blocks",
    ),
    BilevelFamily(
        "ClutteredRetrieval2D",
        "kinder.envs.kinematic2d.clutteredretrieval2d:ClutteredRetrieval2DEnv",
        "clutteredretrieval2d",
        "num_obstructions",
    ),
    BilevelFamily(
        "Motion2D",
        "kinder.envs.kinematic2d.motion2d:Motion2DEnv",
        "motion2d",
        "num_passages",
    ),
    BilevelFamily(
        "DynObstruction2D",
        "kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
        "dynobstruction2d",
        "num_obstructions",
    ),
    BilevelFamily(
        "DynPushPullHook2D",
        "kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
        "dynpushpullhook2d",
        "num_obstructions",
    ),
    BilevelFamily(
        "Transport3D",
        "kinder.envs.kinematic3d.transport3d:Transport3DEnv",
        "transport3d",
        "num_objects",
    ),
    # The kinematic Shelf3D. The MuJoCo Shelf3DEnv under dynamic3d.task_families is a
    # different environment whose models (tidybot3d_shelf3D) are not wired here.
    BilevelFamily(
        "KinematicShelf3D",
        "kinder.envs.kinematic3d.shelf3d:Shelf3DEnv",
        "shelf3d",
        "num_objects",
    ),
    BilevelFamily(
        "Tossing3D",
        "kinder.envs.dynamic3d.task_families:Tossing3DEnv",
        "tidybot3d_tossing3D",
        "num_objects",
    ),
)
_BY_ID_FAMILY = {family.id_family: family for family in _FAMILIES}
_BY_ENV_PATH = {family.env_path: family for family in _FAMILIES}
_BY_NAME = {family.bilevel_env_name: family for family in _FAMILIES}

# e.g. "kinder/Obstruction2D-o2-v0" -> family="Obstruction2D", count=2.
_ENV_ID_RE = re.compile(r"kinder/([A-Za-z0-9]+)-[a-z](\d+)-v\d+")


def infer_bilevel_mapping(env_id: str) -> tuple[str | None, dict[str, int]]:
    """Infer ``(bilevel_env_name, model_kwargs)`` from a kinder env id.

    e.g. ``"kinder/Obstruction2D-o2-v0" -> ("obstruction2d", {"num_obstructions": 2})``.
    Returns ``(None, {})`` for env ids that have no bilevel planning model (e.g.
    Obstruction3D, Packing3D, pushpullhook, mazes). Used as a fallback so a plain
    ``KinderGeom2DEnv``/``KinderGeom3DEnv`` (e.g. one the agent builds to test) can use
    the ``bilevel_models`` primitive without the explicit mapping; the 2D env configs
    still set it explicitly, and a test checks the two agree.
    """
    match = _ENV_ID_RE.fullmatch(env_id)
    if match is None:
        return None, {}
    family = _BY_ID_FAMILY.get(match.group(1))
    if family is None:
        return None, {}
    return family.bilevel_env_name, {family.count_kwarg: int(match.group(2))}


def infer_bilevel_env_name_from_path(env_path: str) -> str | None:
    """The ``bilevel_env_name`` for a ``"module:Class"`` env path, or None if
    unmapped."""
    family = _BY_ENV_PATH.get(env_path)
    return None if family is None else family.bilevel_env_name


def bilevel_count_kwarg(bilevel_env_name: str) -> str:
    """The object-count kwarg of a family's ``create_bilevel_planning_models``."""
    family = _BY_NAME.get(bilevel_env_name)
    assert family is not None, (
        f"{bilevel_env_name!r} is not a known bilevel env family; add it to "
        "robocode.utils.bilevel._FAMILIES"
    )
    return family.count_kwarg


class VariableCountBilevelModels:
    """The ``bilevel_models`` primitive for a variable-object-count env.

    The planning models bake in the object count, so there is no single
    ``SesameModels`` bundle. Call :meth:`models_for_state` (or :meth:`models_for_count`)
    to get the bundle for the current instance's count; bundles are built once per count
    and cached.
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self._by_count: dict[int, Any] = {}

    def models_for_count(self, count: int) -> Any:
        """Return the ``SesameModels`` bundle for a given object count (cached)."""
        if count not in self._by_count:
            self._by_count[count] = self._env.models_for_count(count)
        return self._by_count[count]

    def models_for_state(self, state: Any) -> Any:
        """Return the ``SesameModels`` bundle for the count implied by *state*."""
        return self.models_for_count(self._env.infer_count(state))


def build_sesame_models(
    env: Any,
    *,
    observation_space: Any | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build the `SesameModels` (predicates, operators, skills, transition sim).

    Reads the bilevel env-family name off *env*. The observation space and object-count
    kwargs default to the env's own (the fixed-count case), but a variable-count env
    passes them explicitly so the models are built for the *current* instance's count:
    its per-count `ObjectCentricBoxSpace` and `{count_kwarg: k}`. Fails loudly if the
    env carries no family mapping rather than planning silently.

    A variable-count env (one exposing `models_for_count`) has no single count, so a
    call without explicit `model_kwargs` returns a :class:`VariableCountBilevelModels`
    accessor instead of one bundle; per-count builds pass explicit kwargs and so take
    the normal path below.

    `kinder_bilevel_planning` is imported lazily (it is an optional `bilevel` extra),
    so `import robocode.primitives` works even where the extra is not installed -- e.g.
    a "models OFF" sandbox. Only actually using the bilevel models requires it.
    """
    # We never run under python -O, so this assert fires as a loud config check.
    assert getattr(env, "bilevel_env_name", None) is not None, (
        "bilevel_env_name is not set on the environment; this env family has no "
        "bilevel planning models (or the env class does not carry the mapping)."
    )
    if model_kwargs is None and hasattr(env, "models_for_count"):
        return VariableCountBilevelModels(env)

    # pylint: disable=import-outside-toplevel
    from kinder_bilevel_planning.env_models import create_bilevel_planning_models

    obs_space = (
        observation_space if observation_space is not None else env.observation_space
    )
    kwargs = model_kwargs if model_kwargs is not None else env.bilevel_env_model_kwargs
    return create_bilevel_planning_models(
        env.bilevel_env_name,
        obs_space,
        env.action_space,
        **kwargs,
    )
