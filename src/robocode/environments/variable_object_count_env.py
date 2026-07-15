"""Environment whose object count varies across resets (generalized planning).

A kinder count-parametrized family (e.g. Obstruction2D) fixes its object count at
construction and freezes it into a fixed-length ``Box`` observation. That makes one
generated program unable to span instances of different sizes. This wrapper drives
the *object-centric* layer underneath directly: each reset produces an instance with
a chosen number of count-defining objects, and observations are the variable-length
``ObjectCentricState`` itself, so a single frozen program can run on any count.

Two views of the same instance are exposed:

* the **object-centric view** for the generated program -- ``reset``/``step`` return
  an ``ObjectCentricState`` and ``observation_space`` is a count-invariant
  ``ObjectCentricStateSpace``;
* a **per-count ``Box`` view** for the bilevel planner, whose SeSamE models consume a
  vector and bake in the count -- built on demand via :meth:`models_for_count` and
  :meth:`current_box_obs`.

The count that parameterizes an instance is the family's constructor kwarg value
(``num_obstructions``, ``num_passages``, ...). It is inferred from a bare state by
counting objects whose name starts with ``count_object_prefix`` and looking that up
in a map built from every configured count -- counting by *type* would be wrong
because non-count objects can share a type (e.g. ``target_block`` is a rectangle just
like ``obstruction0``).
"""

from __future__ import annotations

import importlib
from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import RenderFrame
from kinder.core import ConstantObjectKinDEREnv
from numpy.typing import NDArray
from relational_structs import ObjectCentricState

from robocode.environments.base_env import BaseEnv
from robocode.environments.mujoco_gl import configure_gl_backend
from robocode.utils.bilevel import build_sesame_models


def _load_constant_object_env_class(path: str) -> type[ConstantObjectKinDEREnv]:
    """Import a ``ConstantObjectKinDEREnv`` subclass from a ``"module:Class"`` path."""
    module_path, _, class_name = path.partition(":")
    if not class_name:
        raise ValueError(
            f"constant_object_env_path must be 'module:Class', got {path!r}"
        )
    cls = getattr(importlib.import_module(module_path), class_name)
    if not issubclass(cls, ConstantObjectKinDEREnv):
        raise TypeError(f"{path} is not a ConstantObjectKinDEREnv subclass")
    return cls


class VariableObjectCountEnv(BaseEnv[ObjectCentricState, NDArray[Any]]):
    """A robocode env that varies the object count per reset (object-centric obs)."""

    # This wrapper drives the object-centric env that kinder exposes under each
    # ConstantObjectKinDEREnv's `_object_centric_env`; reaching it is the intended
    # (if underscore-named) integration point, so allow the protected access.
    # pylint: disable=protected-access

    def __init__(
        self,
        *,
        constant_object_env_path: str,
        count_kwarg: str,
        count_object_prefix: str,
        design_counts: list[int],
        eval_counts: list[int],
        bilevel_env_name: str | None = None,
        reference_count: int | None = None,
        base_steps: int = 300,
        steps_per_object: int = 150,
        render_dpi: int | None = None,
    ) -> None:
        # Lock the GL backend before building any backend env so rendering works;
        # driving the kinder classes directly needs no gym registry.
        configure_gl_backend()
        self._env_path = constant_object_env_path
        self._env_cls = _load_constant_object_env_class(constant_object_env_path)
        self._count_kwarg = count_kwarg
        self._count_object_prefix = count_object_prefix
        # Evaluation horizon grows with the object count, so a larger instance is not
        # scored as failed merely for running out of steps before it could finish.
        self._base_steps = base_steps
        self._steps_per_object = steps_per_object
        # Optional render-resolution override applied to every backend. Unset, each
        # family keeps its own default DPI; lowering it (e.g. for gif capture) shrinks
        # each frame so long, high-object-count rollouts fit in memory.
        self._render_dpi = render_dpi
        self._design_counts = [int(c) for c in design_counts]
        self._eval_counts = [int(c) for c in eval_counts]
        if not self._design_counts:
            raise ValueError("design_counts must be non-empty")
        if not self._eval_counts:
            raise ValueError("eval_counts must be non-empty")
        # Family name for the bilevel planning models (e.g. "obstruction2d"); None for
        # families with no SeSamE models, which then support the program side only.
        self.bilevel_env_name = bilevel_env_name

        self._backends: dict[int, ConstantObjectKinDEREnv] = {}
        # prefixed-object-count -> constructor-kwarg value, so a bare state (set_state,
        # init_state) can be routed to the right backend. Built for every configured
        # count up front so inference never hits an un-built backend.
        self._prefixed_count_to_kwarg: dict[int, int] = {}
        for count in sorted(set(self._design_counts) | set(self._eval_counts)):
            self._backend_for(count)

        ref_count = (
            reference_count if reference_count is not None else min(self._design_counts)
        )
        ref_env = self._backend_for(ref_count)._object_centric_env
        # The object-centric spaces do not depend on the object count, so they can be
        # fixed once from any backend and are honestly count-invariant.
        self.observation_space = ref_env.observation_space
        self.action_space = ref_env.action_space
        self.type_features = ref_env.type_features
        # The object-centric space carries no feature names on its own; attach them so
        # the blackbox serializer can emit the full type->feature schema.
        setattr(self.observation_space, "type_features", self.type_features)
        # The bare ObjectCentricStateSpace exposes only `.types`; the blackbox client
        # mirror and the object-centric prompts also offer `get_type(name)`. Attach it
        # (KeyError on an unknown name, matching the client) so a program developed
        # against the blackbox client runs unchanged at eval time.
        self._type_by_name = {t.name: t for t in self.type_features}
        setattr(self.observation_space, "get_type", self._type_by_name.__getitem__)
        self._reference_state, _ = ref_env.reset(seed=0)

        self._current_backend: ConstantObjectKinDEREnv | None = None
        self._current_count: int | None = None
        self._current_ocs: ObjectCentricState | None = None
        super().__init__()

    # -- backends & count inference -----------------------------------------

    def _backend_for(self, count: int) -> ConstantObjectKinDEREnv:
        """Return (building + caching on first use) the backend for a given count."""
        backend = self._backends.get(count)
        if backend is None:
            kwargs = {self._count_kwarg: count}
            try:
                backend = self._env_cls(**kwargs)  # type: ignore[arg-type]
            except RuntimeError as exc:
                # kinder raises RuntimeError("Failed to sample initial state ...") when
                # the scene geometry cannot fit this many objects. Surface the count so
                # a config with an over-large count fails with a clear message.
                raise ValueError(
                    f"{self._env_cls.__name__} could not build an instance with "
                    f"{self._count_kwarg}={count}; it likely exceeds the family's "
                    f"feasible object count. Lower the configured counts."
                ) from exc
            self._backends[count] = backend
            if self._render_dpi is not None:
                # The kinematic env's config is a frozen dataclass; set the field
                # through object.__setattr__ (render() reads config.render_dpi fresh).
                object.__setattr__(
                    backend._object_centric_env.config, "render_dpi", self._render_dpi
                )
            exemplar, _ = backend._object_centric_env.reset(seed=0)
            self._prefixed_count_to_kwarg[self._count_prefixed(exemplar)] = count
        return backend

    def _count_prefixed(self, state: ObjectCentricState) -> int:
        prefix = self._count_object_prefix
        return sum(1 for name in state.get_object_names() if name.startswith(prefix))

    def _infer_count(self, state: ObjectCentricState) -> int:
        prefixed = self._count_prefixed(state)
        try:
            return self._prefixed_count_to_kwarg[prefixed]
        except KeyError as exc:
            raise ValueError(
                f"Cannot infer object count: {prefixed} object(s) with prefix "
                f"{self._count_object_prefix!r} match no configured count "
                f"(known prefixed counts: {sorted(self._prefixed_count_to_kwarg)})"
            ) from exc

    def count_for_seed(self, seed: int | None) -> int:
        """A design-range count for an unpinned reset (keeps dev in-distribution).

        Held-out counts are never produced here; they reach the env only through an
        explicit ``options={"object_count": k}`` from the eval harness.
        """
        return int(np.random.default_rng(seed).choice(self._design_counts))

    @property
    def design_counts(self) -> list[int]:
        """The object counts the agent develops against (in-distribution)."""
        return list(self._design_counts)

    @property
    def eval_counts(self) -> list[int]:
        """The object counts swept at evaluation (design plus held-out)."""
        return list(self._eval_counts)

    @property
    def current_count(self) -> int:
        """The object count of the current instance."""
        assert self._current_count is not None, "Must call reset()"
        return self._current_count

    def max_steps_for_count(self, count: int) -> int:
        """Evaluation step budget for an instance of this object count.

        Larger instances need more steps to solve, so the horizon scales with the count;
        otherwise a big instance would be scored as failed just for hitting a fixed cap
        before it could reasonably finish.
        """
        return self._base_steps + self._steps_per_object * int(count)

    # -- gym API ------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObjectCentricState, dict[str, Any]]:
        if options is not None and "object_count" in options:
            count = int(options["object_count"])
            backend = self._backend_for(count)
            ocs, info = backend._object_centric_env.reset(seed=seed)
        elif options is not None and "init_state" in options:
            state = options["init_state"]
            count = self._infer_count(state)
            backend = self._backend_for(count)
            ocs, info = backend._object_centric_env.reset(
                seed=seed, options={"init_state": state}
            )
        else:
            count = self.count_for_seed(seed)
            backend = self._backend_for(count)
            ocs, info = backend._object_centric_env.reset(seed=seed)
        self._current_backend = backend
        self._current_count = count
        self._current_ocs = ocs
        return ocs, {**info, "object_count": count}

    def step(
        self, action: NDArray[Any]
    ) -> tuple[ObjectCentricState, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._current_backend is not None, "Must call reset()"
        action = np.asarray(action, dtype=np.float32)
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self._current_backend._object_centric_env.step(action)
        self._current_ocs = obs
        return (
            obs,
            reward,
            terminated,
            truncated,
            {**info, "object_count": self._current_count},
        )

    def get_state(self) -> ObjectCentricState:
        assert self._current_ocs is not None, "Must call reset()"
        return self._current_ocs.copy()

    def set_state(self, state: ObjectCentricState) -> None:
        count = self._infer_count(state)
        backend = self._backend_for(count)
        ocs, _ = backend._object_centric_env.reset(options={"init_state": state})
        self._current_backend = backend
        self._current_count = count
        self._current_ocs = ocs

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        assert self._current_backend is not None, "Must call reset()"
        inner = self._current_backend._object_centric_env
        return inner.render()  # type: ignore[no-untyped-call,return-value]

    # -- per-count Box view for the bilevel planner -------------------------

    @property
    def current_box_space(self) -> Any:
        """The current instance's fixed-length ``ObjectCentricBoxSpace``."""
        assert self._current_backend is not None, "Must call reset()"
        return self._current_backend.observation_space

    def current_box_obs(self) -> NDArray[Any]:
        """Vectorize the current object-centric observation for the planner."""
        assert self._current_ocs is not None, "Must call reset()"
        return self.current_box_space.vectorize(self._current_ocs)

    def to_box(self, state: ObjectCentricState) -> NDArray[Any]:
        """Vectorize an object-centric state through the current count's Box space."""
        return self.current_box_space.vectorize(state)

    def models_for_count(self, count: int) -> Any:
        """Build the SeSamE models for a given count (per-count Box space + kwargs)."""
        backend = self._backend_for(count)
        return build_sesame_models(
            self,
            observation_space=backend.observation_space,
            model_kwargs={self._count_kwarg: count},
        )

    def infer_count(self, state: ObjectCentricState) -> int:
        """The family's count parameter for a state, inferred by object-name prefix."""
        return self._infer_count(state)

    def models_for_state(self, state: ObjectCentricState) -> Any:
        """Build the SeSamE models for the count implied by *state*."""
        return self.models_for_count(self._infer_count(state))

    # -- description --------------------------------------------------------

    @property
    def env_description(self) -> str:
        return self._describe(include_access=True)

    @property
    def env_description_blackbox(self) -> str:
        return self._describe(include_access=False)

    def _observation_section(self) -> str:
        """A per-TYPE feature table for the object-centric observation.

        Replaces the fixed-count Box index table: in this setting the number of
        objects (and hence any index layout) varies, so the observation is described
        by the types present and their features, not by absolute indices.
        """
        by_type: dict[str, list[str]] = {}
        type_features: dict[str, list[str]] = {}
        for name in sorted(self._reference_state.get_object_names()):
            obj = self._reference_state.get_object_from_name(name)
            by_type.setdefault(obj.type.name, []).append(name)
            type_features.setdefault(obj.type.name, list(self.type_features[obj.type]))
        lines = [
            "Each observation is an `ObjectCentricState`: a set of typed objects, each "
            "with named features. The number of objects VARIES between episodes, so "
            "there is no fixed-length vector and no fixed index layout. Read it with:",
            "",
            "- `state.get_objects(type)` / `state.get_object_names()` / "
            "`state.get_object_from_name(name)` to enumerate objects,",
            "- `state.get(obj, feature)` to read a feature.",
            "",
            "| **Type** | **Features** | **Example objects (this reset)** |",
            "| --- | --- | --- |",
        ]
        for type_name in sorted(by_type):
            feats = ", ".join(type_features[type_name])
            examples = ", ".join(by_type[type_name])
            lines.append(f"| {type_name} | {feats} | {examples} |")
        lines.append("")
        lines.append(
            f"The count-defining objects are named `{self._count_object_prefix}0`, "
            f"`{self._count_object_prefix}1`, ... and their number changes per episode."
        )
        return "\n".join(lines)

    def _generalization_section(self) -> str:
        return (
            "This environment contains a VARIABLE number of objects. Your program "
            "must handle ANY number of them -- in principle unbounded. Do not assume "
            "a fixed object count, a fixed number of objects of any type, or any "
            "fixed index layout; iterate the objects in the state and act on whatever "
            "is there. You will be evaluated on a range of object counts, including "
            "counts larger than those you see while developing."
        )

    def _describe(self, include_access: bool) -> str:
        md = self._backend_for(min(self._design_counts))._object_centric_env.metadata
        description = (
            f"# {self._env_path.rsplit(':', 1)[-1]} (variable object count)\n\n"
            f"{md.get('description', '')}\n\n"
            f"## Generalization\n\n{self._generalization_section()}\n\n"
            f"## Observation\n\n{self._observation_section()}\n\n"
            f"## Action Space\n\n{md.get('action_space_description', '')}\n\n"
            f"## Reward\n\n{md.get('reward_description', '')}\n\n"
        )
        if not include_access:
            return description
        return description + (
            "## Example Usage\n\n"
            "```python\n"
            "from robocode.environments.variable_object_count_env import "
            "VariableObjectCountEnv\n\n"
            "env = VariableObjectCountEnv(\n"
            f'    constant_object_env_path="{self._env_path}",\n'
            f'    count_kwarg="{self._count_kwarg}",\n'
            f'    count_object_prefix="{self._count_object_prefix}",\n'
            f"    design_counts={self._design_counts},\n"
            f"    eval_counts={self._eval_counts},\n"
            ")\n\n"
            "# Unpinned reset samples a design-range count; pin one for testing:\n"
            "state, info = env.reset(seed=0, options={'object_count': 3})\n"
            "for name in state.get_object_names():\n"
            "    obj = state.get_object_from_name(name)\n"
            "    x = state.get(obj, 'x')  # read a feature of an object\n\n"
            "action = env.action_space.sample()\n"
            "state, reward, terminated, truncated, info = env.step(action)\n"
            "saved = env.get_state()      # an ObjectCentricState\n"
            "env.set_state(saved)         # restores it\n"
            "```\n"
        )
