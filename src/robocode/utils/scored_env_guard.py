"""Read-only view of the scored environment, for handing to generated code.

A frozen ``GeneratedApproach`` is scored through ``reset()``/``get_action()``: it must
reach the goal via the actions it returns, not by moving the environment into a
solved state. The approach is not given the environment directly, but some primitives
are bound to it (``check_action_collision`` is a ``partial`` over the live env), so a
program that receives such a primitive can reach the scored environment through its
closure.

:func:`readonly_view` wraps the environment so that the mutating entry points raise
instead. Everything else -- reading state, geometry handles, collision queries --
passes through untouched.

This replaces a source scan that looked for ``.set_state`` in ``approach.py``. That
check was wrong in both directions: it missed the same call in a sibling module the
approach imports, and it rejected programs that build a *private* environment of
their own to plan in, which is the ordinary way to write a TAMP policy and cannot
affect scoring at all. Guarding the object the approach can actually reach is exact:
a private clone is unaffected, and no amount of aliasing gets around it.
"""

from __future__ import annotations

from typing import Any, NoReturn

# Mutators that could move the scored environment into a solved state, or desync it
# from the episode the runner is scoring.
FORBIDDEN = frozenset({"set_state", "sample_next_state", "reset", "step", "close"})


class ScoredEnvMutationError(RuntimeError):
    """Raised when generated code tries to mutate the environment being scored."""


class _ReadOnlyEnv:
    """Attribute proxy that forwards everything except the mutating entry points."""

    __slots__ = ("_env",)

    def __init__(self, env: Any) -> None:
        object.__setattr__(self, "_env", env)

    def __getattribute__(self, name: str) -> Any:
        # Primitives dispatch on isinstance(env, ...) to pick an implementation, so
        # the proxy reports the wrapped environment's class; a proxy advertising its
        # own type would fall through to "unsupported environment". Only the type is
        # borrowed -- every other attribute still routes through the guard below.
        if name == "__class__":
            return object.__getattribute__(self, "_env").__class__
        return object.__getattribute__(self, name)

    @property
    def unwrapped_scored_env(self) -> Any:
        """The wrapped environment, for host-side code that legitimately mutates it."""
        return object.__getattribute__(self, "_env")

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN:
            return _forbid(name)
        return getattr(object.__getattribute__(self, "_env"), name)

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise ScoredEnvMutationError(
            f"Cannot set {name!r} on the environment being scored. Build your own "
            "environment to plan in; this one is scored through the actions your "
            "approach returns."
        )

    def __repr__(self) -> str:
        return f"<read-only {object.__getattribute__(self, '_env')!r}>"


def _forbid(name: str) -> Any:
    def _raise(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise ScoredEnvMutationError(
            f"{name}() is not available on the environment being scored: an approach "
            "must reach the goal through the actions it returns, not by moving the "
            "environment. To plan, construct your own environment and mutate that."
        )

    return _raise


def readonly_view(env: Any) -> Any:
    """Wrap *env* so generated code cannot mutate it.

    Idempotent.
    """
    # `type(...) is` rather than isinstance: the proxy reports the wrapped class.
    if type(env) is _ReadOnlyEnv:  # pylint: disable=unidiomatic-typecheck
        return env
    return _ReadOnlyEnv(env)
