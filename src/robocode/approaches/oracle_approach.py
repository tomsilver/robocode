"""Dispatch to the hand-written oracle approach for the configured environment.

Oracles are reference policies written per environment, so there is no single class
Hydra can point ``approach=oracle`` at. This module maps the Hydra environment choice
name (the same key that locates per-env helper files under
``src/robocode/primitives/<env_name>/``) to the oracle that solves it, and builds it.

An oracle is not a scored method: it exists to establish that a task is solvable and
to give experiments a reference row to compare synthesized approaches against.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

from robocode.approaches.base_approach import BaseApproach

# Hydra environment choice name -> "module:ClassName" of its oracle. Environments
# without an oracle are simply absent; asking for one raises with the known list.
ORACLE_TARGETS: dict[str, str] = {
    "obstruction2d_medium": (
        "robocode.oracles.obstruction2d_medium.approach:Obstruction2DOracleApproach"
    ),
    "clutteredstorage2d_medium": (
        "robocode.oracles.clutteredstorage2d_medium.approach"
        ":ClutteredStorage2DOracleApproach"
    ),
    "stickbutton2d_medium": (
        "robocode.oracles.stickbutton2d_medium.approach:StickButton2DOracleApproach"
    ),
    "pushpullhook2d": (
        "robocode.oracles.pushpullhook2d.approach:PushPullHook2DOracleApproach"
    ),
    "pr2packed_easy": "robocode.oracles.pr2packed.approach:PR2PackedOracleApproach",
    "pr2packed_medium": "robocode.oracles.pr2packed.approach:PR2PackedOracleApproach",
    "pr2packed_hard": "robocode.oracles.pr2packed.approach:PR2PackedOracleApproach",
    "pr2packed_generalized": (
        "robocode.oracles.pr2packed.approach:PR2PackedOracleApproach"
    ),
}


def resolve_oracle_class(env_name: str | None) -> type[BaseApproach[Any, Any]]:
    """Return the oracle class registered for *env_name*."""
    if env_name is None:
        raise ValueError(
            "approach=oracle needs the environment choice name; run it through "
            "experiments/run_experiment.py, which passes env_name."
        )
    try:
        target = ORACLE_TARGETS[env_name]
    except KeyError as exc:
        known = ", ".join(sorted(ORACLE_TARGETS))
        raise ValueError(
            f"No oracle is registered for environment {env_name!r}. "
            f"Environments with an oracle: {known}."
        ) from exc
    module_path, _, class_name = target.partition(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    if not issubclass(cls, BaseApproach):
        raise TypeError(f"{target} is not a BaseApproach subclass")
    return cls


def build_oracle_approach(
    *,
    action_space: Any,
    observation_space: Any,
    seed: int = 0,
    primitives: dict[str, Callable[..., Any]] | None = None,
    env_description_path: str | None = None,
    env_name: str | None = None,
    **kwargs: Any,
) -> BaseApproach[Any, Any]:
    """Build the oracle for the configured environment.

    The runner passes every approach the same wide set of keyword arguments (``env``,
    ``max_steps``, ``mcp_tools``, ...). Most oracles were written before those existed
    and declare only the base constructor's parameters, so the extras are filtered
    against the target's signature rather than forwarded blindly.
    """
    cls = resolve_oracle_class(env_name)
    parameters = inspect.signature(cls).parameters
    accepts_everything = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    extras = {
        name: value
        for name, value in kwargs.items()
        if accepts_everything or name in parameters
    }
    return cls(
        action_space=action_space,
        observation_space=observation_space,
        seed=seed,
        primitives=primitives or {},
        env_description_path=env_description_path,
        **extras,
    )
