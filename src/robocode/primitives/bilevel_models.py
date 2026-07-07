"""The `bilevel_models` primitive: the SeSamE planning models for an environment.

`primitives['bilevel_models']` is the `SesameModels` bundle that
`build_sesame_models(env)` returns (see `robocode.utils.bilevel`): the symbolic
predicates, types, and lifted operators, the parameterized skills the SeSamE
planner is built from, the `transition_fn` simulator, the `state_abstractor` and
`goal_deriver`, and the observation/state converters. The `SesameModels` type is
defined in `bilevel_planning.structs`.
"""

from robocode.utils.bilevel import build_sesame_models

__all__ = ["build_sesame_models"]
