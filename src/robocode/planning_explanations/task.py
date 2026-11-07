"""The known task boundary and candidate computational models.

Reference rollouts here are analyzer work, never measured program computation. Utilities
are terminal values at the lookahead cutoff (no intermediate reward).
"""

# Exact builtin types exclude booleans and user-defined numeric subclasses.
# pylint: disable=unidiomatic-typecheck

from dataclasses import dataclass
from math import isfinite
from random import Random

TRANSITIONS = ((1, 2), (3, 4), (4, 5), (0, 5), (1, 3), (2, 0))

# Model roles in canonical order. Depth 1 compares its leaf evaluations directly:
# a separate "scores" role would only be a copy of the same numbers, and demanding
# a distinct program location for that copy makes support depend on code style.
ROLES = {
    1: ("states_1", "evaluations"),
    2: ("states_1", "states_2", "evaluations", "scores"),
}


def choose(scores):
    """Two actions, maximizing utility, with ties going to action zero."""
    return int(scores[1] > scores[0])


def action_values(roles):
    """The per-action values a model's decision compares."""
    return roles["scores"] if "scores" in roles else roles["evaluations"]


def rollout(state, transitions, utilities, horizon):
    """Compute a candidate model's states, evaluations and (depth 2) backups."""
    if horizon not in ROLES:
        raise ValueError("Only horizons 1 and 2 are supported")
    first = list(transitions[state])
    roles = {"states_1": first}
    leaves = first
    if horizon == 2:
        leaves = [transitions[s][a] for s in first for a in (0, 1)]
        roles["states_2"] = leaves
    roles["evaluations"] = [utilities[s] for s in leaves]
    if horizon == 2:
        roles["scores"] = backup(roles["evaluations"])
    return roles


def backup(evaluations):
    """Maximum backup of each first action's two depth-2 leaf evaluations."""
    return [max(evaluations[:2]), max(evaluations[2:])]


@dataclass(frozen=True)
class Case:
    """An immutable input, including an externally cached policy control."""

    label: str
    state: int
    transitions: tuple
    utilities: tuple
    cached_actions: tuple

    def args(self):
        """Fresh mutable inputs, so every run starts from the same state."""
        return (
            self.state,
            [list(row) for row in self.transitions],
            list(self.utilities),
            list(self.cached_actions),
        )


def make_case(label, state, utilities, transitions=TRANSITIONS):
    """The cache is prepared outside program execution for negative controls."""
    if len(transitions) != 6 or any(len(row) != 2 for row in transitions):
        raise ValueError("Expected six states and two actions")
    if type(state) is not int or not 0 <= state < 6 or len(utilities) != 6:
        raise ValueError("Invalid initial state or utility vector")
    if any(type(v) not in (int, float) or not isfinite(v) for v in utilities):
        raise ValueError("Utilities must be finite numbers")
    if any(type(s) is not int or not 0 <= s < 6 for row in transitions for s in row):
        raise ValueError("Invalid transition")
    transitions = tuple(tuple(row) for row in transitions)
    utilities = tuple(utilities)
    cached = tuple(
        choose(action_values(rollout(s, transitions, utilities, 2))) for s in range(6)
    )
    return Case(label, state, transitions, utilities, cached)


def random_transitions(rng):
    """Two different successors for every state; a state may lead back to itself."""
    return tuple(tuple(rng.sample(range(6), 2)) for _ in range(6))


def cases(seed, prefix):
    """Two fresh contexts for every initial state, reproducibly.

    Each context draws its own utilities AND its own transition table. With one fixed
    table, the states a program visits depend only on the start state, and reading the
    environment's table looks the same as carrying a private copy of it.
    """
    rng = Random(seed)
    result = []
    for repetition in range(2):
        for state in range(6):
            # Distinct nonintegral utilities help separate state IDs from values.
            utilities = [v + 0.25 for v in rng.sample(range(-30, 31), 6)]
            result.append(
                make_case(
                    f"{prefix}-{repetition}-{state}",
                    state,
                    utilities,
                    random_transitions(rng),
                )
            )
    return result


class Navigation:
    """A resettable deterministic navigation environment for manual inspection."""

    def __init__(self, case):
        self.case = case
        self.state = case.state

    def reset(self, state=None):
        """Reset to the case's start or an explicitly supplied state."""
        state = self.case.state if state is None else state
        if type(state) is not int or not 0 <= state < 6:
            raise ValueError("Unknown state")
        self.state = state
        return self.state

    def step(self, action):
        """Move once; the caller chooses the terminal horizon."""
        if action not in (0, 1):
            raise ValueError("Unknown action")
        self.state = self.case.transitions[self.state][action]
        return self.state

    def terminal_utility(self):
        """Utility if the rollout ends at the current state."""
        return self.case.utilities[self.state]
