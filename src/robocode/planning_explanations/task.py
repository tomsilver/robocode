"""The known task boundary and candidate computational models.

Reference rollouts here are analyzer work, never measured program computation.
"""

# Exact builtin types exclude booleans and user-defined numeric subclasses.
# pylint: disable=unidiomatic-typecheck

from dataclasses import dataclass
from math import isfinite
from random import Random

TRANSITIONS = ((1, 2), (3, 4), (4, 5), (0, 5), (1, 3), (2, 0))
DEPTHS = (1, 2, 3)
# How a model values one action path. "leaf": the utility of its last state.
# "sum": the sum of utilities of every state on it, the start state excluded (it is
# the same for both actions).
PATH_VALUES = ("leaf", "sum")
# Whether a model may stop expanding an action early (see ``Model``).
PRUNING = ("none", "cutoff")


def choose(scores):
    """Two actions, maximizing utility, with ties going to action zero."""
    return int(scores[1] > scores[0])


def action_values(roles):
    """The per-action values a model's decision compares."""
    return roles["scores"] if "scores" in roles else roles["evaluations"]


@dataclass(frozen=True)
class Model:
    """Depth-first lookahead to a fixed depth, with a maximum backup.

    Values are node events. ``states_k[i]`` is the state reached by the i-th path of
    k actions (in action order), ``evaluations[i]`` is the value of the i-th
    complete path and, beyond depth 1, ``scores[a]`` is the best evaluated path
    value under first action ``a``. A node the search never expands is ``None``.
    With ``pruning="cutoff"`` the search stops expanding the second action as soon
    as one of its paths is worth strictly more than the first action's score: the
    decision is then already known, so the cutoff never changes it, but which
    nodes are expanded now depends on the values seen.

    Depth 1 compares its evaluations directly: a separate score role would only be
    a copy of the same numbers, and demanding a distinct program location for that
    copy makes support depend on code style. Intermediate maxima (for example per
    depth-2 node at depth 3) are not roles; a program may compute them or not.
    """

    depth: int
    path_value: str = "leaf"
    pruning: str = "none"

    def __post_init__(self):
        if (
            self.depth not in DEPTHS
            or self.path_value not in PATH_VALUES
            or self.pruning not in PRUNING
        ):
            raise ValueError(
                f"Depth must be in {DEPTHS}, path value in {PATH_VALUES}, "
                f"pruning in {PRUNING}"
            )
        if self.depth == 1 and self.path_value == "sum":
            raise ValueError("At depth 1 a path's sum is its leaf utility; use 'leaf'")
        if self.depth == 1 and self.pruning == "cutoff":
            raise ValueError("At depth 1 each action has one path; nothing to cut")

    @property
    def label(self):
        """A short human-readable name."""
        value = "leaf utility" if self.path_value == "leaf" else "sum of utilities"
        cutoff = ", with cutoff" if self.pruning == "cutoff" else ""
        return f"depth {self.depth}, {value}{cutoff}"

    @property
    def roles(self):
        """Role names in canonical order."""
        states = tuple(f"states_{k}" for k in range(1, self.depth + 1))
        return states + (
            ("evaluations",) if self.depth == 1 else ("evaluations", "scores")
        )

    def rollout(self, state, transitions, utilities, patch=None):
        """Node events of one search; ``patch = (role, index, value)`` intervenes.

        Actions are searched in order and each action's paths left to right; a
        state is generated when the search first reaches it. A patched value
        replaces the computed one as soon as it exists, and everything computed
        afterwards, including whether to cut, uses the replacement. Nodes the
        search does not expand stay ``None``.
        """
        depth = self.depth
        roles = {f"states_{k}": [None] * 2**k for k in range(1, depth + 1)}
        roles["evaluations"] = [None] * 2**depth
        if depth > 1:
            roles["scores"] = [None, None]

        def record(role, index, value):
            if patch and tuple(patch[:2]) == (role, index):
                value = patch[2]
            roles[role][index] = value
            return value

        half = 2 ** (depth - 1)
        for action in (0, 1):
            evaluated = []
            for leaf in range(action * half, (action + 1) * half):
                path = []
                for level in range(1, depth + 1):
                    node = leaf >> (depth - level)
                    states = roles[f"states_{level}"]
                    if states[node] is None:
                        parent = (
                            state
                            if level == 1
                            else roles[f"states_{level - 1}"][node >> 1]
                        )
                        record(f"states_{level}", node, transitions[parent][node & 1])
                    path.append(states[node])
                if self.path_value == "leaf":
                    value = utilities[path[-1]]
                else:
                    value = sum(utilities[s] for s in path)
                evaluated.append(record("evaluations", leaf, value))
                if (
                    self.pruning == "cutoff"
                    and action == 1
                    and evaluated[-1] > roles["scores"][0]
                ):
                    break
            if depth > 1:
                record("scores", action, max(evaluated))
        return roles


# Every hypothesis, in the order they are searched and reported.
MODELS = tuple(
    Model(depth, value, pruning)
    for depth in DEPTHS
    for value in PATH_VALUES
    for pruning in PRUNING
    if depth > 1 or (value, pruning) == ("leaf", "none")
)
REFERENCE = Model(2)  # The externally cached control policy.


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
        choose(action_values(REFERENCE.rollout(s, transitions, utilities)))
        for s in range(6)
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
