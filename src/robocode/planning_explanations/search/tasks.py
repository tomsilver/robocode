"""Six-state tasks, bounded ingredients, and exact task-model correspondence.

The task suite uses identity state abstraction. Its edge checks are deliberately
stronger than the legacy line world's existential action witnesses.
"""

# IDs and immutable table shapes are an exact data contract.
# pylint: disable=unidiomatic-typecheck

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Literal

from .engine import Action, Node, SearchSpec

Table = tuple[tuple[int | None, int | None], ...]
LINE: Table = tuple((s - 1 if s else None, s + 1 if s < 5 else None) for s in range(6))
ZERO = (0, 0, 0, 0, 0, 0)
TASK_DOMAIN = {
    "states": 6,
    "actions": [-1, 0, 1],
    "columns": [-1, 1],
    "goal": "exact state",
}


def check_table(table: Table) -> None:
    """Reject malformed/nonfinite transition tables before running a policy."""
    if type(table) is not tuple or len(table) != 6:
        raise ValueError("A task needs an immutable six-state transition table")
    for row in table:
        if (
            type(row) is not tuple
            or len(row) != 2
            or any(
                x is not None and (type(x) is not int or not 0 <= x < 6) for x in row
            )
        ):
            raise ValueError("Each state needs two valid successor IDs or None")


def check_heuristic(values: tuple[int, ...]) -> None:
    """Validate a supplied table without calculating distances or optimal paths."""
    if (
        type(values) is not tuple
        or len(values) != 6
        or any(type(v) is not int or not 0 <= v <= 100 for v in values)
    ):
        raise ValueError("A heuristic needs six integer values in 0..100")


@dataclass(frozen=True)
class TaskCase:
    """One ordinary invocation; names are display labels, not semantic identity."""

    name: str
    start: int
    goal: int
    transitions: Table = LINE

    def __post_init__(self):
        if (
            type(self.name) is not str
            or not self.name
            or any(
                type(s) is not int or not 0 <= s < 6 for s in (self.start, self.goal)
            )
        ):
            raise ValueError("Case needs a name and integer start/goal in 0..5")
        check_table(self.transitions)

    def payload(self) -> dict:
        """Make fresh mutable input data for the original function."""
        return {
            "start": self.start,
            "goal": self.goal,
            "transitions": {
                s: {-1: row[0], 1: row[1]} for s, row in enumerate(self.transitions)
            },
        }

    @property
    def signature(self) -> str:
        """Input identity ignores a cosmetic case name."""
        data = {k: v for k, v in asdict(self).items() if k != "name"}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def at(self, position: int):
        """A resettable invocation at another state of the same task."""
        return replace(self, start=position)

    def move(self, position: int, action: int) -> int:
        """Action zero stays in place; missing edges are invalid actions."""
        if type(position) is not int or not 0 <= position < 6:
            raise ValueError("Invalid physical state")
        if type(action) is not int or action not in (-1, 0, 1):
            raise ValueError("Invalid primitive action")
        if action == 0:
            return position
        target = self.transitions[position][0 if action == -1 else 1]
        if target is None:
            raise ValueError("Action has no transition at this state")
        return target


@dataclass(frozen=True)
class HeuristicCase(TaskCase):
    """A task with a supplied estimate table, not necessarily admissible."""

    heuristic: tuple[int, ...] = ZERO

    def __post_init__(self):
        super().__post_init__()
        check_heuristic(self.heuristic)

    def payload(self) -> dict:
        return {**super().payload(), "heuristic": dict(enumerate(self.heuristic))}


def case_from_data(data: dict) -> TaskCase:
    """Reconstruct immutable inputs while rechecking the contract."""
    if "heuristic" in data:
        return HeuristicCase(
            data["name"],
            data["start"],
            data["goal"],
            tuple(tuple(row) for row in data["transitions"]),
            tuple(data["heuristic"]),
        )
    return TaskCase(
        data["name"],
        data["start"],
        data["goal"],
        tuple(tuple(row) for row in data["transitions"]),
    )


@dataclass(frozen=True)
class BoundTaskSpec(SearchSpec):
    """A concrete task instance of the same first-parent search engine."""

    transitions: Table = LINE
    exact_goal: int = 4
    reactive_action: int = 0

    def task_goal(self, node: int) -> bool:
        return node == self.exact_goal

    def successors(self, node: int) -> tuple[Node, ...]:
        if self.mode == "reactive":
            return (Action(self.reactive_action),)
        return tuple(
            y
            for a in self.order
            if (y := self.transitions[node][0 if a == -1 else 1]) is not None
        )

    def first_action(self, root: int, child: int) -> int:
        """Read an edge's command label, not the numeric difference of node IDs."""
        for action in self.order:
            if self.transitions[root][0 if action == -1 else 1] == child:
                return action
        raise ValueError("Parent path does not correspond to an outgoing action")


@dataclass(frozen=True)
class TaskSearchSpec:
    """Task-aware ingredients; binding only selects supplied data, never solves."""

    mode: Literal["graph", "reactive"] = "graph"
    selection: str = "fifo"
    order: tuple[int, int] = (-1, 1)
    fallback: int = 0
    model_source: str = "input"
    goal_source: str = "input"
    fixed_transitions: Table = LINE
    fixed_goal: int = 4
    reactive_rule: str = "goal_direction"
    lookup: tuple[int, ...] = (1, 1, 1, 1, 0, -1)

    def __post_init__(self):
        SearchSpec(
            block=1,
            goal=self.fixed_goal,
            mode=self.mode,
            selection=self.selection,
            order=self.order,
            fallback=self.fallback,
        )
        check_table(self.fixed_transitions)
        if self.model_source not in ("input", "fixed") or self.goal_source not in (
            "input",
            "fixed",
        ):
            raise ValueError("Unknown task ingredient source")
        if self.reactive_rule not in ("goal_direction", "lookup", "constant"):
            raise ValueError("Unknown reactive rule")
        if (
            type(self.lookup) is not tuple
            or len(self.lookup) != 6
            or any(type(a) is not int or a not in (-1, 0, 1) for a in self.lookup)
        ):
            raise ValueError("A baseline lookup needs six primitive actions")

    @property
    def nodes(self) -> int:
        """Identity abstraction keeps all six environment states."""
        return 6

    def abstract(self, position: int) -> int:
        """Identity abstraction, with no rollout or feature synthesis."""
        return position

    def bind(self, case: TaskCase) -> BoundTaskSpec:
        """Instantiate frozen ingredient sources on this input task."""
        transitions = (
            case.transitions if self.model_source == "input" else self.fixed_transitions
        )
        goal = case.goal if self.goal_source == "input" else self.fixed_goal
        action = self.fallback
        if self.reactive_rule == "goal_direction":
            action = (case.start < goal) - (case.start > goal)
        elif self.reactive_rule == "lookup":
            action = self.lookup[case.start]
        return BoundTaskSpec(
            block=1,
            goal=goal,
            selection=self.selection,
            order=self.order,
            fallback=self.fallback,
            mode=self.mode,
            transitions=transitions,
            exact_goal=goal,
            reactive_action=action,
        )


def task_environment(spec: TaskSearchSpec, case: TaskCase) -> dict:
    """Check all declared goals/edges separately from computational correspondence."""
    bound = spec.bind(case)
    mismatches = []
    if spec.mode == "graph":
        if bound.exact_goal != case.goal:
            mismatches.append(
                {"kind": "goal", "model": bound.exact_goal, "task": case.goal}
            )
        for state in range(6):
            for index, action in enumerate((-1, 1)):
                model, physical = (
                    bound.transitions[state][index],
                    case.transitions[state][index],
                )
                if model != physical:
                    mismatches.append(
                        {
                            "kind": "edge",
                            "state": state,
                            "action": action,
                            "model": model,
                            "task": physical,
                        }
                    )
    return {
        "case": case.name,
        "status": (
            "not_applicable"
            if spec.mode == "reactive"
            else "rejected" if mismatches else "supported"
        ),
        "semantics": "Identity-state, all labeled edges and exact goal; "
        "reactive leaves are decisions",
        "checked_edges": 12 if spec.mode == "graph" else 0,
        "counterexamples": mismatches,
        "attributed_policy_work": 0,
    }
