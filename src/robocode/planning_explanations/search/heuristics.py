"""Bounded priority-search ingredients, unit-cost relaxation and goal timing."""

# Exact built-in scalar types are the representation contract.
# pylint: disable=unidiomatic-typecheck

from dataclasses import asdict, dataclass, field, fields, replace

from .engine import Configuration, Transition, label, read_action
from .tasks import (
    ZERO,
    BoundTaskSpec,
    HeuristicCase,
    TaskCase,
    TaskSearchSpec,
    check_heuristic,
)


@dataclass
class CostConfiguration(Configuration):
    """All priority-search state, selected from actual concrete dictionaries."""

    costs: dict[int, int] = field(default_factory=dict)
    heuristic: dict[int, int] = field(default_factory=dict)

    def extra_data(self) -> dict:
        return {
            "costs": [[label(k), v] for k, v in self.costs.items()],
            "heuristic": [[label(k), v] for k, v in self.heuristic.items()],
        }


@dataclass
class CostTransition(Transition):
    """Selection priorities and strict improvements of already discovered paths."""

    improved: tuple[int, ...] = ()
    priorities: tuple[tuple[int, int], ...] = ()

    def extra_data(self) -> dict:
        return {"improved": list(self.improved), "priorities": list(self.priorities)}


def check_options(selection: str, tie: str, relaxation: str, goal_timing: str):
    """Finite algorithm choices shared by declarations and bound problems."""
    if selection not in ("fifo", "lifo", "greedy", "astar"):
        raise ValueError("Unknown priority-search selection")
    if tie not in ("first", "last") or relaxation not in ("first", "improve"):
        raise ValueError("Unknown tie or path-update rule")
    if goal_timing not in ("selection", "generation"):
        raise ValueError("Unknown goal-test timing")


@dataclass(frozen=True)
class HeuristicProblem(BoundTaskSpec):
    """An independently executable, unit-cost priority-search configuration."""

    selection: str = "astar"
    tie: str = "first"
    relaxation: str = "improve"
    goal_timing: str = "selection"
    heuristic: tuple[int, ...] = ZERO

    def __post_init__(self):
        # Reuse legacy domain checks while the extended selector is checked here.
        data = {f.name: getattr(self, f.name) for f in fields(BoundTaskSpec)}
        data["selection"] = "fifo"
        BoundTaskSpec(**data)
        if self.block != 1 or self.mode != "graph":
            raise ValueError("Priority search uses identity states and graph mode")
        check_options(self.selection, self.tie, self.relaxation, self.goal_timing)
        check_heuristic(self.heuristic)

    @property
    def step_budget(self) -> int:
        return 64

    def initialize(self, root: int) -> CostConfiguration:
        return CostConfiguration(
            root,
            (root,),
            {root: None},
            costs={root: 0},
            heuristic=dict(enumerate(self.heuristic)),
        )

    def advance(self, state: Configuration) -> CostTransition:
        if not isinstance(state, CostConfiguration) or state.phase != "ready":
            raise ValueError("Priority search requires a live cost configuration")
        parents, costs = dict(state.parents), dict(state.costs)
        frontier: list[int] = []
        for represented in state.frontier:
            if not isinstance(represented, int):
                raise ValueError("Priority frontier contains a non-environment node")
            frontier.append(represented)

        def configuration(
            done: bool = False, output: int | None = None
        ) -> CostConfiguration:
            return CostConfiguration(
                state.root,
                tuple(frontier),
                parents,
                "done" if done else "ready",
                output,
                costs,
                dict(state.heuristic),
            )

        if not frontier:
            return CostTransition(configuration(True, self.fallback), "fallback")
        priorities: tuple[tuple[int, int], ...] = ()
        if self.selection in ("fifo", "lifo"):
            index = 0 if self.selection == "fifo" else len(frontier) - 1
        else:
            priorities = tuple(
                (n, state.heuristic[n] + (costs[n] if self.selection == "astar" else 0))
                for n in frontier
            )
            index = min(
                range(len(frontier)),
                key=lambda i: (priorities[i][1], i if self.tie == "first" else -i),
            )
        node = frontier.pop(index)
        if self.task_goal(node):
            return CostTransition(
                configuration(True, read_action(state, node, self)),
                "goal",
                selected=node,
                priorities=priorities,
            )
        generated, added, improved = [], [], []
        for child in self.successors(node):
            if not isinstance(child, int):
                raise ValueError("Priority successors must be environment states")
            generated.append(child)
            candidate = costs[node] + 1
            if child not in parents or (
                self.relaxation == "improve" and candidate < costs[child]
            ):
                if child in parents:
                    improved.append(child)
                parents[child] = node
                costs[child] = candidate
                if child not in frontier:
                    frontier.append(child)
                    added.append(child)
                if self.goal_timing == "generation" and self.task_goal(child):
                    ready = configuration()
                    return CostTransition(
                        replace(
                            ready, phase="done", output=read_action(ready, child, self)
                        ),
                        "goal_generated",
                        node,
                        node,
                        tuple(generated),
                        tuple(added),
                        tuple(improved),
                        priorities,
                    )
        return CostTransition(
            configuration(),
            "expand",
            node,
            node,
            tuple(generated),
            tuple(added),
            tuple(improved),
            priorities,
        )


@dataclass(frozen=True)
class HeuristicSpec(TaskSearchSpec):
    """Frozen priority ingredients; extra fields distinguish this version in JSON."""

    selection: str = "astar"
    tie: str = "first"
    relaxation: str = "improve"
    goal_timing: str = "selection"
    heuristic_source: str = "input"
    fixed_heuristic: tuple[int, ...] = ZERO

    def base(self) -> TaskSearchSpec:
        """Legacy task ingredients, without attempting to use its FIFO engine."""
        data = {f.name: getattr(self, f.name) for f in fields(TaskSearchSpec)}
        data["selection"] = "fifo"
        return TaskSearchSpec(**data)

    def __post_init__(self):
        self.base()
        if self.mode != "graph":
            raise ValueError("The extended priority family is graph search")
        check_options(self.selection, self.tie, self.relaxation, self.goal_timing)
        if self.heuristic_source not in ("input", "fixed", "zero"):
            raise ValueError("Unknown heuristic source")
        check_heuristic(self.fixed_heuristic)

    def bind(self, case: TaskCase) -> HeuristicProblem:
        if not isinstance(case, HeuristicCase):
            raise ValueError("Priority-search ingredients need a HeuristicCase")
        heuristic = {
            "input": case.heuristic,
            "fixed": self.fixed_heuristic,
            "zero": ZERO,
        }[self.heuristic_source]
        data = asdict(self.base().bind(case))
        data["selection"] = self.selection
        return HeuristicProblem(
            **data,
            tie=self.tie,
            relaxation=self.relaxation,
            goal_timing=self.goal_timing,
            heuristic=heuristic,
        )
