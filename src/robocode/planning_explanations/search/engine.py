"""The PDF's fixed search engine, independent of policy traces and mappings."""

# Exact scalar types keep bools distinct from integer state/action labels.
# pylint: disable=unidiomatic-typecheck

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Action:
    """A terminal decision node, not a hypothetical environment state."""

    value: int


Node = int | Action


@dataclass(frozen=True)
class SearchSpec:
    """A finite, executable ingredient language; no arbitrary callbacks."""

    block: int = 2
    goal: int = 2
    selection: str = "fifo"
    order: tuple[int, int] = (-1, 1)
    fallback: int = 0
    mode: Literal["graph", "reactive"] = "graph"

    def __post_init__(self):
        if self.block not in (1, 2, 3) or type(self.block) is not int:
            raise ValueError("Block size must be 1, 2 or 3")
        if type(self.goal) is not int or not 0 <= self.goal < self.nodes:
            raise ValueError("Goal threshold outside the abstract domain")
        if self.selection not in ("fifo", "lifo"):
            raise ValueError("Selection must be FIFO or LIFO")
        if self.order not in ((-1, 1), (1, -1)):
            raise ValueError("Declare one of the two neighbor orders")
        if type(self.fallback) is not int or self.fallback not in (-1, 0, 1):
            raise ValueError("Invalid primitive fallback action")
        if self.mode not in ("graph", "reactive"):
            raise ValueError("Unknown search family")

    @property
    def nodes(self) -> int:
        """Number of abstract environment nodes."""
        return 6 // self.block

    def abstract(self, position: int) -> int:
        """The explicitly declared environment abstractor."""
        return position // self.block

    def task_goal(self, node: int) -> bool:
        """Meaning of the environment goal, distinct from action-node stopping."""
        return node >= self.goal

    def successors(self, node: int) -> tuple[Node, ...]:
        """Neighbor moves or a minimal action-selection expansion."""
        if self.mode == "reactive":
            return (Action(0 if self.task_goal(node) else 1),)
        return tuple(node + a for a in self.order if 0 <= node + a < self.nodes)

    def first_action(self, root: int, child: int) -> int:
        """Decode the first parent edge in the original ordered line world."""
        return 1 if child > root else -1

    @property
    def step_budget(self) -> int:
        """Bound independent prediction without attributing it to the program."""
        return 20

    def initialize(self, root: int) -> "Configuration":
        """The legacy first-parent configuration remains unchanged."""
        return initial(root)

    def advance(self, state: "Configuration") -> "Transition":
        """Run this fixed engine family, without consulting concrete snapshots."""
        return first_parent_step(self, state)


@dataclass
class Configuration:
    """All abstract state needed to predict subsequent execution."""

    root: int
    frontier: tuple[Node, ...]
    parents: dict[Node, Node | None]
    phase: str = "ready"
    output: int | None = None

    def extra_data(self) -> dict:
        """Extended families can expose additional state without changing old JSON."""
        return {}


@dataclass
class Transition:
    """A single fixed-engine iteration and its inspectable effects."""

    state: Configuration
    operation: str
    selected: Node | None = None
    expanded: int | None = None
    generated: tuple[Node, ...] = ()
    added: tuple[Node, ...] = ()

    def extra_data(self) -> dict:
        """Extended families expose decision ingredients and update evidence."""
        return {}


@dataclass
class Execution:
    """Independent abstract run; these reference operations are not policy work."""

    initial: Configuration
    transitions: list[Transition] = field(default_factory=list)


def initial(root: int) -> Configuration:
    """Initialize one root and an ordered frontier."""
    return Configuration(root, (root,), {root: None})


def read_action(
    state: Configuration, node: Node, spec: SearchSpec | None = None
) -> int:
    """Read a terminal action or the first edge on the retained parent path."""
    if isinstance(node, Action):
        return node.value
    if node == state.root:
        return 0
    seen: set[Node] = set()
    while state.parents[node] != state.root:
        parent = state.parents[node]
        if node in seen or parent is None:
            raise ValueError("Invalid parent path")
        seen.add(node)
        node = parent
    if type(node) is not int:
        raise ValueError("An environment path contains an action node")
    return (
        spec.first_action(state.root, node)
        if spec
        else (1 if node > state.root else -1)
    )


def step(spec: SearchSpec, state: Configuration) -> Transition:
    """Advance independently, without observing a future concrete snapshot."""
    return spec.advance(state)


def first_parent_step(spec: SearchSpec, state: Configuration) -> Transition:
    """Original FIFO/LIFO semantics, preserved for existing frozen experiments."""
    if state.phase != "ready":
        raise ValueError("Cannot step a terminated search")
    parents = dict(state.parents)
    frontier = list(state.frontier)
    if not frontier:
        return Transition(
            Configuration(state.root, (), parents, "done", spec.fallback), "fallback"
        )
    node = frontier.pop(0 if spec.selection == "fifo" else -1)
    if isinstance(node, Action) or (spec.mode == "graph" and spec.task_goal(node)):
        output = read_action(state, node, spec)
        return Transition(
            Configuration(state.root, tuple(frontier), parents, "done", output),
            "goal",
            selected=node,
        )
    generated = spec.successors(node)
    added = []
    for child in generated:
        if child not in parents:
            parents[child] = node
            frontier.append(child)
            added.append(child)
    return Transition(
        Configuration(state.root, tuple(frontier), parents),
        "expand",
        node,
        node,
        generated,
        tuple(added),
    )


def run(spec: SearchSpec, root: int, max_steps: int | None = None) -> Execution:
    """Execute a supplied abstract problem, the milestone-1 interface."""
    if type(root) is not int or not 0 <= root < spec.nodes:
        raise ValueError("Invalid root")
    execution = Execution(spec.initialize(root))
    state = execution.initial
    for _ in range(spec.step_budget if max_steps is None else max_steps):
        transition = step(spec, state)
        execution.transitions.append(transition)
        state = transition.state
        if state.phase == "done":
            return execution
    raise RuntimeError("Abstract step budget exhausted")


def label(node: Node | None) -> str:
    """Human-readable node label, also used in JSON's explicit edge lists."""
    if isinstance(node, Action):
        return f"Action({node.value})"
    return "None" if node is None else str(node)


def state_data(state: Configuration) -> dict:
    """Serialize node identities without ambiguous JSON dictionary keys."""
    return {
        "root": label(state.root),
        "frontier": [label(n) for n in state.frontier],
        "parents": [[label(n), label(p)] for n, p in state.parents.items()],
        "phase": state.phase,
        "output": state.output,
        **state.extra_data(),
    }
