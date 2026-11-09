"""Restricted manifests and current-configuration decoding, without a solver."""

# Exact built-in scalar/container types are the representation contract.
# pylint: disable=unidiomatic-typecheck

import hashlib
import json
from dataclasses import asdict, dataclass

from .checkpoints import Snapshot, Trace
from .engine import Action, Configuration, Node, SearchSpec
from .heuristics import CostConfiguration, HeuristicSpec
from .selectors import parse, select
from .tasks import TaskCase, TaskSearchSpec


@dataclass(frozen=True)
class Mapping:
    """Local selections plus one fixed numeric relabeling and alignment rule."""

    site: str
    root: str = ""
    frontier: str = ""
    parents: str = ""
    action: str = ""
    offset: int = 0

    def __post_init__(self):
        if type(self.offset) is not int or self.offset not in (0, 10):
            raise ValueError("The initial decoding grammar allows offsets 0 and 10")
        for name in (self.root, self.frontier, self.parents, self.action):
            if name:
                parse(name)


@dataclass(frozen=True)
class CostMapping(Mapping):
    """Additional literal bindings for actual cost and heuristic dictionaries."""

    costs: str = ""
    heuristic: str = ""

    def __post_init__(self):
        super().__post_init__()
        for name in (self.costs, self.heuristic):
            if name:
                parse(name)


@dataclass(frozen=True)
class Explanation:
    """A complete candidate, shared by enumerative and future LLM proposers."""

    spec: SearchSpec | TaskSearchSpec
    mapping: Mapping

    @property
    def identifier(self) -> str:
        """Stable identity of the entire frozen explanation."""
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]

    def problem(self, invocation: int | TaskCase) -> SearchSpec:
        """Bind fixed task ingredient sources without inspecting a concrete trace."""
        if isinstance(self.spec, TaskSearchSpec):
            if not isinstance(invocation, TaskCase):
                raise ValueError("Task ingredients require a TaskCase input")
            return self.spec.bind(invocation)
        if isinstance(invocation, TaskCase):
            raise ValueError("Legacy line ingredients require a position input")
        return self.spec

    def aligned(self, trace: Trace) -> list[Snapshot]:
        """Apply the same recurring checkpoint rule to any execution length."""
        if self.spec.mode == "graph":
            events = [
                s
                for s in trace.snapshots
                if s.site == self.mapping.site or s.kind == "exit"
            ]
            if not events or events[0].site != self.mapping.site:
                raise ValueError("Missing initialization/loop checkpoint")
        else:
            events = [
                s
                for s in trace.snapshots
                if s.kind in ("entry", "exit") or s.site == self.mapping.site
            ]
            if len(events) != 3 or events[1].site != self.mapping.site:
                raise ValueError("Reactive alignment needs one action checkpoint")
        if events[-1].kind != "exit":
            raise ValueError("Missing termination checkpoint")
        return events

    def decode(self, snapshot: Snapshot, argument: str) -> Configuration:
        """Interpret one snapshot; neither history nor future snapshots are read."""
        mapping, spec = self.mapping, self.spec
        values = snapshot.values
        done = snapshot.kind == "exit"
        output = None
        if done:
            if type(snapshot.output) is not int:
                raise ValueError("The policy must return an integer primitive action")
            output = snapshot.output
        parents: dict[Node, Node | None]
        frontier: tuple[Node, ...]
        if spec.mode == "reactive":
            value = values[argument]
            root = spec.abstract(
                value["start"] if isinstance(spec, TaskSearchSpec) else value
            )
            parents = {root: None}
            frontier = (root,)
            if snapshot.kind != "entry":
                value = select(values, mapping.action)
                if type(value) is not int or value not in (-1, 0, 1):
                    raise ValueError("Invalid represented action")
                child = Action(value)
                parents[child] = root
                frontier = () if done else (child,)
        else:

            def node(value: int) -> int:
                if (
                    type(value) is not int
                    or not 0 <= value - mapping.offset < spec.nodes
                ):
                    raise ValueError("Invalid represented node")
                return value - mapping.offset

            root = node(select(values, mapping.root))
            raw_frontier, raw_parents = (
                select(values, mapping.frontier),
                select(values, mapping.parents),
            )
            if type(raw_frontier) is not list or type(raw_parents) is not dict:
                raise ValueError("Expected a list frontier and dictionary parents")
            frontier = tuple(node(v) for v in raw_frontier)
            parents = {
                node(k): None if v is None else node(v) for k, v in raw_parents.items()
            }
        configuration = Configuration(
            root,
            frontier,
            parents,
            "done" if done else "ready",
            output,
        )
        if isinstance(spec, HeuristicSpec):
            if not isinstance(mapping, CostMapping):
                raise ValueError(
                    "Priority search needs concrete cost/heuristic bindings"
                )
            cost_mapping: CostMapping = mapping

            def table(selector: str) -> dict[int, int]:
                raw = select(values, selector)
                if type(raw) is not dict or any(
                    type(v) is not int or v < 0 for v in raw.values()
                ):
                    raise ValueError(
                        "Costs and heuristics must be nonnegative integer dictionaries"
                    )
                return {node(k): v for k, v in raw.items()}

            # The isinstance check above guarantees these fields; pylint does
            # not narrow types the way mypy does.
            # pylint: disable-next=no-member
            costs, heuristic = table(cost_mapping.costs), table(cost_mapping.heuristic)
            if set(costs) != set(parents) or set(heuristic) != set(range(spec.nodes)):
                raise ValueError(
                    "Cost/heuristic domains do not match represented states"
                )
            return CostConfiguration(
                **asdict(configuration), costs=costs, heuristic=heuristic
            )
        return configuration

    def admissible(self, sites: dict) -> None:
        """Check language shape and the selected source checkpoint kind."""
        if isinstance(self.spec, HeuristicSpec) != isinstance(
            self.mapping, CostMapping
        ):
            raise ValueError(
                "Priority ingredients and mappings must use the same family"
            )
        if isinstance(self.mapping, CostMapping):
            cost_names = (
                self.mapping.root,
                self.mapping.frontier,
                self.mapping.parents,
                self.mapping.costs,
                self.mapping.heuristic,
            )
            if not all(cost_names) or len(set(cost_names)) != 5:
                raise ValueError("Priority mapping requires five distinct selections")
        expected = "loop" if self.spec.mode == "graph" else "statement"
        if (
            self.mapping.site not in sites
            or sites[self.mapping.site]["kind"] != expected
        ):
            raise ValueError("Invalid alignment site")
        if self.spec.mode == "graph":
            names = (self.mapping.root, self.mapping.frontier, self.mapping.parents)
            if not all(names) or len(set(names)) != 3 or self.mapping.action:
                raise ValueError("Graph mapping requires three distinct local bindings")
        elif (
            not self.mapping.action
            or self.mapping.root
            or self.mapping.frontier
            or self.mapping.parents
            or self.mapping.offset
        ):
            raise ValueError("Reactive mapping selects a computed action only")


def from_data(data: dict) -> Explanation:
    """Load a declarative explanation, rechecking its restricted ingredient types."""
    spec = dict(data["spec"])
    spec["order"] = tuple(spec["order"])
    if "model_source" in spec:
        spec["fixed_transitions"] = tuple(
            tuple(row) for row in spec["fixed_transitions"]
        )
        spec["lookup"] = tuple(spec["lookup"])
        if "heuristic_source" in spec:
            spec["fixed_heuristic"] = tuple(spec["fixed_heuristic"])
            return Explanation(HeuristicSpec(**spec), CostMapping(**data["mapping"]))
        return Explanation(TaskSearchSpec(**spec), Mapping(**data["mapping"]))
    return Explanation(SearchSpec(**spec), Mapping(**data["mapping"]))
