"""Grid priority search with float costs and heap entries (milestone 8).

The family is independent of policy traces. A problem is bound from a run's inputs (root
cell, optional goal cell, and the policy's own free-cell predicate or boolean grid), then
stepped one loop iteration at a time. Earlier engines are unchanged.

grid-priority-v2 adds ingredients whose values are proposed from development runs
(`policy_fit`) and then frozen: a move table with step costs, a weighted octile or
straight-line heuristic, the tie-break numbers stored in heap entries, a no-corner-
cutting rule, a parent table without a root entry, and a boolean-grid model. The engine
never reads a concrete snapshot to decide a step.
"""

# Exact built-in types are the representation contract.
# pylint: disable=unidiomatic-typecheck

import heapq
import math
from collections.abc import Callable
from operator import itemgetter
from dataclasses import dataclass, field
from typing import Any

Cell = tuple[int, int]

GRAMMAR = "grid-priority-v2"
FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))
EIGHT = FOUR + ((1, 1), (1, -1), (-1, 1), (-1, -1))
MODELS = ("own", "open", "grid_free", "grid_blocked")
_VIEW = itemgetter(0, -1)
FIELDS = ("g", "neg_g")
H_FORMS = {"octile": 2, "euclidean": 1}


def _tuples(value) -> tuple:
    """JSON lists back to (nested) tuples, so loaded specs compare and hash."""
    if isinstance(value, (list, tuple)):
        return tuple(_tuples(v) for v in value)
    return value


def is_cell(value) -> bool:
    """A grid state is a pair of plain integers."""
    return (
        type(value) is tuple and len(value) == 2 and all(type(v) is int for v in value)
    )


@dataclass(frozen=True)
class GridSpec:
    """Finite ingredient choices; no callbacks are stored in the declaration."""

    selection: str = "min"
    priority: str = "g"
    heuristic: str = "none"
    root_priority: str = "zero"
    goal_timing: str = "selection"
    relaxation: str = "improve"
    stale: str = "reexpand"
    neighborhood: int = 4
    model: str = "own"
    layout: str = "pairs"
    # grid-priority-v2 ingredients; the defaults reproduce grid-priority-v1.
    moves: tuple = ()
    h_form: str = ""
    h_coef: tuple = ()
    corners: str = "cut"
    root_parent: str = "none"
    fields: tuple = ()
    # Where an expansion reads the selected cell's cost: the cost table, or the
    # number stored in its heap entry. Equal in ordinary runs; cost edits differ.
    g_source: str = "costs"

    def __post_init__(self):
        for name in ("moves", "h_coef", "fields"):
            object.__setattr__(self, name, _tuples(getattr(self, name)))
        self._check_v1()
        self._check_v2()

    def _check_v1(self):
        if self.selection not in ("min", "fifo", "lifo"):
            raise ValueError("Unknown grid selection")
        if self.priority not in ("g", "g+h", "h"):
            raise ValueError("Unknown priority rule")
        if self.heuristic not in ("none", "euclidean", "manhattan", "fitted"):
            raise ValueError("Unknown heuristic formula")
        if (self.priority == "g") != (self.heuristic == "none"):
            raise ValueError("A heuristic is used exactly when the priority has h")
        if self.root_priority not in ("zero", "rule"):
            raise ValueError("Unknown root entry priority")
        if self.priority == "g" and self.root_priority != "zero":
            # The root's g is zero, so "rule" would duplicate "zero".
            raise ValueError("A g-priority root entry is canonically zero")
        if self.goal_timing not in ("selection", "generation", "none"):
            raise ValueError("Unknown goal timing")
        if self.goal_timing == "none" and self.heuristic != "none":
            raise ValueError("A heuristic needs a goal cell")
        if self.relaxation not in ("first", "improve"):
            raise ValueError("Unknown path-update rule")
        if self.stale not in ("reexpand", "skip"):
            raise ValueError("Unknown stale-entry rule")
        if self.stale == "skip" and self.selection != "min":
            raise ValueError("Stale skipping is defined for priority selection")
        if type(self.neighborhood) is not int or self.neighborhood not in (0, 4, 8):
            raise ValueError("Neighborhood must be 4 or 8, or 0 with a move table")
        if self.model not in MODELS:
            raise ValueError("Model is the policy's predicate, a boolean grid, or open")
        if self.layout not in ("pairs", "states", "tagged"):
            raise ValueError("Frontier layout must be pairs, states or tagged")
        if self.layout == "tagged" and self.selection != "min":
            # Extra tie-break numbers only matter to a priority heap.
            raise ValueError("A tagged frontier is a priority heap")
        if self.layout == "states" and (
            self.selection == "min"
            or self.priority != "g"
            or self.root_priority != "zero"
        ):
            raise ValueError("A state-list frontier is ordered FIFO/LIFO only")

    def _check_v2(self):
        if (self.neighborhood == 0) != bool(self.moves):
            raise ValueError("A move table replaces the 4/8 neighborhood")
        offsets = [m[:2] for m in self.moves]
        if len(set(offsets)) != len(offsets) or any(
            len(m) != 3
            or type(m[0]) is not int
            or type(m[1]) is not int
            or m[:2] == (0, 0)
            or type(m[2]) not in (int, float)
            or not m[2] > 0
            for m in self.moves
        ):
            raise ValueError("Moves are distinct (di, dj, positive cost) triples")
        if (self.heuristic == "fitted") != bool(self.h_form):
            raise ValueError("A fitted heuristic names its form")
        if self.h_form and (
            self.h_form not in H_FORMS
            or len(self.h_coef) != H_FORMS[self.h_form]
            or not all(type(c) is float for c in self.h_coef)
        ):
            raise ValueError("Fitted heuristics are octile (w, c) or euclidean (w)")
        if not self.h_form and self.h_coef:
            raise ValueError("Only a fitted heuristic has coefficients")
        if self.corners not in ("cut", "no_cut"):
            raise ValueError("Corner rule is cut or no_cut")
        if self.corners == "no_cut" and (
            self.model == "open" or not any(di and dj for di, dj in self.offsets)
        ):
            raise ValueError("No corner cutting needs a model and diagonal moves")
        if self.root_parent not in ("none", "absent"):
            raise ValueError("The root's parent entry is None or absent")
        if self.fields and (
            self.layout != "tagged" or not all(f in FIELDS for f in self.fields)
        ):
            raise ValueError("Entry fields name g or -g numbers of a tagged heap")
        if self.g_source not in ("costs", "entry"):
            raise ValueError("Expansion costs come from the table or the entry")
        if self.g_source == "entry" and not self.fields:
            raise ValueError("Reading g from the entry needs a fitted g field")

    @property
    def offsets(self) -> tuple:
        """Move offsets in generation order."""
        if self.moves:
            return tuple(m[:2] for m in self.moves)
        return FOUR if self.neighborhood == 4 else EIGHT

    def step_cost(self, di: int, dj: int) -> float:
        """A fitted move's cost, or the straight-line length of a v1 move."""
        for move in self.moves:
            if move[:2] == (di, dj):
                return move[2]
        return math.hypot(di, dj)

    @property
    def needs_goal(self) -> bool:
        """Goal tests and heuristics both read a mapped goal cell."""
        return self.goal_timing != "none" or self.heuristic != "none"

    @property
    def ordered(self) -> bool:
        """FIFO/LIFO frontiers compare list order; heaps compare a multiset."""
        return self.selection != "min"


@dataclass
class GridTransition:
    """One loop iteration's inspectable effects; full states stay in the engine."""

    operation: str
    selected: Cell | None = None
    expanded: Cell | None = None
    generated: tuple[Cell, ...] = ()
    added: tuple[Cell, ...] = ()
    improved: tuple[Cell, ...] = ()
    frontier_size: int = 0
    expanded_depth: int | None = None
    discovery_depths: tuple[int, ...] = ()


@dataclass
class GridSearch:
    """A live abstract configuration and its fixed transition rule."""

    spec: GridSpec
    root: Cell
    goal: Cell | None = None
    free: Callable[[Cell], bool] | None = None
    parents: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    frontier: list = field(default_factory=list)
    phase: str = "ready"
    reached_goal: bool = False
    pushes: int = 0

    def __post_init__(self):
        if not is_cell(self.root):
            raise ValueError("A grid root must be an integer cell")
        if self.spec.needs_goal and not is_cell(self.goal):
            raise ValueError("These ingredients need a goal cell")
        if self.spec.model != "open" and not callable(self.free):
            raise ValueError("This model ingredient needs a mapped predicate or grid")
        self.parents = {self.root: None}
        self.costs = {self.root: 0.0}
        if self.spec.layout == "states":
            self.frontier = [self.root]
        else:
            first = 0.0 if self.spec.root_priority == "zero" else self.rank(self.root)
            self.frontier = (
                [(first, *self._fields(self.root, 0), self.root)]
                if self.spec.layout == "tagged"
                else [(first, self.root)]
            )
            self.pushes = 1

    def _fields(self, cell: Cell, push: int) -> tuple:
        """A tagged entry's numbers between priority and cell."""
        if not self.spec.fields:
            return (push,)
        cost = self.costs[cell]
        return tuple(cost if f == "g" else -cost for f in self.spec.fields)

    def _expansion_cost(self, entry, node: Cell) -> float:
        """The selected cell's g used to cost its children."""
        if self.spec.g_source == "costs":
            return self.costs[node]
        position = 1 + self.spec.fields.index(
            "g" if "g" in self.spec.fields else "neg_g"
        )
        value = entry[position]
        return value if self.spec.fields[position - 1] == "g" else -value

    def observed_parents(self) -> dict:
        """Parents as the program stores them (with or without a root entry)."""
        if self.spec.root_parent == "none":
            return self.parents
        return {k: v for k, v in self.parents.items() if k != self.root}

    def h(self, cell: Cell) -> float:
        """Declared heuristic formulas, measured in cells to the goal."""
        if self.spec.heuristic == "none":
            return 0.0
        assert self.goal is not None
        dx, dy = cell[0] - self.goal[0], cell[1] - self.goal[1]
        if self.spec.heuristic == "euclidean":
            return math.hypot(dx, dy)
        if self.spec.h_form == "euclidean":
            return math.hypot(dx, dy) * self.spec.h_coef[0]
        if self.spec.h_form == "octile":
            weight, ratio = self.spec.h_coef
            high, low = max(abs(dx), abs(dy)), min(abs(dx), abs(dy))
            return (high + ratio * low) * weight
        return float(abs(dx) + abs(dy))

    def rank(self, cell: Cell) -> float:
        """The priority an entry for this cell receives now."""
        if self.spec.priority == "g":
            return self.costs[cell]
        if self.spec.priority == "h":
            return self.h(cell)
        return self.costs[cell] + self.h(cell)

    def canonical_frontier(self) -> tuple:
        """Ordered lists compare in order; a heap compares as a sorted multiset.

        A tagged heap is compared as (priority, cell) entries: its tie-break
        numbers are the policy's own bookkeeping, not part of the search.
        """
        entries = (
            list(map(_VIEW, self.frontier))
            if self.spec.layout == "tagged"
            else self.frontier
        )
        return tuple(entries) if self.spec.ordered else tuple(sorted(entries))

    def frontier_cells(self) -> list[Cell]:
        """States currently queued, including duplicates of stale entries."""
        if self.spec.layout == "states":
            return list(self.frontier)
        return [entry[-1] for entry in self.frontier]

    def depth(self, cell: Cell) -> int:
        """Retained parent-chain length; parent cycles are invalid."""
        depth, seen = 0, set()
        while self.parents[cell] is not None:
            if cell in seen:
                raise ValueError("Cyclic retained parent graph")
            seen.add(cell)
            cell = self.parents[cell]
            depth += 1
        return depth

    def _pop(self, prefer: Cell | None = None) -> Any:
        if self.spec.selection != "min":
            return self.frontier.pop(0 if self.spec.selection == "fifo" else -1)
        if prefer is not None and self.spec.layout == "tagged" and not self.spec.fields:
            # Ties in priority go to the cell the run itself selected.
            low = self.frontier[0][0]
            tied = [
                i
                for i, e in enumerate(self.frontier)
                if e[0] == low and e[-1] == prefer
            ]
            if tied:
                entry = self.frontier.pop(min(tied, key=lambda i: self.frontier[i][1]))
                heapq.heapify(self.frontier)
                return entry
        return heapq.heappop(self.frontier)

    def _push(self, cell: Cell) -> None:
        if self.spec.layout == "states":
            if cell not in self.frontier:
                self.frontier.append(cell)
        elif self.spec.layout == "tagged":
            entry = (self.rank(cell), *self._fields(cell, self.pushes), cell)
            heapq.heappush(self.frontier, entry)
            self.pushes += 1
        elif self.spec.selection == "min":
            heapq.heappush(self.frontier, (self.rank(cell), cell))
        else:
            self.frontier.append((self.rank(cell), cell))

    def tied_cells(self) -> set:
        """Distinct cells sharing the lowest priority of a tagged heap."""
        if self.spec.layout != "tagged" or self.spec.fields or not self.frontier:
            # Fitted entry numbers break ties exactly as the program's heap does.
            return set()
        low = self.frontier[0][0]
        return {e[-1] for e in self.frontier if e[0] == low}

    def step(self, prefer: Cell | None = None) -> GridTransition:
        """Advance one loop iteration without reading any concrete snapshot.

        `prefer` resolves a priority tie of a tagged heap (read from the run);
        without it, ties go to the earliest push.
        """
        if self.phase != "ready":
            raise ValueError("Cannot step a terminated search")
        if not self.frontier:
            self.phase = "done"
            return GridTransition("exhausted")
        entry = self._pop(prefer)
        node = entry if self.spec.layout == "states" else entry[-1]
        if self.spec.goal_timing == "selection" and node == self.goal:
            self.phase, self.reached_goal = "done", True
            return GridTransition("goal", node, frontier_size=len(self.frontier))
        if self.spec.stale == "skip" and entry[0] > self.rank(node):
            return GridTransition("skip", node, frontier_size=len(self.frontier))
        generated: list[Cell] = []
        added: list[Cell] = []
        improved: list[Cell] = []
        depths: list[int] = []
        base = self._expansion_cost(entry, node)
        for di, dj in self.spec.offsets:
            child = (node[0] + di, node[1] + dj)
            if self.spec.model != "open":
                assert self.free is not None
                if not self.free(child):
                    continue
                if (
                    self.spec.corners == "no_cut"
                    and di
                    and dj
                    and not (
                        self.free((node[0] + di, node[1]))
                        and self.free((node[0], node[1] + dj))
                    )
                ):
                    continue
            generated.append(child)
            candidate = base + self.spec.step_cost(di, dj)
            known = child in self.parents
            if known and not (
                self.spec.relaxation == "improve" and candidate < self.costs[child]
            ):
                continue
            (improved if known else added).append(child)
            self.parents[child] = node
            self.costs[child] = candidate
            self._push(child)
            depths.append(self.depth(child))
            if self.spec.goal_timing == "generation" and child == self.goal:
                self.phase, self.reached_goal = "done", True
                return GridTransition(
                    "goal_generated",
                    node,
                    node,
                    tuple(generated),
                    tuple(added),
                    tuple(improved),
                    len(self.frontier),
                    self.depth(node),
                    tuple(depths),
                )
        return GridTransition(
            "expand",
            node,
            node,
            tuple(generated),
            tuple(added),
            tuple(improved),
            len(self.frontier),
            self.depth(node),
            tuple(depths),
        )

    def path(self) -> tuple[Cell, ...] | None:
        """Root-to-goal parent path when the goal was reached, otherwise None."""
        if not self.reached_goal:
            return None
        assert self.goal is not None
        cells: list[Cell] = []
        cell: Cell | None = self.goal
        while cell is not None:
            cells.append(cell)
            cell = self.parents[cell]
        return tuple(reversed(cells))

    def result(self) -> object:
        """The decision a run hands on: a goal path, or the whole reached tree."""
        if self.spec.goal_timing == "none":
            return tuple(sorted(self.parents.items(), key=lambda item: item[0]))
        return self.path()


def grid_features(transitions: list[GridTransition]) -> dict:
    """The earlier feature names, computed incrementally for long grid runs."""
    expanded = [t.expanded for t in transitions if t.expanded is not None]
    generated = [c for t in transitions for c in t.generated]
    distinct = set(expanded)
    return {
        "N": len(distinct),
        "M": max(len(distinct) - 1, 0),
        "max_discovery_depth": max(
            (d for t in transitions for d in t.discovery_depths), default=0
        ),
        "max_expanded_depth": max(
            (t.expanded_depth for t in transitions if t.expanded_depth is not None),
            default=0,
        ),
        "generated_alternatives": len(generated),
        "distinct_generated_states": len(set(generated)),
        "max_frontier_size": max([1] + [t.frontier_size for t in transitions]),
        "search_expansions": sum(bool(t.generated) for t in transitions),
        "expansion_events": len(expanded),
        "reexpansions": len(expanded) - len(distinct),
        "cost_improvements": sum(len(t.improved) for t in transitions),
    }
