"""Checkpoint every loop and method of a class-based policy across an episode.

Milestone 8's runner. Unlike `checkpoints.Program` it accepts a whole module (imports,
helpers, closures, NumPy), because agent-written policies are classes with state across
calls. It is trusted-code instrumentation, not a sandbox. Names are never interpreted:
every `while` loop is a part, and its runs are recorded compactly (dictionaries as per-
iteration changes) so a 2,000-cell search does not cost 2,000 full copies.
"""

# Exact built-in types are the restricted value language.
# pylint: disable=unidiomatic-typecheck

import ast
import copy
import hashlib
import heapq
from collections import deque
from itertools import compress, islice
from operator import is_not
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..instrument import UnsupportedProgram
from .selectors import locate

FILENAME = "<search-policy-module>"
RECORDER = "__m8__"
_MISSING = object()


_SCALARS = frozenset({type(None), bool, int, float, str})
_CONTAINERS = (list, dict, deque, set)


def immutable(value, depth: int = 0) -> bool:
    """Scalars, strings and tuples of them: safe to keep by reference."""
    kind = type(value)
    if kind in _SCALARS:
        return True
    if kind is tuple and depth < 4:
        for item in value:
            if type(item) not in _SCALARS and not immutable(item, depth + 1):
                return False
        return True
    return False


def value_like(value, depth: int = 0) -> bool:
    """The restricted language of recorded values, nested at most four levels."""
    if immutable(value, depth):
        return True
    if depth >= 4:
        return False
    if type(value) in (list, deque, tuple):
        # A tuple holding lists (a record of containers) is copied like a list.
        return all(value_like(v, depth + 1) for v in value)
    if type(value) is set:
        return all(immutable(v, depth + 1) for v in value)
    if type(value) is dict:
        return all(
            immutable(k, depth + 1) and value_like(v, depth + 1)
            for k, v in value.items()
        )
    return False


OPAQUE = object()


_NUMPY_NUMBER_TYPES: dict[type, bool] = {}


def _numpy_number(value) -> bool:
    """A numpy boolean, integer or float scalar (recognised without importing numpy)."""
    kind = type(value)
    known = _NUMPY_NUMBER_TYPES.get(kind)
    if known is None:
        dtype = getattr(value, "dtype", None)
        known = (
            kind.__module__ == "numpy"
            and getattr(value, "ndim", None) == 0
            and getattr(dtype, "kind", "?") in ("b", "i", "u", "f")
        )
        if kind.__module__ == "numpy":
            _NUMPY_NUMBER_TYPES[kind] = known
    return known


# A numeric numpy array this small (a point, a pose, a rectangle's corners) is
# recorded as the numbers it holds; larger ones stay opaque.
SMALL_ARRAY = 64


def _small_array(value) -> bool:
    """A numpy array of at most SMALL_ARRAY booleans, integers or floats."""
    dtype = getattr(value, "dtype", None)
    return (
        type(value).__module__ == "numpy"
        and type(value).__name__ == "ndarray"
        and getattr(dtype, "kind", "?") in ("b", "i", "u", "f")
        and 1 <= getattr(value, "ndim", 0) <= 3
        and getattr(value, "size", SMALL_ARRAY + 1) <= SMALL_ARRAY
    )


def _same_items(left: tuple, right: tuple) -> bool:
    """Tuple equality that treats a comparison numpy cannot decide as different."""
    try:
        return bool(left == right)
    except (ValueError, TypeError):
        return False


def _nested_tuple(value):
    return tuple(map(_nested_tuple, value)) if type(value) is list else value


def plain(value, depth: int = 0):
    """A copy with numpy numbers as Python numbers, or OPAQUE if anything is opaque.

    Small numeric arrays become nested tuples of their numbers.
    """
    kind = type(value)
    if kind in _SCALARS:
        return value
    if _numpy_number(value):
        return value.item()
    if _small_array(value):
        return _nested_tuple(value.tolist())
    if depth >= 4:
        return OPAQUE
    if kind is dict:
        pairs = []
        for key, item in value.items():
            key, item = plain(key, depth + 1), plain(item, depth + 1)
            if key is OPAQUE or item is OPAQUE:
                return OPAQUE
            pairs.append((key, item))
        return dict(pairs)
    if kind in (tuple, list, deque, set):
        items = []
        for item in value:
            item = plain(item, depth + 1)
            if item is OPAQUE:
                return OPAQUE
            items.append(item)
        if kind is deque:
            return deque(items, value.maxlen)
        return kind(items)
    return OPAQUE


GRID_LIMIT = 4_000_000


def boolean_grid(value) -> bool:
    """A numpy array of booleans (an occupancy map), recognised without numpy."""
    dtype = getattr(value, "dtype", None)
    return (
        type(value).__module__ == "numpy"
        and type(value).__name__ == "ndarray"
        and getattr(dtype, "kind", "?") == "b"
        and 1 <= getattr(value, "ndim", 0) <= 3
        and getattr(value, "size", GRID_LIMIT + 1) <= GRID_LIMIT
    )


def same_snapshot(left: dict, right: dict) -> bool:
    """Equal frame values; boolean grids compare element by element."""
    if left.keys() != right.keys():
        return False
    for name, value in left.items():
        other = right[name]
        if boolean_grid(value) or boolean_grid(other):
            if not (
                boolean_grid(value)
                and boolean_grid(other)
                and value.shape == other.shape
                and bool((value == other).all())
            ):
                return False
        elif value != other:
            return False
    return True


def holds_mutable(value, depth: int = 0) -> bool:
    """A list, dict, deque or set, or a tuple holding one: state that can change.

    A tuple of numbers or objects cannot itself gain or lose items, so it can
    never be a hidden worklist.
    """
    if type(value) in (list, dict, deque, set):
        return True
    if type(value) is tuple:
        return depth >= 4 or any(holds_mutable(v, depth + 1) for v in value)
    return False


def policy_callable(value) -> bool:
    """A function or lambda written in the analyzed source, never a library call."""
    code = getattr(value, "__code__", None)
    return callable(value) and code is not None and code.co_filename == FILENAME


@dataclass(frozen=True)
class SearchEdit:
    """A declared live edit at one loop checkpoint of one run inside a call."""

    site: str
    ordinal: int
    visit: int
    operation: str
    selector: str
    entry: tuple | None = None
    cell: tuple | None = None
    value: float | None = None
    heap: bool = True


class RunCapture:
    """Frames of one loop run: full first frame, then only what changed.

    Flat lists and dictionaries of immutable items are recorded incrementally:
    the previous frame's raw container is kept, positions or keys whose object
    changed are found with C-level identity comparisons, and only those items are
    converted (each numpy-holding tuple once per run). An immutable item with the
    same identity has the same value, so nothing is missed; a mutable item, an
    opaque one, or a key-order change falls back to a full snapshot.
    `incremental = False` keeps the plain full-snapshot encoder as a reference.
    """

    incremental = True

    def __init__(self) -> None:
        self.records: list[tuple[str, dict]] = []
        self.callables: dict[str, Callable] = {}
        self._previous: dict[str, Any] = {}
        # Values outside the snapshot language must not look like absent state.
        # Accumulate names over the run, including containers lost in later frames.
        self.omitted_containers: set[str] = set()
        # Per frame, the variables whose container is a different object than in
        # the previous frame (a new list, not the old one edited). Ids are compared
        # only between consecutive frames. A reused address can make a replaced
        # container look edited in place, never the reverse, so errors are toward
        # "maybe a worklist".
        self.replaced: list[frozenset[str]] = []
        self._ids: dict[str, int] = {}
        # Raw containers as of the previous frame, and converted immutable items.
        self._raw: dict[str, Any] = {}
        self._memo: dict[int, tuple[Any, Any]] = {}

    def record(self, kind: str, values: dict) -> None:
        """Encode current values against the previous frame of this run."""
        if not self.records:
            self.callables = {
                k: v
                for k, v in values.items()
                if not k.startswith("__") and policy_callable(v)
            }
        encoded: dict[str, tuple] = {}
        # Each remembered value is tagged with its kind: V scalar/tuple, L flat
        # list, C nested list or tuple holding lists, D dictionary (kept as a
        # private copy), Q deque of immutable items, S set of immutable items,
        # A read-only boolean grid; P and U are list and dictionary changes.
        previous = self._previous
        seen = set()
        for name, value in values.items():
            if name.startswith("__"):
                continue
            if self.incremental and self._incremental(name, value, encoded):
                seen.add(name)
                continue
            self._raw.pop(name, None)
            recorded = self._encode(name, value, encoded)
            if not recorded:
                # numpy numbers are recorded as the plain numbers they hold.
                converted = plain(value)
                if converted is not OPAQUE:
                    recorded = self._encode(name, converted, encoded)
            if recorded:
                seen.add(name)
            elif holds_mutable(value):
                # Unrecorded state that could change is not evidence of absence.
                self.omitted_containers.add(name)
        for name in list(previous):
            if name not in seen:
                encoded[name] = ("X",)
                del previous[name]
                self._raw.pop(name, None)
        self.records.append((kind, encoded))
        ids = {
            name: id(value)
            for name, value in values.items()
            if type(value) in _CONTAINERS and not name.startswith("__")
        }
        old_ids = self._ids
        self.replaced.append(
            frozenset(name for name, ident in ids.items() if old_ids.get(name) != ident)
        )
        self._ids = ids

    def _item(self, value):
        """An immutable plain copy of one container item, or OPAQUE."""
        kind = type(value)
        if kind in _SCALARS:
            return value
        if kind is tuple:
            hit = self._memo.get(id(value))
            if hit is not None and hit[0] is value:
                return hit[1]
            converted = plain(value)
            if converted is OPAQUE or not immutable(converted):
                return OPAQUE
            self._memo[id(value)] = (value, converted)
            return converted
        if _numpy_number(value):
            return value.item()
        return OPAQUE

    def _incremental(self, name: str, value, encoded: dict) -> bool:
        """Record a flat list or dictionary from its changes; False to fall back."""
        kind = type(value)
        if kind is not list and kind is not dict:
            return False
        shadow = self._raw.get(name)
        if type(shadow) is kind and name in self._previous:
            delta = self._list_delta if kind is list else self._dict_delta
            if delta(name, value, shadow, encoded):
                return True
        # Full snapshot of the converted container.
        if kind is list:
            items = [self._item(v) for v in value]
            if any(v is OPAQUE for v in items):
                return False
            snapshot = tuple(items)
            if self._previous.get(name) != ("L", snapshot):
                encoded[name] = ("L", snapshot)
            self._raw[name] = list(value)
        else:
            pairs = [(self._item(k), self._item(v)) for k, v in value.items()]
            if any(k is OPAQUE or v is OPAQUE for k, v in pairs):
                return False
            encoded[name] = ("D", dict(pairs))
            self._raw[name] = dict(value)
        # A marker: the plain encoder sees no reusable previous state.
        self._previous[name] = ("R",) if kind is dict else ("L", snapshot)
        return True

    def _list_delta(self, name, value: list, old: list, encoded: dict) -> bool:
        common = min(len(old), len(value))
        changed = list(compress(range(common), map(is_not, value, old)))
        tail = value[len(old) :]
        if not changed and len(value) == len(old):
            return True
        patches = []
        for index in changed:
            item = self._item(value[index])
            if item is OPAQUE:
                return False
            patches.append((index, item))
        appended = [self._item(v) for v in tail]
        if any(v is OPAQUE for v in appended):
            return False
        encoded[name] = ("P", len(value), tuple(patches), tuple(appended))
        for index in changed:
            old[index] = value[index]
        del old[len(value) :]
        old.extend(tail)
        self._previous[name] = ("R",)
        return True

    def _dict_delta(self, name, value: dict, old: dict, encoded: dict) -> bool:
        keys = list(value)
        count = len(old)
        if len(keys) < count or keys[:count] != list(old):
            return False  # removed or reordered keys: take a full snapshot
        values = list(value.values())
        changed = list(
            compress(range(count), map(is_not, islice(values, count), old.values()))
        )
        changes = {}
        for index in (*changed, *range(count, len(keys))):
            key, item = self._item(keys[index]), self._item(values[index])
            if key is OPAQUE or item is OPAQUE:
                return False
            changes[key] = item
        if changes:
            encoded[name] = ("U", changes, ())
            for index in (*changed, *range(count, len(keys))):
                old[keys[index]] = values[index]
        return True

    def _encode(self, name: str, value, encoded: dict) -> bool:
        """Record one value if it is in the snapshot language; True if recorded."""
        previous = self._previous
        old = previous.get(name)
        if type(value) is dict:
            if (
                old is not None
                and old[0] == "D"
                and self._delta(name, value, old[1], encoded)
            ):
                return True
            if value_like(value):
                encoded[name] = ("D", copy.deepcopy(value))
                previous[name] = ("D", copy.deepcopy(value))
                return True
            return False
        if type(value) is list:
            snapshot = tuple(value)
            if old is not None and old[0] == "L" and _same_items(old[1], snapshot):
                # Equal to an already-checked immutable snapshot.
                return True
            if immutable(snapshot):
                encoded[name] = ("L", snapshot)
                previous[name] = ("L", snapshot)
                return True
            if value_like(value):
                if old is None or old[0] != "C" or old[1] != value:
                    encoded[name] = ("C", copy.deepcopy(value))
                    previous[name] = ("C", copy.deepcopy(value))
                return True
            return False
        if boolean_grid(value):
            if (
                old is None
                or old[0] != "A"
                or old[1].shape != value.shape
                or not (old[1] == value).all()
            ):
                grid = value.copy()
                grid.flags.writeable = False
                encoded[name] = ("A", grid)
                previous[name] = ("A", grid)
            return True
        if type(value) in (deque, set) and all(immutable(v) for v in value):
            snapshot = (
                ("Q", tuple(value), value.maxlen)
                if type(value) is deque
                else ("S", frozenset(value))
            )
            if old != snapshot:
                encoded[name] = snapshot
                previous[name] = snapshot
            return True
        if immutable(value):
            if (
                old is None
                or old[0] != "V"
                or type(old[1]) is not type(value)
                or old[1] != value
            ):
                encoded[name] = ("V", value)
                previous[name] = ("V", value)
            return True
        if type(value) is tuple and value_like(value):
            if old is None or old[0] != "C" or not _same_items(old[1], value):
                encoded[name] = ("C", copy.deepcopy(value))
                previous[name] = ("C", copy.deepcopy(value))
            return True
        return False

    @staticmethod
    def _delta(name: str, value: dict, old: dict, encoded: dict) -> bool:
        """Record changed and removed entries of a flat dictionary, if possible."""
        changes = {}
        for key, item in value.items():
            before = old.get(key, _MISSING)
            if before is item:
                continue
            if before is _MISSING or type(before) is not type(item) or before != item:
                if not (immutable(key) and immutable(item)):
                    return False
                changes[key] = item
        removed: tuple = ()
        if len(old) + sum(k not in old for k in changes) != len(value):
            removed = tuple(k for k in old if k not in value)
        if changes or removed:
            encoded[name] = ("U", changes, removed)
            old.update(changes)
            for key in removed:
                del old[key]
        return True

    def frames(self) -> Iterator[tuple[str, dict]]:
        """Reconstruct each frame's values.

        The yielded dict is reused: copy to keep.
        """
        working: dict[str, Any] = {}
        for kind, encoded in self.records:
            for name, item in encoded.items():
                tag = item[0]
                if tag == "V":
                    working[name] = item[1]
                elif tag == "L":
                    working[name] = list(item[1])
                elif tag in ("D", "C"):
                    working[name] = copy.deepcopy(item[1])
                elif tag == "Q":
                    working[name] = deque(item[1], item[2])
                elif tag == "S":
                    working[name] = set(item[1])
                elif tag == "A":
                    working[name] = item[1]
                elif tag == "P":
                    current = working[name]
                    for index, value in item[2]:
                        current[index] = value
                    del current[item[1] :]
                    current.extend(item[3])
                elif tag == "U":
                    current = working.setdefault(name, {})
                    current.update(item[1])
                    for key in item[2]:
                        del current[key]
                else:
                    working.pop(name, None)
            yield kind, working

    @property
    def length(self) -> int:
        """Number of recorded frames, including any edit frame."""
        return len(self.records)


@dataclass
class RunRecord:
    """One execution of one loop inside one harness call."""

    site: str
    ordinal: int
    methods: tuple[str, ...]
    capture: RunCapture = field(default_factory=RunCapture)
    visits: int = 0
    exited: bool = False


class Session:
    """Everything recorded during one harness call or one probe re-execution."""

    def __init__(self, capture: bool = True, edit: SearchEdit | None = None) -> None:
        self.capture = capture
        self.edit = edit
        self.runs: list[RunRecord] = []
        self.stack: list[str] = []
        self.edits = 0
        self._ordinals: dict[str, int] = {}

    def start(self, site: str) -> RunRecord:
        """Begin a run; ordinals number this site's runs within the call."""
        ordinal = self._ordinals.get(site, 0)
        self._ordinals[site] = ordinal + 1
        run = RunRecord(site, ordinal, tuple(self.stack))
        self.runs.append(run)
        return run


class Recorder:
    """The injected global; dispatches to the innermost active session."""

    def __init__(self) -> None:
        self.sessions: list[Session] = []

    def begin(self, method: str) -> None:
        """Method entry, for attributing runs to helpers and harness calls."""
        if self.sessions:
            self.sessions[-1].stack.append(method)

    def end(self) -> None:
        """Method exit, including exits by exception."""
        if self.sessions and self.sessions[-1].stack:
            self.sessions[-1].stack.pop()

    def enter(self, site: str) -> RunRecord | None:
        """A loop statement is about to run: one new run of this part."""
        return self.sessions[-1].start(site) if self.sessions else None

    def visit(self, run: RunRecord | None, values: dict) -> None:
        """Before every loop test, including the final false test."""
        if run is None:
            return
        session = self.sessions[-1]
        index = run.visits
        run.visits += 1
        edit = session.edit
        if session.capture:
            run.capture.record("visit", values)
        if (
            edit is not None
            and edit.site == run.site
            and edit.ordinal == run.ordinal
            and edit.visit == index
        ):
            apply_edit(values, edit)
            session.edits += 1
            if session.capture:
                run.capture.record("edited", values)

    def leave(self, run: RunRecord | None, values: dict, kind: str = "exit") -> None:
        """Loop termination: a false test or break ("exit"), or a return inside it."""
        if run is None:
            return
        run.exited = True
        if self.sessions[-1].capture:
            run.capture.record(kind, values)


def apply_edit(values: dict, edit: SearchEdit) -> None:
    """Mutate the live container so aliases keep Python semantics."""
    container, key = locate(values, edit.selector)
    target = container[key]
    if edit.operation == "raise":
        if type(target) is not dict or edit.cell not in target:
            raise ValueError("Cost edits need a live dictionary entry")
        target[edit.cell] = edit.value
        return
    if type(target) not in (list, deque):
        raise ValueError("Frontier edits require a live list or deque")
    if edit.operation == "clear":
        target.clear()
    elif edit.operation == "drop":
        del target[drop_index(target, edit.entry)]
        if edit.heap:
            # The heap layout's own invariant, restored after removing an entry.
            heapq.heapify(target)
    elif edit.operation == "drop_cell":
        kept = drop_cell_entries(target, edit.cell)
        target.clear()
        target.extend(kept)
        if edit.heap:
            heapq.heapify(target)
    else:
        raise ValueError("Unknown search edit")


def entry_view(entry) -> Any:
    """A frontier entry as (priority, cell), or the cell itself.

    Extra numbers between the priority and the cell (tie-breakers) are dropped.
    """
    if type(entry) is tuple and len(entry) >= 2 and type(entry[-1]) is tuple:
        return (entry[0], entry[-1])
    return entry


def drop_index(frontier, view) -> int:
    """Position of the first entry whose view equals `view`."""
    for index, entry in enumerate(frontier):
        if entry_view(entry) == view:
            return index
    raise ValueError("The frontier holds no such entry")


def entry_cell(entry) -> Any:
    """The cell a frontier entry holds."""
    view = entry_view(entry)
    return view if view is entry else view[1]


def drop_cell_entries(frontier, cell) -> list:
    """Every entry except those of one cell, for any entry layout."""
    kept = [entry for entry in frontier if entry_cell(entry) != cell]
    if len(kept) == len(frontier):
        raise ValueError("The frontier holds no entry for this cell")
    return kept


class _ReturnCheckpoints(ast.NodeTransformer):
    """Record a termination checkpoint before each return inside one loop body."""

    def __init__(self, token: str) -> None:
        self.token = token

    def visit_Return(self, node: ast.Return):  # pylint: disable=invalid-name
        """`return value` ends the loop's run here, as Tom's algorithm does."""
        leave = ast.parse(f"{RECORDER}.leave({self.token}, locals(), 'return')").body[0]
        ast.copy_location(leave, node)
        return [leave, node]

    def _skip(self, node):
        # A nested function's returns do not end this loop.
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = _skip  # type: ignore[assignment]
    visit_ClassDef = visit_Lambda = _skip  # type: ignore[assignment]


class _Transformer(ast.NodeTransformer):
    """Wrap loops and class methods; the original statements are untouched."""

    def __init__(self, source: str) -> None:
        self.lines = source.splitlines()
        self.sites: dict[str, dict] = {}
        self.functions: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):  # pylint: disable=invalid-name
        """Track the enclosing function for site descriptions."""
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()
        return node

    visit_ClassDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_While(self, node: ast.While):  # pylint: disable=invalid-name
        """Before each test, after exit, and a fresh run on every loop entry."""
        self.generic_visit(node)
        if node.orelse:
            raise UnsupportedProgram("Loop else is outside the policy runner")
        site = f"{node.lineno}:{node.col_offset}:While"
        token = f"__run_{node.lineno}_{node.col_offset}__"
        self.sites[site] = {
            "kind": "loop",
            "line": node.lineno,
            "function": ".".join(self.functions),
            "source": self.lines[node.lineno - 1].strip(),
        }
        start = ast.parse(f"{token} = {RECORDER}.enter({site!r})").body[0]
        visit = ast.parse(f"{RECORDER}.visit({token}, locals())").body[0]
        stop = ast.If(ast.UnaryOp(ast.Not(), node.test), [ast.Break()], [])
        leave = ast.parse(f"{RECORDER}.leave({token}, locals())").body[0]
        rewriter = _ReturnCheckpoints(token)
        body = [rewriter.visit(statement) for statement in node.body]
        body = [
            s for item in body for s in (item if isinstance(item, list) else [item])
        ]
        loop = ast.While(ast.Constant(True), [visit, stop, *body], [])
        result = [start, loop, leave]
        for statement in result:
            ast.copy_location(statement, node)
        return result


def _wrap_method(function: ast.FunctionDef) -> None:
    """Record method entry/exit with try/finally, so helpers are attributed."""
    begin = ast.parse(f"{RECORDER}.begin({function.name!r})").body[0]
    end = ast.parse(f"{RECORDER}.end()").body[0]
    body = function.body
    docstring = []
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        docstring, body = [body[0]], body[1:]
    guarded = ast.Try(body or [ast.Pass()], [], [], [end])
    for statement in (begin, guarded, end):
        ast.copy_location(statement, function)
    function.body = [*docstring, begin, guarded]


class PolicyProgram:
    """Plain and instrumented compilations of one trusted policy module."""

    def __init__(self, source: str, class_name: str = "GeneratedApproach") -> None:
        self.source = source
        self.class_name = class_name
        self.digest = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        classes = [
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        ]
        if len(classes) != 1:
            raise UnsupportedProgram(f"Expected one class named {class_name}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.startswith("__m8"):
                raise UnsupportedProgram("Reserved identifier")
        self.plain = compile(tree, FILENAME, "exec")
        transformed = copy.deepcopy(tree)
        transformer = _Transformer(source)
        transformed = transformer.visit(transformed)
        for node in transformed.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        _wrap_method(item)
        self.code = compile(ast.fix_missing_locations(transformed), FILENAME, "exec")
        self.sites = transformer.sites
        self.recorder = Recorder()

    def policy_class(self, instrumented: bool = True) -> type:
        """Execute the module in fresh globals and return the policy class."""
        namespace: dict[str, Any] = {
            "__name__": "search_policy_module",
            "__file__": FILENAME,
            RECORDER: self.recorder,
        }
        exec(  # pylint: disable=exec-used
            self.code if instrumented else self.plain, namespace
        )
        return namespace[self.class_name]

    def call(self, policy, method: str, args: tuple, session: Session):
        """Invoke one harness method under a session; always restore the stack."""
        self.recorder.sessions.append(session)
        try:
            return getattr(policy, method)(*args)
        finally:
            self.recorder.sessions.pop()


def plan_attributes(policy) -> dict[str, tuple]:
    """Instance attributes holding a list of 2D points (empty lists included)."""
    found = {}
    for name, value in vars(policy).items():
        if type(value) not in (list, tuple):
            continue
        points = []
        for item in value:
            try:
                if len(item) != 2:
                    break
                points.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                break
        else:
            found[name] = tuple(points)
    return found


@dataclass
class CallRecord:
    """One harness call: initialization `reset` (index 0) or `get_action`."""

    index: int
    method: str
    runs: list[RunRecord]
    plans_before: dict[str, tuple]
    plans_after: dict[str, tuple]
    action: tuple[float, ...] | None = None


@dataclass
class EpisodeTrace:
    """All calls of one closed-loop episode, in order."""

    case: tuple
    calls: list[CallRecord]
    solved: bool
    steps: int


def action_tuple(action) -> tuple[float, ...]:
    """Portable, comparable copy of an action vector."""
    return tuple(float(v) for v in list(action))


def run_episode(
    program: PolicyProgram,
    domain,
    case: tuple,
    after_call: Callable | None = None,
    instrumented: bool = True,
    capture: bool = True,
) -> EpisodeTrace:
    """Run one episode; callback receives pre-call copies and the live post-state.

    With `capture=False` the instrumented policy only counts runs per part.
    The callback must copy the post-state before any counterfactual execution.
    """
    state, info = domain.reset(case)
    policy = program.policy_class(instrumented)(
        domain.action_space, domain.observation_space, {}
    )
    calls: list[CallRecord] = []

    def invoke(index: int, method: str, args: tuple):
        before = (
            (copy.deepcopy(policy), domain.copy_state(args[0]))
            if after_call is not None
            else None
        )
        session = Session(capture=instrumented and capture)
        plans_before = plan_attributes(policy)
        result = program.call(policy, method, args, session)
        record = CallRecord(
            index,
            method,
            session.runs,
            plans_before,
            plan_attributes(policy),
            None if result is None else action_tuple(result),
        )
        calls.append(record)
        if after_call is not None:
            assert before is not None
            after_call(record, before[0], before[1], args[1:], policy)
        return result

    invoke(0, "reset", (state, info))
    solved, steps = False, 0
    for index in range(1, domain.max_steps(case) + 1):
        action = invoke(index, "get_action", (state,))
        state, terminated, truncated = domain.step(action)
        steps = index
        if terminated or truncated:
            solved = bool(terminated)
            break
    return EpisodeTrace(case, calls, solved, steps)


def observe_episode(program: PolicyProgram, domain, case: tuple) -> EpisodeTrace:
    """Instrumented episode, checked against the genuinely uninstrumented policy."""
    plain = run_episode(program, domain, case, instrumented=False)
    observed = run_episode(program, domain, case)
    if [c.action for c in plain.calls] != [c.action for c in observed.calls]:
        raise AssertionError("Observation changed the policy's actions")
    return observed


def reexecute(
    program: PolicyProgram,
    policy,
    method: str,
    state,
    extra: tuple,
    session: Session,
) -> tuple[Any, Any]:
    """Run one call again on deep copies; returns (output, policy afterwards)."""
    clone = copy.deepcopy(policy)
    output = program.call(clone, method, (state, *extra), session)
    return output, clone
