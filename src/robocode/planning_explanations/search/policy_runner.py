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
import builtins
import copy
import hashlib
import heapq
import random
import sys
import time as real_time
import types
from collections import Counter, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import compress, islice
from operator import is_not
from typing import Any

from ..instrument import UnsupportedProgram
from .selectors import locate

FILENAME = "<search-policy-module>"
RECORDER = "__m8__"
_MISSING = object()


POLICY_MODULE = "search_policy_module"


def own_object(value) -> bool:
    """An instance of a class the policy itself defines (its `self`, a `Tree`)."""
    return getattr(type(value), "__module__", None) == POLICY_MODULE and hasattr(
        value, "__dict__"
    )


def _generator(value) -> bool:
    """A random-number generator: Python's `random.Random`, numpy's Generator or
    RandomState (recognised without importing numpy)."""
    kind = type(value)
    return isinstance(value, random.Random) or (
        kind.__name__ in ("Generator", "RandomState")
        and kind.__module__.startswith("numpy.random")
    )


def _generator_state(value) -> Any:
    """Something that changes whenever the generator draws a number."""
    if isinstance(value, random.Random):
        return hash(value.getstate())
    if type(value).__name__ == "RandomState":
        state = value.get_state()
        return (state[2], hash(state[1].tobytes()))
    state = value.bit_generator.state
    inner = state.get("state")
    if isinstance(inner, dict) and "state" in inner:
        return inner["state"]
    return hash(repr(state))


def _big_array(value) -> bool:
    """A numeric numpy array of more than 64 numbers: snapshots do not copy it."""
    return (
        type(value).__name__ == "ndarray"
        and value.dtype.kind != "b"
        and value.size > 64
    )


def _hidden_value(value) -> Any:
    """A cheap fingerprint of a hidden value that can hold many items, else None.

    Single numbers, strings and small arrays cannot hold a tree or a to-do list
    (a call counter, a pose), so they are not watched.
    """
    kind = type(value)
    if _big_array(value):
        if value.size <= 1_000_000:
            return ("array", value.shape, hash(value.tobytes()))
        return ("array", value.shape, id(value))
    if kind in (list, dict, set, deque, tuple):
        return (kind.__name__, len(value))
    if own_object(value):
        return ("object", id(value))
    return None


def hidden_state(values: dict) -> dict[str, Any]:
    """Fingerprints of what snapshots do not hold and that can hold many items:
    numeric numpy arrays of more than 64 numbers, and the containers, big arrays
    and objects among the attributes of the policy's own objects (`self.x`)."""
    hidden: dict[str, Any] = {}
    for name, value in values.items():
        if name.startswith("__"):
            continue
        if _big_array(value):
            hidden[name] = _hidden_value(value)
        elif own_object(value):
            for key, item in vars(value).items():
                fingerprint = _hidden_value(item)
                if fingerprint is not None:
                    hidden[f"{name}.{key}"] = fingerprint
    return hidden


class RandomWatch:
    """Which part draws random numbers: the innermost running one when a random
    generator the policy can reach moves (Python's `random`, numpy's global
    generator, and generators in the policy's locals, objects and module)."""

    def __init__(self) -> None:
        # Sites that have drawn random numbers: their hidden state is watched.
        self.sites: set[str] = set()
        self.namespace: dict | None = None

    def learn(self, values: dict, generators: dict) -> None:
        """Generators visible from a run's first checkpoint."""
        sources = list(values.values())
        if self.namespace is not None:
            sources.extend(self.namespace.values())
        for value in sources:
            if _generator(value):
                generators[id(value)] = value
            elif own_object(value):
                for item in vars(value).values():
                    if _generator(item):
                        generators[id(item)] = item

    @staticmethod
    def state(generators: dict) -> tuple:
        """The joint state of every generator known so far in this call."""
        parts: list = [hash(random.getstate())]
        numpy = sys.modules.get("numpy")
        if numpy is not None:
            legacy = numpy.random.get_state()
            parts.append((legacy[2], hash(legacy[1].tobytes())))
        parts.extend(_generator_state(g) for g in generators.values())
        return tuple(parts)


class FakeClock:
    """The policy's own view of time while it is analyzed.

    A policy that plans "until 2 seconds have passed" would otherwise plan less
    while it is recorded (recording is slow), so its recorded episode would differ
    from its plain one. Every read of this clock advances it by `TICK` seconds and
    nothing else does, so a time budget becomes a fixed number of clock reads: the
    same with or without recording, on any machine. `sleep` advances it at once.
    """

    TICK = 1e-3
    START = 1_700_000_000.0

    def __init__(self) -> None:
        self.now = self.START
        self.reads = 0
        module = types.ModuleType("time", "Deterministic stand-in for `time`.")
        for name in dir(real_time):
            if not name.startswith("__"):
                setattr(module, name, getattr(real_time, name))
        for name in (
            "time",
            "monotonic",
            "perf_counter",
            "process_time",
            "thread_time",
        ):
            setattr(module, name, self.read)
        for name in ("time_ns", "monotonic_ns", "perf_counter_ns", "process_time_ns"):
            setattr(module, name, self.read_ns)
        module.sleep = self.sleep
        self.module = module

    def reset(self) -> None:
        """Every episode starts at the same time."""
        self.now = self.START

    def read(self) -> float:
        """One look at the clock: time moves on by one tick."""
        self.now += self.TICK
        self.reads += 1
        return self.now

    def read_ns(self) -> int:
        """The same reading in nanoseconds."""
        return int(self.read() * 1e9)

    def sleep(self, seconds: float) -> None:
        """Sleeping passes the time without waiting."""
        self.now += max(0.0, float(seconds))

    def importer(self, real_import: Callable) -> Callable:
        """`import time` (and `from time import ...`) in the policy gets this clock."""

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            # pylint: disable=redefined-builtin
            if level == 0 and name == "time":
                return self.module
            return real_import(name, globals, locals, fromlist, level)

        return fake_import


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


def _members(value) -> Counter | None:
    """The objects a flat container holds, by identity (dictionary keys by value).

    For containers the snapshot language cannot hold by value. None when an item
    is itself a list, dict, deque or set: a worklist could hide inside it.
    """
    kind = type(value)
    if kind is dict:
        if any(type(v) in _CONTAINERS for v in value.values()):
            return None
        return Counter(k if immutable(k) else ("id", id(k)) for k in value)
    if kind in (list, deque, set):
        if any(type(v) in _CONTAINERS for v in value):
            return None
        return Counter(map(id, value))
    return None


def _skeleton(value, depth: int = 0):
    """Which items a container holds at every level: plain items by value, objects
    by identity, nested containers by their own skeleton (dictionary keys only, and
    the skeletons of container values; a tuple only through containers inside it).

    Two equal skeletons mean nothing was gained or lost anywhere inside. Beyond
    depth 4 the result never equals another one.
    """
    if depth >= 4:
        return object()
    kind = type(value)
    if kind is dict:
        return (
            dict,
            tuple(
                (k if immutable(k) else id(k), _skeleton(v, depth + 1))
                for k, v in value.items()
                if holds_mutable(v)
            ),
            tuple(k if immutable(k) else id(k) for k in value),
        )
    if kind is tuple:
        return (
            tuple,
            tuple(_skeleton(v, depth + 1) for v in value if holds_mutable(v)),
        )
    items = tuple(
        (
            _skeleton(v, depth + 1)
            if holds_mutable(v)
            else (v if immutable(v) else ("id", id(v)))
        )
        for v in value
    )
    return (kind, frozenset(items) if kind is set else items)


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
        # Flat containers of opaque items, followed by item identity: per frame,
        # (size, gained an item, lost an item, is a list or deque) since the
        # previous frame.
        self.members: list[dict[str, tuple[int, bool, bool, bool]]] = []
        self._members: dict[str, Counter] = {}
        # Size of every container variable in the previous frame, by value or not.
        self._sizes: dict[str, tuple[bool, int]] = {}
        # Per pass that drew random numbers (as the innermost running part): the
        # hidden values (numpy arrays, the policy's own objects) that changed.
        self.random_changes: list[frozenset[str]] = []
        # Unrecordable containers with containers inside: their last skeleton.
        # They count as omitted only once their skeleton changes in the run.
        self._skeletons: dict[str, Any] = {}

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
        members: dict[str, tuple[int, bool, bool, bool]] = {}
        sizes: dict[str, tuple[bool, int]] = {}
        for name, value in values.items():
            if name.startswith("__"):
                continue
            if type(value) in _CONTAINERS:
                sizes[name] = (False, len(value))
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
            elif holds_mutable(value) and not self._follow(name, value, members, sizes):
                self._compare_skeleton(name, value, sizes)
        self._settle(values, members, sizes)
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

    def _follow(self, name: str, value, members: dict, sizes: dict) -> bool:
        """Follow an unrecordable flat container by item identity.

        False when it must count as omitted: an item is itself a container, or
        the variable held plain items in the previous frame.
        """
        held = _members(value)
        was_opaque, size = self._sizes.get(name, (True, 0))
        if held is None or (not was_opaque and size) or name in self._skeletons:
            return False
        before = self._members.get(name, Counter())
        members[name] = (
            len(value),
            bool(held - before),
            bool(before - held),
            type(value) in (list, deque),
        )
        self._members[name] = held
        sizes[name] = (True, len(value))
        return True

    def _compare_skeleton(self, name: str, value, sizes: dict) -> None:
        """Unrecorded state that changed is not evidence of absence: omitted."""
        skeleton = _skeleton(value)
        sizes[name] = (True, len(value))
        was_opaque, size = self._sizes.get(name, (True, 0))
        old = self._skeletons.get(name)
        if (
            (not was_opaque and size)
            or name in self._members
            or (old is not None and old != skeleton)
        ):
            self.omitted_containers.add(name)
        self._skeletons[name] = skeleton

    def _settle(self, values: dict, members: dict, sizes: dict) -> None:
        """Close the frame's identity records: containers that dropped objects."""
        for name, (opaque, size) in sizes.items():
            was_opaque, old = self._sizes.get(name, (opaque, 0))
            if not was_opaque or opaque or not old:
                continue
            if size:
                # Objects replaced by plain items: not comparable either way.
                self.omitted_containers.add(name)
            else:
                members[name] = (0, False, True, type(values[name]) in (list, deque))
        self._members = {
            k: v for k, v in self._members.items() if k in members and members[k][0]
        }
        self._sizes = sizes
        self.members.append(members)

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

    def finish(self) -> None:
        """Drop what only encoding needs (previous values, raw copies, the
        conversion memo); recorded frames stay readable."""
        self._previous = {}
        self._raw = {}
        self._memo = {}
        self._ids = {}
        self._members = {}
        self._sizes = {}
        self._skeletons = {}

    def release(self) -> None:
        """Drop everything, frames included, once a call's analysis is done."""
        self.finish()
        self.records.clear()
        self.replaced.clear()
        self.members.clear()
        self.random_changes.clear()


@dataclass
class RunRecord:
    """One execution of one loop inside one harness call."""

    site: str
    ordinal: int
    methods: tuple[str, ...]
    capture: RunCapture = field(default_factory=RunCapture)
    visits: int = 0
    exited: bool = False
    # Open calls of a recursive function's run (its "iterations" are calls).
    depth: int = 0
    # Runs started and frames recorded in the call before this run started.
    started_after: tuple[int, int] = (0, 0)
    # Random numbers drawn while this run was the innermost running part, since
    # its last checkpoint; and the hidden state at that checkpoint.
    drew: bool = False
    hidden: dict | None = None


class Session:
    """Everything recorded during one harness call or one probe re-execution."""

    def __init__(
        self,
        capture: bool = True,
        edit: SearchEdit | None = None,
        clock: float | None = None,
    ) -> None:
        self.capture = capture
        self.edit = edit
        # A re-executed call starts from its original call's clock reading.
        self.clock = clock
        self.runs: list[RunRecord] = []
        self.stack: list[str] = []
        self.edits = 0
        self._ordinals: dict[str, int] = {}
        # The run of each recursive function with calls still open.
        self.recursions: dict[str, RunRecord] = {}
        # Frames recorded so far, by every run of this call.
        self.frames = 0
        # Running parts, innermost last; random generators seen in this call.
        self.active: list[RunRecord] = []
        self.generators: dict[int, Any] = {}
        self.random_state: tuple | None = None

    def start(self, site: str) -> RunRecord:
        """Begin a run; ordinals number this site's runs within the call."""
        ordinal = self._ordinals.get(site, 0)
        self._ordinals[site] = ordinal + 1
        run = RunRecord(site, ordinal, tuple(self.stack))
        # An enclosing run still recording later is compared only up to here.
        run.started_after = (len(self.runs), self.frames)
        self.runs.append(run)
        return run

    def record(self, run: RunRecord, kind: str, values: dict) -> None:
        """One frame of one run, counted for the call."""
        run.capture.record(kind, values)
        self.frames += 1


class Recorder:
    """The injected global; dispatches to the innermost active session."""

    def __init__(self) -> None:
        self.sessions: list[Session] = []
        self.random = RandomWatch()

    def _moved(self, session: Session) -> None:
        """Random numbers drawn since the last checkpoint belong to the innermost
        running part (a planner's loop, not the loop that called the planner)."""
        state = self.random.state(session.generators)
        if session.random_state is not None and state != session.random_state:
            while session.active and session.active[-1].exited:
                session.active.pop()
            if session.active:
                session.active[-1].drew = True
        session.random_state = state

    def _pass(self, session: Session, run: RunRecord, values: dict) -> None:
        """At a checkpoint of `run`: did the pass that just ended draw random
        numbers while state the snapshots do not hold changed?"""
        if run.visits <= 1:
            known = len(session.generators)
            self.random.learn(values, session.generators)
            if len(session.generators) != known:
                # New generators join the baseline; finding one is not a draw.
                session.random_state = self.random.state(session.generators)
        if not run.drew and run.site not in self.random.sites:
            run.hidden = None
            return
        hidden = hidden_state(values)
        if run.drew:
            self.random.sites.add(run.site)
            if run.hidden is not None:
                changed = frozenset(
                    k
                    for k, v in hidden.items()
                    if k in run.hidden and run.hidden[k] != v
                )
                if changed:
                    run.capture.random_changes.append(changed)
        run.hidden = hidden
        run.drew = False

    @staticmethod
    def _stop(session: Session, run: RunRecord) -> None:
        """The run is no longer running."""
        run.hidden = None
        if run in session.active:
            session.active.remove(run)

    def begin(self, method: str) -> None:
        """Method entry, for attributing runs to helpers and harness calls."""
        if self.sessions:
            self.sessions[-1].stack.append(method)

    def end(self) -> None:
        """Method exit, including exits by exception."""
        if self.sessions and self.sessions[-1].stack:
            self.sessions[-1].stack.pop()

    def library(self, site: str, function):
        """A library graph search is about to run: one (unreadable) run of it."""
        if self.sessions:
            self.sessions[-1].start(site).exited = True
        return function

    def call(self, site: str, values: dict) -> RunRecord | None:
        """A recursive function was called: its arguments are one visit.

        The outermost call starts a run; calls nested inside it are its later
        visits, like the iterations of a loop.
        """
        if not self.sessions:
            return None
        session = self.sessions[-1]
        if session.capture:
            self._moved(session)
        run = session.recursions.get(site)
        if run is None or run.depth == 0:
            run = session.recursions[site] = session.start(site)
            session.active.append(run)
        run.depth += 1
        run.visits += 1
        if session.capture:
            session.record(run, "visit", values)
            self._pass(session, run, values)
        return run

    def returned(self, run: RunRecord | None) -> None:
        """A call of a recursive function ended (by return or exception)."""
        if run is None:
            return
        run.depth -= 1
        if run.depth == 0:
            # Only call entries are recorded: an exit frame would hold the
            # outermost call's own locals next to the deepest call's arguments.
            session = self.sessions[-1]
            if session.capture:
                self._moved(session)
            self._stop(session, run)
            run.exited = True
            run.capture.finish()

    def enter(self, site: str) -> RunRecord | None:
        """A loop statement is about to run: one new run of this part."""
        if not self.sessions:
            return None
        session = self.sessions[-1]
        if session.capture:
            self._moved(session)
        run = session.start(site)
        session.active.append(run)
        return run

    def visit(self, run: RunRecord | None, values: dict) -> None:
        """Before every loop test, including the final false test."""
        if run is None:
            return
        session = self.sessions[-1]
        index = run.visits
        run.visits += 1
        edit = session.edit
        if session.capture:
            self._moved(session)
            session.record(run, "visit", values)
            self._pass(session, run, values)
        if (
            edit is not None
            and edit.site == run.site
            and edit.ordinal == run.ordinal
            and edit.visit == index
        ):
            apply_edit(values, edit)
            session.edits += 1
            if session.capture:
                session.record(run, "edited", values)

    def leave(self, run: RunRecord | None, values: dict, kind: str = "exit") -> None:
        """Loop termination: a false test or break ("exit"), or a return inside it."""
        if run is None:
            return
        session = self.sessions[-1]
        if session.capture:
            self._moved(session)
            self._pass(session, run, values)
        self._stop(session, run)
        run.exited = True
        if session.capture:
            session.record(run, kind, values)
            # A run is never recorded again after it ends.
            run.capture.finish()


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
        """`return value` ends the loop's run here, as the method's algorithm does."""
        leave = ast.parse(f"{RECORDER}.leave({self.token}, locals(), 'return')").body[0]
        ast.copy_location(leave, node)
        return [leave, node]

    def _skip(self, node):
        # A nested function's returns do not end this loop.
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = _skip  # type: ignore[assignment]
    visit_ClassDef = visit_Lambda = _skip  # type: ignore[assignment]


# Library routines that are graph searches by definition (path searches only).
LIBRARY_SEARCHES = {
    "scipy.sparse.csgraph": {
        "dijkstra",
        "shortest_path",
        "bellman_ford",
        "johnson",
        "floyd_warshall",
        "yen",
        "breadth_first_order",
        "depth_first_order",
        "breadth_first_tree",
        "depth_first_tree",
    },
    "networkx": {
        "shortest_path",
        "shortest_path_length",
        "all_shortest_paths",
        "astar_path",
        "astar_path_length",
        "dijkstra_path",
        "dijkstra_path_length",
        "bidirectional_dijkstra",
        "single_source_dijkstra",
        "single_source_dijkstra_path",
        "single_source_dijkstra_path_length",
        "multi_source_dijkstra",
        "bellman_ford_path",
        "bfs_tree",
        "dfs_tree",
        "bfs_edges",
        "dfs_edges",
        "has_path",
    },
    "skimage.graph": {
        "route_through_array",
        "MCP",
        "MCP_Geometric",
        "MCP_Connect",
        "MCP_Flexible",
        "shortest_path",
    },
}


def import_aliases(tree: ast.AST) -> dict[str, str]:
    """Local names bound by imports anywhere in the module, to their full paths."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    head = alias.name.split(".")[0]
                    aliases[head] = head
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def library_search(func: ast.expr, aliases: dict[str, str]) -> str | None:
    """The full name of a library graph search this callee resolves to, if any."""
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name) or func.id not in aliases:
        return None
    full = ".".join([aliases[func.id], *reversed(parts)])
    for package, names in LIBRARY_SEARCHES.items():
        if full.startswith(package + ".") and full.rsplit(".", 1)[1] in names:
            return full
    return None


def _iterated(iterable) -> str | None:
    """The variable a `for` loop iterates directly (`xs`, or `enumerate(xs)`).

    Python reads such a list lazily, by position, so items appended during the
    loop are visited too.
    """
    if (
        isinstance(iterable, ast.Call)
        and isinstance(iterable.func, ast.Name)
        and iterable.func.id == "enumerate"
        and len(iterable.args) == 1
    ):
        iterable = iterable.args[0]
    return iterable.id if isinstance(iterable, ast.Name) else None


def _names(target) -> list[str]:
    """Plain names bound by a loop target (`i`, `(x, y)`, `[a, b]`)."""
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]


# Expressions that build a new value (never an alias of an existing container).
_NEW_VALUES = (
    ast.Call,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.JoinedStr,
)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _scope_walk(node):
    """Nodes under `node` that belong to its own function (no nested scopes)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(
            c for c in ast.iter_child_nodes(current) if not isinstance(c, _SCOPES)
        )


def read_names(node) -> list[str]:
    """Variables read anywhere in a loop or function (its own scope only)."""
    return sorted(
        {
            n.id
            for n in _scope_walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
    )


def _mentioned(node) -> set[str]:
    return {n.id for n in _scope_walk(node) if isinstance(n, ast.Name)}


def _rebound(target) -> list[str]:
    """Names a plain assignment target rebinds (not `a[i] = ...` or `a.x = ...`)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _rebound(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names = [_rebound(t) for t in target.elts]
        return [n for group in names for n in group] if all(names) else []
    return []


def _self_recursive(function: ast.FunctionDef) -> bool:
    """Calls itself by name, or as `self.name(...)` / `cls.name(...)`."""
    for node in _scope_walk(function):
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            return False
    for node in _scope_walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == function.name:
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr == function.name
            and isinstance(func.value, ast.Name)
            and func.value.id in ("self", "cls")
        ):
            return True
    return False


def _split_docstring(body: list) -> tuple[list, list]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return [body[0]], body[1:]
    return [], body


def _loads(node) -> set[str]:
    """Variables an expression or statement reads (`x += 1` and `x[i] = v` read x)."""
    names = {
        n.id
        for n in _scope_walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    if isinstance(node, ast.AugAssign):
        names.update(_names(node.target))
    return names


def _root(value) -> str | None:
    """`x`, `x.a` or `x[k]` (any depth) is rooted at x; other values have no root."""
    while isinstance(value, (ast.Attribute, ast.Subscript)):
        value = value.value
    return value.id if isinstance(value, ast.Name) else None


def _resets(statement, stored: set) -> set[str]:
    """Variables a simple statement surely rebinds to a value of this iteration.

    Rebinding to `x`, `x.a` or `x[k]` where x is not assigned in the loop is an
    alias of outside state (`q = self.queue`), so it does not count.
    """
    if isinstance(statement, ast.Assign):
        value, targets = statement.value, statement.targets
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        value, targets = statement.value, [statement.target]
    else:
        return set()
    root = _root(value)
    if root is not None and root not in stored:
        return set()
    return {n for t in targets for n in _rebound(t)}


def _late_reads(block: list, assigned: set, stored: set, found: tuple) -> None:
    """Variables read in `block` before this iteration surely rebinds them.

    `found` is (late, rebound): reads not preceded by a rebinding, and every
    variable some statement rebinds (for-statement variables included).
    """
    late, rebound = found
    current = set(assigned)

    def check(*nodes) -> None:
        for node in nodes:
            if node is not None:
                late.update(_loads(node) - current)

    for statement in block:
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            check(statement.iter)
            rebound.update(_names(statement.target))
            inner = current | set(_names(statement.target))
            _late_reads(statement.body, inner, stored, found)
            _late_reads(statement.orelse, current, stored, found)
        elif isinstance(statement, (ast.While, ast.If)):
            check(statement.test)
            _late_reads(statement.body, current, stored, found)
            _late_reads(statement.orelse, current, stored, found)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            check(*(item.context_expr for item in statement.items))
            bound = {
                n
                for item in statement.items
                if item.optional_vars is not None
                for n in _rebound(item.optional_vars)
            }
            _late_reads(statement.body, current | bound, stored, found)
        elif isinstance(statement, ast.Try):
            _late_reads(statement.body, current, stored, found)
            for handler in statement.handlers:
                check(handler.type)
                named = {handler.name} if handler.name else set()
                _late_reads(handler.body, current | named, stored, found)
            _late_reads(statement.orelse, current, stored, found)
            _late_reads(statement.finalbody, current, stored, found)
        elif isinstance(statement, _SCOPES):
            continue
        else:
            check(statement)
            resets = _resets(statement, stored)
            rebound |= resets
            current |= resets


def iteration_local(loop: ast.For | ast.While) -> set[str]:
    """Variables every read of which, in one iteration, follows a rebinding of
    them earlier in that iteration (in the same block or an enclosing one).

    Whatever they held at the end of the previous iteration is never read, so
    they cannot carry items from one iteration to the next.
    """
    stored = {
        n.id
        for n in _scope_walk(loop)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    late: set = set()
    if isinstance(loop, ast.While):
        late |= _loads(loop.test)
        start: set = set()
    else:
        start = set(_names(loop.target))
    rebound = set(start)
    _late_reads(loop.body, start, stored, (late, rebound))
    return rebound - late


def fresh_names(loop: ast.For | ast.While) -> list[str]:
    """Variables that cannot carry items from one iteration of `loop` to the next.

    A `for` statement's variables (the loop's own and nested ones) are set by
    their iterator. A variable whose first use in the body is a top-level
    assignment of a newly built value (a call, a literal, a comprehension, an
    operator result) drops whatever it held: it is a per-iteration value. So
    does any variable that is iteration-local (see `iteration_local`).
    """
    fresh = {
        name
        for node in _scope_walk(loop)
        if isinstance(node, (ast.For, ast.AsyncFor))
        for name in _names(node.target)
    }
    seen = _mentioned(loop.test) if isinstance(loop, ast.While) else set()
    for statement in loop.body:
        value, targets = None, []
        if isinstance(statement, ast.Assign):
            value, targets = statement.value, statement.targets
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            value, targets = statement.value, [statement.target]
        if isinstance(value, _NEW_VALUES):
            used = seen | _mentioned(value)
            fresh.update(n for t in targets for n in _rebound(t) if n not in used)
        seen |= _mentioned(statement)
    return sorted(fresh | iteration_local(loop))


class _Transformer(ast.NodeTransformer):
    """Wrap loops and class methods; the original statements are untouched.

    Every `while` loop is a part. A `for` loop is a part when no other loop of the
    same function encloses it: an inner `for` (a search's neighbour loop) would
    otherwise start a recorded run on every expansion, and the loop around it
    already sees its effects.
    """

    def __init__(self, source: str, aliases: dict[str, str] | None = None) -> None:
        self.lines = source.splitlines()
        self.aliases = aliases or {}
        self.sites: dict[str, dict] = {}
        self.functions: list[str] = []
        # Enclosing loop sites (None for an unwatched loop) of the current function.
        self.loops: list[str | None] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):  # pylint: disable=invalid-name
        """Track the enclosing function; a self-recursive function is a part.

        Its calls are its iterations: each call's arguments are recorded on
        entry, in one run per outermost call.
        """
        recursive = isinstance(node, ast.FunctionDef) and _self_recursive(node)
        self.functions.append(node.name)
        site = None
        if recursive:
            site = f"{node.lineno}:{node.col_offset}:Recursion"
            self.sites[site] = {
                "kind": "recursion",
                "line": node.lineno,
                "function": ".".join(self.functions),
                "source": self.lines[node.lineno - 1].strip(),
                "parent": None,
                "fresh": [],
                "read": read_names(node),
            }
        outer, self.loops = self.loops, []
        self.generic_visit(node)
        self.loops = outer
        self.functions.pop()
        if site is not None:
            token = f"__call_{node.lineno}_{node.col_offset}__"
            enter = ast.parse(f"{token} = {RECORDER}.call({site!r}, locals())").body[0]
            leave = ast.parse(f"{RECORDER}.returned({token})").body[0]
            docstring, body = _split_docstring(node.body)
            guarded = ast.Try(body or [ast.Pass()], [], [], [leave])
            for statement in (enter, guarded, leave):
                ast.copy_location(statement, node)
            node.body = [*docstring, enter, guarded]
        return node

    visit_ClassDef = visit_FunctionDef  # type: ignore[assignment]
    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _site(self, node, kind: str, **extra) -> str:
        site = f"{node.lineno}:{node.col_offset}:{kind}"
        parents = [s for s in self.loops if s]
        self.sites[site] = {
            "kind": "loop",
            "loop": kind.lower(),
            "line": node.lineno,
            "function": ".".join(self.functions),
            "source": self.lines[node.lineno - 1].strip(),
            "parent": parents[-1] if parents else None,
            "fresh": fresh_names(node),
            "read": read_names(node),
            **extra,
        }
        return site

    def visit_Call(self, node: ast.Call):  # pylint: disable=invalid-name
        """A library graph search becomes a site recorded each time it runs."""
        self.generic_visit(node)
        full = library_search(node.func, self.aliases)
        if full is None:
            return node
        site = f"{node.lineno}:{node.col_offset}:Library"
        parents = [s for s in self.loops if s]
        self.sites[site] = {
            "kind": "library",
            "call": full,
            "line": node.lineno,
            "function": ".".join(self.functions),
            "source": self.lines[node.lineno - 1].strip(),
            "parent": parents[-1] if parents else None,
        }
        node.func = ast.Call(
            ast.Attribute(ast.Name(RECORDER, ast.Load()), "library", ast.Load()),
            [ast.Constant(site), node.func],
            [],
        )
        return node

    def visit_For(self, node: ast.For):  # pylint: disable=invalid-name
        """Before each iteration's body, at exit, and at a return inside it."""
        if self.loops:
            self.loops.append(None)
            self.generic_visit(node)
            self.loops.pop()
            return node
        site = self._site(node, "For", iterates=_iterated(node.iter))
        self.loops.append(site)
        self.generic_visit(node)
        self.loops.pop()
        token = f"__run_{node.lineno}_{node.col_offset}__"
        start = ast.parse(f"{token} = {RECORDER}.enter({site!r})").body[0]
        visit = ast.parse(f"{RECORDER}.visit({token}, locals())").body[0]
        leave = ast.parse(f"{RECORDER}.leave({token}, locals())").body[0]
        rewriter = _ReturnCheckpoints(token)
        body = [rewriter.visit(statement) for statement in node.body]
        body = [
            s for item in body for s in (item if isinstance(item, list) else [item])
        ]
        node.body = [visit, *body]
        result = [start, node, leave]
        for statement in result:
            ast.copy_location(statement, node)
        return result

    def visit_While(self, node: ast.While):  # pylint: disable=invalid-name
        """Before each test, after exit, and a fresh run on every loop entry."""
        site = self._site(node, "While")
        self.loops.append(site)
        self.generic_visit(node)
        self.loops.pop()
        if node.orelse:
            raise UnsupportedProgram("Loop else is outside the policy runner")
        token = f"__run_{node.lineno}_{node.col_offset}__"
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
    docstring, body = _split_docstring(function.body)
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
        transformer = _Transformer(source, import_aliases(tree))
        transformed = transformer.visit(transformed)
        for node in transformed.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        _wrap_method(item)
        self.code = compile(ast.fix_missing_locations(transformed), FILENAME, "exec")
        self.sites = transformer.sites
        self.recorder = Recorder()
        self.clock = FakeClock()

    def policy_class(self, instrumented: bool = True) -> type:
        """Execute the module in fresh globals and return the policy class."""
        namespace: dict[str, Any] = {
            "__name__": POLICY_MODULE,
            "__file__": FILENAME,
            "__builtins__": {
                **builtins.__dict__,
                "__import__": self.clock.importer(builtins.__import__),
            },
            RECORDER: self.recorder,
        }
        exec(  # pylint: disable=exec-used
            self.code if instrumented else self.plain, namespace
        )
        if instrumented:
            self.recorder.random.namespace = namespace
        return namespace[self.class_name]

    def call(self, policy, method: str, args: tuple, session: Session):
        """Invoke one harness method under a session; always restore the stack."""
        self.recorder.sessions.append(session)
        resumed = self.clock.now
        if session.clock is not None:
            self.clock.now = session.clock
        try:
            return getattr(policy, method)(*args)
        finally:
            self.recorder.sessions.pop()
            if session.clock is not None:
                # A re-execution never moves the episode's own clock.
                self.clock.now = resumed


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
    # The fake clock's reading when the call started.
    clock: float = FakeClock.START


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
    program.clock.reset()
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
        started = program.clock.now
        result = program.call(policy, method, args, session)
        record = CallRecord(
            index,
            method,
            session.runs,
            plans_before,
            plan_attributes(policy),
            None if result is None else action_tuple(result),
            started,
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
    reference = run_episode(program, domain, case, instrumented=False)
    observed = run_episode(program, domain, case)
    if [c.action for c in reference.calls] != [c.action for c in observed.calls]:
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
