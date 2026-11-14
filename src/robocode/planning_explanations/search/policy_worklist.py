"""Conservative worklist warnings for loops the search grammar cannot bind.

Look for container churn or a growing sequence with an advancing integer that
could be its read cursor. Either makes the loop unreadable (unknown). Snapshot
omissions and inspection limits do too. These are bounded representation checks,
not a proof that arbitrary programs without these signatures cannot search.

Churn means a container both gains and loses items. A variable that is only ever
replaced by a new container of the same size (a point or pose rebuilt each
iteration) is a value, not a worklist. A container edited in place, or replaced
by one of another size (a level-by-level BFS frontier), can still be one.
Integers are leaves: they are kept as possible cursors but never cut inspection.
Tuples that hold containers are looked into; tuples of plain values are leaves.
A container reached by position in a list is compared across two snapshots only
if that list kept its length: after `items = items[1:]`, `items[0]` is another
element, not the same container edited.

A container that cannot be recorded by value (its items are objects) but holds
no lists or dictionaries is followed by item identity: the recorder notes its size
and whether it gained or lost items, and the same churn and cursor tests apply.

A wavefront search keeps its reached set as a boolean grid: a 2-D grid whose True
cells, over a whole run, change only by growing by their own neighbours (each new
cell next to an old one), at least three times, is the grid form of a worklist.
Stamping shapes into a map does not keep growing by neighbours.

A `for` loop over a list it appends to (`for node in queue: queue.append(...)`)
reads that list with its own hidden cursor, so the list growing while it is
iterated is the same sign: breadth-first search written as a for loop.

A loop that draws random numbers (as the innermost running part) while state the
snapshots do not hold changes in the same pass (a numpy array, an attribute of the
policy's own objects) may be a sampling planner whose tree we cannot see, such as
an RRT kept in preallocated arrays or a `Tree` object (`random_with_hidden_state`).

`read` (if given) names the variables the loop's code reads: a read cursor is
read, so an integer whose variable the loop never reads is not one.

`ignore` names variables that cannot hold this loop's worklist: ones rebuilt at
the start of every iteration or set by a `for` statement (they carry nothing from
one iteration to the next), and ones an explained search nested inside the loop
already accounts for.
"""

# Exact builtin types define the supported snapshot language.
# pylint: disable=unidiomatic-typecheck

import json
from collections import Counter, deque

from .policy_runner import boolean_grid

SCAN_DEPTH = 4
SCAN_LIMIT = 4096
CONTAINERS = (list, dict, deque, set)


def _key(item) -> object:
    try:
        hash(item)
    except TypeError:
        return repr(item)
    return item


def _holds_container(value) -> bool:
    """A container, or a tuple that (at any depth) holds one: may hide a worklist."""
    if type(value) in CONTAINERS:
        return True
    return type(value) is tuple and any(_holds_container(v) for v in value)


def _items(value) -> Counter:
    """A container's items as a multiset (a dictionary's keys)."""
    return Counter(_key(item) for item in value)


def _scan(values: dict, depth: int, limit: int, parents: dict | None = None) -> tuple:
    """Container contents, sequence lengths, integer candidates and truncation.

    `parents` (if given) receives each nested path's parent path and whether
    it was reached by position in a list or deque.
    """
    queue = deque((name, value, 0) for name, value in sorted(values.items()))
    found: dict[str, Counter] = {}
    lengths: dict[str, int] = {}
    integers: dict[str, int] = {}
    links = {} if parents is None else parents
    cut = False
    while queue:
        path, value, level = queue.popleft()
        if type(value) is int:
            integers[path] = value
            continue
        if type(value) is tuple:
            # A record of containers is looked into; it cannot churn itself.
            inner = [
                (f"{path}[{i}]", v) for i, v in enumerate(value) if _holds_container(v)
            ]
            if inner and level >= depth:
                cut = True
            elif inner:
                links.update((p, (path, False)) for p, _ in inner)
                queue.extend((p, v, level + 1) for p, v in inner)
            continue
        if type(value) not in CONTAINERS:
            continue
        if len(found) >= limit:
            return found, lengths, integers, True
        found[path] = _items(value)
        if type(value) in (list, deque):
            lengths[path] = len(value)
        if type(value) is dict:
            children = [
                (f"{path}[{json.dumps(k)}]", v)
                for k, v in value.items()
                if type(k) in (str, int)
            ]
        elif type(value) is set:
            children = []
        else:
            children = [(f"{path}[{i}]", v) for i, v in enumerate(value)]
        # Integers need no further descent, so they never count toward the limits.
        integers.update((p, v) for p, v in children if type(v) is int)
        nested = [(p, v) for p, v in children if _holds_container(v)]
        if nested and level >= depth:
            cut = True
            continue
        positional = type(value) in (list, deque)
        links.update((p, (path, positional)) for p, _ in nested)
        queue.extend((p, v, level + 1) for p, v in nested)
    return found, lengths, integers, cut


def _shifted(path: str, parents: dict, lengths: dict, old_lengths: dict) -> bool:
    """Reached by position in a list that changed length: a different element."""
    while path in parents:
        parent, positional = parents[path]
        if positional and lengths.get(parent) != old_lengths.get(parent):
            return True
        path = parent
    return False


def containers(values: dict, depth: int = SCAN_DEPTH, limit: int = SCAN_LIMIT):
    """Container item multisets by path, and whether inspection was incomplete."""
    found, _, _, cut = _scan(values, depth, limit)
    return found, cut


GROWTH_STEPS = 3


def _neighbours(grid):
    """Cells next to (or on) a True cell of a 2-D boolean grid, 8-connected."""
    near = grid.copy()
    near[1:, :] |= grid[:-1, :]
    near[:-1, :] |= grid[1:, :]
    near[:, 1:] |= grid[:, :-1]
    near[:, :-1] |= grid[:, 1:]
    near[1:, 1:] |= grid[:-1, :-1]
    near[:-1, :-1] |= grid[1:, 1:]
    near[1:, :-1] |= grid[:-1, 1:]
    near[:-1, 1:] |= grid[1:, :-1]
    return near


def _grew_by_neighbours(before, grid) -> bool:
    """True cells only added, at least one, each next to an earlier True cell."""
    if before.shape != grid.shape:
        return False
    added = grid & ~before
    return (
        bool(added.any())
        and not bool((before & ~grid).any())
        and not bool((added & ~_neighbours(before)).any())
    )


def _member_churn(
    opaque: dict, old: dict, previous: dict, replaced, sets
) -> str | None:
    """The churn test on containers followed by item identity."""
    gained, lost, stored, growing = sets
    for name, (size, plus, minus, sequence) in opaque.items():
        if name in old:
            old_size = old[name][0]
        elif name in previous:
            old_size = sum(previous[name].values())
        else:
            continue
        if plus:
            gained.add(name)
        if minus:
            lost.add(name)
        if (plus or minus) and (name not in replaced or size != old_size):
            stored.add(name)
        if name in gained and name in lost and name in stored:
            return name
        if sequence and size > old_size:
            growing.add(name)
    return None


def _grid_changes(values: dict, grids: dict, growth: Counter, stamped: set) -> None:
    """Count neighbour-growth steps of boolean grids; note any other change."""
    for name, grid in values.items():
        before = grids.get(name)
        if before is None or not boolean_grid(grid) or grid.ndim != 2:
            continue
        if _grew_by_neighbours(before, grid):
            growth[name] += 1
        elif before.shape != grid.shape or (before != grid).any():
            stamped.add(name)


def _iterated_grows(run, name: str) -> bool:
    """The list a `for` loop iterates gains items while it is being iterated."""
    members = run.capture.members
    old = None
    for index, ((kind, frame), replaced) in enumerate(
        zip(run.capture.frames(), run.capture.replaced)
    ):
        if old is not None and name in replaced:
            # Rebound: later frames no longer show the list being iterated.
            return False
        value = frame.get(name)
        entry = members[index].get(name) if index < len(members) else None
        if type(value) is list:
            size = len(value)
        else:
            size = entry[0] if entry is not None and entry[3] else None
        if kind != "edited" and None not in (old, size) and size > old:
            return True
        old = size
    return False


def worklist_signature(
    run,
    depth: int = SCAN_DEPTH,
    limit: int = SCAN_LIMIT,
    ignore: frozenset = frozenset(),
    read: frozenset | None = None,
    iterated: str | None = None,
):
    """Churn, a growing sequence with a possible read cursor, or a wavefront.

    `iterated` is the variable a `for` loop iterates directly, if any.
    """
    gained: set = set()
    lost: set = set()
    # Containers changed in place or resized: not a value rebuilt each iteration.
    # Only variables can be seen to be replaced; nested containers always count.
    stored: set = set()
    previous = None
    old_lengths: dict = {}
    old_integers: dict = {}
    growing: set = set()
    omitted = sorted(set(run.capture.omitted_containers) - set(ignore))
    cut = bool(omitted)
    if iterated is not None and _iterated_grows(run, iterated):
        return {
            "worklist": iterated,
            "kind": "iterated_list_grows",
            "cut": cut,
            "omitted": omitted,
        }
    grids: dict = {}
    growth: Counter = Counter()
    # Grids that changed other than by growing by neighbours in this run.
    stamped: set = set()
    members = run.capture.members
    old_members: dict = {}
    for index, ((kind, frame), replaced) in enumerate(
        zip(run.capture.frames(), run.capture.replaced)
    ):
        values = {k: v for k, v in frame.items() if k not in ignore}
        opaque = {
            k: v
            for k, v in (members[index] if index < len(members) else {}).items()
            if k not in ignore
        }
        parents: dict = {}
        current, lengths, integers, clipped = _scan(values, depth, limit, parents)
        cut = cut or clipped
        if previous is not None and kind != "edited":
            shifted = {p for p in current if _shifted(p, parents, lengths, old_lengths)}
            for path, items in current.items():
                before = previous.get(path)
                if before is None or items == before or path in shifted:
                    continue
                if items - before:
                    gained.add(path)
                if before - items:
                    lost.add(path)
                if path not in replaced or sum(items.values()) != sum(before.values()):
                    stored.add(path)
                if path in gained and path in lost and path in stored:
                    return {"worklist": path, "cut": cut, "omitted": omitted}
            churned = _member_churn(
                opaque, old_members, previous, replaced, (gained, lost, stored, growing)
            )
            if churned is not None:
                return {"worklist": churned, "cut": cut, "omitted": omitted}
            lengths.update((k, v[0]) for k, v in opaque.items() if v[3])
            growing.update(
                p
                for p, size in lengths.items()
                if p in old_lengths and size > old_lengths[p] and p not in shifted
            )
            # Integers that moved forward since the last frame: possible cursors.
            advanced = [
                (name, start, value)
                for name, value in integers.items()
                if (start := old_integers.get(name)) is not None
                and 0 <= start < value
                and (read is None or name.split("[", 1)[0] in read)
            ]
            for path in sorted(growing & lengths.keys() & old_lengths.keys()):
                for name, start, value in advanced:
                    # A read cursor trails the end while items wait to be read;
                    # a counter equal to the length has nothing left to read.
                    if value < lengths[path] and start <= old_lengths[path]:
                        return {
                            "worklist": path,
                            "cursor": name,
                            "kind": "growing_sequence_with_cursor",
                            "cut": cut,
                            "omitted": omitted,
                        }
            _grid_changes(values, grids, growth, stamped)
        else:
            lengths.update((k, v[0]) for k, v in opaque.items() if v[3])
        grids = {k: v for k, v in values.items() if boolean_grid(v)}
        previous = current
        old_lengths, old_integers, old_members = lengths, integers, opaque
    for name, steps in sorted(growth.items()):
        if steps >= GROWTH_STEPS and name not in stamped:
            return {
                "worklist": name,
                "kind": "growing_reached_grid",
                "cut": cut,
                "omitted": omitted,
            }
    for names in getattr(run.capture, "random_changes", ()):
        kept = sorted(n for n in names if n.split(".", 1)[0] not in ignore)
        if kept:
            return {
                "worklist": kept[0],
                "kind": "random_with_hidden_state",
                "cut": cut,
                "omitted": omitted,
            }
    return {"worklist": None, "cut": cut, "omitted": omitted}


def scan_runs(
    runs,
    ignore: frozenset = frozenset(),
    read: frozenset | None = None,
    iterated: str | None = None,
) -> dict:
    """Combine the signature over every run of one part."""
    cut = False
    omitted: set[str] = set()
    for run in runs:
        signature = worklist_signature(run, ignore=ignore, read=read, iterated=iterated)
        if signature["worklist"] is not None:
            return {**signature, "run": run.ordinal}
        cut = cut or signature["cut"]
        omitted.update(signature["omitted"])
    return {"worklist": None, "cut": cut, "omitted": sorted(omitted)}


def unreadable_reason(signature: dict) -> str | None:
    """Why a loop without a binding cannot be counted as zero, if it cannot."""
    if signature["worklist"] is not None:
        return signature.get("kind", "worklist_without_binding")
    if signature.get("omitted"):
        return "snapshot_omission"
    if signature["cut"]:
        return "inspection_cut"
    return None
