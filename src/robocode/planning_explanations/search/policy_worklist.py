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
"""

# Exact builtin types define the supported snapshot language.
# pylint: disable=unidiomatic-typecheck

import json
from collections import Counter, deque

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


def _scan(values: dict, depth: int, limit: int) -> tuple:
    """Container contents, sequence lengths, integer candidates and truncation."""
    queue = deque((name, value, 0) for name, value in sorted(values.items()))
    found: dict[str, Counter] = {}
    lengths: dict[str, int] = {}
    integers: dict[str, int] = {}
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
        queue.extend((p, v, level + 1) for p, v in nested)
    return found, lengths, integers, cut


def containers(values: dict, depth: int = SCAN_DEPTH, limit: int = SCAN_LIMIT):
    """Container item multisets by path, and whether inspection was incomplete."""
    found, _, _, cut = _scan(values, depth, limit)
    return found, cut


def worklist_signature(run, depth: int = SCAN_DEPTH, limit: int = SCAN_LIMIT):
    """Churn or a growing sequence with a possible advancing read cursor."""
    gained: set = set()
    lost: set = set()
    # Containers changed in place or resized: not a value rebuilt each iteration.
    # Only variables can be seen to be replaced; nested containers always count.
    stored: set = set()
    previous = None
    old_lengths: dict = {}
    old_integers: dict = {}
    growing: set = set()
    omitted = sorted(run.capture.omitted_containers)
    cut = bool(omitted)
    for (kind, values), replaced in zip(run.capture.frames(), run.capture.replaced):
        current, lengths, integers, clipped = _scan(values, depth, limit)
        cut = cut or clipped
        if previous is not None and kind != "edited":
            for path, items in current.items():
                before = previous.get(path)
                if before is None or items == before:
                    continue
                if items - before:
                    gained.add(path)
                if before - items:
                    lost.add(path)
                if path not in replaced or sum(items.values()) != sum(before.values()):
                    stored.add(path)
                if path in gained and path in lost and path in stored:
                    return {"worklist": path, "cut": cut, "omitted": omitted}
            growing.update(
                p
                for p, size in lengths.items()
                if p in old_lengths and size > old_lengths[p]
            )
            # Integers that moved forward since the last frame: possible cursors.
            advanced = [
                (name, start, value)
                for name, value in integers.items()
                if (start := old_integers.get(name)) is not None and 0 <= start < value
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
        previous = current
        old_lengths, old_integers = lengths, integers
    return {"worklist": None, "cut": cut, "omitted": omitted}


def scan_runs(runs) -> dict:
    """Combine the signature over every run of one part."""
    cut = False
    omitted: set[str] = set()
    for run in runs:
        signature = worklist_signature(run)
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
