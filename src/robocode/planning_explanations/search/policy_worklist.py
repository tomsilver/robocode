"""Does a loop the grammar cannot bind keep a to-do list? A shape-free test.

A search keeps a worklist: some container gains items and loses items while the
loop runs. A loop with no such container (for example one that only pops
waypoints, or only appends while walking back through parents) is not a search
under this definition, so counting it as zero planning is supported by evidence.
A loop with one is reported as unreadable (unknown), never as zero. So is a loop
whose containers could not all be inspected within the scan limits.
"""

import json
from collections import Counter, deque

SCAN_DEPTH = 4
SCAN_LIMIT = 4096
CONTAINERS = (list, dict, deque, set)


def _key(item):
    try:
        hash(item)
    except TypeError:
        return repr(item)
    return item


def _items(value) -> Counter:
    """A container's items as a multiset (a dictionary's keys)."""
    return Counter(_key(item) for item in value)


def containers(values: dict, depth: int = SCAN_DEPTH, limit: int = SCAN_LIMIT):
    """Item multisets of every reachable list, dict, deque and set, by path.

    Returns the inventory and whether the depth or count limit hid any container.
    """
    queue = deque((name, value, 0) for name, value in sorted(values.items()))
    found: dict[str, Counter] = {}
    cut = False
    while queue:
        path, value, level = queue.popleft()
        if type(value) not in CONTAINERS:
            continue
        if len(found) >= limit:
            return found, True
        found[path] = _items(value)
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
        children = [(p, v) for p, v in children if type(v) in CONTAINERS]
        if children and level >= depth:
            cut = True
            continue
        queue.extend((p, v, level + 1) for p, v in children)
    return found, cut


def worklist_signature(run, depth: int = SCAN_DEPTH, limit: int = SCAN_LIMIT):
    """The first container that both gained and lost items during one run."""
    gained: set = set()
    lost: set = set()
    previous = None
    cut = False
    for kind, values in run.capture.frames():
        current, clipped = containers(values, depth, limit)
        cut = cut or clipped
        if previous is not None and kind != "edited":
            for path, items in current.items():
                before = previous.get(path)
                if before is None:
                    continue
                if items - before:
                    gained.add(path)
                if before - items:
                    lost.add(path)
                if path in gained and path in lost:
                    return {"worklist": path, "cut": cut}
        previous = current
    return {"worklist": None, "cut": cut}


def scan_runs(runs) -> dict:
    """Combine the signature over every run of one part."""
    cut = False
    for run in runs:
        signature = worklist_signature(run)
        if signature["worklist"] is not None:
            return {**signature, "run": run.ordinal}
        cut = cut or signature["cut"]
    return {"worklist": None, "cut": cut}


def unreadable_reason(signature: dict) -> str | None:
    """Why a loop without a binding cannot be counted as zero, if it cannot."""
    if signature["worklist"] is not None:
        return "worklist_without_binding"
    if signature["cut"]:
        return "inspection_cut"
    return None
