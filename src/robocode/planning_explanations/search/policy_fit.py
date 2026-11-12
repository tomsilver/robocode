"""Ingredient values proposed from development runs (grid-priority-v2).

Following the planning-explanations method, proposal search may use development
cases; the frozen explanation is then checked on held-out inputs. This module only
proposes. It reads a bounded prefix of each development run through one binding's
paths and returns numbers for the fixed search algorithm: a move table with step
costs, heuristic coefficients, and the numbers a heap entry stores between its
priority and its cell. Nothing here accepts an explanation: every proposal is
verified by stepping the engine independently beside each run (`compare_run`),
which never reads a later snapshot to decide a step.

Constants are snapped to the shortest decimal that reproduces every observation
exactly (1.414 rather than 1.4139999999999997), so the engine repeats the
program's own arithmetic. If no short decimal does, the raw estimate is kept and
verification compares numbers up to rounding.
"""

# Exact built-in types are the representation contract.
# pylint: disable=unidiomatic-typecheck

import math
from collections import Counter

from .grid_engine import EIGHT, is_cell
from .selectors import select

FIT_FRAMES = 64
SNAP_DIGITS = range(1, 18)
FIT_TOL = 1e-9


def _snap(raw: float, reproduces) -> float:
    """The shortest decimal constant reproducing every observation, else `raw`."""
    for digits in SNAP_DIGITS:
        value = float(f"{raw:.{digits}g}")
        if reproduces(value):
            return value
    return raw


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=FIT_TOL, abs_tol=FIT_TOL)


def _numbers(entry) -> bool:
    return all(type(v) in (int, float) for v in entry[:-1])


def observe(role: dict, runs, frames: int = FIT_FRAMES) -> dict:
    """Parent-edge costs and newly pushed heap entries in each run's first frames."""
    steps: dict = {}
    entries: list = []
    for run in runs:
        previous = None
        for index, (kind, values) in enumerate(run.capture.frames()):
            if index > frames or kind == "edited":
                break
            try:
                frontier = select(values, role["frontier"])
                parents = select(values, role["parents"])
                costs = select(values, role["costs"])
                goals = {path: select(values, path) for path in role["goals"]}
            except (KeyError, IndexError, TypeError, ValueError):
                break
            if type(parents) is not dict or type(costs) is not dict:
                break
            current = (list(frontier), dict(parents), dict(costs))
            if previous is not None:
                old_frontier, old_parents, old_costs = previous
                for child, parent in parents.items():
                    if parent is None or not (is_cell(child) and is_cell(parent)):
                        continue
                    if old_parents.get(child) == parent and old_costs.get(
                        child
                    ) == costs.get(child):
                        continue
                    if child in costs and parent in costs:
                        offset = (child[0] - parent[0], child[1] - parent[1])
                        steps.setdefault(offset, []).append(
                            (costs[parent], costs[child])
                        )
                pushed = Counter(e for e in frontier if type(e) is tuple) - Counter(
                    e for e in old_frontier if type(e) is tuple
                )
                for entry in pushed.elements():
                    cell = entry[-1]
                    if len(entry) >= 2 and is_cell(cell) and _numbers(entry):
                        entries.append((entry, costs.get(cell), goals))
            previous = current
    return {"steps": steps, "entries": entries}


AXIS, DIAGONAL = EIGHT[:4], EIGHT[4:]


def fit_moves(steps: dict) -> tuple | None:
    """One cost per offset, consistent across every observed parent edge.

    Unit moves of one kind (axis or diagonal) that a bounded prefix never showed
    get that kind's cost when every observed move of the kind agrees on it;
    verification checks the completed table like any other proposal.
    """
    if not steps:
        return None
    costs = {}
    for offset, pairs in steps.items():
        raws = sorted(child - parent for parent, child in pairs)
        raw = raws[len(raws) // 2]
        if not raw > 0 or not all(_close(r, raw) for r in raws):
            return None
        costs[offset] = _snap(raw, lambda c, p=pairs: all(a + c == b for a, b in p))
    for kind in (AXIS, DIAGONAL):
        seen = {costs[o] for o in kind if o in costs}
        if len(seen) == 1:
            (cost,) = seen
            for offset in kind:
                costs.setdefault(offset, cost)
    order = {offset: i for i, offset in enumerate(EIGHT)}
    return tuple(
        (o[0], o[1], costs[o])
        for o in sorted(costs, key=lambda o: (order.get(o, len(order)), o))
    )


def _distances(cell, goal) -> tuple[int, int, float]:
    dx, dy = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
    return max(dx, dy), min(dx, dy), math.hypot(dx, dy)


def _octile(points: list) -> tuple | None:
    """Weight and ratio of (high + ratio * low) * weight through every point."""
    for i, (hi1, lo1, _, t1) in enumerate(points):
        for hi2, lo2, _, t2 in points[i + 1 :]:
            det = hi1 * lo2 - hi2 * lo1
            if det == 0:
                continue
            weight = (t1 * lo2 - t2 * lo1) / det
            scaled = (hi1 * t2 - hi2 * t1) / det
            if weight > 0:
                return weight, scaled / weight
    return None


def fit_heuristics(entries: list, goal_path: str) -> list[tuple]:
    """(priority, form, coefficients) consistent with every pushed entry's number."""
    rows = []
    for entry, cost, goals in entries:
        goal = goals.get(goal_path)
        if cost is None or not is_cell(goal):
            continue
        high, low, straight = _distances(entry[-1], goal)
        rows.append((entry[0], cost, high, low, straight))
    if not rows:
        return []
    fits = []
    if all(_close(p, g) for p, g, *_ in rows):
        fits.append(("g", "", ()))
    for priority in ("g+h", "h"):

        def base(g, priority=priority) -> float:
            return g if priority == "g+h" else 0.0

        targets = [(hi, lo, s, p - base(g)) for p, g, hi, lo, s in rows]
        found = _octile(targets)
        if found is not None:
            weight, ratio = found
            if all(_close((hi + ratio * lo) * weight, t) for hi, lo, _, t in targets):
                weight, ratio = _snap_pair(weight, ratio, rows, base)
                fits.append((priority, "octile", (weight, ratio)))
        scaled = [(t / s) for _, _, s, t in targets if s > 0]
        if scaled and scaled[0] > 0 and all(_close(w, scaled[0]) for w in scaled):
            weight = _snap(
                scaled[0],
                lambda w: all(base(g) + s * w == p for p, g, _, _, s in rows if s > 0),
            )
            fits.append((priority, "euclidean", (float(weight),)))
    return fits


def _snap_pair(weight: float, ratio: float, rows: list, base) -> tuple:
    """Shortest (weight, ratio) decimals that reproduce every priority exactly."""
    for digits in SNAP_DIGITS:
        w = float(f"{weight:.{digits}g}")
        for other in SNAP_DIGITS:
            r = float(f"{ratio:.{other}g}")
            if all(base(g) + (hi + r * lo) * w == p for p, g, hi, lo, _ in rows):
                return w, r
    return weight, ratio


def fit_fields(entries: list) -> tuple:
    """Names of the numbers between a tagged entry's priority and its cell."""
    tagged = [(e, g) for e, g, _ in entries if len(e) >= 3 and g is not None]
    if not tagged or len({len(e) for e, _ in tagged}) != 1:
        return ()
    names = []
    for position in range(1, len(tagged[0][0]) - 1):
        if all(_close(e[position], g) for e, g in tagged):
            names.append("g")
        elif all(_close(e[position], -g) for e, g in tagged):
            names.append("neg_g")
        else:
            return ()
    return tuple(names)


def fit_role(role: dict, runs) -> dict:
    """Every numeric proposal for one binding, from development runs only."""
    seen = observe(role, runs)
    return {
        "moves": fit_moves(seen["steps"]),
        "heuristics": {
            path: fit_heuristics(seen["entries"], path) for path in role["goals"]
        },
        "fields": fit_fields(seen["entries"]) if role["layout"] == "tagged" else (),
        "observed_edges": sum(len(v) for v in seen["steps"].values()),
        "observed_entries": len(seen["entries"]),
    }
