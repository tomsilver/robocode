"""Verify grid search explanations of loop runs, cached plans and plan use.

Three separate evidence axes, none of which discovers or repairs anything:
1. computation: every loop iteration of a run against the abstract engine,
   plus sampled paired edits of the live frontier/costs (and the detour probes
   `policy_causality` derives from the stored plan);
2. candidate read-out: a cached plan's waypoints follow a parent chain;
3. local cache use: rotating it about the robot rotates the planar command.

Neither (2) nor (3) attributes search to later actions. `policy_causality`
checks that entire chain using the same search intervention across calls.
"""

# Exact built-in types are the representation contract.
# pylint: disable=unidiomatic-typecheck

import copy
import hashlib
import heapq
import json
import math
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from itertools import chain
from operator import itemgetter
from typing import Any

import numpy as np

from .grid_engine import GridSearch, GridSpec, grid_features, is_cell
from .policy_runner import (
    CallRecord,
    PolicyProgram,
    RunRecord,
    SearchEdit,
    Session,
    action_tuple,
    drop_cell_entries,
    drop_index,
    entry_view,
    plan_attributes,
    same_snapshot,
)
from .selectors import parse, select

ERRORS = (ValueError, KeyError, TypeError, IndexError, RuntimeError, AttributeError)
# Checkpoint kinds that end a run: a false loop test or break, or a return inside it.
ENDINGS = ("exit", "return")
# Costs and priorities agree up to floating-point rounding of the same arithmetic.
REL_TOL = ABS_TOL = 1e-9
_VIEW = itemgetter(0, -1)
_MIDDLE = itemgetter(slice(1, -1))
_NUMBERS = frozenset({int, float})
KEEP_RECORDS = 4
ROTATION_COSINE = 0.9
WORLD_SHIFT = 0.05


@dataclass(frozen=True)
class GridMapping:
    """Literal selections at one loop site.

    The model is a callable local's name, or the path of a boolean grid the run
    already holds (a mapping decodes an existing representation).
    """

    site: str
    root: str
    frontier: str
    parents: str
    costs: str
    goal: str = ""
    model: str = ""

    def __post_init__(self):
        names = (self.root, self.frontier, self.parents, self.costs)
        if not all(names) or len(set(names)) != 4:
            raise ValueError("Grid mappings select four distinct values")
        for name in (*names, self.goal):
            if name:
                parse(name)
        if self.model:
            parse(self.model)


@dataclass(frozen=True)
class GridExplanation:
    """Ingredients plus a selection-only mapping, frozen as one identity."""

    spec: GridSpec
    mapping: GridMapping

    def __post_init__(self):
        if self.spec.needs_goal != bool(self.mapping.goal):
            raise ValueError("A goal is mapped exactly when the ingredients use one")
        if (self.spec.model != "open") != bool(self.mapping.model):
            raise ValueError("A predicate or grid model needs exactly one mapping")
        if self.spec.model == "own" and not self.mapping.model.isidentifier():
            raise ValueError("The own model is a plain local callable")

    @property
    def identifier(self) -> str:
        """Stable identity of the entire frozen explanation."""
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]


def explanation_from_data(data: dict) -> GridExplanation:
    """Load a declarative explanation, rechecking its restricted ingredients."""
    return GridExplanation(GridSpec(**data["spec"]), GridMapping(**data["mapping"]))


def grid_predicate(grid, true_means_free: bool):
    """Free-cell test read off a 2-D boolean grid; outside the grid is blocked."""
    shape = getattr(grid, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("A grid model is a two-dimensional boolean array")
    rows, columns = shape

    def free(cell) -> bool:
        i, j = cell
        if not (0 <= i < rows and 0 <= j < columns):
            return False
        return bool(grid[i, j]) == true_means_free

    return free


def bind(explanation: GridExplanation, values: dict, callables: dict) -> GridSearch:
    """A run's inputs are read once, at its first loop checkpoint."""
    mapping, spec = explanation.mapping, explanation.spec
    goal = select(values, mapping.goal) if mapping.goal else None
    if spec.model == "own":
        free = callables[mapping.model]
    elif spec.model in ("grid_free", "grid_blocked"):
        free = grid_predicate(select(values, mapping.model), spec.model == "grid_free")
    else:
        free = None
    return GridSearch(spec, select(values, mapping.root), goal, free)


def decode(explanation: GridExplanation, values: dict) -> tuple:
    """Current root, canonical frontier, parents and costs; nothing is computed."""
    mapping, spec = explanation.mapping, explanation.spec
    raw = select(values, mapping.frontier)
    parents = select(values, mapping.parents)
    costs = select(values, mapping.costs)
    if (
        type(raw) not in (list, deque)
        or type(parents) is not dict
        or type(costs) is not dict
    ):
        raise ValueError("Expected a list/deque frontier and dictionary parents/costs")
    for entry in raw:
        if spec.layout == "states":
            if not is_cell(entry):
                raise ValueError("A state frontier holds cells")
        elif not (
            type(entry) is tuple
            and (len(entry) == 2 if spec.layout == "pairs" else len(entry) >= 3)
            and all(type(v) in (int, float) for v in entry[:-1])
            and is_cell(entry[-1])
        ):
            raise ValueError("A heap frontier holds (priority, ..., cell) entries")
    # Tie-break numbers of tagged entries are not part of the search state.
    views = [entry_view(e) for e in raw] if spec.layout == "tagged" else list(raw)
    frontier = tuple(views) if spec.ordered else tuple(sorted(views))
    return select(values, mapping.root), frontier, parents, costs


def _near(a, b) -> bool:
    return a == b or math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _same_costs(observed: dict, predicted: dict) -> bool:
    if observed == predicted:
        return True
    return observed.keys() == predicted.keys() and all(
        _near(v, predicted[k]) for k, v in observed.items()
    )


def _same_frontier(observed: tuple, predicted: tuple, spec: GridSpec) -> bool:
    """Same entries; priorities may differ by float rounding only."""
    if observed == predicted:
        return True
    if len(observed) != len(predicted) or spec.layout == "states":
        return False
    if spec.ordered:
        return all(
            a[-1] == b[-1] and _near(a[0], b[0]) for a, b in zip(observed, predicted)
        )

    def by_cell(entries) -> dict:
        grouped: dict = {}
        for priority, cell in entries:
            grouped.setdefault(cell, []).append(priority)
        return {cell: sorted(ps) for cell, ps in grouped.items()}

    left, right = by_cell(observed), by_cell(predicted)
    return left.keys() == right.keys() and all(
        len(ps) == len(right[c]) and all(map(_close, ps, right[c]))
        for c, ps in left.items()
    )


def _quick_frontier(spec: GridSpec, raw) -> tuple | None:
    """The observed canonical frontier without per-entry validation, if it reads."""
    try:
        views = list(map(_VIEW, raw)) if spec.layout == "tagged" else list(raw)
        return tuple(views) if spec.ordered else tuple(sorted(views))
    except (TypeError, IndexError):
        return None


def _well_formed_tags(raw) -> bool:
    """Tagged entries hold at least one number between priority and cell."""
    return not raw or (
        min(map(len, raw)) >= 3
        and all(
            map(
                _NUMBERS.__contains__, map(type, chain.from_iterable(map(_MIDDLE, raw)))
            )
        )
    )


def _same_parents(parents: dict, engine: GridSearch) -> bool:
    if engine.spec.root_parent == "none":
        return parents == engine.parents
    return (
        len(parents) == len(engine.parents) - 1
        and engine.root not in parents
        and parents.items() <= engine.parents.items()
    )


def matches(explanation: GridExplanation, engine: GridSearch, values: dict) -> bool:
    """Root, parents and frontier cells exactly; costs and priorities up to rounding.

    The observed frontier is first compared with the prediction, which is well
    formed by construction; only a frontier that differs is validated entry by
    entry (`decode`), so malformed state is still reported as such.
    """
    spec, mapping = explanation.spec, explanation.mapping
    raw = select(values, mapping.frontier)
    parents = select(values, mapping.parents)
    costs = select(values, mapping.costs)
    predicted = engine.canonical_frontier()
    observed = (
        _quick_frontier(spec, raw)
        if type(raw) in (list, deque) and type(parents) is dict and type(costs) is dict
        else None
    )
    if (
        observed is not None
        and len(observed) == len(predicted)
        and (spec.layout != "tagged" or _well_formed_tags(raw))
        and _same_frontier(observed, predicted, spec)
    ):
        return (
            select(values, mapping.root) == engine.root
            and _same_parents(parents, engine)
            and _same_costs(costs, engine.costs)
        )
    decode(explanation, values)  # raises for malformed state
    return False


def difference(explanation: GridExplanation, engine: GridSearch, values) -> dict:
    """A short, inspectable counterexample instead of two 2,000-entry tables."""
    root, frontier, parents, costs = decode(explanation, values)
    predicted = engine.canonical_frontier()
    expected = engine.observed_parents()

    def few(items) -> list:
        return [repr(item) for item in list(items)[:5]]

    return {
        "root": {"predicted": repr(engine.root), "observed": repr(root)},
        "frontier_only_predicted": few(e for e in predicted if e not in frontier),
        "frontier_only_observed": few(e for e in frontier if e not in predicted),
        "parents_differ": few(
            k
            for k in set(parents) | set(expected)
            if parents.get(k, "-") != expected.get(k, "-")
        ),
        "costs_differ": few(
            k
            for k in set(costs) | set(engine.costs)
            if k not in costs
            or k not in engine.costs
            or not _near(costs[k], engine.costs[k])
        ),
        "sizes": {
            "predicted_frontier": len(predicted),
            "observed_frontier": len(frontier),
            "predicted_parents": len(expected),
            "observed_parents": len(parents),
        },
    }


def abstract_edit(engine: GridSearch, edit: SearchEdit) -> None:
    """The abstract counterpart of a declared live edit."""
    if engine.phase != "ready":
        raise ValueError("Interventions require a live search configuration")
    if edit.operation == "clear":
        engine.frontier.clear()
    elif edit.operation == "drop":
        del engine.frontier[drop_index(engine.frontier, edit.entry)]
        if not engine.spec.ordered:
            heapq.heapify(engine.frontier)
    elif edit.operation == "drop_cell":
        engine.frontier[:] = drop_cell_entries(engine.frontier, edit.cell)
        if not engine.spec.ordered:
            heapq.heapify(engine.frontier)
    elif edit.operation == "raise":
        if edit.cell not in engine.costs or edit.value is None:
            raise ValueError("Cost edit names an unrepresented state")
        engine.costs[edit.cell] = edit.value
    else:
        raise ValueError("Unknown abstract intervention")


def _summary(transition, step: int, kind: str) -> dict:
    return {
        "step": step,
        "checkpoint": kind,
        "operation": transition.operation,
        "selected": repr(transition.selected),
        "generated": len(transition.generated),
        "added": [repr(c) for c in transition.added[:4]],
        "improved": [repr(c) for c in transition.improved[:4]],
        "frontier_size": transition.frontier_size,
    }


def _selected(explanation: GridExplanation, engine: GridSearch, values) -> Any:
    """For a tied tagged heap, the tied cell the run took out (read from `values`).

    The run's own tie-break numbers decide ties; which tied cell went is the one
    whose (priority, cell) count dropped in the next snapshot. Anything else is
    still predicted and compared.
    """
    tied = engine.tied_cells()
    if len(tied) < 2:
        return None
    try:
        after = Counter(decode(explanation, values)[1])
    except ERRORS:
        return None
    low = engine.frontier[0][0]
    before = Counter(entry_view(e) for e in engine.frontier)
    gone = [c for c in tied if after[(low, c)] < before[(low, c)]]
    return gone[0] if len(gone) == 1 else None


def compare_run(
    explanation: GridExplanation, run: RunRecord, patch: SearchEdit | None = None
) -> tuple[dict, GridSearch | None]:
    """Step the engine beside every frame of a run; stop at the first mismatch."""
    result: dict = {
        "site": run.site,
        "ordinal": run.ordinal,
        "status": "supported",
        "records": [],
    }
    engine: GridSearch | None = None
    transitions: list = []
    step = 0
    edited = False

    def reject(reason: str, values=None, **extra) -> None:
        result.update(status="rejected", reason=reason, step=step, **extra)
        if values is not None and engine is not None:
            try:
                result["counterexample"] = difference(explanation, engine, values)
            except ERRORS as exc:
                result["counterexample"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        frames = run.capture.frames()
        _, values = next(frames)
        engine = bind(explanation, values, run.capture.callables)
        if not matches(explanation, engine, values):
            reject("configuration_mismatch", values)
            return result, None
        for kind, values in frames:
            if kind == "edited":
                if patch is None:
                    raise ValueError("Unexpected edit frame")
                abstract_edit(engine, patch)
                edited = True
                if not matches(explanation, engine, values):
                    reject("edit_correspondence", values)
                    return result, None
                continue
            if engine.phase == "done":
                reject("step_count", extra_frames=True)
                return result, None
            transition = engine.step(_selected(explanation, engine, values))
            transitions.append(transition)
            step += 1
            if step <= KEEP_RECORDS:
                result["records"].append(_summary(transition, step, kind))
            if (kind in ENDINGS) != (engine.phase == "done"):
                reject("step_count", predicted_phase=engine.phase, checkpoint=kind)
                return result, None
            if not matches(explanation, engine, values):
                reject("configuration_mismatch", values)
                return result, None
        if not run.exited:
            reject("no_termination_checkpoint")
            return result, None
        if patch is not None and not edited:
            reject("edit_not_observed")
            return result, None
    except StopIteration:
        reject("empty_run")
        return result, None
    except ERRORS as exc:
        reject("decode_or_alignment", error=f"{type(exc).__name__}: {exc}")
        return result, None
    result.update(
        steps=step,
        reached_goal=engine.reached_goal,
        features=grid_features(transitions),
    )
    return result, engine


def choose_edits(
    explanation: GridExplanation, run: RunRecord, seed: str, count: int
) -> list[SearchEdit]:
    """Edit points and edits drawn with a fixed seed from the ordinary run."""
    rng = random.Random(seed)
    visits = list(range(run.visits))
    points = sorted(rng.sample(visits, min(count, len(visits))))
    kinds = ["drop", "raise", "drop", "raise", "clear"][: len(points)]
    rng.shuffle(kinds)
    wanted = dict(zip(points, kinds))
    mapping, spec = explanation.mapping, explanation.spec
    edits = []
    index = -1
    for kind, values in run.capture.frames():
        if kind != "visit":
            continue
        index += 1
        if index not in wanted:
            continue
        root, frontier, parents, costs = decode(explanation, values)
        if not frontier:
            continue
        operation = wanted[index]
        if operation == "raise":
            cells = frontier if spec.layout == "states" else [e[1] for e in frontier]
            internal = set(parents.values())
            leaves = sorted(
                {c for c in cells if c != root and c not in internal and c in costs}
            )
            if leaves:
                cell = rng.choice(leaves)
                edits.append(
                    SearchEdit(
                        run.site,
                        run.ordinal,
                        index,
                        "raise",
                        mapping.costs,
                        cell=cell,
                        value=float(costs[cell]) + 2.0,
                    )
                )
                continue
            operation = "drop"
        if operation == "drop":
            entry = rng.choice(list(frontier))
            edits.append(
                SearchEdit(
                    run.site,
                    run.ordinal,
                    index,
                    "drop",
                    mapping.frontier,
                    entry=entry,
                    heap=not spec.ordered,
                )
            )
        else:
            edits.append(
                SearchEdit(run.site, run.ordinal, index, "clear", mapping.frontier)
            )
    targeted = _raise_next(explanation, run)
    if targeted is not None and targeted not in edits:
        edits.append(targeted)
    # Route-changing producer probes need the stored plan; see
    # `policy_causality.detour_edits`.
    return edits


def _raise_next(explanation: GridExplanation, run: RunRecord) -> SearchEdit | None:
    """Raise the cost of the cell selected next, at the first later checkpoint.

    Reading the selected cell's cost from the table or from its heap entry gives
    the same ordinary run; this edit separates the two.
    """
    spec, mapping = explanation.spec, explanation.mapping
    index = -1
    for kind, values in run.capture.frames():
        if kind != "visit":
            continue
        index += 1
        if index == 0:
            continue
        root, frontier, _, costs = decode(explanation, values)
        if not frontier:
            return None
        if spec.layout == "states":
            cell = frontier[0] if spec.selection == "fifo" else frontier[-1]
        elif spec.selection == "min":
            cell = frontier[0][1]
        else:
            cell = (frontier[0] if spec.selection == "fifo" else frontier[-1])[1]
        if cell == root or cell not in costs:
            return None
        return SearchEdit(
            run.site,
            run.ordinal,
            index,
            "raise",
            mapping.costs,
            cell=cell,
            value=float(costs[cell]) + 2.0,
        )
    return None


def same_frames(ordinary: RunRecord, patched: RunRecord, count: int) -> bool:
    """The first `count` frames of two runs carry identical values."""
    if ordinary.capture.length < count or patched.capture.length < count:
        return False
    for _, (left, right) in zip(
        range(count), zip(ordinary.capture.frames(), patched.capture.frames())
    ):
        if left[0] != right[0] or not same_snapshot(left[1], right[1]):
            return False
    return True


def execute_edit(
    program: PolicyProgram, call: CallRecord, before: tuple, edit: SearchEdit, domain
) -> dict:
    """Re-run only the enclosing call from deep copies with one live edit."""
    policy, state, extra = before
    session = Session(capture=True, edit=edit, clock=call.clock)
    clone = copy.deepcopy(policy)
    output, error = None, None
    try:
        output = program.call(
            clone, call.method, (domain.copy_state(state), *extra), session
        )
    except ERRORS as exc:
        # The edited run may still have finished before the policy failed.
        error = f"{type(exc).__name__}: {exc}"
    target = None
    prefix = session.edits == 1
    for run in session.runs:
        if (run.site, run.ordinal) == (edit.site, edit.ordinal):
            target = run
            break
    if target is None:
        for run in session.runs:
            run.capture.release()
        return {"target": None, "prefix_preserved": False, "error": error}
    position = session.runs.index(target)
    ordinary = next(
        r for r in call.runs if (r.site, r.ordinal) == (edit.site, edit.ordinal)
    )
    # Everything before the run starts, then its frames 0..visit (pre-edit,
    # inclusive), must replay unchanged. Runs enclosing it (an outer loop, a
    # recursion) keep recording after the edit, so they are compared by what
    # they had recorded when it started.
    prefix = (
        prefix
        and [(r.site, r.ordinal) for r in session.runs[:position]]
        == [(r.site, r.ordinal) for r in call.runs[:position]]
        and target.started_after == ordinary.started_after
        and same_frames(ordinary, target, _frame_index(ordinary, edit.visit) + 1)
    )
    # Only the edited run is compared later; a planning call can hold dozens of
    # other recorded searches, kept until the call's analysis ends otherwise.
    for run in session.runs:
        if run is not target:
            run.capture.release()
    plans = plan_attributes(clone)
    return {
        "target": target,
        # Runtime-only counterfactual state, never serialized into evidence.
        "policy": clone,
        "plans": plans,
        "prefix_preserved": prefix,
        "error": error,
        "plans_changed": sorted(
            k
            for k in set(plans) | set(call.plans_after)
            if plans.get(k) != call.plans_after.get(k)
        ),
        "action_changed": (
            None
            if call.action is None or output is None
            else not _close(action_tuple(output), call.action)
        ),
    }


def _frame_index(run: RunRecord, visit: int) -> int:
    """Frame position of the given loop visit in an ordinary (unedited) run."""
    seen = -1
    for position, (kind, _) in enumerate(run.capture.frames()):
        if kind == "visit":
            seen += 1
            if seen == visit:
                return position
    raise ValueError("Edit visit outside the ordinary run")


def _close(a, b, tolerance: float = 1e-9) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))


def check_edit(
    explanation: GridExplanation, executed: dict, edit: SearchEdit, baseline
) -> dict:
    """Compare a patched continuation with the explanation's patched prediction."""
    record = {
        "edit": {
            "visit": edit.visit,
            "operation": edit.operation,
            "selector": edit.selector,
            "entry": repr(edit.entry) if edit.entry is not None else None,
            "cell": repr(edit.cell) if edit.cell is not None else None,
            "value": edit.value,
        },
        "prefix_preserved": executed["prefix_preserved"],
    }
    if executed["target"] is None:
        record.update(
            passed=False, reason="edited_run_missing", error=executed.get("error")
        )
        return record
    comparison, engine = compare_run(explanation, executed["target"], edit)
    passed = executed["prefix_preserved"] and comparison["status"] == "supported"
    record.update(
        passed=passed,
        comparison={k: v for k, v in comparison.items() if k != "records"},
        predicted_result_changed=(
            None if engine is None else engine.result() != baseline
        ),
        observed_plans_changed=executed["plans_changed"],
        observed_action_changed=executed["action_changed"],
    )
    return record


def _array_point(array):
    rows, columns = array.shape[:2]

    def point(cell) -> tuple[float, float]:
        i, j = cell
        if not (0 <= i < rows and 0 <= j < columns):
            raise IndexError("Cell outside the point array")
        return (float(array[i, j, 0]), float(array[i, j, 1]))

    return point


def point_functions(callables: dict, policy) -> dict:
    """Cell-to-point readouts: policy callables, and arrays the policy holds.

    An array attribute of shape (rows, columns, 2) is read as `point[i, j]`
    (for example a grid of waypoint coordinates); the mapping decodes an existing
    representation and computes nothing itself.
    """
    found = dict(callables)
    for name, value in vars(policy).items() if hasattr(policy, "__dict__") else ():
        dtype = getattr(value, "dtype", None)
        if (
            type(value).__module__ == "numpy"
            and type(value).__name__ == "ndarray"
            and getattr(value, "ndim", 0) == 3
            and value.shape[2] == 2
            and getattr(dtype, "kind", "?") == "f"
        ):
            found[f"self.{name}"] = _array_point(value)
    return found


def readout_shape(function, cell) -> bool:
    """A read-out maps a cell to a pair of floats."""
    try:
        point = function(cell)
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return (
        type(point) in (tuple, list)
        and len(point) == 2
        and all(type(v) is float for v in point)
    )


def link_plan(points: tuple, parents: dict, readout) -> bool:
    """Waypoints, except at most one trailing target, read out a parent chain."""
    if not points:
        return False
    inverse = {}
    for cell in parents:
        try:
            x, y = readout(cell)
        except ERRORS:
            return False
        inverse[(float(x), float(y))] = cell
    cells = [inverse.get(point) for point in points]
    mapped = 0
    while mapped < len(cells) and cells[mapped] is not None:
        mapped += 1
    if mapped == 0 or len(cells) - mapped > 1:
        return False
    return all(parents.get(b) == a for a, b in zip(cells[:mapped], cells[1:mapped]))


def _rotated(original, points: list[tuple[float, float]]):
    """Replace contents keeping the container and element types."""

    def like(item: Any, point: tuple[float, float]) -> Any:
        if type(item) is list:
            return [point[0], point[1]]
        if hasattr(item, "dtype"):
            return np.array(point, dtype=item.dtype)
        return (point[0], point[1])

    values = [like(item, point) for item, point in zip(original, points)]
    if type(original) is list:
        original[:] = values
        return original
    return tuple(values)


def follow_probe(
    program: PolicyProgram,
    domain,
    call: CallRecord,
    before: tuple,
    attribute: str,
) -> dict:
    """Rotate a cached plan by 90 degrees about the robot; the command should too."""
    policy, state, _ = before
    points = call.plans_before[attribute]
    record: dict = {"attribute": attribute, "waypoints": len(points)}
    baseline = domain.planar(call.action)
    norm = math.hypot(*baseline)
    if norm < 1e-9:
        record.update(passed=False, reason="no_planar_command")
        return record
    rx, ry = domain.robot_xy(state)
    rotated = [(rx - (y - ry), ry + (x - rx)) for x, y in points]
    clone = copy.deepcopy(policy)
    value = getattr(clone, attribute)
    replaced = _rotated(value, rotated)
    if replaced is not value:
        setattr(clone, attribute, replaced)
    try:
        output = program.call(
            clone,
            "get_action",
            (domain.copy_state(state),),
            Session(capture=False, clock=call.clock),
        )
    except ERRORS as exc:
        record.update(passed=False, reason="policy_error", error=str(exc))
        return record
    probe = domain.planar(action_tuple(output))
    expected = (-baseline[1], baseline[0])
    size = math.hypot(*probe)
    cosine = (
        0.0
        if size < 1e-12
        else (probe[0] * expected[0] + probe[1] * expected[1]) / (size * norm)
    )
    record.update(cosine=round(cosine, 6), passed=cosine >= ROTATION_COSINE)
    if record["passed"]:
        shifted = domain.shift_world(domain.copy_state(state), WORLD_SHIFT)
        try:
            moved = program.call(
                copy.deepcopy(policy),
                "get_action",
                (shifted,),
                Session(capture=False, clock=call.clock),
            )
            record["world_insensitive"] = _close(action_tuple(moved), call.action)
        except ERRORS as exc:
            record["world_insensitive"] = None
            record["world_error"] = str(exc)
    return record
