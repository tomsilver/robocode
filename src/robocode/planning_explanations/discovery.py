"""Hypothesis search and paired causal experiments.

Candidate mappings only select trace values. Task-model computations live here, outside
measured execution. No source names or fixture identities are consulted.

The search is factorized. Every model coordinate (one state, evaluation or score) is
matched and patch-tested on its own: any recorded numeric value, at any expression,
visit or container element, that equals the model's value in every discovery run.
Complete mappings are then assembled from survivors that are pairwise compatible
(patching one leaves the other with the value the model predicts). The full paired
test is a conjunction of exactly these single-location and pairwise checks, so the
cost grows with the number of candidate locations instead of their product.
"""

# Exclude booleans and arbitrary subclasses from numeric mapping coordinates.
# pylint: disable=unidiomatic-typecheck

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from math import isfinite, log2
from random import Random

from .instrument import Patch, Selector
from .task import (
    DEPTHS,
    MODELS,
    Model,
    UnknownTransition,
    action_values,
    choose,
    random_transitions,
)

RUN_ERRORS = (KeyError, IndexError, TypeError, ValueError, RuntimeError)
_MISSING = object()
_EXAMPLES = 16
NUMERIC = ("evaluations", "scores")
# Distance from the decision's switching point for boundary patches. Generated
# utilities are a whole number apart, so a smaller step lands strictly between them.
BOUNDARY_STEP = 0.125


@dataclass(frozen=True)
class Limits:
    """Explicit search and evidence budgets, recorded in the report."""

    locations: int = 64  # matching locations kept per coordinate
    candidates: int = 128  # complete mappings checked for a decision effect
    steps: int = 100_000  # partial assignments explored while assembling
    intervention_cases: int = 2
    tables: int = 64  # own transition tables inferred per route and depth

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Search and evidence budgets must be positive integers")


@dataclass(frozen=True)
class Explanation:
    """A frozen computation and selection-only alignment for one route.

    ``horizon``, ``path_value``, ``pruning`` and ``table`` name the model (see
    ``task.Model``); ``table`` is the program's own inferred transition table, or
    None for the environment's. ``mapping`` names one recorded location per model
    coordinate, or ``None`` for a node the model does not expand on this route.
    ``interchangeable`` lists ``(role, coordinate, locations)``: other locations
    that passed the same tests when swapped in for that single coordinate. They are
    copies of the same quantity read at different moments, not different
    computations.
    """

    route: str
    horizon: int
    mapping: tuple
    interchangeable: tuple = ()
    path_value: str = "leaf"
    pruning: str = "none"
    table: tuple | None = None

    @property
    def model(self):
        """The computation this explanation claims the program performs."""
        return Model(self.horizon, self.path_value, self.pruning, self.table)

    @property
    def identifier(self):
        """A stable identity for the entire model and mapping.

        Model parameters added later are left out at their default value, so
        adding a parameter does not rename existing explanations.
        """
        fields = asdict(self)
        if fields["pruning"] == "none":
            del fields["pruning"]
        if fields["table"] is None:
            del fields["table"]
        payload = json.dumps(fields, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def selectors(self):
        """View the immutable mapping by role."""
        return dict(self.mapping)

    def read(self, trace, strict=True):
        """Select and rearrange existing values, with no model computation.

        Unexpanded nodes read as ``None``. With ``strict=False`` a location the
        execution never reached (for example after a patch triggered a cutoff)
        also reads as ``None`` instead of raising.
        """
        roles = {}
        for role, selectors in self.mapping:
            roles[role] = []
            for selector in selectors:
                try:
                    value = None if selector is None else selector.read(trace)
                except (KeyError, IndexError, TypeError):
                    if strict:
                        raise
                    value = None
                roles[role].append(value)
        return roles

    def swaps(self):
        """Yield each single-coordinate substitution of an interchangeable location."""
        for role, coordinate, locations in self.interchangeable:
            for location in locations:
                mapping = tuple(
                    (
                        name,
                        tuple(
                            location if (name, i) == (role, coordinate) else selector
                            for i, selector in enumerate(selectors)
                        ),
                    )
                    for name, selectors in self.mapping
                )
                yield role, coordinate, location, replace(
                    self, mapping=mapping, interchangeable=()
                )


@dataclass(frozen=True)
class FrozenBank:
    """Discovery output; validation never adds or repairs an explanation."""

    program_digest: str
    discovery_inputs: tuple
    explanations: tuple
    audit: dict

    @property
    def digest(self):
        """Freeze the source and complete explanation library together."""
        payload = {
            "source": self.program_digest,
            "explanations": [asdict(e) for e in self.explanations],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def leaves(value, path=()):
    """Inventory numeric scalar locations, without interpreting identifiers."""
    if type(value) in (int, float):
        yield path
    elif type(value) in (list, tuple, dict):
        keys = value.keys() if type(value) is dict else range(len(value))
        for key in keys:
            yield from leaves(value[key], path + (key,))


def value_table(trace):
    """Every recorded numeric scalar, keyed by checkpoint, visit and container path."""
    table = {}
    for event in trace.events:
        for path in leaves(event.value):
            value = event.value
            for key in path:
                value = value[key]
            table[Selector(event.site, event.occurrence, path)] = value
    return table


def order_constraints(model):
    """Temporal requirements as ``(reason, earlier, later, strict)`` coordinates.

    A state precedes the two states it leads to, a path's last state precedes the
    path's evaluation, and an evaluation is no later than its action's score.
    """
    depth = model.depth
    constraints = []
    for level in range(1, depth):
        for i in range(2**level):
            for child in (2 * i, 2 * i + 1):
                constraints.append(
                    (
                        "state_order",
                        (f"states_{level}", i),
                        (f"states_{level + 1}", child),
                        True,
                    )
                )
    for leaf in range(2**depth):
        constraints.append(
            ("evaluation_order", (f"states_{depth}", leaf), ("evaluations", leaf), True)
        )
        if depth > 1:
            action = leaf >> (depth - 1)
            constraints.append(
                ("backup_order", ("evaluations", leaf), ("scores", action), False)
            )
    return constraints


def out_of_order(trace, earlier, later, strict):
    """Whether ``earlier`` fails to precede ``later`` in this execution."""
    first, second = trace.event(earlier).index, trace.event(later).index
    return first > second or (strict and first == second)


def ordinary(explanation, case, trace):
    """Check task semantics, temporal order and output on an ordinary execution."""
    expected = explanation.model.rollout(case.state, case.transitions, case.utilities)
    try:
        observed = explanation.read(trace)
        if observed != expected:
            return {
                "reason": "role_values",
                "case": case.label,
                "predicted": expected,
                "observed": observed,
            }
        mapping = explanation.selectors()
        for reason, (r1, i1), (r2, i2), strict in order_constraints(explanation.model):
            first, second = mapping[r1][i1], mapping[r2][i2]
            if first and second and out_of_order(trace, first, second, strict):
                return {"reason": reason, "case": case.label}
        if trace.output != choose(action_values(expected)):
            return {
                "reason": "ordinary_output",
                "case": case.label,
                "predicted": choose(action_values(expected)),
                "observed": trace.output,
            }
    except (KeyError, IndexError, TypeError) as exc:
        return {"reason": "missing_location", "case": case.label, "detail": str(exc)}
    return None


def predict_patch(case, model, role, index, value):
    """Intervene in the candidate computation and propagate its effects."""
    return model.rollout(
        case.state, case.transitions, case.utilities, (role, index, value)
    )


def compared_numbers(case, model):
    """Every number the model's decision is computed from, in increasing order."""
    roles = model.rollout(case.state, case.transitions, case.utilities)
    return sorted(
        {v for role in NUMERIC if role in roles for v in roles[role] if v is not None}
    )


def switch_point(case, model, role, index):
    """Where the model's decision flips as this one value moves, or None.

    The decision can only flip where the patched value crosses another number the
    model compares, and those are the model's own evaluations and scores. ``step``
    is small enough that no other compared number lies between ``point - step``
    and ``point + step``.
    """
    numbers = compared_numbers(case, model)
    gaps = [b - a for a, b in zip(numbers, numbers[1:])]
    step = min([BOUNDARY_STEP] + [g / 4 for g in gaps])

    def decision(value):
        return choose(action_values(predict_patch(case, model, role, index, value)))

    for point in numbers:
        if decision(point - step) != decision(point + step):
            return point, step
    return None


def predictable(case, model, role, index, value):
    """Whether the model can predict this patch (its own table may lack a row)."""
    try:
        predict_patch(case, model, role, index, value)
    except UnknownTransition:
        return False
    return True


def covers(model, case):
    """Whether the model can predict this input's search at all."""
    try:
        model.rollout(case.state, case.transitions, case.utilities)
    except UnknownTransition:
        return False
    return True


def infer_tables(runs, depth, limits):
    """Fixed transition tables that could be the program's own model.

    Candidate states are recorded integers 0-5 present in every run of the route,
    taken as value series across runs, so copies of one value are one candidate.
    Constants (literals, loop counters) and re-reads of the parent state are not
    successors: a candidate must take two values across inputs and differ from its
    parent on at least one. Most varied candidates are tried first. A
    first state must be a fixed function of the start state, a deeper state a fixed
    function of its parent's value, and all of them together one table: the same
    (state, action) always leads to the same state, whatever the input's
    environment says. A node with no fitting candidate is left out with its subtree
    (the program may not expand it). Tables that agree with every run's environment
    are dropped: they are indistinguishable from reading the environment, which the
    environment models already test. Every entry must be seen on at least two
    inputs, so a route needs each of its start states twice; on fewer inputs almost
    anything looks like a fixed function. Whether the program actually searches
    with a table is decided afterwards by the usual checks.

    Returns distinct tables (six rows of two successors, None where unobserved),
    whether a budget stopped the enumeration, and why nothing was tried, if so.
    """
    values = [value_table(trace) for _, trace in runs]
    series = sorted(
        {
            tuple(table[selector] for table in values)
            for selector in values[0]
            if all(
                type(table.get(selector)) is int and 0 <= table[selector] < 6
                for table in values
            )
        },
        key=lambda candidate: (-len(set(candidate)), candidate),
    )
    series = [candidate for candidate in series if len(set(candidate)) > 1]
    starts = tuple(case.state for case, _ in runs)
    if min(Counter(starts).values()) < 2:
        return [], False, "a start state of this route has only one input"
    environments = [case.transitions for case, _ in runs]
    nodes = [(level, i) for level in range(1, depth + 1) for i in range(2**level)]
    found = {}
    steps = 0

    def extend(entries, parent, action, children):
        """The table plus these observations, or None if they contradict it.

        Entries map (state, action) to (successor, inputs where it was seen).
        """
        added = dict(entries)
        for run, (start, child) in enumerate(zip(parent, children)):
            seen, inputs = added.get((start, action), (child, frozenset()))
            if seen != child:
                return None
            added[(start, action)] = (child, inputs | {run})
        return added

    def visit(position, chosen, entries, differs):
        nonlocal steps
        if len(found) >= limits.tables or steps >= limits.steps:
            return
        if position == len(nodes):
            if differs and all(len(inputs) > 1 for _, inputs in entries.values()):
                table = tuple(
                    tuple(entries.get((s, a), (None,))[0] for a in (0, 1))
                    for s in range(6)
                )
                found.setdefault(table, None)
            return
        level, i = nodes[position]
        parent = starts if level == 1 else chosen.get((level - 1, i >> 1))
        if parent is None:  # Its parent is not expanded, so neither is this node.
            visit(position + 1, chosen, entries, differs)
            return
        fitting = []
        for children in series:
            steps += 1
            if children == parent:
                continue  # Re-reading the parent is not a successor.
            added = extend(entries, parent, i & 1, children)
            if added is not None:
                fitting.append((children, added))
        if not fitting:
            visit(position + 1, chosen, entries, differs)
        for children, added in fitting:
            # Evidence of an own model: a successor the environment does not have.
            new = differs or any(
                env[start][i & 1] != child
                for env, start, child in zip(environments, parent, children)
            )
            visit(position + 1, {**chosen, (level, i): children}, added, new)

    visit(0, {}, {}, False)
    truncated = len(found) >= limits.tables or steps >= limits.steps
    return list(found), truncated, None


# Random environments tried after the targeted changes in the no-lookahead test.
RANDOM_FUTURES = 16


def future_probes(case):
    """Inputs that differ from ``case`` only in its future.

    The current state, its own utility and the cached actions stay as they are.
    What changes is where actions lead and how good the other states are: each
    other state's utility pushed far up and far down, each transition redirected to
    every other state, then random tables and utilities. Yields (change, input).
    """
    state, utilities = case.state, case.utilities
    for x in range(6):
        if x == state:
            continue
        for value in (max(utilities) + 100, min(utilities) - 100):
            changed = list(utilities)
            changed[x] = value
            yield f"u[{x}] → {value}", replace(case, utilities=tuple(changed))
    for x in range(6):
        for a in (0, 1):
            for y in range(6):
                if y == case.transitions[x][a]:
                    continue
                rows = [list(row) for row in case.transitions]
                rows[x][a] = y
                yield (
                    f"t[{x}][{a}]: {case.transitions[x][a]} → {y}",
                    replace(case, transitions=tuple(tuple(r) for r in rows)),
                )
    rng = Random(case.label)
    for i in range(RANDOM_FUTURES):
        values = [v + 0.25 for v in rng.sample(range(-30, 31), 6)]
        values[state] = utilities[state]
        yield f"random future {i + 1}", replace(
            case, transitions=random_transitions(rng), utilities=tuple(values)
        )


def lookahead_test(program, case, output):
    """Whether changing only the future of ``case`` can change the program's action.

    If no change does, the action depends only on the current state, its utility
    and stored control data: the input does not look ahead. ``witness`` is the
    first change that moved the action.
    """
    tried = 0
    for change, probe in future_probes(case):
        tried += 1
        try:
            changed = program.output(probe)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # The program read the changed future, so its behavior depends on it.
            changed = f"error: {type(exc).__name__}"
        if changed != output:
            witness = {"change": change, "action": changed}
            return {"depends": True, "probes": tried, "witness": witness}
    return {"depends": False, "probes": tried, "witness": None}


# Tree-shaped futures tried for the behavioral lower bound on compared paths.
TREE_FUTURES = 64


def tree_future(case, rng):
    """A random future in which the first two moves of every path are its own.

    The start state has no incoming transitions, and each first successor is
    entered only from it; every other transition stays among the remaining three
    states. So (start, a) is used only by paths starting with action a, and
    (successor of a, b) only by paths starting with a, b. Returns the input, the
    two first successors and the remaining states.
    """
    others = [x for x in range(6) if x != case.state]
    rng.shuffle(others)
    first, rest = others[:2], others[2:]
    rows = [()] * 6
    rows[case.state] = tuple(first)
    for x in first + rest:
        rows[x] = tuple(rng.sample(rest, 2))
    values = [v + 0.25 for v in rng.sample(range(-30, 31), 6)]
    values[case.state] = case.utilities[case.state]
    future = replace(case, transitions=tuple(rows), utilities=tuple(values))
    return future, first, rest


def redirected(case, state, action, target):
    """``case`` with one transition sent elsewhere."""
    rows = [list(row) for row in case.transitions]
    rows[state][action] = target
    return replace(case, transitions=tuple(tuple(row) for row in rows))


def paths_lower_bound(program, case, route):
    """The fewest complete paths the program's decision on ``case`` compares.

    Evidence comes only from tree-shaped futures in which the program takes the
    same route as on ``case``, so it computes with the same structure. If sending
    (start, a) elsewhere moves the action, the decision depends on a path starting
    with a; if sending (successor of a, b) elsewhere moves it, on a path starting
    with a, b. Different starts are different paths, so the bound is at most 4.
    """
    rng = Random(f"{case.label}/tree")
    found = set()
    matched = 0

    def moves(probe, output):
        try:
            return program.output(probe) != output
        except Exception:  # pylint: disable=broad-exception-caught
            return True  # It read the changed transition.

    for _ in range(TREE_FUTURES):
        future, first, rest = tree_future(case, rng)
        try:
            trace = program.run(future)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if trace.route != route:
            continue
        matched += 1
        for a in (0, 1):
            for b in (0, 1):
                current = future.transitions[first[a]][b]
                if (a, b) not in found and any(
                    moves(redirected(future, first[a], b, y), trace.output)
                    for y in rest
                    if y != current
                ):
                    found.add((a, b))
            if not any(p[0] == a for p in found) and any(
                moves(redirected(future, case.state, a, y), trace.output) for y in rest
            ):
                found.add((a,))
        if len([p for p in found if len(p) == 2]) == 4:
            break
    paths = sum(
        max(sum((a, b) in found for b in (0, 1)), int((a,) in found)) for a in (0, 1)
    )
    return {
        "paths": paths,
        "effective_depth": round(log2(paths), 4) if paths else 0.0,
        "prefixes": [list(p) for p in sorted(found)],
        "matching_futures": matched,
        "tried_futures": TREE_FUTURES,
    }


def environment_agreement(model, cases):
    """How often an own table agrees with each input's environment.

    Counts the transitions each search actually takes, so a pair used twice counts
    twice. ``example`` is the first disagreement.
    """
    agreeing = total = 0
    example = None
    for case in cases:
        roles = model.rollout(case.state, case.transitions, case.utilities)
        for level in range(1, model.depth + 1):
            for i, child in enumerate(roles[f"states_{level}"]):
                if child is None:
                    continue
                parent = (
                    case.state if level == 1 else roles[f"states_{level - 1}"][i >> 1]
                )
                actual = case.transitions[parent][i & 1]
                total += 1
                if actual == child:
                    agreeing += 1
                elif example is None:
                    example = {
                        "case": case.label,
                        "state": parent,
                        "action": i & 1,
                        "believed": child,
                        "environment": actual,
                    }
    return {"agreeing": agreeing, "total": total, "example": example}


def untestable_patches(explanation, case, trace):
    """State patches the model cannot predict, because its table lacks a row."""
    count = 0
    for role, selectors in explanation.mapping:
        if not role.startswith("states_"):
            continue
        for i, selector in enumerate(selectors):
            if selector is None:
                continue
            current = int(selector.read(trace))
            count += sum(
                not predictable(case, explanation.model, role, i, s)
                for s in range(6)
                if s != current
            )
    return count


def replacements(case, model, role, index, current):
    """Labelled substitute values for one model coordinate.

    States: every other state the model can predict, so a decision-changing state
    is tried whenever one exists. Values: far below and far above every utility and
    every number the model compares, plus just below, exactly at, and just above
    the point where the model's decision flips.
    """
    if role.startswith("states_"):
        return tuple(
            ("other state", s)
            for s in range(6)
            if s != int(current) and predictable(case, model, role, index, s)
        )
    numbers = list(case.utilities) + compared_numbers(case, model)
    values = [("low", min(numbers) - 100), ("high", max(numbers) + 100)]
    found = switch_point(case, model, role, index)
    if found:
        point, step = found
        values += [
            ("below switch", point - step),
            ("at switch", point),
            ("above switch", point + step),
        ]
    return tuple(values)


def interventions(explanation, case, trace):
    """Every labelled single-coordinate substitution for every mapped coordinate."""
    for role, selectors in explanation.mapping:
        for i, selector in enumerate(selectors):
            if selector is None:
                continue  # A node this route never expands has nothing to patch.
            current = selector.read(trace)
            for kind, value in replacements(case, explanation.model, role, i, current):
                yield role, i, kind, Patch(selector, value)


def paired_test(program, explanation, case, trace):
    """Compare model predictions to actual downstream checkpoints and output."""
    records = []
    for role, i, kind, patch in interventions(explanation, case, trace):
        predicted = predict_patch(case, explanation.model, role, i, patch.value)
        record = {
            "case": case.label,
            "role": role,
            "coordinate": i,
            "kind": kind,
            "target": asdict(patch.selector),
            "old_value": patch.selector.read(trace),
            "replacement": patch.value,
            "baseline_output": trace.output,
            "predicted_output": choose(action_values(predicted)),
            # The model expects a different action, so the action alone can fail it.
            "informative": choose(action_values(predicted)) != trace.output,
            "predicted_roles": predicted,
        }
        try:
            patched = program.run(case, patch)
            # A patch may stop the program from reaching a location (a cutoff):
            # that reads as None, which matches a node the model no longer expands.
            observed = explanation.read(patched, strict=False)
            # Earlier snapshots are historical facts, not overwritten retroactively.
            target_time = trace.event(patch.selector).index
            checks = []
            for name, selectors in explanation.mapping:
                for coordinate, selector in enumerate(selectors):
                    if selector is not None and (
                        trace.event(selector).index > target_time
                        or selector == patch.selector
                    ):
                        checks.append(
                            {
                                "role": name,
                                "coordinate": coordinate,
                                "predicted": predicted[name][coordinate],
                                "observed": observed[name][coordinate],
                            }
                        )
            record.update(
                {
                    "observed_output": patched.output,
                    "observed_roles": observed,
                    "downstream_checks": checks,
                    "patch_count": patched.patch_count,
                    "output_changed": patched.output != trace.output,
                    "passed": patched.patch_count == 1
                    and patched.output == record["predicted_output"]
                    and all(c["predicted"] == c["observed"] for c in checks),
                }
            )
        except RUN_ERRORS as exc:
            record.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
        records.append(record)
        if not record["passed"]:
            return records, {
                "reason": "intervention_mismatch",
                "counterexample": record,
            }
    if not any(r["output_changed"] for r in records):
        return records, {"reason": "no_decision_effect", "case": case.label}
    return records, None


class _BudgetExhausted(Exception):
    """A search budget ran out before the search reached a conclusion."""


def _coordinate_name(coordinate):
    return f"{coordinate[0]}[{coordinate[1]}]"


def _names(coordinates):
    return [_coordinate_name(c) for c in sorted(coordinates)]


def expanded(model, roles):
    """The model's coordinates that this search actually computed, in role order."""
    return [
        (role, i)
        for role in model.roles
        for i, value in enumerate(roles[role])
        if value is not None
    ]


class ModelSearch:
    """Factorized search for one route and one model.

    1. The program's ordinary outputs must equal the model's decisions. This does
       not depend on any mapping, so a wrong model is rejected before matching.
    2. Each coordinate is matched to every recorded value equal to the model's value
       in all of the route's discovery runs.
    3. Each matching location is patched on its own; the action must change exactly
       as the model predicts. The patched runs are kept.
    4. Survivors are assembled into one complete mapping by backtracking. Every
       chosen pair must respect the model's temporal order and be compatible: under
       each patch of one, the other must hold the model's predicted value.
    5. Other survivors that fit in place of one coordinate are interchangeable.
    6. The assembled mapping is re-checked with the full ``ordinary`` and
       ``paired_test`` functions, which also produce the reported evidence.
    """

    def __init__(self, program, route, model, runs, limits):
        self.program = program
        self.route = route
        self.model = model
        self.runs = runs
        self.limits = limits
        self.cases = runs[: limits.intervention_cases]
        try:
            self.expected = [
                model.rollout(c.state, c.transitions, c.utilities) for c, _ in runs
            ]
        except UnknownTransition:
            self.expected = None  # An own table that cannot predict every run.
        # Only nodes the model expands are coordinates; the run() checks that the
        # model expands the same nodes on every input of the route.
        self.coordinates = expanded(model, self.expected[0]) if self.expected else []
        present = set(self.coordinates)
        self.constraints = defaultdict(list)
        for reason, earlier, later, strict in order_constraints(model):
            if earlier in present and later in present:
                self.constraints[frozenset((earlier, later))].append(
                    (reason, earlier, strict)
                )
        self.effects = {}
        self.survivors = {}
        self._pairs = {}
        self._undecided = 0
        self.audit = {
            "model": model.label,
            "horizon": model.depth,
            "path_value": model.path_value,
            "pruning": model.pruning,
            "table": None if model.table is None else [list(r) for r in model.table],
            "result": None,
            "truncated": False,
            "coordinates": [],
            "rejected_locations": [],
            "conflicts": {},
            "conflict_examples": [],
            "candidates_tested": 0,
            "steps": 0,
            "explanation": None,
            "interchangeable": 0,
            "interventions": [],
        }

    def run(self):
        """Return a verified explanation, or None; ``audit`` records why."""
        if self.expected is None:
            return self._finish("table_incomplete")
        for (case, trace), roles in zip(self.runs, self.expected):
            if trace.output != choose(action_values(roles)):
                return self._finish(
                    "ordinary_output",
                    counterexample={
                        "case": case.label,
                        "predicted": choose(action_values(roles)),
                        "observed": trace.output,
                    },
                )
        for (case, _), roles in zip(self.runs, self.expected):
            other = expanded(self.model, roles)
            if other != self.coordinates:
                # A fixed mapping needs the same events on every input of a route.
                first = self.runs[0][0].label
                return self._finish(
                    "shape_varies",
                    counterexample={
                        "cases": [first, case.label],
                        "only_in_first": _names(set(self.coordinates) - set(other)),
                        "only_in_second": _names(set(other) - set(self.coordinates)),
                    },
                )
        candidates = self._match()
        missing = [c for c in self.coordinates if not candidates[c]]
        if missing:
            return self._finish(
                "no_matching_location", missing=[_coordinate_name(c) for c in missing]
            )
        self._screen(candidates)
        dead = [c for c in self.coordinates if not self.survivors[c]]
        if dead:
            return self._finish(
                "no_surviving_location", dead=[_coordinate_name(c) for c in dead]
            )
        try:
            chosen = self._assemble([])
        except _BudgetExhausted:
            self.audit["truncated"] = True
            return self._finish("budget")
        if chosen is None:
            return self._finish("no_compatible_mapping")
        explanation = self._explanation(dict(chosen))
        failure = next(
            (f for c, t in self.runs if (f := ordinary(explanation, c, t))), None
        )
        evidence = []
        for case, trace in [] if failure else self.cases:
            records, failure = paired_test(self.program, explanation, case, trace)
            evidence.extend(records)
            if failure:
                break
        if failure:
            # The factorized checks and the full test disagree: report, never hide.
            return self._finish("verification_failed", counterexample=failure)
        self.audit.update(
            explanation=explanation.identifier,
            interchangeable=sum(
                len(locs) for _, _, locs in explanation.interchangeable
            ),
            interventions=evidence,
        )
        self._finish("supported")
        return explanation

    def _finish(self, result, **details):
        """Record the outcome; unsupported outcomes return no explanation."""
        counts = Counter(c["reason"] for c in self._pairs.values() if c)
        if self._undecided:
            counts["no_decision_effect"] = self._undecided
        self.audit.update(result=result, conflicts=dict(counts), **details)

    def _match(self):
        tables = [value_table(trace) for _, trace in self.runs]
        first = self.runs[0][1]

        def rank(selector):
            # Whole expression results before container elements, then earliest.
            return (bool(selector.path), first.event(selector).index, repr(selector))

        candidates = {}
        for role, i in self.coordinates:
            values = [roles[role][i] for roles in self.expected]
            found = sorted(
                (
                    selector
                    for selector in tables[0]
                    if all(
                        table.get(selector, _MISSING) == value
                        for table, value in zip(tables, values)
                    )
                ),
                key=rank,
            )
            if len(found) > self.limits.locations:
                self.audit["truncated"] = True
                found = found[: self.limits.locations]
            candidates[(role, i)] = found
        return candidates

    def _screen(self, candidates):
        for coordinate in self.coordinates:
            self.survivors[coordinate] = []
            for selector in candidates[coordinate]:
                effects, counterexample = self._patch_effects(coordinate, selector)
                if counterexample:
                    self.audit["rejected_locations"].append(
                        {
                            "role": coordinate[0],
                            "coordinate": coordinate[1],
                            "location": asdict(selector),
                            "counterexample": counterexample,
                        }
                    )
                else:
                    self.effects[(coordinate, selector)] = effects
                    self.survivors[coordinate].append(selector)
            self.audit["coordinates"].append(
                {
                    "role": coordinate[0],
                    "coordinate": coordinate[1],
                    "matched": len(candidates[coordinate]),
                    "survived": len(self.survivors[coordinate]),
                }
            )

    def _patch_effects(self, coordinate, selector):
        """Patch one location; the action must follow the model's prediction."""
        role, index = coordinate
        effects = []
        for case, trace in self.cases:
            old = selector.read(trace)
            for kind, value in replacements(case, self.model, role, index, old):
                predicted = predict_patch(case, self.model, role, index, value)
                record = {
                    "case": case.label,
                    "role": role,
                    "coordinate": index,
                    "kind": kind,
                    "target": asdict(selector),
                    "old_value": old,
                    "replacement": value,
                    "baseline_output": trace.output,
                    "predicted_output": choose(action_values(predicted)),
                }
                try:
                    patched = self.program.run(case, Patch(selector, value))
                except RUN_ERRORS as exc:
                    return None, {**record, "error": f"{type(exc).__name__}: {exc}"}
                record.update(
                    observed_output=patched.output, patch_count=patched.patch_count
                )
                if (
                    patched.patch_count != 1
                    or patched.output != record["predicted_output"]
                ):
                    return None, record
                effects.append(
                    {
                        "case": case.label,
                        "trace": trace,
                        "patched": patched,
                        "predicted": predicted,
                        "replacement": value,
                        "changed": patched.output != trace.output,
                    }
                )
        return effects, None

    def _mismatch(self, source, target):
        """How patching ``source`` contradicts the model at ``target``, if at all."""
        source_coordinate, source_location = source
        target_coordinate, target_location = target
        for effect in self.effects[source]:
            trace = effect["trace"]
            if target_location != source_location and (
                trace.event(target_location).index <= trace.event(source_location).index
            ):
                continue  # Earlier snapshots are historical facts.
            try:
                observed = target_location.read(effect["patched"])
            except (KeyError, IndexError, TypeError):
                observed = None  # Never reached, like a node the model skips.
            predicted = effect["predicted"][target_coordinate[0]][target_coordinate[1]]
            if observed != predicted:
                return {
                    "case": effect["case"],
                    "patched": _coordinate_name(source_coordinate),
                    "replacement": effect["replacement"],
                    "checked": _coordinate_name(target_coordinate),
                    "predicted": predicted,
                    "observed": observed,
                }
        return None

    def _conflict(self, first, second):
        """An order violation or patch-effect contradiction between two choices."""
        key = frozenset((first, second))
        if key not in self._pairs:
            conflict = None
            for reason, earlier, strict in self.constraints.get(
                frozenset((first[0], second[0])), ()
            ):
                before, after = (
                    (first, second) if first[0] == earlier else (second, first)
                )
                if any(
                    out_of_order(trace, before[1], after[1], strict)
                    for _, trace in self.runs
                ):
                    conflict = {"reason": reason}
                    break
            if conflict is None:
                detail = self._mismatch(first, second) or self._mismatch(second, first)
                if detail:
                    conflict = {"reason": "intervention_mismatch", **detail}
            if conflict and len(self.audit["conflict_examples"]) < _EXAMPLES:
                self.audit["conflict_examples"].append(
                    {
                        **conflict,
                        "locations": {
                            _coordinate_name(c): asdict(s) for c, s in (first, second)
                        },
                    }
                )
            self._pairs[key] = conflict
        return self._pairs[key]

    def _fits(self, item, chosen):
        return all(self._conflict(other, item) is None for other in chosen)

    def _decides(self, items):
        """On every intervention case, some patch must change the action."""
        return all(
            any(
                effect["changed"]
                for item in items
                for effect in self.effects[item]
                if effect["case"] == case.label
            )
            for case, _ in self.cases
        )

    def _assemble(self, chosen):
        if len(chosen) == len(self.coordinates):
            self.audit["candidates_tested"] += 1
            if self._decides(chosen):
                return list(chosen)
            self._undecided += 1
            if self.audit["candidates_tested"] >= self.limits.candidates:
                raise _BudgetExhausted
            return None
        coordinate = self.coordinates[len(chosen)]
        for selector in self.survivors[coordinate]:
            self.audit["steps"] += 1
            if self.audit["steps"] > self.limits.steps:
                raise _BudgetExhausted
            item = (coordinate, selector)
            if self._fits(item, chosen):
                found = self._assemble(chosen + [item])
                if found:
                    return found
        return None

    def _explanation(self, representative):
        mapping = tuple(
            (
                role,
                tuple(
                    representative.get((role, i))
                    for i in range(len(self.expected[0][role]))
                ),
            )
            for role in self.model.roles
        )
        interchangeable = []
        for coordinate in self.coordinates:
            others = [
                (c, representative[c]) for c in self.coordinates if c != coordinate
            ]
            locations = tuple(
                selector
                for selector in self.survivors[coordinate]
                if selector != representative[coordinate]
                and self._fits((coordinate, selector), others)
                and self._decides(others + [(coordinate, selector)])
            )
            if locations:
                interchangeable.append((coordinate[0], coordinate[1], locations))
        return Explanation(
            self.route,
            self.model.depth,
            mapping,
            tuple(interchangeable),
            self.model.path_value,
            self.model.pruning,
            self.model.table,
        )


def discover(program, inputs, limits=Limits()):
    """Discover on this split only, then return an immutable explanation bank."""
    if not inputs:
        raise ValueError("Discovery inputs cannot be empty")
    runs = [(c, program.check_observation(c)) for c in inputs]
    routes = defaultdict(list)
    for case, trace in runs:
        routes[trace.route].append((case, trace))
    audit = {"limits": asdict(limits), "routes": {}, "observation_checks": len(runs)}
    accepted = []
    for route, route_runs in routes.items():
        searches = [
            ModelSearch(program, route, model, route_runs, limits) for model in MODELS
        ]
        accepted.extend(e for e in (s.run() for s in searches) if e is not None)
        inference = []
        for depth in DEPTHS:
            # The same searches with the program's own transitions, one per table.
            tables, truncated, note = infer_tables(route_runs, depth, limits)
            own = [
                ModelSearch(
                    program, route, replace(model, table=table), route_runs, limits
                )
                for table in tables
                for model in MODELS
                if model.depth == depth
            ]
            accepted.extend(e for e in (s.run() for s in own) if e is not None)
            early = ("ordinary_output", "table_incomplete")
            inference.append(
                {
                    "horizon": depth,
                    "tables": len(tables),
                    "truncated": truncated,
                    "note": note,
                    "rejected_before_matching": sum(
                        s.audit["result"] in early for s in own
                    ),
                }
            )
            # Keep audits of own-table searches that got past the cheap checks.
            searches += [s for s in own if s.audit["result"] not in early]
        audit["routes"][route] = {
            "branches": route_runs[0][1].branches,
            "cases": [c.label for c, _ in route_runs],
            "truncated": any(s.audit["truncated"] for s in searches)
            or any(i["truncated"] for i in inference),
            "models": [s.audit for s in searches],
            "table_inference": inference,
        }
    return FrozenBank(program.digest, tuple(inputs), tuple(accepted), audit)


def _full_test(program, explanation, matching, intervention_cases):
    """Ordinary checks on every matching run, then paired tests on the first few."""
    failures = [
        failure
        for c, t in matching
        if (failure := ordinary(explanation, c, t)) is not None
    ]
    records = []
    if not failures:
        for case, trace in matching[:intervention_cases]:
            evidence, failure = paired_test(program, explanation, case, trace)
            records.extend(evidence)
            if failure:
                failures.append(failure)
                break
    status = "supported" if matching and not failures and records else "unresolved"
    return ("rejected" if failures else status), records, failures


def validate(program, bank, inputs, state_weights=None, intervention_cases=2):
    """Validate frozen mappings and describe ordinary execution features.

    No enumeration or mapping repair happens here. Unknown routes remain unknown. State
    weights define the ordinary population, never the patched experiments. Each frozen
    interchangeable location is re-tested as a single-coordinate swap.
    """
    if program.digest != bank.program_digest:
        raise ValueError("Frozen explanations belong to a different program")
    if type(intervention_cases) is not int or intervention_cases < 1:
        raise ValueError("At least one validation intervention case is required")
    discovery_fingerprints = {
        (c.state, c.transitions, c.utilities, c.cached_actions)
        for c in bank.discovery_inputs
    }
    if any(
        (c.state, c.transitions, c.utilities, c.cached_actions)
        in discovery_fingerprints
        for c in inputs
    ):
        raise ValueError(
            "Validation inputs must be fresh, not relabeled discovery inputs"
        )
    if not inputs:
        raise ValueError("Validation inputs cannot be empty")
    weights = {s: 1 / 6 for s in range(6)} if state_weights is None else state_weights
    if (
        set(weights) != set(range(6))
        or any(not isfinite(w) or w < 0 for w in weights.values())
        or abs(sum(weights.values()) - 1) > 1e-9
    ):
        raise ValueError("Supply six nonnegative state weights summing to one")
    counts = defaultdict(int)
    runs = [(c, program.check_observation(c)) for c in inputs]
    for case, _ in runs:
        counts[case.state] += 1
    if any(weights[s] > 0 and not counts[s] for s in weights):
        raise ValueError("Missing validation inputs for a positive-mass state")
    validations = []
    supported = defaultdict(list)
    has_supported_swap = {}
    for explanation in bank.explanations:
        on_route = [(c, t) for c, t in runs if t.route == explanation.route]
        # An own table may lack a row an input needs: such inputs are not covered.
        matching = [(c, t) for c, t in on_route if covers(explanation.model, c)]
        status, records, failures = _full_test(
            program, explanation, matching, intervention_cases
        )
        if status == "supported":
            supported[explanation.route].append(explanation)
        swaps = []
        for role, coordinate, location, variant in explanation.swaps():
            swap_status, _, swap_failures = _full_test(
                program, variant, matching, intervention_cases
            )
            swaps.append(
                {
                    "role": role,
                    "coordinate": coordinate,
                    "location": asdict(location),
                    "status": swap_status,
                    "counterexamples": swap_failures,
                }
            )
        has_supported_swap[explanation.identifier] = any(
            s["status"] == "supported" for s in swaps
        )
        validations.append(
            {
                "id": explanation.identifier,
                "status": status,
                "ordinary_cases": len(matching),
                "uncovered_cases": len(on_route) - len(matching),
                "environment_agreement": (
                    None
                    if explanation.table is None
                    else environment_agreement(
                        explanation.model, [c for c, _ in matching]
                    )
                ),
                "untestable_patches": sum(
                    untestable_patches(explanation, c, t)
                    for c, t in matching[:intervention_cases]
                ),
                "interventions": records,
                "informative_interventions": sum(r["informative"] for r in records),
                "counterexamples": failures,
                "interchangeable": swaps,
            }
        )
    # Routes without a supported explanation: test every frozen explanation that
    # was not falsified on its own route. Nothing is fitted to the fresh inputs;
    # the same full test decides, so a transfer is held-out evidence.
    transfers = []
    usable = [
        e for e, v in zip(bank.explanations, validations) if v["status"] != "rejected"
    ]
    by_route = defaultdict(list)
    for case, trace in runs:
        by_route[trace.route].append((case, trace))
    for route, route_runs in by_route.items():
        if supported[route]:
            continue
        for explanation in usable:
            if explanation.route == route:
                continue
            matching = [(c, t) for c, t in route_runs if covers(explanation.model, c)]
            status, _, failures = _full_test(
                program, explanation, matching, intervention_cases
            )
            transfers.append(
                {
                    "route": route,
                    "id": explanation.identifier,
                    "from_route": explanation.route,
                    "status": status,
                    "cases": len(matching),
                    "counterexample": failures[0] if failures else None,
                }
            )
            if status == "supported":
                supported[route].append(explanation)
    reasons = UnresolvedReasons(program, bank, validations)
    population = []
    for case, trace in runs:
        explanations = [e for e in supported[trace.route] if covers(e.model, case)]
        depths = sorted({effective_depth(e) for e in explanations})
        population.append(
            {
                "case": case.label,
                "state": case.state,
                "weight": weights[case.state] / counts[case.state],
                "route": trace.route,
                "branches": trace.branches,
                "output": trace.output,
                "original_checkpoints": len(trace.events),
                "status": "supported_planning" if explanations else "unresolved",
                "unresolved_reason": (
                    None if explanations else reasons.reason(trace.route, case)
                ),
                "supported_explanations": [e.identifier for e in explanations],
                "transferred_from": sorted(
                    {e.route for e in explanations if e.route != trace.route}
                ),
                "features": {
                    "supported_horizons": sorted({e.horizon for e in explanations}),
                    "path_values": sorted({e.path_value for e in explanations}),
                    "pruning": sorted({e.pruning for e in explanations}),
                    "transitions": sorted(
                        {
                            "environment" if e.table is None else "own"
                            for e in explanations
                        }
                    ),
                    # Paths this route's search evaluates; fewer when it prunes.
                    "evaluated_alternatives": sorted(
                        {
                            sum(s is not None for s in e.selectors()["evaluations"])
                            for e in explanations
                        }
                    ),
                    "effective_depth": depths,
                    "root_actions": [2] if explanations else [],
                    "amount_if_unresolved": None,
                },
                "ambiguity": len(explanations) > 1
                or any(has_supported_swap[e.identifier] for e in explanations),
            }
        )
    for row, (case, trace) in zip(population, runs):
        row["lookahead"] = lookahead_test(program, case, trace.output)
        if row["status"] == "unresolved" and not row["lookahead"]["depends"]:
            # Verified not to look ahead: its amount is known, and it is zero.
            row["status"] = "no_lookahead"
            row["unresolved_reason"] = None
            row["features"]["effective_depth"] = [0.0]
        bound = (
            paths_lower_bound(program, case, trace.route)
            if row["lookahead"]["depends"]
            else {"paths": 0, "effective_depth": 0.0}
        )
        row["lookahead"]["lower_bound"] = bound
        row["features"]["effective_depth_lower_bound"] = bound["effective_depth"]
    # The bound comes from behavior alone, so it must never exceed a verified amount.
    bound_violations = [
        r["case"]
        for r in population
        if r["status"] == "supported_planning"
        and r["features"]["effective_depth_lower_bound"]
        > r["features"]["effective_depth"][0]
    ]
    unresolved = [r for r in population if r["status"] == "unresolved"]
    by_reason = defaultdict(float)
    for row in unresolved:
        by_reason[row["unresolved_reason"]] += row["weight"]
    return {
        "frozen_digest": bank.digest,
        "state_distribution": weights,
        "utility_distribution": (
            "Uniform over the listed fresh contexts conditional on state"
        ),
        "observation_checks": len(runs),
        "explanations": validations,
        "transfers": transfers,
        "ordinary_population": population,
        "evidence_coverage": {
            "supported_mass": sum(
                r["weight"] for r in population if r["status"] == "supported_planning"
            ),
            "no_lookahead_mass": sum(
                r["weight"] for r in population if r["status"] == "no_lookahead"
            ),
            "unresolved_mass": sum(r["weight"] for r in unresolved),
            "unresolved_by_reason": dict(sorted(by_reason.items())),
            "meaning": (
                "Supported: a lower bound on the share of the state distribution with "
                "a verified planning explanation. No lookahead: the action never "
                "changed when only the future changed. Unresolved: the action depends "
                "on the future, but no model was verified."
            ),
        },
        "planningness": planningness(population),
        "lower_bound_violations": bound_violations,
    }


def effective_depth(explanation):
    """log2 of the complete paths the explanation's search evaluates on its route.

    A full depth-d search over two actions scores d; pruning lowers it.
    """
    paths = sum(s is not None for s in explanation.selectors()["evaluations"])
    return round(log2(paths), 4)


# The largest effective depth any model in the family can have.
MAX_EFFECTIVE_DEPTH = max(DEPTHS)


def planningness(population):
    """Weighted effective lookahead depth, as an interval over what is unknown.

    Inputs with a verified amount contribute it: explained planning its effective
    depth (the smallest and largest, if several explanations differ), no lookahead
    zero. Unresolved inputs look ahead by an unknown amount: they count their
    behavioral lower bound (log2 of the paths their decision provably compares) at
    the lower end and the family's maximum at the upper end. ``verified`` is the
    average over inputs with a verified amount; ``verified_mass`` is their share.
    """
    known = [
        r for r in population if r["status"] in ("supported_planning", "no_lookahead")
    ]
    mass = sum(r["weight"] for r in known)
    low = sum(r["weight"] * r["features"]["effective_depth"][0] for r in known)
    low += sum(
        r["weight"] * r["features"]["effective_depth_lower_bound"]
        for r in population
        if r["status"] == "unresolved"
    )
    high = sum(r["weight"] * r["features"]["effective_depth"][-1] for r in known)
    unknown = sum(r["weight"] for r in population) - mass
    return {
        "unit": "effective lookahead depth: log2 of complete paths evaluated",
        "interval": [
            round(float(low), 4),
            round(float(high + unknown * MAX_EFFECTIVE_DEPTH), 4),
        ],
        "verified": (
            None if not known else [round(low / mass, 4), round(high / mass, 4)]
        ),
        "verified_mass": round(mass, 4),
        "maximum_in_family": MAX_EFFECTIVE_DEPTH,
    }


class UnresolvedReasons:
    """Why an input has no supported explanation, from the frozen bank alone.

    The discovery inputs are rerun (never refitted) to learn which routes
    discovery saw; the bank's audit, when present, tells whether a search there
    was truncated.
    """

    def __init__(self, program, bank, validations):
        self.seen = {program.run(c).route for c in bank.discovery_inputs}
        self.truncated = {
            route
            for route, audit in bank.audit.get("routes", {}).items()
            if audit["truncated"]
        }
        self.status = defaultdict(list)
        for explanation, validation in zip(bank.explanations, validations):
            self.status[explanation.route].append((explanation, validation["status"]))

    def reason(self, route, case):
        """One of: unseen_route, not_covered, rejected, budget, no_model_fits."""
        if route not in self.seen:
            return "unseen_route"
        own = self.status[route]
        if any(s == "supported" and not covers(e.model, case) for e, s in own):
            return "not_covered"
        if own and all(s == "rejected" for _, s in own):
            return "rejected"
        if not own and route in self.truncated:
            return "budget"
        return "no_model_fits"
