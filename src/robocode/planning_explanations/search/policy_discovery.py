"""Milestone 8: per-part banks on development episodes, frozen, then validated.

A part is one `while` loop; a run is one execution of it. Bindings come from value
shapes at each run's first checkpoint, never from names. Explanations that match every
development run and pass sampled paired edits form the part's bank, which is frozen with
the validation episodes before any of them is executed. Accounting follows the agreed
rules: effort is summed over an episode's runs, depth is a maximum, planning events are
counted, the planned share of actions is a fraction, and coverage is kept separate.
"""

# Exact built-in types are the restricted value language.
# pylint: disable=unidiomatic-typecheck

import itertools
import math
import json
import random
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .discover import digest
from .grid_engine import (
    EIGHT,
    FOUR,
    GRAMMAR,
    MODELS,
    GridSearch,
    GridSpec,
    is_cell,
)
from .policy_causality import (
    CHAIN_RULES,
    candidate_templates,
    check_plan_chain,
    detour_edits,
    finish_chains,
    propose_rules,
)
from .policy_runner import (
    boolean_grid,
    CallRecord,
    PolicyProgram,
    RunRecord,
    observe_episode,
    run_episode,
)
from .policy_verify import (
    ROTATION_COSINE,
    WORLD_SHIFT,
    GridExplanation,
    GridMapping,
    check_edit,
    choose_edits,
    compare_run,
    execute_edit,
    explanation_from_data,
    follow_probe,
    link_plan,
    point_functions,
    readout_shape,
)
from .policy_fit import fit_role
from .policy_worklist import (
    SCAN_DEPTH,
    SCAN_LIMIT,
    scan_runs,
    unreadable_reason,
    worklist_signature,
)
from .selectors import inventory

RUN_SELECTION = (
    "per part and episode, at most `probed_runs` runs: first the last run of each "
    "call that stored a new non-empty plan (from an uninstrumented pass), then "
    "seeded draws among the rest"
)
SELECTION_RULE = (
    "Largest development mean M over the part's runs; identifier breaks ties"
)
SEARCH_PARTS = ("search", "no_explanation")
# A loop the grammar cannot bind that still shows a worklist, or that could not
# be fully inspected: unknown, never zero.
UNREADABLE = "unreadable_loop"
LOOP_RULES = (
    "numpy numbers are recorded as plain numbers, and only a list, dict, deque, "
    "set or a tuple holding one counts as omitted container data; "
    "a loop with no grammar binding is unreadable if container data was omitted, "
    "a container gains and loses items, or a growing sequence has a possible "
    "advancing integer read cursor; inspection limits also make it unreadable "
    f"(depth {SCAN_DEPTH}, {SCAN_LIMIT} containers and integer candidates per "
    "snapshot). These are bounded representation checks. Validation runs of "
    "not-search-shaped parts are re-checked the same way"
)
BANK_SCHEMA = 11
# A call following a cached plan takes the strongest evidence among the plan
# versions it follows, in this order.
FOLLOWED = {
    "supported": "followed_witnessed_plan",
    "unresolved": "followed_unresolved_plan",
    "rejected": "followed_disproved_plan",
    "unmatched": "followed_unmatched_plan",
}
CALL_CATEGORIES = (
    "ran_search",
    *FOLLOWED.values(),
    "unresolved_loop",
    "neither",
)
EFFORT = (
    "expansion_events",
    "N",
    "generated_alternatives",
    "reexpansions",
    "cost_improvements",
)
DEPTH = ("max_expanded_depth", "max_discovery_depth")
# Totals withheld when a search-shaped run is unresolved. Each keeps its explained
# part (`explained_only_*`), a lower bound: an unknown run can only add to it.
WITHHELD = (
    "planning_events",
    *EFFORT,
    *DEPTH,
    "attributed_expansion_events",
    "initialization_expansion_events",
    "search_score",
)


def search_score(runs: list[dict], calls: int) -> float:
    """Mean states expanded beyond the start, per policy call.

    Per call, N sums the distinct states expanded by its supported search runs,
    and the call scores max(N - 1, 0); a call without a search scores 0 (a
    reactive decision). The mean is over every call of the episode: the reset
    call and each action call.
    """
    expanded: dict[int, int] = {}
    for run in runs:
        if run["status"] == "supported":
            expanded[run["call"]] = expanded.get(run["call"], 0) + run["features"]["N"]
    return sum(max(n - 1, 0) for n in expanded.values()) / calls if calls else 0.0


@dataclass(frozen=True)
class PolicyLimits:
    """Explicit bounds on proposals, inventories and sampled probes."""

    candidates: int = 6000
    bindings: int = 64
    mapping_depth: int = 3
    selectors: int = 1024
    edits_per_run: int = 5
    probed_runs: int = 8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (0 if name == "mapping_depth" else 1):
                raise ValueError("Budgets must be positive integers")
        if self.mapping_depth > 4:
            raise ValueError("Mapping depth must be between zero and four")


def _returns_bool(function, cell) -> bool:
    try:
        return type(function(cell)) is bool
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _one_entry(value, root, sizes) -> bool:
    """A frontier holding one (priority, ..., root) entry; extra numbers break ties."""
    return (
        type(value) in (list, deque)
        and len(value) == 1
        and type(value[0]) is tuple
        and len(value[0]) in sizes
        and all(type(x) in (int, float) for x in value[0][:-1])
        and value[0][-1] == root
    )


def grid_bindings(runs: list[RunRecord], limits: PolicyLimits) -> tuple[list, dict]:
    """Shape-only role candidates, common to every development run of a part."""
    firsts, cut = [], []
    for run in runs:
        _, values = next(run.capture.frames())
        found, truncated = inventory(values, limits.mapping_depth, limits.selectors)
        if truncated:
            cut.append(run.ordinal)
        firsts.append((found, run.capture.callables))
    common = sorted(set.intersection(*(set(found) for found, _ in firsts)))
    named = sorted(set.intersection(*(set(c) for _, c in firsts)))
    roles: list[dict] = []
    limited = False
    for root in common:
        roots = [found[root] for found, _ in firsts]
        if not all(is_cell(v) for v in roots):
            continue

        def paths(test, root=root, roots=roots) -> list[str]:
            return [
                p
                for p in common
                if p != root
                and all(test(found[p], r) for (found, _), r in zip(firsts, roots))
            ]

        pairs = paths(lambda v, r: _one_entry(v, r, (2,)))
        tagged = paths(lambda v, r: _one_entry(v, r, range(3, 8)))
        states = paths(lambda v, r: type(v) in (list, deque) and list(v) == [r])
        # The root's parent entry is None, or absent as in the PDF's algorithm
        # ("store no parent for it").
        parents = [
            (p, "none") for p in paths(lambda v, r: type(v) is dict and v == {r: None})
        ] + [(p, "absent") for p in paths(lambda v, r: type(v) is dict and not v)]
        costs = paths(
            lambda v, r: type(v) is dict
            and len(v) == 1
            and r in v
            and type(v[r]) in (int, float)
            and v[r] == 0
        )
        goals = paths(lambda v, r: is_cell(v))
        models = [
            n
            for n in named
            if all(_returns_bool(c[n], r) for (_, c), r in zip(firsts, roots))
        ]
        readouts = [
            n
            for n in named
            if all(readout_shape(c[n], r) for (_, c), r in zip(firsts, roots))
        ]
        # A boolean array the run already holds, indexed by the root cell.
        grids = paths(
            lambda v, r: boolean_grid(v)
            and v.ndim == 2
            and 0 <= r[0] < v.shape[0]
            and 0 <= r[1] < v.shape[1]
        )
        for layout, frontiers in (
            ("pairs", pairs),
            ("tagged", tagged),
            ("states", states),
        ):
            for frontier, (parent, root_parent), cost in itertools.product(
                frontiers, parents, costs
            ):
                if len(roles) == limits.bindings:
                    limited = True
                    break
                roles.append(
                    {
                        "root": root,
                        "frontier": frontier,
                        "parents": parent,
                        "root_parent": root_parent,
                        "costs": cost,
                        "layout": layout,
                        "goals": goals,
                        "models": models,
                        "grids": grids,
                        "readouts": readouts,
                    }
                )
    audit = {"inventory_cut_runs": cut, "binding_limit": limited}
    return roles, audit


# Fitted ingredients that equal a v1 menu item are written as that item, so the
# same explanation never appears under two identities.
_MENU_MOVES = {
    tuple((di, dj, math.hypot(di, dj)) for di, dj in offsets): size
    for offsets, size in ((FOUR, 4), (EIGHT, 8))
}
_MENU_HEURISTICS = {
    ("euclidean", (1.0,)): "euclidean",
    ("octile", (1.0, 1.0)): "manhattan",
}


def grid_specifications(
    layout: str, root_parent: str = "none", fit: dict | None = None, goal: str = ""
) -> Iterator[GridSpec]:
    """The finite ingredient grammar for one frontier layout.

    Without `fit`, these are the grid-priority-v1 menu items (4/8 neighbors with
    straight-line step costs; straight-line or city-block heuristics). With `fit`,
    the move table, heuristic and entry fields proposed from development runs
    replace those menu items; every other choice is enumerated the same way.
    """
    selections = {
        "pairs": ("min", "fifo", "lifo"),
        "tagged": ("min",),
        "states": ("fifo", "lifo"),
    }[layout]
    if fit is None:
        priorities: tuple = (
            (
                ("g", "none", "", ()),
                ("g+h", "euclidean", "", ()),
                ("g+h", "manhattan", "", ()),
                ("h", "euclidean", "", ()),
                ("h", "manhattan", "", ()),
            )
            if layout in ("pairs", "tagged")
            else (("g", "none", "", ()),)
        )
        neighborhoods: tuple = (4, 8)
        extra: dict = {}
    else:
        fitted = []
        for p, form, coef in fit["heuristics"].get(goal, []):
            menu = _MENU_HEURISTICS.get((form, coef))
            if p == "g":
                fitted.append((p, "none", "", ()))
            elif menu:
                fitted.append((p, menu, "", ()))
            else:
                fitted.append((p, "fitted", form, coef))
        priorities = tuple(
            dict.fromkeys([("g", "none", "", ()), *fitted])
            if layout in ("pairs", "tagged")
            else [("g", "none", "", ())]
        )
        size = _MENU_MOVES.get(fit["moves"])
        neighborhoods = (size,) if size else (0,)
        extra = {"moves": () if size else fit["moves"], "fields": fit["fields"]}
    for selection, (priority, heuristic, h_form, h_coef) in itertools.product(
        selections, priorities
    ):
        for (
            root_priority,
            timing,
            relaxation,
            stale,
            neighborhood,
            model,
        ) in itertools.product(
            ("zero",) if priority == "g" else ("zero", "rule"),
            ("selection", "generation", "none"),
            ("first", "improve"),
            ("reexpand", "skip") if selection == "min" else ("reexpand",),
            neighborhoods,
            MODELS,
        ):
            sources = ("costs", "entry") if extra.get("fields") else ("costs",)
            for corners, g_source in itertools.product(("cut", "no_cut"), sources):
                try:
                    yield GridSpec(
                        selection,
                        priority,
                        heuristic,
                        root_priority,
                        timing,
                        relaxation,
                        stale,
                        neighborhood,
                        model,
                        layout,
                        h_form=h_form,
                        h_coef=h_coef,
                        corners=corners,
                        root_parent=root_parent,
                        g_source=g_source,
                        **extra,
                    )
                except ValueError:
                    continue


def _model_paths(role: dict, model: str) -> list[str]:
    if model == "own":
        return role["models"]
    if model in ("grid_free", "grid_blocked"):
        return role["grids"]
    return [""]


def proposals(
    site: str, roles: list[dict], fits: list[dict] | None = None
) -> Iterator[GridExplanation]:
    """Every bounded ingredient/binding combination; no names are consulted.

    Ingredients fitted on development runs come first (one set per binding and
    goal), then the fixed menu. Both are verified the same way.
    """
    fits = fits or [None] * len(roles)
    seen: set = set()
    for explanation in _proposals(site, roles, fits):
        if explanation.identifier not in seen:
            seen.add(explanation.identifier)
            yield explanation


def _proposals(site: str, roles: list[dict], fits: list) -> Iterator[GridExplanation]:
    for role, fit in zip(roles, fits):
        if fit is None or fit["moves"] is None:
            continue
        for goal in role["goals"] or [""]:
            for spec in grid_specifications(
                role["layout"], role["root_parent"], fit, goal
            ):
                if spec.needs_goal and not goal:
                    continue
                uses_goal = spec.needs_goal or spec.heuristic == "fitted"
                for model in _model_paths(role, spec.model):
                    if not uses_goal and goal != (role["goals"] or [""])[0]:
                        continue
                    yield GridExplanation(
                        spec,
                        GridMapping(
                            site,
                            role["root"],
                            role["frontier"],
                            role["parents"],
                            role["costs"],
                            goal if spec.needs_goal else "",
                            model,
                        ),
                    )
    for role in roles:
        for spec in grid_specifications(role["layout"], role["root_parent"]):
            goals = role["goals"] if spec.needs_goal else [""]
            models = _model_paths(role, spec.model)
            for goal, model in itertools.product(goals, models):
                yield GridExplanation(
                    spec,
                    GridMapping(
                        site,
                        role["root"],
                        role["frontier"],
                        role["parents"],
                        role["costs"],
                        goal,
                        model,
                    ),
                )


def check_partition(development, validation) -> None:
    """Full (count, seed) cases must be disjoint and non-empty."""
    if not development or not validation:
        raise ValueError("Development and validation episodes are both required")
    if set(development) & set(validation):
        raise ValueError("Validation reuses a development episode")


@dataclass(frozen=True)
class FrozenPolicyBank:
    """Everything fixed before validation: banks, choices and held-out episodes."""

    source_digest: str
    domain: dict
    development: tuple[tuple[int, int], ...]
    validation: tuple[tuple[int, int], ...]
    parts: dict
    limits: PolicyLimits
    audit: dict = field(default_factory=dict, compare=False)

    def manifest(self) -> dict:
        """Only fixed experiment choices, excluding the discovery audit."""
        return {
            "schema": BANK_SCHEMA,
            "grammar": GRAMMAR,
            "domain": self.domain,
            "source_digest": self.source_digest,
            "development": [list(c) for c in self.development],
            "validation": [list(c) for c in self.validation],
            "weights": [1 / len(self.validation)] * len(self.validation),
            "parts": {
                site: {
                    "status": part["status"],
                    "explanations": [asdict(e) for e in part["explanations"]],
                    "selected": part["selected"],
                    "readout": part["readout"],
                    "plan_rules": part.get("plan_rules", []),
                }
                for site, part in sorted(self.parts.items())
            },
            "limits": asdict(self.limits),
            "probe_rules": {
                "edits_per_run": self.limits.edits_per_run,
                "probed_runs_per_part_per_episode": self.limits.probed_runs,
                "rotation_degrees": 90,
                "rotation_cosine": ROTATION_COSINE,
                "world_shift": WORLD_SHIFT,
                "seeds": "phase:case:call:site:ordinal strings; runs by phase:case:site",
                "run_selection": RUN_SELECTION,
                "loops_without_binding": LOOP_RULES,
                "cross_call": CHAIN_RULES,
            },
            "selection_rule": SELECTION_RULE,
        }

    @property
    def fingerprint(self) -> str:
        """Freeze source, grammar, banks, choices and validation episodes together."""
        return digest(self.manifest())

    def save(self, path: Path) -> None:
        """Persist the manifest before validation executes the policy."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.manifest(), "fingerprint": self.fingerprint}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_policy_bank(path: Path) -> FrozenPolicyBank:
    """Reload a bank and verify its integrity hash; nothing is rediscovered."""
    data = json.loads(path.read_text())
    if data.get("schema") != BANK_SCHEMA:
        raise ValueError(
            f"Rediscover this bank: schema {BANK_SCHEMA} proposes move tables, "
            "heuristic weights and heap-entry numbers from development runs, reads "
            "boolean-grid models, records small numpy arrays and tuples of lists, "
            "and reads partial (smoothed) plans and point arrays"
        )
    if data.get("grammar") != GRAMMAR:
        raise ValueError("Frozen bank uses a different grammar")
    parts = {
        site: {
            "status": part["status"],
            "explanations": [explanation_from_data(e) for e in part["explanations"]],
            "selected": part["selected"],
            "readout": part["readout"],
            "plan_rules": part["plan_rules"],
        }
        for site, part in data["parts"].items()
    }
    bank = FrozenPolicyBank(
        data["source_digest"],
        data["domain"],
        tuple(tuple(c) for c in data["development"]),
        tuple(tuple(c) for c in data["validation"]),
        parts,
        PolicyLimits(**data["limits"]),
    )
    if bank.fingerprint != data["fingerprint"]:
        raise ValueError("Frozen bank integrity check failed")
    return bank


def _sample_runs(counts: dict, producers: dict, phase: str, case, limit: int) -> dict:
    """Which runs of each part get paired edits; fixed before any result.

    Runs that may have produced a new stored plan come first; seeded draws fill
    the remaining places (or choose among producers when they exceed the limit).
    """
    chosen = {}
    for site, count in counts.items():
        rng = random.Random(f"{phase}:{case}:{site}:runs")
        first = sorted(producers.get(site, ()))
        if len(first) >= limit:
            chosen[site] = set(rng.sample(first, limit))
            continue
        taken = set(first)
        rest = [i for i in range(count) if i not in taken]
        fill = rng.sample(rest, min(limit - len(first), len(rest)))
        chosen[site] = taken | set(fill)
    return chosen


def _new_plan(old: tuple, new: tuple) -> bool:
    """A call stored a new, non-empty plan rather than consuming the old one."""
    if not new or new == old:
        return False
    consumed = len(new) < len(old) and old[-len(new) :] == new
    return not consumed


def _count_runs(program: PolicyProgram, domain, case) -> tuple[dict, dict, list]:
    """Runs per part, and the runs that may have produced a new stored plan.

    In a call that stores a new non-empty plan, the last run of each part is a
    candidate producer: a later run of the same loop would overwrite the plan.
    Only the policy's ordinary behavior is used, never a probe's result.
    """
    trace = run_episode(program, domain, case, capture=False)
    counts: dict[str, int] = {}
    producers: dict[str, set] = {}
    for call in trace.calls:
        last: dict[str, int] = {}
        for run in call.runs:
            last[run.site] = counts.get(run.site, 0)
            counts[run.site] = last[run.site] + 1
        if any(
            _new_plan(call.plans_before.get(a, ()), call.plans_after.get(a, ()))
            for a in set(call.plans_before) | set(call.plans_after)
        ):
            for site, index in last.items():
                producers.setdefault(site, set()).add(index)
    return counts, producers, [c.action for c in trace.calls]


class EpisodeAnalyzer:
    """Online, per-call analysis; each call's frames are released afterwards.

    `parts[site]` holds the explanations to check, an optional preselected identifier,
    and the read-out names to try when linking plans.
    """

    def __init__(self, program, domain, parts: dict, case, phase: str, limits) -> None:
        self.program, self.domain, self.parts = program, domain, parts
        self.case, self.phase, self.limits = case, phase, limits
        self.sampled: dict[str, set] = {}
        self.part_index: dict[str, int] = {}
        self.runs: list[dict] = []
        self.calls: list[dict] = []
        self.versions: dict[str, dict] = {}
        self.history: list[dict] = []
        self.actions: list = []
        self.executions: dict = {}
        self.replays: list = []

    def __call__(self, call: CallRecord, policy, state, extra, after) -> None:
        self.actions.append(call.action)
        if self.phase == "development":
            # Point arrays the policy holds are read-out candidates, like callables.
            arrays = sorted(point_functions({}, after))
            for part in self.parts.values():
                if part["status"] not in SEARCH_PARTS:
                    continue
                # The two lists may be one object; extend each list only once.
                lists = [part["readouts"]]
                candidates = part.get("readout_candidates")
                if candidates is not None and candidates is not part["readouts"]:
                    lists.append(candidates)
                for names in lists:
                    names.extend(n for n in arrays if n not in names)
        # Replays stop when a recognised search runs again. An unknown loop does
        # not stop them: if it replaces the cache, the replay's cache check stops
        # it, and the plan-only and restoration controls test mediation anyway.
        search_sites = {s for s, p in self.parts.items() if p["status"] in SEARCH_PARTS}
        for replay in self.replays:
            replay.step(self.program, self.domain, call, state, extra, search_sites)
        self.replays = [r for r in self.replays if r.evidence["status"] == "pending"]
        trees = []
        statuses = set()
        unexplained = False
        for run in call.runs:
            entry, tree = self._run(call, run, (policy, state, extra), after)
            entry["index"] = len(self.runs)
            self.runs.append(entry)
            statuses.add(entry["part_status"])
            # A loop that may be a search but is not explained could have made
            # any plan this call stores.
            unexplained |= (
                entry["status"] != "supported"
                and entry["part_status"] != "not_search_shaped"
            )
            if tree is not None:
                trees.append(
                    (entry, tree, point_functions(run.capture.callables, after))
                )
        self._versions(call, trees, after, unexplained)
        if call.method == "get_action":
            self.calls.append(self._classify(call, (policy, state, extra), statuses))
        # Release frames only now: probes of a later run replay the earlier ones.
        for run in call.runs:
            run.capture.release()
        self.executions.clear()

    def _run(self, call, run, before, after) -> tuple[dict, GridSearch | None]:
        index = self.part_index.get(run.site, 0)
        self.part_index[run.site] = index + 1
        part = self.parts.get(run.site)
        entry: dict = {
            "call": call.index,
            "method": call.method,
            "methods": list(run.methods),
            "site": run.site,
            "ordinal": run.ordinal,
            "visits": run.visits,
            "part_status": part["status"] if part else "unseen_part",
            "explanations": {},
            "probed": False,
        }
        if not part or part["status"] != "search":
            shaped = part is not None and part["status"] == "not_search_shaped"
            if shaped:
                # The frozen "not a search" verdict is re-checked on this run.
                signature = worklist_signature(run)
                reason = unreadable_reason(signature)
                if reason:
                    entry.update(
                        part_status=UNREADABLE, unreadable=reason, worklist=signature
                    )
                    shaped = False
            entry["status"] = "not_search_shaped" if shaped else "unresolved"
            return entry, None
        engines = {}
        for explanation in part["explanations"]:
            result, engine = compare_run(explanation, run)
            entry["explanations"][explanation.identifier] = result
            if engine is not None:
                engines[explanation.identifier] = (explanation, engine)
        if index in self.sampled.get(run.site, ()):
            entry["probed"] = True
            self._probe(call, run, before, after, entry, engines)
        selected = part["selected"]
        if selected is None:
            supported = [
                i
                for i, r in entry["explanations"].items()
                if r["status"] == "supported"
            ]
            selected = supported[0] if supported else None
        chosen = entry["explanations"].get(selected) if selected else None
        entry["selected"] = selected
        if chosen is None:
            entry["status"] = "unresolved"
        elif chosen["status"] != "supported":
            entry["status"] = "rejected"
        else:
            entry.update(
                status="supported",
                features=chosen["features"],
                reached_goal=chosen["reached_goal"],
            )
            return entry, engines[selected][1]
        return entry, None

    def _probe(self, call, run, before, after, entry, engines) -> None:
        seed = f"{self.phase}:{self.case}:{call.index}:{run.site}:{run.ordinal}"
        executed: dict = {}
        part = self.parts[run.site]
        plans = {
            a: v
            for a, v in call.plans_after.items()
            if v and v != call.plans_before.get(a)
        }
        templates = (
            part.get("plan_rules", [])
            if self.phase == "validation"
            else candidate_templates(plans, part["readouts"])
        )
        for identifier, (explanation, engine) in engines.items():
            result = entry["explanations"][identifier]
            records = []
            runtime = []
            edits = choose_edits(explanation, run, seed, self.limits.edits_per_run)
            # The stored plan is read with the policy as the call left it.
            functions = point_functions(run.capture.callables, after)
            for edit in detour_edits(
                explanation, run, engine, plans, templates, functions
            ):
                if edit not in edits:
                    edits.append(edit)
            for edit in edits:
                if edit not in executed:
                    executed[edit] = execute_edit(
                        self.program, call, before, edit, self.domain
                    )
                record = check_edit(explanation, executed[edit], edit, engine.result())
                records.append(record)
                runtime.append((edit, executed[edit], record))
            self.executions[(run.site, run.ordinal, identifier)] = runtime
            result["probes"] = records
            failed = [r for r in records if not r["passed"]]
            if failed:
                result.update(
                    status="rejected", reason="intervention", counterexample=failed[0]
                )

    def _versions(self, call: CallRecord, trees: list, after, unexplained) -> None:
        """A plan version starts when a call replaces (not merely consumes) a plan."""
        for attribute in sorted(set(call.plans_before) | set(call.plans_after)):
            old = call.plans_before.get(attribute, ())
            new = call.plans_after.get(attribute, ())
            if old == new:
                continue
            links: dict[str, int] = {}
            producer = None
            for entry, tree, functions in trees:
                for name in self.parts[entry["site"]]["readouts"]:
                    if name in functions and link_plan(
                        new, tree.parents, functions[name]
                    ):
                        links.setdefault(name, entry["index"])
                        producer = entry["index"]
            consumed = bool(new) and len(new) < len(old) and old[-len(new) :] == new
            if producer is None and consumed:
                continue
            # A plan no search in this call produced that is identical to another
            # plan the policy already holds (`self.path = self.delivery`) is a
            # copy: it carries that plan's provenance, not "no source".
            copied_from = None
            if producer is None and new:
                for other, points in sorted(call.plans_after.items()):
                    if other != attribute and points == new and other in self.versions:
                        copied_from = self.versions[other]["index"]
                        break
            version: dict = {
                "attribute": attribute,
                "call": call.index,
                "waypoints": len(new),
                "producer": producer,
                "links": links,
                "consumers": 0,
                "world_insensitive": 0,
                "chains": [],
                "causal_status": "unresolved",
                "unexplained_loop_in_call": unexplained,
                "copied_from": copied_from,
            }
            for entry, engine, functions in trees:
                part = self.parts[entry["site"]]
                if self.phase == "development":
                    rules = part.setdefault("plan_rules", [])
                    for rule in propose_rules(
                        attribute, new, engine, functions, part["readouts"]
                    ):
                        if rule not in rules:
                            rules.append(rule)
                explanation = next(
                    e for e in part["explanations"] if e.identifier == entry["selected"]
                )
                runtime = self.executions.get(
                    (entry["site"], entry["ordinal"], entry["selected"]), []
                )
                for rule in part.get("plan_rules", []):
                    if rule["attribute"] != attribute:
                        continue
                    chain, replays = check_plan_chain(
                        rule, new, engine, functions, explanation, runtime, after
                    )
                    chain["producer"] = entry["index"]
                    version["chains"].append(chain)
                    self.replays.extend(replays)
            version["index"] = len(self.history)
            self.history.append(version)
            self.versions[attribute] = version

    def _classify(self, call: CallRecord, before, statuses: set) -> dict:
        record: dict = {"call": call.index, "probes": []}
        if statuses & set(SEARCH_PARTS):
            record["category"] = "ran_search"
            return record
        # A loop not known to be a search (or not): it may drive this action, so
        # the call is not "neither". It does not hide plan evidence, though.
        unknown = bool(statuses - {"not_search_shaped"})
        record["unknown_loop"] = unknown
        category = "neither"
        for attribute, points in sorted(call.plans_before.items()):
            if not points:
                continue
            probe = follow_probe(self.program, self.domain, call, before, attribute)
            version = self.versions.get(attribute)
            # Structural tree-shape match only; never causal evidence by itself.
            probe["linked"] = bool(version and version["producer"] is not None)
            probe["version"] = version["index"] if version else None
            record["probes"].append(probe)
            if not probe["passed"] or version is None:
                continue
            if unknown and version["call"] >= call.index:
                # The unknown loop may have replaced this plan during the call.
                probe["stored_this_call"] = True
                continue
            version["consumers"] += 1
            version["world_insensitive"] += bool(probe.get("world_insensitive"))
            # Provisional: the evidence category is known once replays finish.
            category = "followed_plan"
        if category == "neither" and unknown:
            category = "unresolved_loop"
        record["category"] = category
        return record


def analyze_episode(
    program: PolicyProgram, domain, parts: dict, case, phase: str, limits
) -> dict:
    """Plain run, a counting run, then one replay with every check and probe."""
    plain = run_episode(program, domain, case, instrumented=False)
    counts, producers, counted = _count_runs(program, domain, case)
    analyzer = EpisodeAnalyzer(program, domain, parts, case, phase, limits)
    analyzer.sampled = _sample_runs(counts, producers, phase, case, limits.probed_runs)
    full = run_episode(program, domain, case, after_call=analyzer)
    actions = [c.action for c in plain.calls]
    if actions != counted or actions != analyzer.actions:
        raise AssertionError("Observation changed the policy's actions")
    finish_chains(analyzer.history)
    # Rotation identifies cache consumers; the followed version's chain evidence
    # says whether that cache is known to come from search.
    for call in analyzer.calls:
        if call["category"] == "followed_plan":
            _resolve_followed(call, analyzer.history)
    _attribute_use(analyzer.runs, analyzer.history)
    return {
        "case": list(case),
        "solved": full.solved,
        "steps": full.steps,
        "runs": analyzer.runs,
        "calls": analyzer.calls,
        "versions": analyzer.history,
        "run_counts": counts,
    }


def _resolve_followed(call: dict, versions: list[dict]) -> None:
    """A plan-following call takes the strongest evidence among followed versions."""
    followed = [p for p in call["probes"] if p["passed"] and p["version"] is not None]
    statuses = {versions[p["version"]]["causal_status"] for p in followed}
    status = next(s for s in FOLLOWED if s in statuses)
    call["category"] = FOLLOWED[status]
    if status == "supported":
        call["world_insensitive"] = any(
            p.get("world_insensitive")
            for p in followed
            if versions[p["version"]]["causal_status"] == "supported"
        )


def _attribute_use(runs: list[dict], versions: list[dict]) -> None:
    """How each supported run's result reached behavior, if it did."""
    followed = {
        c["producer"]
        for v in versions
        for c in v["chains"]
        if c["status"] == "supported"
    }
    for run in runs:
        if run["status"] != "supported":
            continue
        probes = (run["explanations"].get(run["selected"]) or {}).get("probes", [])
        if run["index"] in followed:
            run["use"] = "plan_followed"
        elif any(p.get("observed_action_changed") for p in probes):
            run["use"] = "edit_changed_action"
        elif any(p.get("observed_plans_changed") for p in probes):
            run["use"] = "edit_changed_plan"
        elif run["probed"]:
            run["use"] = "no_use_found"
        else:
            run["use"] = "untested"


def _mean(values: list[dict], key: str) -> float | None:
    if not values or any(v.get(key) is None for v in values):
        return None
    return sum(float(v[key]) for v in values) / len(values)


def account(episode: dict) -> dict:
    """Per-part tables plus the program row, combined by what each number means."""
    parts: dict[str, dict] = {}
    for run in episode["runs"]:
        row = parts.get(run["site"])
        if run["part_status"] == "not_search_shaped":
            if row is None:
                row = parts[run["site"]] = {"runs": 0, "status": "not_search_shaped"}
            row["runs"] += 1
            continue
        if row is None or "supported" not in row:
            # A "not a search" part can have runs re-checked as unreadable in
            # validation; its row then counts every run, with the unknown ones
            # under unresolved_or_rejected.
            row = parts[run["site"]] = {
                "runs": row["runs"] if row else 0,
                "status": run["part_status"],
                "supported": 0,
                "unresolved_or_rejected": 0,
                "initialization_runs": 0,
                **{k: 0 for k in EFFORT},
                **{k: 0 for k in DEPTH},
                "attributed_expansion_events": 0,
                "plan_versions_linked": 0,
            }
        row["runs"] += 1
        row["initialization_runs"] += run["call"] == 0
        if run["status"] != "supported":
            row["unresolved_or_rejected"] += 1
            continue
        row["supported"] += 1
        for key in EFFORT:
            row[key] += run["features"][key]
        for key in DEPTH:
            row[key] = max(row[key], run["features"][key])
        if run["use"] in ("plan_followed", "edit_changed_action"):
            row["attributed_expansion_events"] += run["features"]["expansion_events"]
        row["plan_versions_linked"] += len(
            [v for v in episode["versions"] if v["producer"] == run["index"]]
        )
    search_runs = [
        r for r in episode["runs"] if r["part_status"] != "not_search_shaped"
    ]
    complete = all(r["status"] == "supported" for r in search_runs)
    supported = [r for r in search_runs if r["status"] == "supported"]
    calls = episode["calls"]
    categories = {
        name: sum(c["category"] == name for c in calls) for name in CALL_CATEGORIES
    }
    followed = [c for c in calls if c["category"] == "followed_witnessed_plan"]
    # The share's max adds calls whose plan evidence is unresolved and calls that
    # ran a loop never classified in development. Disproved and unmatched plans
    # are excluded from both ends.
    possible = (
        categories["followed_witnessed_plan"]
        + categories["followed_unresolved_plan"]
        + categories["unresolved_loop"]
    )
    program = {
        "complete": complete,
        "search_runs": len(search_runs),
        "supported_runs": len(supported),
        "planning_events": sum(
            r["features"]["expansion_events"] > 0 for r in supported
        ),
        **{k: sum(r["features"][k] for r in supported) for k in EFFORT},
        **{k: max((r["features"][k] for r in supported), default=0) for k in DEPTH},
        "initialization_runs": sum(r["call"] == 0 for r in search_runs),
        "initialization_expansion_events": sum(
            r["features"]["expansion_events"] for r in supported if r["call"] == 0
        ),
        "attributed_expansion_events": sum(
            r["features"]["expansion_events"]
            for r in supported
            if r["use"] in ("plan_followed", "edit_changed_action")
        ),
        "runs_with_untested_use": sum(r["use"] == "untested" for r in supported),
        "runs_without_action_witness": sum(
            r["use"] not in ("plan_followed", "edit_changed_action") for r in supported
        ),
        "action_calls": len(calls),
        **categories,
        "followed_plan_share": (
            categories["followed_witnessed_plan"] / len(calls) if calls else None
        ),
        "followed_plan_share_max": possible / len(calls) if calls else None,
        "searched_share": categories["ran_search"] / len(calls) if calls else None,
        "world_insensitive_share": (
            sum(bool(c.get("world_insensitive")) for c in followed) / len(followed)
            if followed
            else None
        ),
        "non_search_loop_runs": len(episode["runs"]) - len(search_runs),
        # Every call counts: the reset call plus each action call.
        "search_score": search_score(search_runs, len(calls) + 1),
    }
    if not complete:
        # An unresolved run is not zero; withhold totals rather than undercount.
        for key in WITHHELD:
            program[f"explained_only_{key}"] = program[key]
            program[key] = None
    return {"parts": parts, "program": program}


def known_part(row: dict, key: str):
    """A total, or if it is withheld, its explained part (a lower bound)."""
    value = row.get(key)
    return row.get(f"explained_only_{key}") if value is None else value


def summarize(episodes: list[dict]) -> dict:
    """Uniform means over validation episodes; withheld if any episode is."""
    rows = [e["accounting"]["program"] for e in episodes]
    bounds = [{k: known_part(r, k) for k in WITHHELD} for r in rows]
    keys = (
        "search_score",
        "planning_events",
        *EFFORT,
        "attributed_expansion_events",
        "runs_with_untested_use",
        "runs_without_action_witness",
        "initialization_expansion_events",
        "action_calls",
        "followed_plan_share",
        "followed_plan_share_max",
        "searched_share",
    )
    return {
        "episodes": len(rows),
        "complete": all(r["complete"] for r in rows),
        "mean": {k: _mean(rows, k) for k in keys},
        "max": {
            k: (
                None
                if any(r[k] is None for r in rows)
                else max((r[k] for r in rows), default=0)
            )
            for k in DEPTH
        },
        # Means and maxima of known parts: lower bounds where totals are withheld.
        "lower_bound": {
            k: (
                _mean(bounds, k)
                if k not in DEPTH
                else (
                    None
                    if any(b[k] is None for b in bounds)
                    else max((b[k] for b in bounds), default=0)
                )
            )
            for k in WITHHELD
        },
        "solved": sum(e["solved"] for e in episodes),
    }


def _best(scores: dict) -> str:
    """The highest score; the identifier or name breaks ties."""
    return min(scores, key=lambda key: (-scores[key], key))


def _bank_parts(bank_parts: dict) -> dict:
    """Validation view: frozen explanations, preselection and one read-out."""
    return {
        site: {
            "status": part["status"],
            "explanations": part["explanations"],
            "selected": part["selected"],
            "readouts": [part["readout"]] if part["readout"] else [],
            "plan_rules": part.get("plan_rules", []),
        }
        for site, part in bank_parts.items()
    }


def discover_policy(
    program: PolicyProgram,
    domain,
    development,
    validation,
    limits: PolicyLimits = PolicyLimits(),
) -> FrozenPolicyBank:
    """Bank every part on development episodes; held-out episodes stay unexecuted."""
    development, validation = tuple(development), tuple(validation)
    check_partition(development, validation)
    traces = {case: observe_episode(program, domain, case) for case in development}
    runs: dict[str, list[RunRecord]] = {}
    for trace in traces.values():
        for call in trace.calls:
            for run in call.runs:
                runs.setdefault(run.site, []).append(run)
    audit: dict = {"parts": {}, "tested": 0, "candidate_budget": False}
    parts: dict[str, dict] = {}
    for site in sorted(program.sites):
        if site not in runs:
            parts[site] = {"status": "not_executed_in_development", "explanations": []}
            continue
        roles, bound = grid_bindings(runs[site], limits)
        record = {"runs": len(runs[site]), "bindings": len(roles), **bound, "tested": 0}
        audit["parts"][site] = record
        if not roles:
            signature = scan_runs(runs[site])
            record["worklist"] = signature
            reason = unreadable_reason(signature)
            record["unreadable"] = reason
            status = UNREADABLE if reason else "not_search_shaped"
            parts[site] = {"status": status, "explanations": []}
            continue
        survivors = []
        # Numbers for the fixed algorithm, proposed from development runs only.
        fits = [fit_role(role, runs[site]) for role in roles]
        record["fitted"] = [
            {
                "moves": fit["moves"],
                "heuristics": fit["heuristics"],
                "fields": fit["fields"],
                "observed_edges": fit["observed_edges"],
                "observed_entries": fit["observed_entries"],
            }
            for fit in fits
        ]
        for explanation in proposals(site, roles, fits):
            if audit["tested"] >= limits.candidates:
                audit["candidate_budget"] = True
                break
            audit["tested"] += 1
            record["tested"] += 1
            if all(
                compare_run(explanation, run)[0]["status"] == "supported"
                for run in runs[site]
            ):
                survivors.append(explanation)
        record["ordinary_survivors"] = len(survivors)
        readouts = sorted({name for role in roles for name in role["readouts"]})
        parts[site] = {
            "status": "search" if survivors else "no_explanation",
            "explanations": survivors,
            "selected": None,
            "readouts": readouts,
            "readout_candidates": readouts,
        }
    del traces, runs
    # Paired edits and plan links on development; validation is still untouched.
    analyses = [
        analyze_episode(program, domain, parts, case, "development", limits)
        for case in development
    ]
    for site, part in parts.items():
        if part["status"] != "search":
            part.setdefault("selected", None)
            part["readout"] = ""
            continue
        site_runs = [r for a in analyses for r in a["runs"] if r["site"] == site]
        bank, means = [], {}
        for explanation in part["explanations"]:
            identifier = explanation.identifier
            results = [r["explanations"][identifier] for r in site_runs]
            if all(r["status"] == "supported" for r in results):
                bank.append(explanation)
                means[identifier] = sum(r["features"]["M"] for r in results) / len(
                    results
                )
        audit["parts"][site]["bank"] = len(bank)
        part["explanations"] = bank
        part["status"] = "search" if bank else "no_explanation"
        part["selected"] = _best(means) if means else None
        counts = {
            name: sum(
                name in v["links"]
                for a in analyses
                for v in a["versions"]
                if v["producer"] is not None
                and a["runs"][v["producer"]]["site"] == site
            )
            for name in part["readout_candidates"]
        }
        audit["parts"][site]["readout_links"] = counts
        best = _best(counts) if counts else ""
        part["readout"] = best if best and counts[best] else ""
    frozen = {
        site: {
            "status": p["status"],
            "explanations": p["explanations"],
            "selected": p.get("selected"),
            "readout": p.get("readout", ""),
            "plan_rules": p.get("plan_rules", []),
        }
        for site, p in parts.items()
    }
    audit["development"] = [
        {"case": a["case"], "accounting": account(a)} for a in analyses
    ]
    return FrozenPolicyBank(
        program.digest,
        domain.describe(),
        development,
        validation,
        frozen,
        limits,
        audit,
    )


def validate_policy(program: PolicyProgram, domain, bank: FrozenPolicyBank) -> dict:
    """Fresh episodes under the frozen bank; nothing is refitted or reselected."""
    if program.digest != bank.source_digest:
        raise ValueError("Frozen bank belongs to a different source")
    parts = _bank_parts(bank.parts)
    episodes = []
    for case in bank.validation:
        episode = analyze_episode(
            program, domain, parts, case, "validation", bank.limits
        )
        episode["accounting"] = account(episode)
        episodes.append(episode)
    statuses = {}
    for site, part in bank.parts.items():
        if part["status"] != "search":
            continue
        runs = [r for e in episodes for r in e["runs"] if r["site"] == site]
        statuses[site] = {
            e.identifier: (
                "not_executed"
                if not runs
                else (
                    "supported"
                    if all(
                        r["explanations"][e.identifier]["status"] == "supported"
                        for r in runs
                    )
                    else "rejected"
                )
            )
            for e in part["explanations"]
        }
    return {
        "fingerprint": bank.fingerprint,
        "episodes": episodes,
        "bank_validation": statuses,
        "summary": summarize(episodes),
    }
