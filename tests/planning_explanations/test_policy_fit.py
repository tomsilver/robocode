"""Ingredients proposed from development runs, then verified strictly (schema 9)."""

import heapq
import json
import unittest
from dataclasses import asdict

import numpy as np

from robocode.planning_explanations.search.grid_engine import GridSearch, GridSpec
from robocode.planning_explanations.search.policy_discovery import (
    discover_policy,
    validate_policy,
)
from robocode.planning_explanations.search.policy_domains import DiscWorld
from robocode.planning_explanations.search.policy_fit import (
    fit_fields,
    fit_heuristics,
    fit_moves,
)
from robocode.planning_explanations.search.policy_fixtures import GRID_OCTILE
from robocode.planning_explanations.search.policy_runner import (
    PolicyProgram,
    Recorder,
    RunCapture,
    Session,
    RECORDER,
    same_snapshot,
)
from robocode.planning_explanations.search.policy_verify import (
    GridExplanation,
    GridMapping,
    explanation_from_data,
    grid_predicate,
)

EIGHT_MOVES = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, 1.414),
    (1, -1, 1.414),
    (-1, 1, 1.414),
    (-1, -1, 1.414),
)


def octile(cell, goal, weight=1.02, ratio=0.414) -> float:
    """Astra's heuristic, written the way its source computes it."""
    ax, ay = abs(goal[0] - cell[0]), abs(goal[1] - cell[1])
    return (max(ax, ay) + ratio * min(ax, ay)) * weight


class RecorderTest(unittest.TestCase):
    """Snapshots needed by the new mappings."""

    def test_a_return_inside_a_loop_ends_its_run(self):
        """A return checkpoint closes the outer loop; nested function returns do not."""
        source = (
            "class GeneratedApproach:\n"
            "    def __init__(self, a, o, p):\n"
            "        pass\n"
            "    def find(self, n):\n"
            "        todo = [0]\n"
            "        while todo:\n"
            "            x = todo.pop()\n"
            "            helper = lambda: 1\n"
            "            def inner():\n"
            "                return 2\n"
            "            if x == n:\n"
            "                return x + helper() + inner()\n"
            "            todo.append(x + 1)\n"
        )
        program = PolicyProgram(source)
        recorder, session = Recorder(), Session()
        policy_class = program.policy_class()
        policy_class.find.__globals__[RECORDER] = recorder
        recorder.sessions.append(session)
        self.assertEqual(policy_class(None, None, None).find(2), 5)
        (run,) = session.runs
        kinds = [kind for kind, _ in run.capture.frames()]
        self.assertTrue(run.exited)
        self.assertEqual(kinds, ["visit", "visit", "visit", "return"])

    def test_boolean_grids_are_recorded_read_only(self):
        """An occupancy grid is kept once, read-only, and compared by value."""
        grid = np.zeros((3, 4), dtype=bool)
        capture = RunCapture()
        capture.record("visit", {"bad": grid})
        grid[1, 1] = True
        capture.record("visit", {"bad": grid})
        first, second = (dict(v) for _, v in capture.frames())
        self.assertFalse(first["bad"][1, 1] or first["bad"].flags.writeable)
        self.assertTrue(second["bad"][1, 1])
        self.assertTrue(same_snapshot({"bad": grid.copy()}, {"bad": grid}))
        self.assertFalse(same_snapshot({"bad": grid}, {"bad": ~grid}))


class IncrementalRecorderTest(unittest.TestCase):
    """Recording only what changed decodes to exactly the full snapshots."""

    def test_changes_decode_like_full_snapshots(self):
        """Heaps, tables, deletions, rebinding, type changes and nested lists."""
        cell = lambda i, j: (np.int64(i), np.int64(j))  # noqa: E731
        heap = [(0.0, 0.0, cell(0, 0))]
        cost, parent, nested = {cell(0, 0): 0.0}, {}, [[1]]
        incremental, reference = RunCapture(), RunCapture()
        reference.incremental = False
        for i in range(1, 30):
            if heap and i % 3:
                heapq.heappop(heap)
            c = cell(i, i % 5)
            cost[c], parent[c] = float(i), cell(i - 1, 0)
            heapq.heappush(heap, (float(i) + 0.5, float(i), c))
            if i == 10:
                del cost[cell(0, 0)]  # a removed key
            if i == 12:
                heap = list(heap)  # rebound to a new list object
            if i == 15:
                parent[cell(15, 0)] = 1  # an int ...
            if i == 16:
                parent[cell(15, 0)] = 1.0  # ... becomes an equal float
            if i == 20:
                nested[0].append(i)  # mutable items fall back to full copies
            values = {"front": heap, "cost": cost, "parent": parent, "nested": nested}
            # Both encoders see the live containers at this step.
            incremental.record("visit", values)
            reference.record("visit", values)
        for (k1, v1), (k2, v2) in zip(incremental.frames(), reference.frames()):
            self.assertEqual(k1, k2)
            self.assertTrue(same_snapshot(v1, v2))
            self.assertEqual(
                {k: type(v) for k, v in v1.items()}, {k: type(v) for k, v in v2.items()}
            )
            self.assertEqual(
                [type(v) for v in v1["parent"].values()],
                [type(v) for v in v2["parent"].values()],
            )
        self.assertEqual(incremental.length, reference.length)


class FitTest(unittest.TestCase):
    """Numbers proposed from observations; nothing here accepts an explanation."""

    def test_step_costs_snap_to_the_program_constant(self):
        """1.414 is recovered exactly from noisy cost differences."""
        steps = {
            (1, 1): [(2.0, 2.0 + 1.414), (7.414, 7.414 + 1.414)],
            (1, 0): [(0.0, 1.0)],
        }
        moves = fit_moves(steps)
        self.assertEqual(moves, EIGHT_MOVES)
        self.assertIsNone(fit_moves({(1, 0): [(0.0, 1.0), (1.0, 3.0)]}))

    def test_weighted_octile_heuristics_are_recovered(self):
        """Priority = g + (high + 0.414 low) * 1.02 is fitted exactly."""
        goal = (9, 4)
        entries = []
        for cell, g in (((1, 1), 1.414), ((2, 1), 2.414), ((1, 3), 3.0), ((5, 0), 6.0)):
            entries.append(((g + octile(cell, goal), g, cell), g, {"dst": goal}))
        fits = fit_heuristics(entries, "dst")
        self.assertIn(("g+h", "octile", (1.02, 0.414)), fits)
        self.assertNotIn(("g", "", ()), fits)
        self.assertEqual(fit_fields(entries), ("g",))

    def test_a_cost_ordered_heap_needs_no_heuristic(self):
        """Dijkstra priorities equal the stored costs."""
        entries = [((g, cell), g, {"dst": (5, 5)}) for cell, g in (((1, 0), 1.0),)]
        self.assertEqual(fit_heuristics(entries, "dst")[0], ("g", "", ()))
        self.assertEqual(fit_fields(entries), ())


class EngineTest(unittest.TestCase):
    """The fixed algorithm with fitted ingredients."""

    def spec(self, **changes) -> GridSpec:
        values = {
            "selection": "min",
            "priority": "g+h",
            "heuristic": "fitted",
            "neighborhood": 0,
            "model": "grid_blocked",
            "layout": "tagged",
            "moves": EIGHT_MOVES,
            "h_form": "octile",
            "h_coef": (1.02, 0.414),
            "corners": "no_cut",
            "root_parent": "absent",
            "fields": ("g",),
            "g_source": "entry",
            "stale": "skip",
        }
        values.update(changes)
        return GridSpec(**values)

    def test_invalid_combinations_are_refused(self):
        """Each new ingredient states what it needs."""
        for bad in (
            {"neighborhood": 8},
            {"h_form": ""},
            {"model": "open"},
            {"fields": (), "g_source": "entry"},
            {"g_source": "table"},
        ):
            with self.assertRaises(ValueError):
                self.spec(**bad)

    def test_json_round_trip_keeps_identity(self):
        """Lists from JSON become tuples again, so frozen identities survive."""
        explanation = GridExplanation(
            self.spec(),
            GridMapping("s", "src", "front", "parent", "cost", "dst", "bad"),
        )
        data = json.loads(json.dumps(asdict(explanation)))
        loaded = explanation_from_data(data)
        self.assertEqual(loaded, explanation)
        self.assertEqual(loaded.identifier, explanation.identifier)

    def test_corner_rule_grid_model_and_entry_numbers(self):
        """A blocked side cell forbids the diagonal; entries carry (f, g, cell)."""
        bad = np.zeros((5, 5), dtype=bool)
        bad[2, 1] = True
        free = grid_predicate(bad, true_means_free=False)
        engine = GridSearch(self.spec(), (1, 1), (4, 4), free)
        self.assertEqual(engine.frontier, [(0.0, 0.0, (1, 1))])
        transition = engine.step()
        self.assertNotIn((2, 2), transition.generated)
        self.assertNotIn((2, 1), transition.generated)
        self.assertIn((1, 2), transition.generated)
        self.assertNotIn((1, 1), engine.observed_parents())
        entry = next(e for e in engine.frontier if e[-1] == (0, 0))
        self.assertEqual(entry, (1.414 + octile((0, 0), (4, 4)), 1.414, (0, 0)))
        self.assertFalse(free((5, 0)))


class AstraStyleTest(unittest.TestCase):
    """The disc-world copy of Astra's A* is found, measured and validated."""

    @classmethod
    def setUpClass(cls):
        cls.program = PolicyProgram(GRID_OCTILE)
        cls.bank = discover_policy(
            cls.program, DiscWorld(), ((1, 0), (3, 1)), ((3, 100),)
        )
        cls.validation = validate_policy(cls.program, DiscWorld(), cls.bank)

    def test_ingredients_are_proposed_from_development(self):
        """Moves, weights and entry numbers come from data, not a menu."""
        search = min(self.bank.parts, key=lambda s: self.program.sites[s]["line"])
        (fit,) = self.bank.audit["parts"][search]["fitted"]
        self.assertEqual(fit["moves"], EIGHT_MOVES)
        self.assertEqual(fit["heuristics"]["dst"], [("g+h", "octile", (1.02, 0.414))])
        self.assertEqual(fit["fields"], ("g",))

    def test_interventions_choose_where_costs_are_read(self):
        """Only explanations reading g from the heap entry survive the probes."""
        part = self.bank.parts[
            min(self.bank.parts, key=lambda s: self.program.sites[s]["line"])
        ]
        self.assertEqual(part["status"], "search")
        specs = [e.spec for e in part["explanations"]]
        self.assertTrue(specs)
        self.assertEqual({s.g_source for s in specs}, {"entry"})
        self.assertEqual(
            {(s.model, s.corners, s.root_parent) for s in specs},
            {("grid_blocked", "no_cut", "absent")},
        )

    def test_validation_is_complete(self):
        """Held-out runs are explained by the frozen explanation."""
        row = self.validation["episodes"][0]["accounting"]["program"]
        self.assertTrue(row["complete"])
        self.assertGreater(row["expansion_events"], 0)


if __name__ == "__main__":
    unittest.main()
