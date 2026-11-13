"""Milestone 8: class-based policies, cached plans, and once-only counting."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from robocode.planning_explanations.search.__main__ import main
from robocode.planning_explanations.search.grid_engine import (
    GridSearch,
    GridSpec,
    grid_features,
)
from robocode.planning_explanations.search.policy_discovery import (
    PolicyLimits,
    account,
    analyze_episode,
    check_partition,
    discover_policy,
    load_policy_bank,
    validate_policy,
)
from robocode.planning_explanations.search.policy_domains import DiscWorld
from robocode.planning_explanations.search.policy_fixtures import (
    PLAN_ONCE,
    POLICY_FIXTURES,
)
from robocode.planning_explanations.search.policy_runner import (
    PolicyProgram,
    observe_episode,
)
from robocode.planning_explanations.search.policy_verify import (
    GridExplanation,
    GridMapping,
    compare_run,
)

DEVELOPMENT = ((1, 0), (3, 1))
VALIDATION = ((3, 100),)

# A frontier that is only a view: rebuilt from `book` every iteration. Every
# ordinary iteration matches uniform-cost search, but a dropped entry returns.
VIEW_FRONTIER = PLAN_ONCE.replace(
    """        frontier = [(0.0, root)]
        cost, parent = {root: 0.0}, {root: None}
        while frontier:
            _, cur = heapq.heappop(frontier)
            if cur == target:
                break""",
    """        frontier = [(0.0, root)]
        cost, parent = {root: 0.0}, {root: None}
        book = {root: 0.0}
        while frontier:
            _, cur = heapq.heappop(frontier)
            del book[cur]
            if cur == target:
                break""",
).replace(
    """                    h = math.hypot(nxt[0] - target[0], nxt[1] - target[1])
                    heapq.heappush(frontier, (nc + h, nxt))""",
    """                    book[nxt] = nc
            frontier = [(p, c) for c, p in book.items()]
            heapq.heapify(frontier)""",
)


# A loop that only runs with ten or more obstacles: never seen in development.
UNSEEN_LOOP = PLAN_ONCE.replace(
    "    def get_action(self, state):\n        return self._follow(state)",
    "    def get_action(self, state):\n"
    '        if len(state["obstacles"]) >= 10:\n'
    "            k = 0\n"
    "            while k < 2:\n"
    "                k += 1\n"
    "        return self._follow(state)",
)


def run_to_end(spec: GridSpec, root, goal, free) -> tuple[GridSearch, dict]:
    """Step an abstract search to termination and compute its features."""
    engine = GridSearch(spec, root, goal, free)
    transitions = []
    while engine.phase == "ready":
        transitions.append(engine.step())
    return engine, grid_features(transitions)


def search_site(program: PolicyProgram) -> str:
    """The `while frontier:` loop; a test-only convenience, never used by discovery."""
    return next(
        s for s, info in program.sites.items() if info["source"] == "while frontier:"
    )


class GridEngineTest(unittest.TestCase):
    """The abstract family on its own, without any policy."""

    def test_explore_everything_expands_each_reachable_cell_once(self):
        """Without a goal test, every reachable cell of a 3x3 room is expanded once."""
        engine, features = run_to_end(
            GridSpec(goal_timing="none"),
            (0, 0),
            None,
            lambda c: 0 <= c[0] <= 2 and 0 <= c[1] <= 2,
        )
        self.assertEqual(len(engine.parents), 9)
        self.assertEqual((features["N"], features["expansion_events"]), (9, 9))
        self.assertIsNone(engine.path())

    def test_stale_entries_are_reexpanded_or_skipped_by_declaration(self):
        """Lazy-deletion entries are re-expanded or skipped as declared."""
        walls = {(2, 1), (2, 2), (2, 3), (4, 3), (4, 5), (5, 3), (5, 5)}
        walls |= {(3, 3), (3, 4), (3, 5)}

        def free(c):
            return 0 <= c[0] <= 5 and 0 <= c[1] <= 5 and c not in walls

        results = {}
        for stale in ("reexpand", "skip"):
            spec = GridSpec(
                priority="g+h",
                heuristic="euclidean",
                stale=stale,
                neighborhood=8,
            )
            engine, features = run_to_end(spec, (0, 0), (5, 4), free)
            self.assertFalse(engine.reached_goal)
            results[stale] = features
        self.assertEqual(results["reexpand"]["reexpansions"], 9)
        self.assertEqual(results["skip"]["reexpansions"], 0)
        self.assertEqual(results["skip"]["N"], results["reexpand"]["N"])

    def test_declarations_reject_redundant_or_incoherent_choices(self):
        """Redundant or incoherent ingredient combinations are not in the grammar."""
        for bad in (
            {"priority": "g", "root_priority": "rule"},
            {"priority": "g+h", "heuristic": "none"},
            {"goal_timing": "none", "priority": "h", "heuristic": "euclidean"},
            {"stale": "skip", "selection": "fifo"},
            {"layout": "states", "selection": "min"},
        ):
            with self.assertRaises(ValueError):
                GridSpec(**bad)


class RunnerTest(unittest.TestCase):
    """Instrumentation of a whole class, across calls."""

    def test_runs_are_recorded_per_call_and_frames_reconstruct(self):
        """The A* run belongs to initialization; later calls only follow the plan."""
        program = PolicyProgram(PLAN_ONCE)
        trace = observe_episode(program, DiscWorld(), (1, 0))
        site = search_site(program)
        initialization = trace.calls[0]
        self.assertEqual(initialization.method, "reset")
        searches = [r for r in initialization.runs if r.site == site]
        self.assertEqual(len(searches), 1)
        self.assertEqual(searches[0].methods, ("reset", "_plan"))
        later = [r for c in trace.calls[1:] for r in c.runs if r.site == site]
        self.assertEqual(later, [])
        kinds = [kind for kind, _ in searches[0].capture.frames()]
        self.assertEqual(kinds[-1], "exit")
        self.assertEqual(kinds.count("visit"), searches[0].visits)
        _, first = next(searches[0].capture.frames())
        self.assertEqual(first["frontier"], [(0.0, first["root"])])
        self.assertIn("free", searches[0].capture.callables)

    def test_ingredients_and_models_are_checked_on_every_iteration(self):
        """The right explanation matches every iteration; near misses do not."""
        program = PolicyProgram(PLAN_ONCE)
        trace = observe_episode(program, DiscWorld(), (3, 1))
        site = search_site(program)
        run = next(r for r in trace.calls[0].runs if r.site == site)
        mapping = GridMapping(
            site, "root", "frontier", "parent", "cost", "target", "free"
        )
        right = GridSpec(priority="g+h", heuristic="euclidean")
        result, _ = compare_run(GridExplanation(right, mapping), run)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["steps"], run.visits)
        open_grid = GridMapping(site, "root", "frontier", "parent", "cost", "target")
        for spec, chosen in (
            (GridSpec(priority="g+h", heuristic="manhattan"), mapping),
            (GridSpec(priority="g"), mapping),
            (
                GridSpec(selection="fifo", priority="g+h", heuristic="euclidean"),
                mapping,
            ),
            (GridSpec(priority="g+h", heuristic="euclidean", model="open"), open_grid),
        ):
            result, _ = compare_run(GridExplanation(spec, chosen), run)
            self.assertEqual(result["status"], "rejected", spec)


class ControlPolicyTest(unittest.TestCase):
    """Each control plants one claim; labels never reach the analyzer."""

    results: dict = {}

    @classmethod
    def setUpClass(cls):
        for name in (
            "plan_once",
            "replan_every_call",
            "unused_plan",
            "reactive",
            "reach_then_pick",
            "nested_renamed",
        ):
            program = PolicyProgram(POLICY_FIXTURES[name])
            bank = discover_policy(program, DiscWorld(), DEVELOPMENT, VALIDATION)
            cls.results[name] = (
                program,
                bank,
                validate_policy(program, DiscWorld(), bank),
            )

    def program_row(self, name: str) -> dict:
        """The single validation episode's program-level accounting."""
        return self.results[name][2]["episodes"][0]["accounting"]["program"]

    def selected(self, name: str):
        """The preselected explanation of the only search part."""
        _, bank, _ = self.results[name]
        (part,) = [p for p in bank.parts.values() if p["status"] == "search"]
        return next(e for e in part["explanations"] if e.identifier == part["selected"])

    def test_planning_at_initialization_is_counted_once(self):
        """Initialization search counts once and drives every later action."""
        row = self.program_row("plan_once")
        self.assertTrue(row["complete"])
        self.assertEqual(row["planning_events"], 1)
        self.assertGreater(row["expansion_events"], 50)
        self.assertEqual(
            row["initialization_expansion_events"], row["expansion_events"]
        )
        self.assertEqual(row["attributed_expansion_events"], row["expansion_events"])
        self.assertEqual(row["followed_plan_share"], 1.0)
        self.assertEqual(row["world_insensitive_share"], 1.0)
        spec = self.selected("plan_once").spec
        self.assertEqual(
            (spec.selection, spec.priority, spec.heuristic, spec.goal_timing),
            ("min", "g+h", "euclidean", "selection"),
        )

    def test_replanning_counts_one_search_per_call(self):
        """A planner that replans each call gets one search per call, none shared."""
        row = self.program_row("replan_every_call")
        self.assertTrue(row["complete"])
        # One search per call; a search that starts on the goal cell selects it
        # without expanding anything, so it is not a planning event.
        self.assertEqual(row["search_runs"], row["action_calls"])
        self.assertEqual(row["ran_search"], row["action_calls"])
        self.assertGreaterEqual(row["planning_events"], row["action_calls"] - 1)
        self.assertEqual(row["initialization_expansion_events"], 0)
        self.assertEqual(row["searched_share"], 1.0)
        self.assertEqual(row["followed_plan_share"], 0.0)
        self.assertGreater(
            row["expansion_events"], self.program_row("plan_once")["expansion_events"]
        )

    def test_an_unused_plan_is_verified_but_not_attributed_to_actions(self):
        """A verified, linked plan that no action follows is attributed nothing."""
        row = self.program_row("unused_plan")
        self.assertEqual(row["planning_events"], 1)
        self.assertEqual(row["followed_plan_share"], 0.0)
        self.assertEqual(row["attributed_expansion_events"], 0)
        versions = self.results["unused_plan"][2]["episodes"][0]["versions"]
        self.assertTrue(any(v["producer"] is not None for v in versions))

    def test_reactive_policy_has_no_search_parts(self):
        """A policy without search-shaped loops has no planning events."""
        _, bank, validation = self.results["reactive"]
        self.assertFalse([p for p in bank.parts.values() if p["status"] == "search"])
        self.assertEqual(self.program_row("reactive")["planning_events"], 0)
        # Only its watched obstacle for loop runs, never as a search.
        runs = validation["episodes"][0]["runs"]
        self.assertEqual({r["status"] for r in runs}, {"not_search_shaped"})

    def test_exhaustive_reachability_has_no_goal_test(self):
        """Search is recovered; a destination-keeping detour witnesses its plan."""
        spec = self.selected("reach_then_pick").spec
        self.assertEqual((spec.priority, spec.goal_timing), ("g", "none"))
        row = self.program_row("reach_then_pick")
        self.assertEqual(row["followed_plan_share"], 1.0)
        self.assertEqual(row["attributed_expansion_events"], row["expansion_events"])
        versions = self.results["reach_then_pick"][2]["episodes"][0]["versions"]
        probes = [p for v in versions for c in v["chains"] for p in c["probes"]]
        # Edits that lose or move the destination are out of the readout's domain.
        self.assertFalse([p for p in probes if p["status"] == "rejected"])
        self.assertTrue(
            any(
                p["edit"]["operation"] == "drop_cell" and p["status"] == "supported"
                for p in probes
            )
        )

    def test_renamed_nested_storage_gives_identical_counts(self):
        """Renamed, nested search storage changes no count."""
        mapping = self.selected("nested_renamed").mapping
        self.assertTrue(all("[" in path for path in (mapping.frontier, mapping.costs)))
        for key in ("planning_events", "expansion_events", "max_expanded_depth"):
            self.assertEqual(
                self.program_row("nested_renamed")[key],
                self.program_row("plan_once")[key],
            )

    def test_loops_without_search_shape_are_listed_not_counted(self):
        """Path-reconstruction and waypoint loops are listed, not counted."""
        _, bank, validation = self.results["plan_once"]
        statuses = sorted(p["status"] for p in bank.parts.values())
        self.assertEqual(statuses.count("not_search_shaped"), 2)
        row = self.program_row("plan_once")
        self.assertGreater(row["non_search_loop_runs"], 0)
        self.assertEqual(row["search_runs"], 1)
        self.assertTrue(validation["summary"]["complete"])

    def test_frozen_bank_round_trips_and_detects_tampering(self):
        """Replaying the frozen bank reproduces validation; edits are detected."""
        program, bank, validation = self.results["plan_once"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            bank.save(path)
            loaded = load_policy_bank(path)
            self.assertEqual(loaded.fingerprint, bank.fingerprint)
            replay = validate_policy(program, DiscWorld(), loaded)
            self.assertEqual(replay["summary"], validation["summary"])
            data = json.loads(path.read_text())
            data["validation"] = [[3, 999]]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_policy_bank(path)


class VerificationTest(unittest.TestCase):
    """Paired edits reject what ordinary agreement alone cannot."""

    def test_a_rebuilt_frontier_view_fails_paired_edits(self):
        """A frontier rebuilt every iteration matches ordinarily but fails edits."""
        self.assertNotEqual(VIEW_FRONTIER, PLAN_ONCE)
        program = PolicyProgram(VIEW_FRONTIER)
        site = search_site(program)
        mapping = GridMapping(
            site, "root", "frontier", "parent", "cost", "target", "free"
        )
        explanation = GridExplanation(GridSpec(goal_timing="selection"), mapping)
        trace = observe_episode(program, DiscWorld(), (1, 0))
        run = next(r for r in trace.calls[0].runs if r.site == site)
        self.assertEqual(compare_run(explanation, run)[0]["status"], "supported")
        parts = {
            site: {
                "status": "search",
                "explanations": [explanation],
                "selected": explanation.identifier,
                "readouts": [],
            }
        }
        episode = analyze_episode(
            program, DiscWorld(), parts, (1, 0), "development", PolicyLimits()
        )
        (search,) = [r for r in episode["runs"] if r["site"] == site]
        result = search["explanations"][explanation.identifier]
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "intervention")
        self.assertEqual(result["counterexample"]["edit"]["operation"], "drop")
        self.assertIsNone(account(episode)["program"]["expansion_events"])

    def test_a_loop_unseen_in_development_is_unresolved(self):
        """A loop that first runs in validation is unresolved, not zero."""
        self.assertNotEqual(UNSEEN_LOOP, PLAN_ONCE)
        program = PolicyProgram(UNSEEN_LOOP)
        bank = discover_policy(program, DiscWorld(), ((1, 0),), ((15, 101),))
        episode = validate_policy(program, DiscWorld(), bank)["episodes"][0]
        row = episode["accounting"]["program"]
        self.assertFalse(row["complete"])
        self.assertIsNone(row["expansion_events"])
        self.assertEqual(row["explained_only_expansion_events"], 232)
        # The unseen loop runs in every call but does not erase plan evidence:
        # the plan stored at reset is witnessed, so every call stays credited.
        self.assertTrue(all(c["unknown_loop"] for c in episode["calls"]))
        self.assertEqual(row["followed_witnessed_plan"], row["action_calls"])
        self.assertEqual(row["unresolved_loop"], 0)

    def test_episodes_must_be_disjoint(self):
        """Validation episodes cannot reuse a development episode."""
        with self.assertRaises(ValueError):
            check_partition(((1, 0),), ((1, 0),))
        with self.assertRaises(ValueError):
            check_partition((), ((1, 0),))


class CommandLineTest(unittest.TestCase):
    """The suite writes a frozen bank before validation, and its reports."""

    def test_policy_suite_writes_reports(self):
        """The CLI freezes the bank and writes every report file."""
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                code = main(
                    [
                        "--suite",
                        "policy",
                        "--fixture",
                        "plan_once",
                        "--output-dir",
                        directory,
                    ]
                )
            self.assertEqual(code, 0)
            root = Path(directory) / "plan_once"
            for name in ("frozen.json", "report.json", "report.md", "report.html"):
                self.assertTrue((root / name).exists(), name)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary[0]["finding"]["code"], "search_supported")
            self.assertEqual(summary[0]["mean_planning_events"], 1.0)


@unittest.skipUnless(importlib.util.find_spec("kinder"), "KinDER is not installed")
class KinderPolicyTest(unittest.TestCase):
    """The agent-written ClutteredRetrieval2D policy on one short episode pair."""

    def test_both_searches_are_recovered(self):
        """Reachability search and carry A* are both explained on real episodes."""
        # pylint: disable-next=import-outside-toplevel
        from robocode.planning_explanations.search.policy_domains import KinderDomain

        # pylint: disable-next=import-outside-toplevel
        from robocode.planning_explanations.search.policy_fixtures import (
            kinder_source,
        )

        program = PolicyProgram(kinder_source())
        domain = KinderDomain()
        bank = discover_policy(program, domain, ((1, 0),), ((3, 100),))
        search = {
            program.sites[s]["function"]: p
            for s, p in bank.parts.items()
            if p["status"] == "search"
        }
        self.assertEqual(
            sorted(search),
            ["GeneratedApproach._make_carry_path", "GeneratedApproach.reset"],
        )

        def chosen(part):
            return next(
                e for e in part["explanations"] if e.identifier == part["selected"]
            )

        reach = chosen(search["GeneratedApproach.reset"]).spec
        carry = chosen(search["GeneratedApproach._make_carry_path"]).spec
        self.assertEqual((reach.priority, reach.goal_timing), ("g", "none"))
        self.assertEqual(
            (carry.priority, carry.heuristic, carry.goal_timing),
            ("g+h", "euclidean", "selection"),
        )
        development = bank.audit["development"][0]["accounting"]["program"]
        self.assertTrue(development["complete"])
        self.assertEqual(development["initialization_runs"], 1)
        self.assertGreater(
            development["followed_witnessed_plan"]
            + development["followed_unresolved_plan"],
            0,
        )
        self.assertEqual(development["followed_disproved_plan"], 0)


if __name__ == "__main__":
    unittest.main()
