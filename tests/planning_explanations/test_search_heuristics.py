"""Priority search, causal ingredient checks, goal timing and frozen validation."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from unittest import mock

from robocode.planning_explanations.search.__main__ import main
from robocode.planning_explanations.search.benchmark import (
    check_partition,
    load_task_bank,
    validate_tasks,
)
from robocode.planning_explanations.search.checkpoints import Program
from robocode.planning_explanations.search.discover import Limits, digest
from robocode.planning_explanations.search.engine import run
from robocode.planning_explanations.search.evidence import computation_features
from robocode.planning_explanations.search.explanation import (
    CostMapping,
    Explanation,
    from_data,
)
from robocode.planning_explanations.search.heuristic_discovery import (
    discover_heuristics,
)
from robocode.planning_explanations.search.heuristic_fixtures import (
    ASTAR,
    FUNNEL,
    HEURISTIC_FIXTURES,
    IGNORED_H,
    REOPEN_H,
    heuristic_cases,
    policy_source,
)
from robocode.planning_explanations.search.heuristics import (
    CostConfiguration,
    HeuristicSpec,
)
from robocode.planning_explanations.search.report import (
    describe_truncation,
    paired_svg,
)
from robocode.planning_explanations.search.routes import (
    discover_routes,
    load_route_bank,
    validate_routes,
)
from robocode.planning_explanations.search.tasks import (
    ZERO,
    HeuristicCase,
    TaskCase,
    case_from_data,
)
from robocode.planning_explanations.search.verify import compare, intervene, verify


def reference(selection="astar", timing="selection", relaxation="improve", tie="last"):
    """A supplied reference is only for focused verifier tests, never discovery."""
    return Explanation(
        HeuristicSpec(
            selection=selection, goal_timing=timing, relaxation=relaxation, tie=tie
        ),
        CostMapping(
            "9:4:While", "root", "todo", "parent", costs="distance", heuristic="h"
        ),
    )


class PriorityEngineTests(unittest.TestCase):
    """Expected computations use an explicitly inspectable merging-path example."""

    def test_reopening_changes_parents_and_counts_repeated_expansion(self):
        """State 4 is expanded twice; its cheaper parent becomes state 2."""
        case = heuristic_cases()[0][0]
        execution = run(HeuristicSpec(tie="last").bind(case), case.start)
        expanded = [t.expanded for t in execution.transitions if t.expanded is not None]
        self.assertEqual(expanded, [0, 1, 3, 4, 2, 4])
        final = execution.transitions[-1].state
        self.assertIsInstance(final, CostConfiguration)
        self.assertEqual(final.parents[4], 2)
        self.assertEqual(final.costs[5], 3)
        self.assertEqual(final.output, 1)
        features = computation_features(execution.transitions)
        self.assertEqual((features["N"], features["M"]), (5, 4))
        self.assertEqual(
            (
                features["expansion_events"],
                features["reexpansions"],
                features["cost_improvements"],
            ),
            (6, 1, 2),
        )

    def test_first_parent_retention_keeps_longer_path(self):
        """An A* ranking alone does not imply cost improvement or reopening."""
        case = heuristic_cases()[0][0]
        execution = run(HeuristicSpec(tie="last", relaxation="first").bind(case), 0)
        final = execution.transitions[-1].state
        self.assertEqual(final.parents[4], 3)
        self.assertEqual(final.costs[5], 4)
        self.assertEqual(final.output, -1)
        self.assertEqual(computation_features(execution.transitions)["reexpansions"], 0)

    def test_goal_generation_can_stop_before_cheaper_path(self):
        """Early stopping returns the four-edge path before its improvement."""
        case = heuristic_cases()[0][0]
        execution = run(
            HeuristicSpec(tie="last", goal_timing="generation").bind(case), 0
        )
        self.assertEqual(execution.transitions[-1].operation, "goal_generated")
        self.assertEqual(execution.transitions[-1].state.costs[5], 4)
        self.assertEqual(execution.transitions[-1].state.output, -1)
        self.assertEqual([t.expanded for t in execution.transitions], [0, 1, 3, 4])

    def test_generation_counts_only_successors_actually_generated(self):
        """A goal in the first action slot stops before the second alternative."""
        case = HeuristicCase("first_child_goal", 0, 5, ((5, 2), *FUNNEL[1:]), ZERO)
        hypothesis = reference(timing="generation")
        execution = run(hypothesis.problem(case), 0)
        self.assertEqual(execution.transitions[0].generated, (5,))
        self.assertEqual(
            computation_features(execution.transitions)["generated_alternatives"], 1
        )
        result = verify(
            Program(policy_source(timing="generation")), hypothesis, (case,)
        )
        self.assertEqual(result["status"], "supported")

    def test_root_goal_is_not_an_expansion(self):
        """Both goal timings have a defined zero-search root-goal execution."""
        case = HeuristicCase("root_goal", 5, 5, FUNNEL, REOPEN_H)
        for timing in ("selection", "generation"):
            result = verify(
                Program(policy_source(timing=timing)), reference(timing=timing), (case,)
            )
            self.assertEqual(result["status"], "supported")
            self.assertEqual(result["ordinary"][0]["expansion_events"], 0)
            self.assertEqual(result["ordinary"][0]["search_expansions"], 0)

    def test_tie_rule_is_visible_despite_identical_output(self):
        """First and last priority ties can choose different computation paths."""
        case = heuristic_cases()[0][0]
        program = Program(ASTAR)
        first, last = reference(tie="first"), reference(tie="last")
        self.assertEqual(
            run(first.problem(case), 0).transitions[-1].state.output,
            program.original(case),
        )
        self.assertEqual(
            compare(program, last, program.observe(case))["status"], "supported"
        )
        self.assertEqual(
            compare(program, first, program.observe(case))["status"], "rejected"
        )

    def test_ignored_heuristic_rejected_by_paired_intervention(self):
        """A* and FIFO match on zero-h ordinary traces; changing h separates them."""
        case = HeuristicCase("zero", 0, 5, FUNNEL, ZERO)
        program, hypothesis = Program(IGNORED_H), reference(tie="first")
        self.assertEqual(
            compare(program, hypothesis, program.observe(case))["status"], "supported"
        )
        result = verify(program, hypothesis, (case,))
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "intervention")
        self.assertEqual(result["counterexample"]["edit"]["role"], "heuristic")

    def test_actual_cost_and_h_tables_are_mapped_and_intervened_on(self):
        """Predicted changes lift to live dictionaries and match downstream control."""
        case = heuristic_cases()[0][0]
        result = verify(Program(ASTAR), reference(), (case,))
        self.assertEqual(result["status"], "supported")
        self.assertEqual(
            {i["edit"]["role"] for i in result["interventions"]},
            {"", "costs", "heuristic"},
        )
        for intervention in result["interventions"]:
            self.assertTrue(intervention["passed"])
            self.assertEqual(
                intervention["predicted_control_effect"],
                intervention["observed_control_effect"],
            )
        self.assertGreater(
            result["ordinary"][0]["priority_control_effects"]["heuristic"], 0
        )
        with self.assertRaises(ValueError):
            intervene(reference().problem(case).initialize(0), "heuristic", -1, 2)
        with self.assertRaises(ValueError):
            HeuristicSpec(selection="arbitrary_solver")

    def test_heuristic_input_identity_and_reset(self):
        """Heuristic-only changes are fresh inputs; policy mutations do not persist."""
        development, heldout = heuristic_cases()
        changed = replace(development[0], name="changed", heuristic=ZERO)
        check_partition(development, (changed,))
        self.assertNotEqual(changed.signature, development[0].signature)
        with self.assertRaises(ValueError):
            check_partition(development, (replace(development[0], name="renamed"),))
        self.assertEqual(case_from_data(asdict(changed)), changed)
        payload = changed.payload()
        payload["heuristic"][0] = 99
        self.assertEqual(changed.payload()["heuristic"][0], 0)
        for source in HEURISTIC_FIXTURES.values():
            program = Program(source)
            for case in (*development, *heldout):
                self.assertEqual(program.original(case), program.observe(case).output)


class PriorityDiscoveryTests(unittest.TestCase):
    """Discovery cannot infer priority roles from names or refit held-out evidence."""

    @classmethod
    def setUpClass(cls):
        cls.names = (
            "astar_reopen",
            "goal_generation",
            "nested_astar",
            "nested_costs",
            "ignored_heuristic",
            "unused_search",
        )
        cls.programs = {name: Program(HEURISTIC_FIXTURES[name]) for name in cls.names}
        cls.banks = {
            name: discover_heuristics(program, *heuristic_cases())
            for name, program in cls.programs.items()
        }
        cls.results = {
            name: validate_tasks(cls.programs[name], bank)
            for name, bank in cls.banks.items()
        }

    def test_joint_discovery_recovers_priority_and_timing(self):
        """All surviving A* candidates share the meaningful engine ingredients."""
        for name, timing, selection in (
            ("astar_reopen", "selection", "astar"),
            ("goal_generation", "generation", "astar"),
            ("nested_costs", "selection", "astar"),
            ("ignored_heuristic", "selection", "fifo"),
        ):
            bank, result = self.banks[name], self.results[name]
            identifiers = {
                r["id"] for r in result["explanations"] if r["status"] == "supported"
            }
            surviving = [h for h in bank.explanations if h.identifier in identifiers]
            self.assertTrue(surviving)
            self.assertTrue(
                all(
                    h.spec.selection == selection and h.spec.goal_timing == timing
                    for h in surviving
                )
            )
            self.assertEqual(result["finding"]["code"], "search_supported")
            self.assertIsNone(result["planningness"])
        self.assertTrue(
            all(
                h.spec.relaxation == "improve"
                for h in self.banks["astar_reopen"].explanations
                if isinstance(h.spec, HeuristicSpec)
            )
        )

    def test_nested_names_do_not_change_attributed_work(self):
        """Renamed nested storage yields the same selected ordinary feature vector."""

        def means(name):
            result = self.results[name]
            return next(
                r["mechanism_mean_features"]
                for r in result["explanations"]
                if r["id"] == result["selected"]
            )

        self.assertEqual(means("astar_reopen"), means("nested_astar"))
        self.assertEqual(means("astar_reopen"), means("nested_costs"))
        self.assertTrue(
            any(
                "[" in h.mapping.frontier
                for h in self.banks["nested_astar"].explanations
            )
        )
        # With g and h moved too, every priority role is a container path.
        for hypothesis in self.banks["nested_costs"].explanations:
            mapping = hypothesis.mapping
            assert isinstance(mapping, CostMapping)
            for role in ("root", "frontier", "parents", "costs", "heuristic"):
                self.assertIn("[", getattr(mapping, role), role)

    def test_truncation_names_the_bound_that_was_reached(self):
        """A reached bound is reported with its reason, never as silent zero."""
        clean = self.banks["astar_reopen"].audit
        self.assertFalse(clean["truncated"])
        self.assertEqual(
            clean["truncation"],
            {"inventory_sites": [], "binding_limit": False, "candidate_budget": False},
        )
        small = discover_heuristics(
            self.programs["astar_reopen"],
            *heuristic_cases(),
            Limits(candidates=1, selectors=64),
        )
        cut = small.audit["truncation"]
        self.assertTrue(small.audit["truncated"])
        self.assertTrue(cut["candidate_budget"])
        # At 64 paths only late action-readout checkpoints are cut; the
        # mapped loop checkpoint still fits.
        self.assertTrue(cut["inventory_sites"])
        self.assertNotIn("9:4:While", cut["inventory_sites"])
        text = describe_truncation(small.audit)
        self.assertIn("value inventory cut at", text)
        self.assertIn("candidate budget exhausted", text)
        self.assertEqual(describe_truncation({}), "audit unavailable (loaded bank)")

    def test_unused_search_remains_reactive_evidence(self):
        """Dead search computation cannot receive attributed search expansions."""
        result = self.results["unused_search"]
        self.assertEqual(result["finding"]["code"], "reactive_only")
        for row in result["explanations"]:
            if row["status"] == "supported":
                self.assertEqual(row["mechanism_mean_features"]["search_expansions"], 0)

    def test_frozen_bank_roundtrip_and_no_validation_refitting(self):
        """Full heuristic tables, source and candidate selections are immutable."""
        bank, program = self.banks["astar_reopen"], self.programs["astar_reopen"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            bank.save(path)
            restored = load_task_bank(path)
            with mock.patch(
                "robocode.planning_explanations.search.heuristic_discovery."
                "discover_heuristics",
                side_effect=AssertionError("refit"),
            ):
                result = validate_tasks(program, restored)
            self.assertEqual(result["selected"], bank.selected)
            self.assertEqual(restored.fingerprint, bank.fingerprint)
            payload = json.loads(path.read_text())
            payload["validation_cases"][0]["heuristic"][0] += 1
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_task_bank(path)

    def test_discovery_never_executes_heldout_tasks_or_heuristics(self):
        """A tiny proposal budget also remains unknown rather than zero planning."""
        program = self.programs["astar_reopen"]
        development, heldout = heuristic_cases()
        original = program.observe

        def development_only(case):
            if case not in development:
                raise AssertionError("held-out execution during discovery")
            return original(case)

        with mock.patch.object(program, "observe", side_effect=development_only):
            bank = discover_heuristics(
                program, development, heldout, Limits(candidates=1)
            )
        self.assertTrue(bank.audit["truncated"])
        result = validate_tasks(program, bank)
        self.assertEqual(result["finding"]["code"], "unresolved")
        self.assertIsNone(result["planningness"])

    def test_old_manifest_and_explanation_identity_are_unchanged(self):
        """Adding a versioned family does not add fields to old explanation hashes."""
        old = {
            "spec": {
                "block": 2,
                "goal": 2,
                "selection": "fifo",
                "order": [-1, 1],
                "fallback": 0,
                "mode": "graph",
            },
            "mapping": {
                "site": "5:4:While",
                "root": "root",
                "frontier": "todo",
                "parents": "parent",
                "action": "",
                "offset": 0,
            },
        }
        self.assertEqual(from_data(old).identifier, digest(old)[:16])
        self.assertEqual(json.loads(json.dumps(asdict(from_data(old)))), old)
        self.assertNotIn("heuristic", asdict(TaskCase("legacy", 0, 4)))

    def test_priority_family_composes_with_frozen_route_dispatch(self):
        """Milestone-6 gates can dispatch priority explanations without refitting."""
        program = self.programs["astar_reopen"]
        bank = discover_routes(
            program, *heuristic_cases(), limits=Limits(candidates=4000)
        )
        self.assertEqual(len(bank.routes), 1)
        self.assertEqual(bank.routes[0].bank.grammar, "unit-cost-priority-v1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            bank.save(path)
            restored = load_route_bank(path)
            self.assertEqual(
                validate_routes(program, restored)["supported_route_coverage"], 1
            )

    def test_cli_and_svgs_expose_costs_reparenting_and_probes(self):
        """Reports show all extended ingredients and every original expansion count."""
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            frozen = Path(directory) / "bank.json"
            self.banks["astar_reopen"].save(frozen)
            self.assertEqual(
                main(
                    [
                        "--suite",
                        "heuristics",
                        "--fixture",
                        "astar_reopen",
                        "--frozen",
                        str(frozen),
                        "--output-dir",
                        directory,
                    ]
                ),
                0,
            )
            root = Path(directory) / "astar_reopen"
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(
                report["validation"]["selected"], self.banks["astar_reopen"].selected
            )
            page = (root / "report.html").read_text()
            for text in (
                "Expansion events",
                "Re-expansions",
                "Cost improvements",
                "goal test: selection",
                "g: cost so far",
                "h: estimate",
            ):
                self.assertTrue(text in page, text)
            ET.parse(root / "ordinary.svg")
            ET.parse(root / "intervention.svg")
            selected = next(
                r
                for r in report["validation"]["explanations"]
                if r["id"] == report["validation"]["selected"]
            )
            self.assertGreater(len(selected["interventions"]), 4)
            row = selected["ordinary"][0]
            graphic = paired_svg(row, "Reopening example")
            self.assertIn("improved existing paths: [4]", graphic)


if __name__ == "__main__":
    unittest.main()
