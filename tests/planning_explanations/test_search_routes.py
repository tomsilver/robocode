"""Nested selections, causal edits, frozen route dispatch and unknown coverage."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest import mock

from robocode.planning_explanations.search.__main__ import main
from robocode.planning_explanations.search.checkpoints import Edit, Program
from robocode.planning_explanations.search.discover import Limits, digest
from robocode.planning_explanations.search.explanation import Mapping
from robocode.planning_explanations.search.route_fixtures import (
    ROUTE_FIXTURES,
    nested,
    route_cases,
)
from robocode.planning_explanations.search.routes import (
    discover_routes,
    load_route_bank,
    route_prefix,
    validate_routes,
)
from robocode.planning_explanations.search.selectors import inventory, select
from robocode.planning_explanations.search.task_fixtures import DECOY, benchmark_cases
from robocode.planning_explanations.search.tasks import TaskCase


class SelectorTests(unittest.TestCase):
    """Mappings only select; patches edit live Python containers with their aliases."""

    def test_literal_paths_and_language_restrictions(self):
        """Selectors cannot run code, inspect attributes or compute index expressions."""
        values = {"x": [{"non identifier": 5}, 7]}
        self.assertEqual(select(values, 'x[0]["non identifier"]'), 5)
        self.assertEqual(select(values, "x[-1]"), 7)
        for expression in (
            "x[solve()]",
            "x[index]",
            "x.copy()",
            "x.attr",
            "x[True]",
            "__result",
            "x[0]+1",
            "x[0][0][0][0][0]",
        ):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                Mapping("site", action=expression)

    def test_inventory_bounds_and_no_solver(self):
        """Depth and visit bounds limit selection to existing scalar/container values."""
        values = {"x": {"a": [2, 4], "b": {"c": 3}}}
        shallow, _ = inventory(values, 1, 64)
        self.assertNotIn('x["b"]["c"]', shallow)
        deep, truncated = inventory(values, 2, 64)
        self.assertFalse(truncated)
        self.assertEqual(deep['x["a"][1]'], 4)
        limited, truncated = inventory(values, 2, 2)
        self.assertTrue(truncated)
        self.assertEqual(len(limited), 2)

    def test_nested_list_edit_preserves_live_aliases(self):
        """A nested clear reaches every alias, while a later invocation resets state."""
        program = Program(
            'def p(x):\n    box = {"q": [x, 2]}\n    alias = box["q"]\n'
            "    result = len(alias)\n    return result\n"
        )
        trace = program.observe(1)
        self.assertEqual(trace.output, 2)
        patched = program.run(1, Edit("4:4:Assign", 0, 'box["q"]', "clear"))
        self.assertEqual(patched.output, 0)
        checkpoint = next(s for s in patched.snapshots if s.site == "4:4:Assign")
        self.assertIs(checkpoint.values["box"]["q"], checkpoint.values["alias"])
        self.assertEqual(checkpoint.before["box"]["q"], [1, 2])
        self.assertEqual(program.observe(1).output, 2)

    def test_nested_scalar_edit_preserves_container_aliases(self):
        """Setting a nested scalar mutates its shared parent, not a detached copy."""
        program = Program(
            'def p(x):\n    box = [{"v": x}]\n    alias = box[0]\n'
            '    result = alias["v"]\n    return result\n'
        )
        patched = program.run(1, Edit("4:4:Assign", 0, 'box[0]["v"]', "set", -1))
        self.assertEqual(patched.output, -1)
        self.assertEqual(patched.edits, 1)
        self.assertEqual(program.observe(1).output, 1)


class RouteTests(unittest.TestCase):
    """Route support and ordinary counts cannot be chosen after seeing validation."""

    @classmethod
    def setUpClass(cls):
        cls.programs = {
            name: Program(source) for name, source in ROUTE_FIXTURES.items()
        }
        cls.banks = {
            name: discover_routes(program, *route_cases(name))
            for name, program in cls.programs.items()
        }
        cls.results = {
            name: validate_routes(cls.programs[name], bank)
            for name, bank in cls.banks.items()
        }

    def test_nested_and_renamed_recover_same_mechanism_counts(self):
        """Names and keys do not supply roles; supported paired edits recover them."""
        left, right = (self.results[n] for n in ("nested", "nested_renamed"))
        self.assertEqual(left["finding"]["code"], "search_supported")
        self.assertEqual(right["finding"]["code"], "search_supported")
        self.assertEqual(
            left["preselected_mean_features"], right["preselected_mean_features"]
        )
        for name in ("nested", "nested_renamed"):
            route = self.banks[name].routes[0]
            self.assertTrue(
                any(h.mapping.frontier.count("[") == 2 for h in route.bank.explanations)
            )
            evidence = self.results[name]["routes"][0]["explanations"]
            supported = [e for e in evidence if e["status"] == "supported"]
            self.assertTrue(supported)
            self.assertTrue(all(e["interventions"] for e in supported))
            self.assertTrue(
                all(i["passed"] for e in supported for i in e["interventions"])
            )

    def test_mixed_routes_have_separate_counts_and_correct_weights(self):
        """Both mechanisms receive ordinary distribution weights and distinct counts."""
        result = self.results["mixed"]
        self.assertEqual(result["finding"]["code"], "mixed_routes")
        self.assertEqual(result["supported_route_coverage"], 1)
        self.assertEqual(result["preselected_supported_coverage"], 1)
        rows = {a["case"]: a for a in result["assignments"]}
        for name in ("reactive_left", "reactive_at_goal"):
            self.assertEqual(rows[name]["finding"]["code"], "reactive_only")
            self.assertEqual(rows[name]["features"]["search_expansions"], 0)
        for name in ("reversed_search", "new_goal_search"):
            self.assertEqual(rows[name]["finding"]["code"], "search_supported")
            self.assertGreater(rows[name]["features"]["search_expansions"], 0)
        for feature, mean in result["preselected_mean_features"].items():
            self.assertAlmostEqual(
                mean, sum(a["weight"] * a["features"][feature] for a in rows.values())
            )
        self.assertIsNone(result["planningness"])

    def test_unseen_route_is_unknown_without_zero_imputation(self):
        """The uncovered fifth case prevents an unconditional numerical average."""
        result = self.results["unseen_route"]
        unseen = next(a for a in result["assignments"] if a["case"] == "unseen_prefix")
        self.assertIsNone(unseen["route"])
        self.assertIsNone(unseen["features"])
        self.assertEqual(unseen["finding"]["code"], "unresolved")
        self.assertAlmostEqual(result["supported_route_coverage"], 0.8)
        self.assertIsNone(result["preselected_mean_features"])
        self.assertTrue(
            all(r["preselected_mean_features"] is not None for r in result["routes"])
        )

    def test_nested_and_flat_mixed_programs_agree(self):
        """Packing storage preserves observation parity and attributed computation."""
        self.assertEqual(
            self.results["mixed"]["preselected_mean_features"],
            self.results["mixed_nested"]["preselected_mean_features"],
        )
        for case in sum(route_cases("mixed"), ()):
            self.assertEqual(
                self.programs["mixed"].original(case),
                self.programs["mixed_nested"].observe(case).output,
            )

    def test_gate_uses_no_values_outputs_or_loop_iterations(self):
        """A prefix alone dispatches, despite different outputs and execution lengths."""
        program = self.programs["mixed"]
        first, second = program.observe(TaskCase("first", 0, 4)), program.observe(
            TaskCase("second", 3, 0)
        )
        self.assertNotEqual(len(first.snapshots), len(second.snapshots))
        self.assertNotEqual(first.output, second.output)
        self.assertEqual(route_prefix(first), route_prefix(second))
        cut = next(i for i, s in enumerate(first.snapshots) if s.kind == "loop")
        minimal = replace(
            first,
            snapshots=[
                replace(s, values={}, output=None) for s in first.snapshots[: cut + 1]
            ],
            output=999,
        )
        self.assertEqual(route_prefix(first), route_prefix(minimal))

    def test_discovery_does_not_execute_heldout_inputs(self):
        """Only development cases supply gates, hypotheses and preselection evidence."""
        program = self.programs["mixed"]
        development, heldout = route_cases("mixed")
        native = program.observe

        def development_only(case):
            if case not in development:
                raise AssertionError("heldout executed during discovery")
            return native(case)

        with mock.patch.object(program, "observe", side_effect=development_only):
            bank = discover_routes(program, development, heldout)
        self.assertEqual(bank.validation_inputs, heldout)

    def test_frozen_reload_does_not_refit_or_reselect(self):
        """Replay preserves identities; gate calls occur only on ordinary cases."""
        bank, program = self.banks["mixed"], self.programs["mixed"]
        with self.assertRaises(ValueError):
            validate_routes(
                program, replace(bank, validation_inputs=bank.discovery_inputs)
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            bank.save(path)
            restored = load_route_bank(path)
            with (
                mock.patch(
                    "robocode.planning_explanations.search.routes.discover_tasks",
                    side_effect=AssertionError("refit"),
                ),
                mock.patch(
                    "robocode.planning_explanations.search.routes.route_prefix",
                    wraps=route_prefix,
                ) as gate,
            ):
                result = validate_routes(program, restored)
            self.assertEqual(gate.call_count, len(bank.validation_inputs))
            self.assertEqual(restored.fingerprint, bank.fingerprint)
            self.assertEqual(
                [r["selected"] for r in result["routes"]],
                [r.bank.selected for r in bank.routes],
            )
            payload = json.loads(path.read_text())
            payload["routes"][0]["prefix"][0] = "tampered"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_route_bank(path)
            payload = json.loads(
                json.dumps({**bank.manifest(), "gate_rule": "choose largest heldout M"})
            )
            payload["digest"] = digest(payload)
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_route_bank(path)

    def test_route_and_mapping_budgets_remain_unknown(self):
        """Missing language coverage cannot turn into a claim of zero planning."""
        program = self.programs["mixed_nested"]
        development, heldout = route_cases("mixed_nested")
        bank = discover_routes(program, development, heldout, max_routes=1)
        self.assertTrue(bank.audit["unresolved"])
        self.assertLess(validate_routes(program, bank)["supported_route_coverage"], 1)
        bank = discover_routes(program, development, heldout, prefix_limit=1)
        self.assertEqual(bank.routes, ())
        result = validate_routes(program, bank)
        self.assertEqual(result["supported_route_coverage"], 0)
        self.assertIsNone(result["preselected_mean_features"])
        bank = discover_routes(
            self.programs["nested"],
            *route_cases("nested"),
            limits=Limits(mapping_depth=0),
        )
        self.assertEqual(
            validate_routes(self.programs["nested"], bank)["finding"]["code"],
            "partially_resolved_routes",
        )

    def test_nested_decoy_has_no_causally_supported_search(self):
        """Nested storage cannot rescue search that has no causal decision role."""
        program = Program(nested(DECOY))
        bank = discover_routes(program, *benchmark_cases("decoy"))
        result = validate_routes(program, bank)
        self.assertEqual(result["finding"]["code"], "reactive_only")
        rejected = bank.routes[0].bank.audit["rejected"]
        self.assertTrue(
            any(
                r["explanation"]["spec"]["mode"] == "graph"
                and r["reason"] == "intervention"
                for r in rejected
            )
        )

    def test_failed_preselection_not_replaced_by_survivor(self):
        """A surviving alternative does not silently supply a selected mean."""
        bank = self.banks["nested"]
        route = bank.routes[0]
        failed = next(
            r["id"]
            for r in self.results["nested"]["routes"][0]["explanations"]
            if r["status"] != "supported"
        )
        altered = replace(
            bank, routes=(replace(route, bank=replace(route.bank, selected=failed)),)
        )
        result = validate_routes(self.programs["nested"], altered)
        self.assertEqual(result["supported_route_coverage"], 1)
        self.assertEqual(result["preselected_supported_coverage"], 0)
        self.assertIsNone(result["preselected_mean_features"])
        self.assertEqual(result["routes"][0]["selected"], failed)

    def test_cli_reports_link_all_route_evidence(self):
        """The runnable demo preserves full banks, typed traces and paired SVGs."""
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            self.assertEqual(
                main(
                    [
                        "--suite",
                        "routes",
                        "--fixture",
                        "mixed_nested",
                        "--output-dir",
                        directory,
                    ]
                ),
                0,
            )
            root = Path(directory) / "mixed_nested"
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["summary"]["finding"]["code"], "mixed_routes")
            page = (root / "report.html").read_text()
            self.assertIn("Validation of the preselected explanation", page)
            self.assertIn("Attributed search expansions", page)
            for route in report["validation"]["routes"]:
                child = root / route["route"]
                self.assertIn(f'{route["route"]}/report.html', page)
                child_report = json.loads((child / "report.json").read_text())
                self.assertEqual(
                    child_report["validation"]["selected"], route["selected"]
                )
                ET.parse(child / "ordinary.svg")
                ET.parse(child / "intervention.svg")
                traces = json.loads((child / "ordinary-traces.json").read_text())[
                    "traces"
                ]
                names = {t["input"]["name"] for t in traces}
                self.assertTrue(all(c["name"] in names for c in route["cases"]))
            self.assertEqual(
                main(
                    [
                        "--suite",
                        "routes",
                        "--fixture",
                        "mixed_nested",
                        "--frozen",
                        str(root / "frozen.json"),
                        "--output-dir",
                        str(Path(directory) / "reload"),
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
