"""Behavioral tests for the bounded planning-explanation experiment.

Run with unittest (stdlib only), or collect these tests with the repo's pytest.
"""

import ast
import dataclasses
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
from math import log2
from pathlib import Path
from random import Random
from unittest import mock

from robocode.planning_explanations import stress
from robocode.planning_explanations.__main__ import main
from robocode.planning_explanations.discovery import (
    Explanation,
    Limits,
    discover,
    future_probes,
    infer_tables,
    ordinary,
    paired_test,
    tree_future,
    validate,
)
from robocode.planning_explanations.fixtures import FIXTURES, ONE_STEP, TWO_STEP
from robocode.planning_explanations.instrument import (
    Patch,
    Program,
    ProgramFailure,
    Selector,
    UnsupportedProgram,
)
from robocode.planning_explanations.report import (
    frozen_payload,
    load_frozen,
    save_report,
    write_json,
)
from robocode.planning_explanations.task import (
    MODELS,
    TRANSITIONS,
    Model,
    Navigation,
    action_values,
    cases,
    choose,
    make_case,
)

# The two-step planner, but successors come from a private copy of the default table.
OWN_TABLE = (
    TWO_STEP.replace("    scores = []", f"    table = {TRANSITIONS}\n    scores = []")
    .replace("t[s][a]", "table[s][a]")
    .replace("t[successor][b]", "table[successor][b]")
)
STRESS = Path(stress.__file__).parent


def stress_source(name):
    """A program from the stress folder; its header records the prediction."""
    return (STRESS / f"{name}.py").read_text(encoding="utf-8")


def search_for(bank, depth, path_value="leaf", pruning="none"):
    """The discovery audit of one model on a single-route bank."""
    (audit,) = bank.audit["routes"].values()
    label = Model(depth, path_value, pruning).label
    return next(s for s in audit["models"] if s["model"] == label)


class PrototypeTests(unittest.TestCase):
    """Exercise positive, negative, ambiguous and unresolved cases together."""

    @classmethod
    def setUpClass(cls):
        cls.training = cases(17, "discovery")
        cls.fresh = cases(29, "validation")
        cls.programs = {name: Program(source) for name, source in FIXTURES.items()}
        cls.banks = {
            name: discover(p, cls.training) for name, p in cls.programs.items()
        }
        cls.results = {
            name: validate(p, cls.banks[name], cls.fresh)
            for name, p in cls.programs.items()
        }

    def test_observation_preserves_all_fixture_outputs_and_inputs(self):
        """Trace every fixture without changing outputs or input mutation."""
        for name, program in self.programs.items():
            for case in self.training + self.fresh:
                with self.subTest(program=name, case=case.label):
                    self.assertEqual(program.check_observation(case).patch_count, 0)

    def test_observation_preserves_aliases_mutation_and_short_circuiting(self):
        """Hooks must preserve Python object and evaluation semantics."""
        program = Program("""\
def p(s, t, u, c):
    x = t[0]
    y = x
    y[0] = 5
    z = False and u[100]
    return x[0] + t[0][0]
""")
        case = make_case("aliases", 0, [0.25] * 6, TRANSITIONS)
        trace = program.check_observation(case)
        self.assertEqual(trace.output, 10)
        self.assertEqual(trace.args_after[1][0][0], 5)
        self.assertEqual(case.transitions[0][0], 1)

    def test_nested_expression_checkpoints_are_distinct(self):
        """Nested subscripts sharing a start position need distinct sites."""
        program = self.programs["two_step"]
        sites = [
            site
            for site, info in program.sites.items()
            if info["expression"] in ("t[s]", "t[s][a]")
        ]
        self.assertEqual(len(sites), 2)
        self.assertNotEqual(sites[0], sites[1])

    def test_targeted_scalar_and_dictionary_interventions(self):
        """A dictionary leaf patch changes one run, preserving prior snapshots."""
        program = Program("""\
def p(s, t, u, c):
    d = {0: 3, 1: 7}
    return d[0] + d[1]
""")
        trace = program.run(self.training[0])
        event = next(e for e in trace.events if program.sites[e.site]["kind"] == "Dict")
        patched = program.run(
            self.training[0], Patch(Selector(event.site, 0, (0,)), 10)
        )
        self.assertEqual(trace.output, 10)
        self.assertEqual(patched.output, 17)
        self.assertEqual(patched.patch_count, 1)
        self.assertEqual(trace.event(Selector(event.site, 0)).value[0], 3)

    def test_recover_both_horizons_and_keep_ambiguity(self):
        """Recover horizon and alternatives without arbitrarily selecting one map."""
        for name, horizon in (
            ("one_step", 1),
            ("two_step", 2),
            ("renamed_dictionary", 2),
        ):
            with self.subTest(name=name):
                result = self.results[name]
                # One computation; its copies are interchangeable locations rather
                # than separate explanations.
                (explanation,) = self.banks[name].explanations
                self.assertTrue(explanation.interchangeable)
                for row in result["ordinary_population"]:
                    self.assertEqual(row["status"], "supported_planning")
                    self.assertEqual(row["features"]["supported_horizons"], [horizon])
                    self.assertEqual(
                        row["features"]["evaluated_alternatives"], [2**horizon]
                    )
                    self.assertTrue(row["ambiguity"])

    def test_arbitrary_identifier_renaming_does_not_change_recovery(self):
        """Remove all meaningful local names and recover the same horizon."""
        tree = ast.parse(TWO_STEP)
        names = sorted(
            {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            - {"range", "max", "int"}
        )
        names = {name: f"v{i}" for i, name in enumerate(names)}

        class Rename(ast.NodeTransformer):
            """Apply an arbitrary consistent renaming without adding role labels."""

            def visit_Name(self, node):  # pylint: disable=invalid-name
                """Rename identifier uses and assignments."""
                node.id = names.get(node.id, node.id)
                return node

            def visit_arg(self, node):
                """Rename input parameters consistently."""
                node.arg = names.get(node.arg, node.arg)
                return node

        tree = Rename().visit(tree)
        tree.body[0].name = "anonymous"
        program = Program(ast.unparse(tree))
        bank = discover(program, self.training)
        self.assertTrue(bank.explanations)
        self.assertEqual({e.horizon for e in bank.explanations}, {2})

    def test_rearrangement_recovers_reverse_action_enumeration(self):
        """Mappings must accommodate a different traversal order."""
        source = ONE_STEP.replace("scores = []", "scores = [0, 0]")
        source = source.replace("range(2)", "[1, 0]")
        source = source.replace("scores.append(value)", "scores[a] = value")
        program = Program(source)
        bank = discover(program, self.training)
        result = validate(program, bank, self.fresh)
        self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)

    def test_decoy_and_lookup_are_observationally_identical_to_planner(self):
        """Negative controls agree on outputs but receive no planning support."""
        for case in self.training + self.fresh:
            expected = self.programs["two_step"].run(case).output
            self.assertEqual(self.programs["decoy"].run(case).output, expected)
            self.assertEqual(self.programs["lookup"].run(case).output, expected)
        self.assertFalse(self.banks["decoy"].explanations)
        self.assertFalse(self.banks["lookup"].explanations)

    def test_same_planner_mapping_is_rejected_for_decoy_by_causal_effect(self):
        """Transfer the exact planner alignment and falsify its decision effect."""
        explanation = self.banks["two_step"].explanations[0]
        decoy = self.programs["decoy"]
        for case in self.training:
            self.assertIsNone(ordinary(explanation, case, decoy.run(case)))
        evidence, failure = paired_test(
            decoy, explanation, self.training[0], decoy.run(self.training[0])
        )
        self.assertEqual(failure["reason"], "intervention_mismatch")
        witness = evidence[-1]
        self.assertNotEqual(witness["predicted_output"], witness["observed_output"])
        self.assertEqual(witness["observed_output"], witness["baseline_output"])

    def test_internal_effects_are_checked_even_when_outputs_match(self):
        """Equal outputs cannot hide incorrect downstream state predictions."""
        program = self.programs["two_step"]
        supported = self.banks["two_step"].explanations[0]
        # An early row snapshot contains both states, but changing its second
        # element does not affect the later read used by the second branch.
        row = next(
            s for s, info in program.sites.items() if info["expression"] == "t[s]"
        )
        mapping = dict(supported.mapping)
        mapping["states_1"] = (Selector(row, 0, (0,)), Selector(row, 0, (1,)))
        explanation = Explanation(
            supported.route,
            2,
            tuple((role, mapping[role]) for role, _ in supported.mapping),
        )
        case = make_case("equal-utilities", 0, [0.25] * 6)
        evidence, failure = paired_test(program, explanation, case, program.run(case))
        self.assertEqual(failure["reason"], "intervention_mismatch")
        self.assertEqual(
            evidence[-1]["predicted_output"], evidence[-1]["observed_output"]
        )
        self.assertTrue(
            any(
                c["predicted"] != c["observed"]
                for c in evidence[-1]["downstream_checks"]
            )
        )

    def test_unrolled_code_is_supported(self):
        """Values read by separate expressions can still form one role."""
        for source, horizon in (
            ("def p(s, t, u, c):\n    return int(u[t[s][1]] > u[t[s][0]])\n", 1),
            (
                "def p(s, t, u, c):\n"
                "    left = t[s][0]\n"
                "    right = t[s][1]\n"
                "    a = max(u[t[left][0]], u[t[left][1]])\n"
                "    b = max(u[t[right][0]], u[t[right][1]])\n"
                "    return int(b > a)\n",
                2,
            ),
        ):
            with self.subTest(horizon=horizon):
                program = Program(source)
                bank = discover(program, self.training)
                result = validate(program, bank, self.fresh)
                self.assertEqual({e.horizon for e in bank.explanations}, {horizon})
                self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)

    def test_unused_copy_is_rejected_by_its_own_patch(self):
        """A read whose element the decision never uses cannot stand for a score."""
        program = self.programs["two_step"]
        # In `return int(scores[1] > scores[0])` the first read of `scores` only
        # feeds `scores[1]`, so its element 0 is never used by the decision.
        first_read = min(
            (
                site
                for site, info in program.sites.items()
                if info["expression"] == "scores" and info["line"] == 11
            ),
            key=lambda site: program.sites[site]["column"],
        )
        search = search_for(self.banks["two_step"], 2)
        rejected = [
            r["location"]
            for r in search["rejected_locations"]
            if (r["role"], r["coordinate"]) == ("scores", 0)
        ]
        self.assertIn(dataclasses.asdict(Selector(first_read, 0, (0,))), rejected)

    def test_search_cost_follows_locations_not_their_combinations(self):
        """Per-coordinate screening finds the mapping without enumerating products."""
        search = search_for(self.banks["two_step"], 2)
        self.assertEqual(search["result"], "supported")
        self.assertEqual(search["candidates_tested"], 1)
        # Matching keeps every copy: about four locations per coordinate, whose
        # product (millions) is never enumerated.
        self.assertGreater(sum(c["matched"] for c in search["coordinates"]), 40)

    def test_each_interchangeable_location_passes_the_full_test(self):
        """Pairwise compatibility must agree with the full paired test it replaces."""
        program = self.programs["two_step"]
        explanation = self.banks["two_step"].explanations[0]
        runs = [(c, program.run(c)) for c in self.training]
        swaps = list(explanation.swaps())
        self.assertTrue(swaps)
        for role, coordinate, location, variant in swaps:
            with self.subTest(role=role, coordinate=coordinate, location=location):
                for case, trace in runs:
                    self.assertIsNone(ordinary(variant, case, trace))
                for case, trace in runs[:2]:
                    self.assertIsNone(paired_test(program, variant, case, trace)[1])

    def test_every_input_has_its_own_transition_table(self):
        """Generated inputs vary the table, reproducibly, with two distinct moves."""
        tables = {c.transitions for c in self.training + self.fresh}
        self.assertGreater(len(tables), 20)
        for table in tables:
            self.assertTrue(all(len(set(row)) == 2 for row in table))
        self.assertEqual(cases(17, "discovery"), self.training)

    def test_private_copy_of_the_table_is_told_apart_only_when_tables_vary(self):
        """With one fixed table, a private copy of it looks like reading ``t``.

        With a table per input, every environment model is rejected and the copy is
        recovered as the program's own table.
        """
        program = Program(OWN_TABLE)
        fixed = [
            make_case(c.label, c.state, c.utilities, TRANSITIONS) for c in self.training
        ]
        (explanation,) = discover(program, fixed).explanations
        self.assertEqual((explanation.horizon, explanation.table), (2, None))
        bank = discover(program, self.training)
        (explanation,) = bank.explanations
        self.assertEqual(explanation.table, TRANSITIONS)
        (audit,) = bank.audit["routes"].values()
        environment = [s for s in audit["models"] if s["table"] is None]
        self.assertEqual({s["result"] for s in environment}, {"ordinary_output"})

    def test_boundary_patches_reject_a_different_decision_rule(self):
        """Action 1 needing a lead above 0.5 agrees on every whole-number difference.

        Only a patch just past the model's switching point separates the two rules.
        """
        source = TWO_STEP.replace(
            "scores[1] > scores[0]", "scores[1] > scores[0] + 0.5"
        )
        program = Program(source)
        for case in self.training:
            self.assertEqual(
                program.run(case).output, self.programs["two_step"].run(case).output
            )
        bank = discover(program, self.training)
        self.assertFalse(bank.explanations)
        depth_two = search_for(bank, 2)
        self.assertEqual(depth_two["result"], "no_surviving_location")
        values = [
            r["counterexample"]
            for r in depth_two["rejected_locations"]
            if r["role"] in ("evaluations", "scores")
        ]
        self.assertTrue(
            any(c["kind"] in ("below switch", "above switch") for c in values)
        )

    def test_patch_set_covers_states_and_the_switching_point(self):
        """Every other state, and values around the point where the decision flips."""
        (validation,) = self.results["two_step"]["explanations"]
        records = validation["interventions"]
        self.assertEqual(
            validation["informative_interventions"],
            sum(r["predicted_output"] != r["baseline_output"] for r in records),
        )
        by_coordinate = defaultdict(list)
        for record in records:
            key = (record["case"], record["role"], record["coordinate"])
            by_coordinate[key].append(record)
        switches = 0
        for (_, role, _), group in by_coordinate.items():
            kinds = {r["kind"]: r for r in group}
            if role.startswith("states_"):
                self.assertEqual(set(kinds), {"other state"})
                self.assertEqual(len({r["replacement"] for r in group}), 5)
                continue
            self.assertLessEqual({"low", "high"}, set(kinds))
            if "at switch" in kinds:
                switches += 1
                # Just below and just above the switching point, the model's
                # decisions differ, so one of these patches must change the action.
                self.assertNotEqual(
                    kinds["below switch"]["predicted_output"],
                    kinds["above switch"]["predicted_output"],
                )
        self.assertGreater(switches, 0)

    def test_model_family_is_depth_by_path_value(self):
        """Depths 1-3, leaf or summed path values, with or without the cutoff.

        At depth 1 the path values coincide and there is nothing to cut.
        """
        self.assertEqual(
            [(m.depth, m.path_value, m.pruning) for m in MODELS],
            [(1, "leaf", "none")]
            + [
                (depth, value, pruning)
                for depth in (2, 3)
                for value in ("leaf", "sum")
                for pruning in ("none", "cutoff")
            ],
        )
        for invalid in ((1, "sum"), (1, "leaf", "cutoff"), (2, "leaf", "beam")):
            with self.assertRaises(ValueError):
                Model(*invalid)
        case = self.training[0]
        model = Model(3, "sum")
        roles = model.rollout(case.state, case.transitions, case.utilities)
        self.assertEqual(model.roles, tuple(roles))
        self.assertEqual([len(roles[r]) for r in model.roles], [2, 4, 8, 8, 2])
        # Path 5 = actions (1, 0, 1): its states are at indices 1, 2 and 5.
        path = (roles["states_1"][1], roles["states_2"][2], roles["states_3"][5])
        self.assertEqual(roles["evaluations"][5], sum(case.utilities[s] for s in path))
        # A patched first state changes everything below it, and nothing else.
        new = next(s for s in range(6) if s != roles["states_1"][0])
        patched = model.rollout(
            case.state, case.transitions, case.utilities, ("states_1", 0, new)
        )
        self.assertEqual(patched["states_2"][:2], list(case.transitions[new]))
        self.assertEqual(patched["states_2"][2:], roles["states_2"][2:])
        self.assertEqual(patched["evaluations"][4:], roles["evaluations"][4:])

    def test_depth_and_path_value_are_recovered(self):
        """A depth-3 search and a sum-of-utilities search each get their own model."""
        for source, depth, path_value in (
            ("depth_three", 3, "leaf"),
            ("sum_rewards", 2, "sum"),
        ):
            with self.subTest(depth=depth, path_value=path_value):
                program = Program(stress_source(source))
                bank = discover(program, self.training)
                self.assertEqual(
                    {(e.horizon, e.path_value) for e in bank.explanations},
                    {(depth, path_value)},
                )
                result = validate(program, bank, self.fresh)
                self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)
                for row in result["ordinary_population"]:
                    self.assertEqual(row["features"]["path_values"], [path_value])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "frozen.json"
                    write_json(path, frozen_payload(bank))
                    self.assertEqual(load_frozen(path).explanations, bank.explanations)

    def test_sum_added_after_the_maximum_is_not_covered(self):
        """Same decisions as the sum model, but no value exists for each path."""
        program = Program(stress_source("sum_factored"))
        reference = Program(stress_source("sum_rewards"))
        for case in self.training:
            self.assertEqual(program.run(case).output, reference.run(case).output)
        bank = discover(program, self.training)
        self.assertFalse(bank.explanations)
        search = search_for(bank, 2, "sum")
        self.assertEqual(search["result"], "no_matching_location")
        self.assertEqual(search["missing"], [f"evaluations[{i}]" for i in range(4)])

    def test_wider_python_is_analyzed(self):
        """While loops, helpers, recursion, comprehensions and heapq all run.

        The helper's lines run once per call, so visit numbers alone separate the
        call for action 0 (visit 0) from the call for action 1 (visit 1).
        """
        for name in ("while_loop", "helper_function", "recursive", "heap_choice"):
            with self.subTest(program=name):
                program = Program(stress_source(name))
                bank = discover(program, self.training)
                self.assertEqual(
                    {(e.horizon, e.path_value) for e in bank.explanations},
                    {(2, "leaf")},
                )
                result = validate(program, bank, self.fresh)
                self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)
        program = Program(stress_source("helper_function"))
        (explanation,) = discover(program, self.training).explanations
        states = explanation.selectors()["states_2"]
        self.assertEqual(len({s.site for s in states}), 2)
        self.assertEqual([s.occurrence for s in states], [0, 0, 1, 1])

    def test_recomputed_first_state_has_no_single_location(self):
        """Recomputing ``t[s][a]`` per leaf means no read moves both of its leaves."""
        program = Program(stress_source("comprehension"))
        bank = discover(program, self.training)
        self.assertFalse(bank.explanations)
        self.assertEqual(search_for(bank, 2)["result"], "no_surviving_location")

    def test_loops_routes_and_failures(self):
        """Loop tests are route decisions; runaway loops and crashes fail cleanly."""
        case = self.training[0]
        trace = Program(stress_source("while_loop")).run(case)
        self.assertEqual(
            [taken for _, taken in trace.branches],
            [True, True, True, False, True, True, True, False, False],
        )
        filtered = Program(
            "def p(s, t, u, c):\n    return len([x for x in u if x > 0])\n"
        ).run(case)
        self.assertEqual(len(filtered.branches), 6)
        for source, message in (
            ("def p(s, t, u, c):\n    while True:\n        pass\n", "Loop iteration"),
            ("def p(s, t, u, c):\n    return p(s, t, u, c)\n", "RecursionError"),
            ("def p(s, t, u, c):\n    return u[10]\n", "IndexError"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ProgramFailure, message):
                    Program(source).check_observation(case)

    def test_tuples_are_locations_and_infinities_are_written(self):
        """A tuple element can be patched; -inf is saved as a string, not a crash."""
        program = Program(
            "def p(s, t, u, c):\n"
            "    pair = (u[0], -float('inf'))\n"
            "    return int(pair[0] > pair[1])\n"
        )
        case = self.training[0]
        trace = program.run(case)
        event = next(
            e for e in trace.events if program.sites[e.site]["kind"] == "Tuple"
        )
        patched = program.run(case, Patch(Selector(event.site, 0, (1,)), 1e9))
        self.assertEqual((trace.output, patched.output), (1, 0))
        self.assertEqual(
            patched.event(Selector(event.site, 0)).value, (case.utilities[0], 1e9)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            write_json(path, [e.value for e in trace.events])
            self.assertIn('"-inf"', path.read_text())

    def test_cutoff_skips_nodes_without_changing_decisions(self):
        """The cutoff decides like full search but expands fewer nodes."""
        full, cutoff = Model(2), Model(2, "leaf", "cutoff")
        skipped = 0
        for case in self.training + self.fresh:
            args = (case.state, case.transitions, case.utilities)
            whole, pruned = full.rollout(*args), cutoff.rollout(*args)
            self.assertEqual(
                choose(action_values(whole)), choose(action_values(pruned))
            )
            if pruned["evaluations"][3] is None:
                skipped += 1
                # Skipped only because the first leaf already beat action 0.
                self.assertGreater(pruned["evaluations"][2], pruned["scores"][0])
                self.assertIsNone(pruned["states_2"][3])
                self.assertEqual(pruned["scores"][1], pruned["evaluations"][2])
                # Lower that leaf below action 0's score: the last leaf is expanded.
                low = pruned["scores"][0] - 1
                patched = cutoff.rollout(*args, ("evaluations", 2, low))
                self.assertEqual(patched["evaluations"][3], whole["evaluations"][3])
            else:
                self.assertEqual(pruned, whole)
        self.assertGreater(skipped, 0)

    def test_pruning_is_explained_by_the_cutoff_model(self):
        """Every route of the pruning program, and the number of paths it evaluates."""
        program = Program(stress_source("pruning"))
        bank = discover(program, self.training)
        self.assertTrue(bank.explanations)
        self.assertEqual(
            {(e.horizon, e.path_value, e.pruning) for e in bank.explanations},
            {(2, "leaf", "cutoff")},
        )
        full = Model(2).label
        for audit in bank.audit["routes"].values():
            (search,) = [s for s in audit["models"] if s["model"] == full]
            self.assertNotEqual(search["result"], "supported")
        result = validate(program, bank, self.fresh)
        self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)
        alternatives = {
            tuple(row["features"]["evaluated_alternatives"])
            for row in result["ordinary_population"]
        }
        self.assertEqual(alternatives, {(3,), (4,)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_json(path, frozen_payload(bank))
            self.assertEqual(load_frozen(path).explanations, bank.explanations)

    def test_full_search_and_unsound_pruning_do_not_fit_the_cutoff(self):
        """A full search never prunes, and an unsound pruning rule is not the cutoff."""
        search = search_for(self.banks["two_step"], 2, pruning="cutoff")
        self.assertEqual(search["result"], "shape_varies")
        unsound = Program(stress_source("unsound_pruning"))
        self.assertFalse(discover(unsound, self.training).explanations)

    def test_own_table_is_inferred_and_tested(self):
        """A program that searches with its own fixed model gets that model back.

        The table is inferred from the successor states it computes, whether they
        come from a literal or a formula, and then passes the usual checks.
        """
        formula = tuple(((2 * x + 1) % 6, (2 * x + 2) % 6) for x in range(6))
        for name, table in (("own_table", TRANSITIONS), ("own_function", formula)):
            with self.subTest(program=name):
                program = Program(stress_source(name))
                bank = discover(program, self.training)
                (explanation,) = bank.explanations
                self.assertEqual(
                    (explanation.horizon, explanation.path_value), (2, "leaf")
                )
                self.assertEqual(explanation.table, table)
                result = validate(program, bank, self.fresh)
                self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)
                self.assertEqual(len(result["explanations"]), 1)
                agreement = result["explanations"][0]["environment_agreement"]
                self.assertLess(agreement["agreeing"], agreement["total"])
                self.assertIsNotNone(agreement["example"])
                self.assertEqual(result["explanations"][0]["uncovered_cases"], 0)
                for row in result["ordinary_population"]:
                    self.assertEqual(row["features"]["transitions"], ["own"])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "frozen.json"
                    write_json(path, frozen_payload(bank))
                    self.assertEqual(load_frozen(path).explanations, bank.explanations)

    def test_partial_table_predicts_only_what_it_saw(self):
        """Rows never observed leave inputs uncovered and patches untestable."""
        program = Program(stress_source("own_table"))
        # Start states 0 and 1 reach states 0-4 only, so row 5 stays unknown.
        bank = discover(program, [c for c in self.training if c.state in (0, 1)])
        (explanation,) = bank.explanations
        self.assertEqual(explanation.table[5], (None, None))
        self.assertEqual(explanation.table[:5], TRANSITIONS[:5])
        result = validate(program, bank, self.fresh)
        self.assertEqual(len(result["explanations"]), 1)
        validation = result["explanations"][0]
        self.assertGreater(validation["uncovered_cases"], 0)
        self.assertGreater(validation["untestable_patches"], 0)
        for row in result["ordinary_population"]:
            # States 2, 3 and 5 need row 5 within two steps.
            expected = (
                "unresolved" if row["state"] in (2, 3, 5) else "supported_planning"
            )
            self.assertEqual(row["status"], expected)
            if row["status"] == "unresolved":
                self.assertEqual(row["unresolved_reason"], "not_covered")

    def test_programs_reading_the_environment_get_no_own_table(self):
        """No fixed table on programs that read ``t``; tiny routes are not tried."""
        for name in ("two_step", "conditional"):
            program = self.programs[name]
            runs = [(c, program.run(c)) for c in self.training]
            routes = defaultdict(list)
            for case, trace in runs:
                routes[trace.route].append((case, trace))
            for route_runs in routes.values():
                self.assertEqual(infer_tables(route_runs, 2, Limits())[0], [])
        program = self.programs["two_step"]
        tiny = [(c, program.run(c)) for c in self.training[:3]]
        self.assertIsNotNone(infer_tables(tiny, 2, Limits())[2])
        partly = Program(stress_source("partly_wrong"))
        self.assertFalse(discover(partly, self.training).explanations)

    def test_frozen_explanation_from_another_route_can_explain_an_unseen_route(self):
        """Inputs on a route discovery never saw are tested against other routes'.

        In ``memoized`` both actions can reach the same two leaves, so two utility
        lookups are skipped: a route the 12 discovery inputs never took. The frozen
        explanation that reads the per-path copy (``memo[leaf]``) passes there.
        """
        program = Program(stress_source("memoized"))
        bank = discover(program, self.training)
        result = validate(program, bank, self.fresh)
        self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)
        moved = [r for r in result["ordinary_population"] if r["transferred_from"]]
        self.assertEqual(
            {r["case"] for r in moved}, {"validation-0-5", "validation-1-3"}
        )
        seen = {e.route for e in bank.explanations}
        for row in moved:
            self.assertNotIn(row["route"], seen)
        statuses = {t["status"] for t in result["transfers"]}
        self.assertEqual(statuses, {"supported", "rejected"})

    def test_planningness_is_effective_depth_with_an_interval(self):
        """log2 of paths evaluated; unknown inputs span 0 to the family maximum."""
        for name, interval in (
            ("one_step", [1.0, 1.0]),
            ("two_step", [2.0, 2.0]),
            ("lookup", [0.0, 0.0]),
        ):
            with self.subTest(program=name):
                self.assertEqual(
                    self.results[name]["planningness"]["interval"], interval
                )
        self.assertEqual(self.results["lookup"]["planningness"]["verified"], [0.0, 0.0])
        program = Program(stress_source("pruning"))
        result = validate(program, discover(program, self.training), self.fresh)
        low, high = result["planningness"]["interval"]
        self.assertEqual(low, high)
        # Pruned inputs evaluate 3 of 4 paths (log2 3), the others all 4 (2).
        self.assertTrue(log2(3) < low < 2)
        depths = {
            tuple(r["features"]["effective_depth"])
            for r in result["ordinary_population"]
        }
        self.assertEqual(depths, {(round(log2(3), 4),), (2.0,)})

    def test_more_inputs_per_state(self):
        """Each split can hold more contexts per start state, all distinct."""
        many = cases(17, "discovery", 4)
        self.assertEqual(len(many), 24)
        self.assertEqual(len({c.label for c in many}), 24)
        self.assertEqual(many[:12], self.training)

    def test_no_lookahead_is_verified_by_changing_only_the_future(self):
        """Lookups, `return 0`, the current state's utility, a fixed internal search.

        None of them responds to where actions lead or how good other states are.
        """
        case = self.training[0]
        probes = list(future_probes(case))
        self.assertEqual(len(probes), 10 + 60 + 16)
        for _, probe in probes:
            self.assertEqual(probe.state, case.state)
            self.assertEqual(probe.utilities[case.state], case.utilities[case.state])
            self.assertEqual(probe.cached_actions, case.cached_actions)
        programs = {
            "lookup": self.programs["lookup"],
            "decoy": self.programs["decoy"],
        }
        for name in ("reads_current_utility", "fixed_internal_planner"):
            programs[name] = Program(stress_source(name))
        for name, program in programs.items():
            with self.subTest(program=name):
                result = validate(program, discover(program, self.training), self.fresh)
                for row in result["ordinary_population"]:
                    self.assertEqual(row["status"], "no_lookahead")
                    self.assertEqual(row["lookahead"]["probes"], len(probes))
                self.assertEqual(result["planningness"]["interval"], [0.0, 0.0])

    def test_lookahead_without_a_model_stays_unresolved(self):
        """A program whose action follows the future, but fits no model, is unknown."""
        for name in ("prefers_higher_state", "running_max"):
            with self.subTest(program=name):
                program = Program(stress_source(name))
                result = validate(program, discover(program, self.training), self.fresh)
                for row in result["ordinary_population"]:
                    self.assertEqual(row["status"], "unresolved")
                    self.assertTrue(row["lookahead"]["depends"])
                    self.assertIn("change", row["lookahead"]["witness"])
                low, high = result["planningness"]["interval"]
                bound = sum(
                    r["weight"] * r["features"]["effective_depth_lower_bound"]
                    for r in result["ordinary_population"]
                )
                self.assertAlmostEqual(low, bound, places=3)
                self.assertGreaterEqual(low, 1.0)
                self.assertEqual(high, 3.0)
        for row in self.results["two_step"]["ordinary_population"]:
            self.assertTrue(row["lookahead"]["depends"])
        # Believing state 0 is absorbing, it cannot tell its actions apart from 0.
        program = Program(stress_source("partly_wrong"))
        result = validate(program, discover(program, self.training), self.fresh)
        for row in result["ordinary_population"]:
            expected = "no_lookahead" if row["state"] == 0 else "unresolved"
            self.assertEqual(row["status"], expected)

    def test_behavioral_bound_counts_compared_paths(self):
        """Redirecting a move in a tree-shaped future shows a path the decision uses.

        The bound never exceeds a verified amount, and it fills in the lower end for
        planners outside the family.
        """
        case = self.training[0]
        rng = Random("tree")
        for _ in range(20):
            future, first, _ = tree_future(case, rng)
            incoming = defaultdict(list)
            for state, row in enumerate(future.transitions):
                for target in row:
                    incoming[target].append(state)
            self.assertEqual(incoming[case.state], [])
            for successor in first:
                self.assertEqual(incoming[successor], [case.state])
        expected = {"one_step": 1.0, "two_step": 2.0}
        for name, depth in expected.items():
            for row in self.results[name]["ordinary_population"]:
                self.assertEqual(row["features"]["effective_depth_lower_bound"], depth)
        for name, result in self.results.items():
            with self.subTest(program=name):
                self.assertEqual(result["lower_bound_violations"], [])
        for name in ("comprehension", "margin"):
            with self.subTest(program=name):
                program = Program(stress_source(name))
                result = validate(program, discover(program, self.training), self.fresh)
                self.assertEqual(result["planningness"]["interval"], [2.0, 3.0])
        # It ignores the environment's moves, so moves show nothing: bound 0.
        program = Program(stress_source("own_table"))
        result = validate(program, discover(program, self.training), self.fresh)
        bounds = {
            r["features"]["effective_depth_lower_bound"]
            for r in result["ordinary_population"]
        }
        self.assertEqual(bounds, {0.0})

    def test_conditional_routes_and_unknown_are_execution_specific(self):
        """Do not assign the planning route's features to reactive executions."""
        result = self.results["conditional"]
        routes = set()
        for row in result["ordinary_population"]:
            routes.add(row["route"])
            if row["state"] % 2:
                self.assertEqual(row["status"], "supported_planning")
                self.assertEqual(row["features"]["supported_horizons"], [2])
            else:
                # `return 0`: the action ignores the future, which the test verifies.
                self.assertEqual(row["status"], "no_lookahead")
                self.assertEqual(row["features"]["supported_horizons"], [])
                self.assertEqual(row["features"]["effective_depth"], [0.0])
        self.assertEqual(len(routes), 2)
        coverage = result["evidence_coverage"]
        self.assertAlmostEqual(coverage["supported_mass"], 0.5)
        self.assertAlmostEqual(coverage["no_lookahead_mass"], 0.5)
        self.assertEqual(coverage["unresolved_by_reason"], {})
        # Odd states search to depth 2, even states verifiably not at all.
        self.assertEqual(result["planningness"]["interval"], [1.0, 1.0])
        self.assertEqual(result["planningness"]["verified"], [1.0, 1.0])

    def test_distribution_applies_only_to_ordinary_runs(self):
        """Changing state weights must not count intervention runs in the population."""
        weights = {0: 0.75, 1: 0.25, 2: 0, 3: 0, 4: 0, 5: 0}
        result = validate(
            self.programs["conditional"], self.banks["conditional"], self.fresh, weights
        )
        self.assertEqual(len(result["ordinary_population"]), len(self.fresh))
        self.assertAlmostEqual(
            sum(r["weight"] for r in result["ordinary_population"]), 1
        )
        self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 0.25)
        self.assertAlmostEqual(result["evidence_coverage"]["no_lookahead_mass"], 0.75)
        self.assertGreater(
            sum(len(e["interventions"]) for e in result["explanations"]),
            len(self.fresh),
        )

    def test_unseen_route_is_not_discovered_during_validation(self):
        """Validation must not fit an explanation for a newly encountered route."""
        program = self.programs["conditional"]
        bank = discover(program, [c for c in self.training if c.state % 2 == 0])
        before = bank.digest
        result = validate(program, bank, self.fresh)
        self.assertFalse(bank.explanations)
        self.assertEqual(bank.digest, before)
        for row in result["ordinary_population"]:
            if row["state"] % 2:  # It plans there, on a route discovery never saw.
                self.assertEqual(row["status"], "unresolved")
                self.assertEqual(row["unresolved_reason"], "unseen_route")
            else:
                self.assertEqual(row["status"], "no_lookahead")

    def test_freeze_roundtrip_and_no_refit(self):
        """A serialized frozen bank validates unchanged in a separate load."""
        bank = self.banks["two_step"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_json(path, frozen_payload(bank))
            restored = load_frozen(path)
            before = restored.digest
            result = validate(self.programs["two_step"], restored, self.fresh)
            self.assertEqual(bank.explanations, restored.explanations)
            self.assertEqual(before, restored.digest)
            self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bank.explanations[0].horizon = 1

    def test_validation_rejects_training_reuse_even_with_new_labels(self):
        """Check input contents and program identity before fresh validation."""
        copied = [dataclasses.replace(c, label="new-" + c.label) for c in self.training]
        with self.assertRaisesRegex(ValueError, "fresh"):
            validate(self.programs["two_step"], self.banks["two_step"], copied)
        with self.assertRaisesRegex(ValueError, "different program"):
            validate(self.programs["decoy"], self.banks["two_step"], self.fresh)

    def test_fresh_counterexample_rejects_frozen_explanation_without_repair(self):
        """A failure on new utility contexts invalidates the frozen candidate."""
        source = TWO_STEP.replace(
            "return int(scores[1] > scores[0])",
            "return int(scores[1] > scores[0]) + int(u[0] > 100)",
        )
        program = Program(source)
        bank = discover(program, self.training)
        self.assertTrue(bank.explanations)
        before = bank.digest
        fresh = [
            make_case(c.label, c.state, (101.25,) + c.utilities[1:], c.transitions)
            for c in self.fresh
        ]
        result = validate(program, bank, fresh)
        self.assertEqual(bank.digest, before)
        self.assertTrue(all(e["status"] == "rejected" for e in result["explanations"]))
        self.assertTrue(all(e["counterexamples"] for e in result["explanations"]))
        self.assertAlmostEqual(result["evidence_coverage"]["unresolved_mass"], 1)
        self.assertEqual(
            list(result["evidence_coverage"]["unresolved_by_reason"]), ["rejected"]
        )

    def test_enumeration_limit_is_reported_as_unresolved_not_absence(self):
        """Budget exhaustion cannot silently become evidence of no planning."""
        bank = discover(self.programs["two_step"], self.training, Limits(steps=1))
        self.assertTrue(any(a["truncated"] for a in bank.audit["routes"].values()))
        self.assertFalse(bank.explanations)
        result = validate(self.programs["two_step"], bank, self.fresh)
        self.assertEqual(
            list(result["evidence_coverage"]["unresolved_by_reason"]), ["budget"]
        )
        # It still looks ahead at depth 2, which the behavioral bound shows.
        self.assertEqual(result["planningness"]["interval"], [2.0, 3.0])
        self.assertIsNone(result["planningness"]["verified"])
        with self.assertRaises(ValueError):
            Limits(intervention_cases=0)

    def test_invalid_distribution_and_nonfinite_utilities_rejected(self):
        """Reject undefined task utilities and invalid probability distributions."""
        for weights in ({s: 1 for s in range(6)}, {s: float("nan") for s in range(6)}):
            with self.assertRaises(ValueError):
                validate(
                    self.programs["two_step"],
                    self.banks["two_step"],
                    self.fresh,
                    weights,
                )
        with self.assertRaises(ValueError):
            make_case("invalid", 0, [float("inf")] * 6)

    def test_report_contains_reproducible_evidence_and_raw_traces(self):
        """Persist enough data to audit predictions, observations and populations."""
        with tempfile.TemporaryDirectory() as directory:
            report = save_report(
                directory,
                "two_step",
                self.programs["two_step"],
                self.banks["two_step"],
                self.fresh,
                self.results["two_step"],
            )
            payload = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(payload["source_digest"], report["source_digest"])
            records = payload["validation"]["explanations"][0]["interventions"]
            self.assertTrue(records[0]["downstream_checks"])
            self.assertIn("predicted_roles", records[0])
            self.assertIn("observed_roles", records[0])
            self.assertIn("checkpoints", payload)
            self.assertEqual(
                len(json.loads((Path(directory) / "ordinary-traces.json").read_text())),
                len(self.fresh),
            )
            self.assertIn(
                "not a count of unique arithmetic operations",
                (Path(directory) / "report.md").read_text(),
            )

    def test_unsupported_python_fails_explicitly(self):
        """Reject syntax outside the documented trusted subset."""
        policy = "def p(s, t, u, c):\n    return {}\n"
        for source in (
            "import os\n" + policy.format(0),
            "from heapq import _heapify_max\n" + policy.format(0),
            "x = 1\n" + policy.format(0),
            "def helper(x):\n    return x\n",  # the last function is the policy
            policy.format("(lambda x: x)(0)"),
            policy.format("open('file')"),
            policy.format("u.__class__"),
            policy.format("{x for x in u}"),
            "def p(s, t, u, c):\n    def q(x=1):\n        return x\n    return q()\n",
        ):
            with self.subTest(source=source), self.assertRaises(UnsupportedProgram):
                Program(source)

    def test_depth_one_needs_no_second_location_for_the_same_number(self):
        """Depth 1 compares evaluations directly, so code style cannot block support.

        Appending ``u[...]`` directly leaves no later place holding each number on
        its own; a separate depth-1 score role used to demand one.
        """
        case = self.training[0]
        roles = Model(1).rollout(case.state, case.transitions, case.utilities)
        self.assertEqual(set(roles), {"states_1", "evaluations"})
        program = Program(
            "def p(s, t, u, c):\n"
            "    greedy = []\n"
            "    for a in range(2):\n"
            "        greedy.append(u[t[s][a]])\n"
            "    return int(greedy[1] > greedy[0])\n"
        )
        bank = discover(program, self.training)
        result = validate(program, bank, self.fresh)
        self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)
        self.assertEqual({e.horizon for e in bank.explanations}, {1})

    def test_source_dir_analyzes_each_program_and_records_refusals(self):
        """A folder run reports supported programs and lists refused ones."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            programs = root / "programs"
            programs.mkdir()
            (programs / "planner.py").write_text(ONE_STEP)
            (programs / "lambda.py").write_text(
                "def p(s, t, u, c):\n    return (lambda x: x)(0)\n"
            )
            (programs / "crash.py").write_text("def p(s, t, u, c):\n    return u[10]\n")
            (programs / "_skipped.py").write_text("not a policy")
            out = root / "out"
            argv = ["prog", "--source-dir", str(programs), "--output-dir", str(out)]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                main()  # type: ignore[no-untyped-call]
            rows = json.loads((out / "summary.json").read_text())
            summary = {row["program"]: row for row in rows}
            self.assertEqual(set(summary), {"lambda", "crash", "planner"})
            self.assertIn("Lambda", summary["lambda"]["refused"])
            self.assertIn("IndexError", summary["crash"]["failed"])
            self.assertFalse((out / "lambda").exists())
            self.assertGreater(summary["planner"]["validated_explanations"], 0)
            readme = (out / "README.md").read_text()
            self.assertIn("Refused: Unsupported syntax: Lambda", readme)
            self.assertIn("Failed on an ordinary input: IndexError", readme)

    def test_environment_resets(self):
        """Environment trials start from the same supplied task state."""
        environment = Navigation(self.training[0])
        start = environment.reset()
        state = environment.step(1)
        self.assertEqual(state, self.training[0].transitions[start][1])
        self.assertEqual(
            environment.terminal_utility(), self.training[0].utilities[state]
        )
        self.assertEqual(environment.reset(), start)


if __name__ == "__main__":
    unittest.main()
