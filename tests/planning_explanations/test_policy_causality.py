"""Regression tests for the complete search -> cache -> later-action chain."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from robocode.planning_explanations.search.grid_engine import GridSearch, GridSpec
from robocode.planning_explanations.search.policy_causality import (
    TAIL_REPLACES,
    Binding,
    PlanRule,
    bind_plan,
    judge_probe,
    read_plan,
)
from robocode.planning_explanations.search.policy_discovery import (
    _count_runs,
    _sample_runs,
    discover_policy,
    load_policy_bank,
    validate_policy,
)
from robocode.planning_explanations.search.policy_domains import DiscWorld
from robocode.planning_explanations.search.policy_fixtures import (
    DISCARDED_SEARCH,
    ENDPOINT_REPLACED,
    PLAN_ONCE,
    UNUSED_PLAN,
)
from robocode.planning_explanations.search.policy_runner import PolicyProgram

STORE = '        self.path = self._plan(state["robot"], self.goal, state["obstacles"])'
FOLLOW = "    def get_action(self, state):\n        return self._follow(state)"


def analyze(source, development=((0, 0),), validation=((0, 1),)):
    """Discover once; validate on disjoint episodes with observation parity."""
    program, domain = PolicyProgram(source), DiscWorld()
    bank = discover_policy(program, domain, development, validation)
    result = validate_policy(program, domain, bank)
    return bank, result["episodes"][0]


def probes(episode):
    """Every stored-plan intervention including rejected predictions."""
    return [p for v in episode["versions"] for c in v["chains"] for p in c["probes"]]


class CrossCallCausalityTest(unittest.TestCase):
    """Shape, changed cache, and changed actions must not be conflated."""

    def test_discarded_search_no_longer_gets_credit(self):
        """Independent goal waypoints match a subtree and rotate actions: no credit."""
        _, episode = analyze(DISCARDED_SEARCH)
        row = episode["accounting"]["program"]
        self.assertTrue(row["complete"])
        self.assertEqual(row["expansion_events"], 459)
        self.assertEqual(row["attributed_expansion_events"], 0)
        self.assertEqual(row["followed_plan_share"], 0)
        # No frozen readout reads its cache as a search path: not even the max.
        self.assertEqual(row["followed_plan_share_max"], 0)
        self.assertGreater(row["followed_unmatched_plan"], 0)
        self.assertEqual(row["runs_without_action_witness"], 1)

    def test_genuine_plan_predicts_stored_value_and_mediates_action(self):
        """The very same search perturbation reaches both cache and later command."""
        bank, episode = analyze(PLAN_ONCE)
        witnesses = [p for p in probes(episode) if p["status"] == "supported"]
        self.assertTrue(witnesses)
        for probe in witnesses:
            self.assertEqual(probe["predicted_plan"], probe["observed_plan"])
            self.assertTrue(probe["changed"])
            action = probe["future"][-1]
            self.assertGreater(action["call"], 0)
            self.assertNotEqual(action["ordinary_action"], action["observed_action"])
            self.assertEqual(action["predicted_action"], action["observed_action"])
            self.assertEqual(action["restored_action"], action["ordinary_action"])
        # Reference/probe executions never add to the ordinary expansion count.
        self.assertTrue(any(p["edit"]["operation"] == "drop_cell" for p in witnesses))
        row = episode["accounting"]["program"]
        self.assertEqual(row["planning_events"], 1)
        self.assertEqual(row["attributed_expansion_events"], row["expansion_events"])
        self.assertEqual(row["followed_plan_share"], 1.0)
        self.assertEqual(row["followed_plan_share_max"], 1.0)
        self.assertEqual(bank.manifest()["schema"], 12)

    def test_delayed_consumer_is_checked_across_multiple_calls(self):
        """Identical early actions do not stop the experiment prematurely."""
        source = PLAN_ONCE.replace(STORE, STORE + "\n        self.wait = 3").replace(
            FOLLOW,
            "    def get_action(self, state):\n"
            "        if self.wait:\n"
            "            self.wait -= 1\n"
            "            return (0.0, 0.0)\n"
            "        return self._follow(state)",
        )
        _, episode = analyze(source)
        witnesses = [p for p in probes(episode) if p["status"] == "supported"]
        self.assertTrue(witnesses)
        self.assertTrue(all(p["future"][-1]["call"] >= 4 for p in witnesses))
        self.assertTrue(all(not p["future"][0]["changed"] for p in witnesses))

    def test_changed_but_unused_plan_is_unresolved_not_attributed(self):
        """Search really changes the cache, but no later action uses it."""
        _, episode = analyze(UNUSED_PLAN)
        changed = [p for p in probes(episode) if p.get("changed")]
        self.assertTrue(changed)
        self.assertTrue(all(p["prediction_matches"] for p in changed))
        self.assertTrue(all(p["status"] == "unresolved" for p in changed))
        self.assertEqual(
            episode["accounting"]["program"]["attributed_expansion_events"], 0
        )

    def test_unrelated_flag_cannot_substitute_for_plan_mediation(self):
        """Search alters a heading and an unused plan; only the heading acts."""
        source = UNUSED_PLAN.replace(
            STORE, STORE + "\n        self.angle = sum(x + 2 * y for x, y in self.path)"
        ).replace(
            "    def get_action(self, state):\n",
            "    def get_action(self, state):\n"
            "        return (0.035 * math.cos(self.angle), 0.035 * math.sin(self.angle))\n",
        )
        _, episode = analyze(source)
        rejected = [
            p
            for p in probes(episode)
            if p.get("reason") == "plan_does_not_mediate_action"
        ]
        self.assertTrue(rejected)
        self.assertTrue(any(p["future"][-1]["changed"] for p in rejected))
        self.assertEqual(
            episode["accounting"]["program"]["attributed_expansion_events"], 0
        )

    def test_replaced_cache_does_not_inherit_search_credit(self):
        """An independently replaced cache can drive actions without using search."""
        source = PLAN_ONCE.replace(
            FOLLOW,
            "    def get_action(self, state):\n"
            "        self.path = [self.goal]\n"
            "        return self._follow(state)",
        )
        _, episode = analyze(source)
        self.assertFalse(any(p["status"] == "supported" for p in probes(episode)))
        self.assertEqual(
            episode["accounting"]["program"]["attributed_expansion_events"], 0
        )

    def test_fallback_after_a_lost_endpoint_is_not_a_counterexample(self):
        """What a policy stores once its search fails is outside the readout."""
        source = PLAN_ONCE.replace(
            STORE,
            STORE + "\n        if len(self.path) == 1:"
            "\n            self.path = [(-1.0, -1.0)]",
        )
        _, episode = analyze(source)
        found = probes(episode)
        self.assertTrue(any(p.get("reason") == "endpoint_lost" for p in found))
        self.assertFalse([p for p in found if p["status"] == "rejected"])
        row = episode["accounting"]["program"]
        self.assertEqual(row["attributed_expansion_events"], row["expansion_events"])

    def test_only_the_search_that_produced_the_plan_gets_credit(self):
        """Two identical searches; the stored plan comes from the second only."""
        source = PLAN_ONCE.replace(
            STORE,
            '        self._plan(state["robot"], self.goal, state["obstacles"])\n'
            + STORE,
        )
        _, episode = analyze(source)
        (version,) = episode["versions"]
        first, second = version["chains"]
        # Editing the first search leaves the plan as the second one made it.
        self.assertEqual(first["status"], "rejected")
        self.assertTrue(
            any(
                p.get("reason") == "stored_plan_prediction_mismatch"
                for p in first["probes"]
            )
        )
        self.assertEqual(second["status"], "supported")
        self.assertEqual(version["causal_status"], "supported")
        row = episode["accounting"]["program"]
        self.assertEqual(row["followed_plan_share"], 1.0)
        self.assertEqual(
            2 * row["attributed_expansion_events"], row["expansion_events"]
        )

    def test_final_point_replacing_the_endpoint_cell_is_read(self):
        """A stored path missing its endpoint cell, then an exact point inside it."""
        _, episode = analyze(ENDPOINT_REPLACED)
        replaced = [
            c
            for v in episode["versions"]
            for c in v["chains"]
            if c["rule"]["tail"] == TAIL_REPLACES
        ]
        self.assertTrue(replaced)
        self.assertTrue(all(c["endpoint_replaced_by_final_point"] for c in replaced))
        self.assertTrue(any(c["status"] == "supported" for c in replaced))
        row = episode["accounting"]["program"]
        self.assertEqual(row["followed_plan_share"], 1.0)

    def test_the_plan_producing_search_is_always_probed(self):
        """A search that stores the plan is probed, however many others run."""
        source = PLAN_ONCE.replace(
            FOLLOW,
            "    def get_action(self, state):\n"
            '        self._plan(state["robot"], self.goal, state["obstacles"])\n'
            "        return self._follow(state)",
        )
        program = PolicyProgram(source)
        counts, producers, _ = _count_runs(program, DiscWorld(), (0, 1))
        (site,) = [
            s
            for s, info in program.sites.items()
            if info["source"] == "while frontier:"
        ]
        self.assertGreater(counts[site], 8)
        # Only `reset` stores a plan; the searches in later calls are discarded.
        self.assertEqual(producers[site], {0})
        for seed in range(5):
            chosen = _sample_runs(counts, producers, "validation", seed, 8)
            self.assertIn(0, chosen[site])
            self.assertEqual(len(chosen[site]), 8)

    def test_producer_first_selection_fills_and_caps_with_seeded_draws(self):
        counts = {"a": 20, "b": 5}
        many = {"a": set(range(12))}
        chosen = _sample_runs(counts, many, "validation", (0, 1), 8)
        self.assertTrue(chosen["a"] <= set(range(12)))
        self.assertEqual(len(chosen["a"]), 8)
        self.assertEqual(chosen["b"], set(range(5)))
        none = _sample_runs(counts, {}, "validation", (0, 1), 8)
        self.assertEqual(len(none["a"]), 8)

    def test_readout_templates_are_frozen_and_legacy_banks_require_rediscovery(self):
        """Validation cannot invent a readout, or silently upgrade an old bank."""
        program, domain = PolicyProgram(PLAN_ONCE), DiscWorld()
        bank = discover_policy(program, domain, ((0, 0),), ((0, 1),))
        for part in bank.parts.values():
            part["plan_rules"] = []
        fingerprint = bank.fingerprint
        result = validate_policy(program, domain, bank)
        self.assertEqual(bank.fingerprint, fingerprint)
        self.assertEqual(result["summary"]["mean"]["attributed_expansion_events"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            bank.save(path)
            self.assertEqual(load_policy_bank(path).fingerprint, fingerprint)
            data = json.loads(path.read_text())
            data["schema"] = 1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Rediscover"):
                load_policy_bank(path)


def _box(cell) -> bool:
    return 0 <= cell[0] < 5 and 0 <= cell[1] < 5


def _point(cell) -> tuple:
    return (float(cell[0]), float(cell[1]))


class JudgeProbeTest(unittest.TestCase):
    """Only a wrong plan for an endpoint the edited search keeps is a refutation."""

    def setUp(self):
        spec = GridSpec(priority="g", goal_timing="none")
        self.engine = GridSearch(spec, (0, 0), None, _box)
        while self.engine.phase == "ready":
            self.engine.step()
        self.rule = PlanRule("path", "point", 0)
        self.points = read_plan(self.engine, _point, Binding(0, (2, 2), ()))
        self.binding = bind_plan(self.rule, self.points, self.engine, _point)
        # The same tree, except (2, 2) now hangs off its other neighbour.
        self.rerouted = copy.deepcopy(self.engine)
        old = self.engine.parents[(2, 2)]
        self.rerouted.parents[(2, 2)] = (2, 1) if old == (1, 2) else (1, 2)

    def judge(self, patched, observed) -> dict:
        return judge_probe(
            self.rule,
            self.binding,
            self.points,
            self.engine,
            patched,
            _point,
            observed,
        )

    def test_binding_reads_a_final_point_in_place_of_the_endpoint(self):
        """The final point's own cell extends the stored chain by one cell."""
        rule = PlanRule("path", "point", TAIL_REPLACES)
        stored = self.points[:-1] + ((2.1, 1.9),)
        binding = bind_plan(rule, stored, self.engine, _point)
        self.assertEqual((binding.endpoint, binding.replaced), ((2, 2), True))
        self.assertEqual(read_plan(self.engine, _point, binding), stored)
        with self.assertRaises(ValueError):
            # (0, 0) holds this point, but it does not extend the chain.
            bind_plan(rule, self.points[:-1] + ((0.1, 0.1),), self.engine, _point)

    def test_binding_reads_a_plan_missing_its_root(self):
        """A root waypoint consumed in the same call is a bound value, not a rule."""
        binding = bind_plan(self.rule, self.points[1:], self.engine, _point)
        self.assertEqual((binding.skip, binding.endpoint), (1, (2, 2)))

    def test_predicted_change_is_replayed(self):
        predicted = read_plan(self.rerouted, _point, self.binding)
        outcome = self.judge(self.rerouted, predicted)
        self.assertEqual(outcome["status"], "replay")
        self.assertTrue(outcome["changed"])

    def test_unchanged_plan_is_unresolved(self):
        outcome = self.judge(self.engine, self.points)
        self.assertEqual(outcome["reason"], "plan_unchanged")

    def test_lost_endpoint_is_unresolved_whatever_is_stored(self):
        patched = copy.deepcopy(self.engine)
        del patched.parents[(2, 2)]
        for observed in ((), ((-1.0, -1.0),), self.points):
            outcome = self.judge(patched, observed)
            self.assertEqual(
                (outcome["status"], outcome["reason"]),
                ("unresolved", "endpoint_lost"),
            )

    def test_path_to_another_destination_is_unresolved(self):
        other = read_plan(self.engine, _point, Binding(0, (3, 2), ()))
        outcome = self.judge(self.engine, other)
        self.assertEqual(outcome["reason"], "destination_changed")
        self.assertEqual(outcome["rebound"]["endpoint"], (3, 2))

    def test_stale_plan_for_a_kept_endpoint_is_a_counterexample(self):
        outcome = self.judge(self.rerouted, self.points)
        self.assertEqual(
            (outcome["status"], outcome["reason"]),
            ("rejected", "stored_plan_prediction_mismatch"),
        )


if __name__ == "__main__":
    unittest.main()
