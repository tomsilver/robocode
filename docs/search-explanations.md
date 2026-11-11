# Search explanations: implementation plan and contract

Recorded 2026-11-08; milestone 8 added 2026-11-09. Implements milestones 1–8
of the agreed design, following Sections 2–5 of *Planning Explanations for
Python Policies* (the supplied PDF). This is a separate model family from the
existing lookahead analyzer.

The original `line` suite covers milestones 1–3. Milestones 4–5 add explicit
evidence findings and a `tasks` suite with held-out transition tables and goals.
Milestone 6 adds bounded nested-container mappings and a `routes` suite with
development-discovered, frozen checkpoint-prefix gates. All three suites use
the same checkpoint machinery and paired verification. Milestone 7 adds an
opt-in priority-search engine family while preserving the earlier engine.
Milestone 8 adds a `policy` suite: whole class-based policies, closed-loop
episodes, cached plans, and a real agent-written KinDER policy.

Issue 1 corrected 2026-11-10: cross-call attribution now requires a complete
**search intervention → predicted stored plan → later action** witness.
See the implementation decisions below. Older policy banks must be
rediscovered; schema 4 freezes readout templates (the root offset is bound per
plan; a final point may follow or replace the endpoint cell), the
destination-keeping detour probe, plan-producer-first probe selection, and the
four followed-plan categories. Issue 2 (also 2026-11-10, schema 5): a loop the
analyzer cannot read is reported unknown, never zero planning.

## Run milestone 8: a stateful KinDER policy

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.search --suite policy
open outputs/search-policy-benchmark/summary.md
open outputs/search-policy-benchmark/clutteredretrieval2d/report.html
```

The whole suite takes about two minutes. `--fixture clutteredretrieval2d` runs
only the KinDER policy (about 70 seconds); `--fixture plan_once` runs one
control in a few seconds. Replay a frozen bank without discovering again:

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.search --suite policy --fixture clutteredretrieval2d --frozen outputs/search-policy-benchmark/clutteredretrieval2d/frozen.json --output-dir outputs/search-policy-replay
```

Another cached ClutteredRetrieval2D policy runs with `--source path/to/approach.py`
(and `--env-config` for a different configuration). Other KinDER environments
need their own domain adapter, which declares the robot's position and which
objects are obstacles; that is environment meaning, not policy analysis.

The problem addressed here: real policies plan inside helpers, keep the plan
in an attribute, and spend most calls following it. Milestones 1–7 explain
one function call at a time, so planning done in `reset` would either vanish
or, if attributed to every action, be counted hundreds of times.

### The KinDER policy, walked through

The policy (529 lines, `search/policies/PROVENANCE.md`) has six `while` loops.
Discovery reads only value shapes at each loop's first checkpoint, never names:

| Part | Where | Frozen explanation |
| --- | --- | --- |
| `while frontier:` | `reset` | Uniform-cost search with **no goal test**: it explores every cell its own `free()` allows, then picks a stance. Plans read out through `point` |
| `while q:` | `_make_carry_path` | A* with a Euclidean heuristic, root entry 0, goal tested on selection, improving updates. Plans read out through `to_p` |
| four others | `reset`, `get_action`, `_make_carry_path` | Path reconstruction and waypoint popping; not search-shaped, listed in coverage |

Each search part keeps two explanations that differ only in how stale heap
entries are handled; no development run popped a stale entry, so both remain.
Every iteration of every run is compared (about 2,000 per reachability search).
Up to five seeded edits per sampled run, plus one destination-keeping detour
per stored plan it may have produced, must predict the whole continuation.

Validation results after the Issue-1 follow-ups
(`outputs/search-policy-issue1/summary.md`). The episodes were not seen during
discovery. The followed-plan share is a range: witnessed calls, up to the max
that also counts calls whose plan evidence is unresolved.

| Episode (obstacles, seed) | Solved | Planning events | Expansion events | At initialization | Followed plan: witnessed / unresolved / disproved / unmatched | Followed-plan share (witnessed – max) |
| --- | --- | --- | --- | --- | --- | --- |
| (3, 100) | no | 11 | 11,264 | 1,931 | 286 / 240 / 0 / 51 | 0.38–0.70 |
| (15, 101) | no | 97 | 17,392 | 1,636 | 97 / 0 / 0 / 0 | 0.04 |
| (20, 102) | no | 127 | 127 | 19 | 0 / 0 / 0 / 0 | 0.00 |

The original milestone-8 shares (0.70, 0.04, 0.00) used structural links and
are not causal evidence; they equal the maxima above. In (3, 100) the robot
gets stuck and replans every 121 steps (`mission_steps > 120`). The 240
unresolved calls follow two of those plans: their changed stored paths were
predicted exactly, but the earliest cell a detour could remove without cutting
off the destination is the third waypoint, and the robot did not get past the
second within the 32 replayed calls. The 51 unmatched calls follow a "path"
that is only the standing point (the stance cell was the robot's own cell). In
(15, 101) one reset call runs the reachability search eight times, ignoring one
more obstacle each time, and stores the last run's path. Detours on earlier
runs leave that plan unchanged, so the readout "this run made the plan" is
correctly refuted for them, while the last run's chain witnesses the version.

Read the last row carefully. The policy "plans" 127 times, but each search
expands one cell: in its own inflated map the robot's cell has no free
neighbor, so no plan forms and it resets again. A count of planning events
alone would call this the most planning-heavy episode; the expansions and the
followed-plan share show it is not. The carry A* never ran in validation (the
policy never got that far), so its part is reported as not executed, not zero.
Every followed call was also world-insensitive: shifting all obstacles left the
command unchanged, as open-loop plan execution predicts.

### Controls

| Control (disc world) | Planted claim | What the analyzer reports |
| --- | --- | --- |
| `plan_once` | A* in `reset`, then follow | 1 planning event, all at initialization; every call follows the plan |
| `replan_every_call` | A* in every call | One search per call; 0 at initialization; searched share 1.0 |
| `unused_plan` | A* in `reset`, then steer straight | Search verified and plan linked, yet no call follows it: 0 attributed |
| `discarded_search` | Compute A*, discard it, build goal waypoints independently | Search verified, cache affects commands, but no readout reads the cache as a search path: plans "unmatched", followed-plan share 0 with max 0, 0 attributed |
| `reactive` | No search | No search-shaped loops |
| `reach_then_pick` | Explore all, then a tree path | Search recovered with no goal test; a destination-keeping detour witnesses the plan in every validation episode (share 1.0) |
| `nested_renamed` | `plan_once` with renamed nested storage | Identical counts to `plan_once` |
| `endpoint_replaced` | `plan_once` storing its path without the goal cell, then the exact goal (as the KinDER policy does) | Read with the final point in place of the endpoint cell; share 1.0 |
| `tiebreak_astar` | `plan_once` with `(f, tie-break counter, cell)` heap entries | Read as a tagged heap, ties taken from the run; share 1.0. Mean expansions 298.3 vs 297.7 for `plan_once`: its counter breaks ties differently, so it really explores slightly differently |
| `deque_bfs` | Breadth-first search with a `deque` frontier; plan once and follow | Read as first-in-first-out search; share 1.0 |
| `bfs_without_costs` | Breadth-first search with no cost map (outside the grammar) | Unreadable: its frontier gains and loses items, so totals are withheld (—), not zero |

A test-only control rebuilds its heap from a dictionary every iteration. It
matches uniform-cost search on every ordinary iteration, but a dropped entry
comes back, so paired edits reject it and its episode totals are withheld.

### What is counted, and how numbers combine

| Number | Rule |
| --- | --- |
| Expansion events, N, generated alternatives, re-expansions, cost improvements | Summed over an episode's runs; each run counted once, in its own call |
| Max expanded and discovery depth | Maximum over runs |
| Planning events | Runs with at least one expansion |
| At initialization | The part of those totals from the harness `reset` |
| Followed-plan share | A range over action calls. Low end: rotation-supported cache consumers whose plan version has a complete cross-call search witness. Max: adds consumers of versions with unresolved evidence and calls running a loop never classified in development. Disproved and unmatched versions count in neither end |
| Searched share | Action calls that contained a search-shaped run |
| Attributed to actions | Runs with a complete cross-call witness, or whose paired edit changed the same call's command; ≥ marks a lower bound when any run lacks an action witness |
| Coverage | Search-shaped runs supported, rejected or unresolved; non-search loop runs listed |

Means are uniform over validation episodes and withheld when any episode is.
Task success is reported separately and never enters these numbers.

Inspect `search/grid_engine.py` for the abstract family, `search/policy_runner.py`
for instrumentation and replay, `search/policy_verify.py` for search and local
cache probes, `search/policy_causality.py` for cross-call mediation,
`search/policy_discovery.py` for banks, freezing and accounting, and
`search/policy_fixtures.py` for the controls. Tests are in `test_search_policy.py`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/planning_explanations -p 'test_search_policy.py'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/planning_explanations -p 'test_policy_causality.py'
```

Remaining restrictions: one grid family (integer cells, `hypot` step costs);
the transition model is the policy's own predicate and must be a callable
local; plans must be attribute lists of points; a part's run is aligned only
at its loop boundary; plan-following is tested for planar commands only. A
loop without a grammar binding counts as no search only when none of its
containers both gains and loses items and every container was inspected;
otherwise it is unreadable and totals are withheld (Issue 2, below). Loop
snapshots record numbers, strings, tuples, lists, dicts, deques and sets of
those; a worklist kept on `self` or inside another object is invisible.

## Run milestone 7: heuristic search and goal timing

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.search --suite heuristics
open outputs/search-heuristic-benchmark/astar_reopen/report.html
```

For a shorter run, add `--fixture astar_reopen`. The full suite includes eight
programs and thousands of bounded proposals; it can take a few minutes.
Replay without discovering or selecting again:

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.search --suite heuristics --fixture astar_reopen --frozen outputs/search-heuristic-benchmark/astar_reopen/frozen.json --output-dir outputs/search-heuristic-replay
```

The problem addressed here is that a program may choose the lowest-priority
frontier state instead of the first/last state, improve an existing path, or stop
as soon as it generates a goal. A FIFO/LIFO first-parent model cannot explain
those state updates even when it returns the same action.

### One inspectable example

The `dev_reopen` case has two paths from state 0 to goal 5:

```text
action -1: 0 -> 1 -> 3 -> 4 -> 5    (four unit-cost edges)
action +1: 0 -> 2 ------> 4 -> 5    (three unit-cost edges)
```

The supplied heuristic is `h = [0, 0, 2, 0, 0, 0]`. The A* fixture ranks queued
states by `g[state] + h[state]`, where `g` is the program's own cost dictionary.
It chooses the last queued state on equal priorities.

1. Expand 0, discovering 1 and 2 at cost 1. Their priorities are 1 and 3, so
   select 1.
2. Expand 1 and then 3, reaching 4 at cost 3. Queued states 2 and 4 both have
   priority 3. The last-on-ties rule selects 4.
3. Expanding 4 discovers goal 5 at cost 4. In the goal-on-selection variant,
   search continues: state 2 has priority 3, whereas goal 5 has priority 4.
4. Expanding 2 improves state 4 from cost 3 to 2 and changes its parent from
   3 to 2. Because 4 has already been expanded, it is placed back in the queue.
5. Expanding 4 again improves goal 5 from cost 4 to 3. Selecting the goal now
   returns action +1, following the cheaper path through state 2.

This is **six expansion events but only five distinct expanded states**.
N=5, M=4, attributed search expansions=6, re-expansions=1, and cost improvements=2.
These are features of this single development execution; the default report's
means average its separate four-case validation distribution.

The comparison fixtures make each ingredient testable:

| Fixture | Behavior on this example | First action | Expansion events |
| --- | --- | --- | --- |
| `astar_reopen` | Reopens 4 after finding its cheaper path | +1 | 6 |
| `first_parent` | Keeps 4's original parent despite the cheaper candidate | -1 | 5 |
| `goal_generation` | Stops as soon as expanding 4 first generates goal 5 | -1 | 4 |
| `greedy` | Ranks by h alone and selects the goal before exploring 2 | -1 | 4 |
| `nested_astar` | Same A*, with renamed nested search storage | +1 | 6 |
| `nested_costs` | Same A*; the g and h tables are also moved into the renamed container | +1 | 6 |
| `ignored_heuristic` | Computes a priority but selects by FIFO | +1 | 5 |
| `unused_search` | Computes search, then replaces the decision with 0 | 0 | No search attributed to the decision |

### What is discovered and what is supplied

The family enumerates selection (FIFO, LIFO, greedy h, A* g+h), first/last ties,
first-parent versus strict cost improvement, and goal-on-selection versus
goal-on-generation. Generation checking occurs immediately after an accepted
child is recorded and enqueued. It stops the rest of that successor loop;
unexecuted alternatives are not counted. Both variants handle an initial goal.
Cost improvement can update an already queued state or reopen an expanded state.

Mappings now select five roles: root, frontier, parents, costs and heuristic.
All bindings come from existing current values, including nested values. Names
such as `distance` or `h` supply no role annotations; `nested_costs` removes
both locals, and its explanations map every role to a container path, with the
same features as `astar_reopen`. The initial singleton cost
map and six-state numeric table constrain the inventory, then full ordinary and
paired-intervention checks filter the joint hypotheses. Aliased selections such
as local `h` versus `case["heuristic"]` can both survive and remain explicit.

Each invocation supplies a `heuristic` dictionary alongside start, goal and
transitions. A hypothesis uses that input table, a fixed development table, or
zeros. The analyzer does not compute shortest-path distances as a mapping or
attribute its own predictions to the program. This version uses identity states,
input transitions/goals and unit edge costs. It does not infer an arbitrary
heuristic formula, prove heuristic admissibility, or certify optimality.

Development includes two heuristic tables and multiple starting states. Held-out
cases reverse action labels, change the heuristic to zeros, change the goal, and
include a start already at the goal. Full inputs and all choices are frozen
before validation; a heuristic-only change is a meaningful held-out input in
this suite. Changing only a name or start is still insufficient.

### Interventions and report inspection

In addition to clear/reverse/drop frontier probes, the verifier changes an
existing frontier state's heuristic value, and raises the represented cost of
an unexpanded leaf. The latter restriction avoids constructing parent cycles
through arbitrary inconsistent cost edits. Each concrete edit must match its
abstract edit and preserve the execution prefix. The entire continuation is
checked, including cost tables, queue order, parent changes and final action.

The `ignored_heuristic` control illustrates why output matching is insufficient:
with all-zero h, its ordinary FIFO execution can match A*. Raising one queued
state's h predicts a different A* choice, while the concrete FIFO policy ignores
that change. The candidate is rejected by this paired intervention. A supported
FIFO explanation remains. Unused search receives only reactive support within
the current grammar, with zero attributed search expansions.

SVG rows show mapped g/h tables, selection priorities, and improved paths next
to the predicted/observed configurations. The HTML contains every validation
intervention; standalone SVGs are representative views. JSON additionally
records `predicted_control_effect`, `observed_control_effect`, and per-case
`priority_control_effects`. These compare downstream queue/parent/output paths,
excluding the directly edited g/h values themselves. They measure evidence,
not planning amount, and never enter the ordinary input distribution.

This suite's defaults are 4,000 candidates and 128 value paths per checkpoint;
other bounds are shared with milestone 6. Late action-readout checkpoints in
these programs hold up to 69 paths. At milestone 6's 64-path bound the inventory
was cut there, never at the mapped loop, and every bank and selection was
identical; the larger bound makes a reported truncation mean that a bound was
actually reached. `unit-cost-priority-v1` manifests record the extended
family explicitly. Earlier frozen line/task/route files and explanation IDs
remain loadable. The route API also selects this family when supplied
`HeuristicCase` inputs, so frozen prefix gates can compose with priority search.

Inspect `search/heuristics.py` for the executable semantics,
`search/heuristic_discovery.py` for proposals, and `search/heuristic_fixtures.py`
for the original programs. Tests are in `test_search_heuristics.py`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/planning_explanations -p 'test_search_heuristics.py'
```

Remaining restrictions include weighted costs, heaps, helper functions, general
alignment, symbolic route guards and stateful caching (milestone 8 adds heaps,
helpers and caching for its separate grid family). A* goal-on-generation
and first-parent variants are described by their ingredients; their names do
not imply shortest-path guarantees.

## Run milestone 6

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.search --suite routes
open outputs/search-route-benchmark/mixed_nested/report.html
```

For one fixture, add `--fixture mixed_nested`. For an unknown-route example,
use `--fixture unseen_route`. The overview links each route's full report:
all mappings, ordinary predicted/observed configurations, every paired intervention,
and rejected-candidate counterexamples. Each route directory also contains
representative `ordinary.svg` and `intervention.svg`; those are not the whole
intervention set. The overview's `report.json` contains assignments, weights,
coverage and results; each child `report.json` contains its discovery audit.

Replay the **same** frozen gates and banks without discovery:

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.search --suite routes --fixture mixed_nested --frozen outputs/search-route-benchmark/mixed_nested/frozen.json --output-dir outputs/search-route-replay
```

Custom trusted policies use `--suite routes --source path/to/policy.py` and the
same declared mixed-route task distribution. No names in the policy are treated
as search roles. This is still the restricted one-function, one-task-dictionary
interface; it is not a runner for arbitrary Python or Kinder yet.

### Mixed-route walkthrough

1. The fixture returns an immediate goal-direction action when `start >= 4`.
   Otherwise it searches. Its nested version stores search values in
   `work["payload"]["root"]`, `work["payload"]["queue"]`, and
   `work["payload"]["parents"]`.
2. Development cases are line-world `(start, goal)` pairs `(0,4)`, `(2,4)`,
   `(5,4)`, and `(4,5)`. The last two distinguish the reactive rule from a
   constant or a fixed lookup. Instrumentation first checks observation parity.
3. Discovery groups ordinary checkpoint prefixes through the **first loop or
   pre-return checkpoint**. It reads checkpoint locations, not values, output,
   or search-loop iteration counts. It does not infer the symbolic predicate
   `start >= 4`; the fixed gate recognizes which prefix actually executed.
4. Within each group, a bounded inventory supplies scalar/list/dictionary paths.
   Candidate bindings must fit the engine's initial root/frontier/parent shape.
   Ordinary and paired intervention checks filter the joint model/mapping bank.
   Names and dictionary keys carry no predefined role. Selection only reads
   existing values; it cannot run a rollout. Equal-valued root selections may
   both survive and remain explicit ambiguity.
5. The complete banks, prefix gates, preselected explanations, held-out tasks
   and uniform weights are written to `frozen.json` before validation.
6. Held-out cases change action labels, goals or edges. Two invoke search and
   two invoke the reactive branch. A queue-clear intervention mutates the live
   nested queue, predicts the abstract continuation, and compares subsequent
   checkpoints and output. A reactive-action intervention changes the selected
   action value. Both retain the ordinary execution's explanation: no gate is
   refitted or reselected on patched runs.
7. Counts use the four ordinary invocations only. In the mixed fixture, the
   supported preselected explanations attribute `4, 5, 0, 0` search expansions.
   Their mean is **2.25**, while mean N is **2.75** and mean M is **1.75**.
   Separate closed-loop task success is reported without adding those extra
   policy invocations to these means. These features are not a final scalar
   planningness metric.

| Fixture | Expected supported interpretation | Mean attributed search expansions | Supported-route coverage |
| --- | --- | --- | --- |
| `nested` | Search with dictionary paths | 3.5 | 100% |
| `nested_renamed` | Same mechanism with renamed local and keys | 3.5 | 100% |
| `mixed` | Search and immediate-action routes | 2.25 | 100% |
| `mixed_nested` | Same routes, nested search storage | 2.25 | 100% |
| `unseen_route` | Two known routes plus an unseen prefix | Unresolved overall | 80% |

The unseen-route fixture has a third branch for `start == 1` that never executes
during development. Its fifth held-out case is assigned no explanation, and its
counts are `null`, not zero. Known-route conditional means remain inspectable.
An overall preselected mean is withheld whenever any case's preselected
explanation is rejected, unresolved or absent. Surviving alternatives are shown
prominently but never silently replace the preselection.

The default bounds are 2 dictionary/list levels, 64 visited selectors per
snapshot, 64 bindings per family, and 2,000 tested hypotheses **per route**, with
at most 8 routes and 64 prefix checkpoints. CLI flags are `--mapping-depth`
(0–4), `--max-selectors`, `--max-bindings`, `--max-candidates`, `--max-routes`
(1–64) and `--prefix-limit` (1–256). Truncation is recorded with its reason:
`audit.truncation` lists the checkpoints whose value inventory was cut and
whether the binding limit or candidate budget was reached, and reports name
each reached bound. Plain-local frozen
manifests and explanation IDs remain compatible; a selector path is stored in
the existing mapping field as a restricted literal subscript expression.

Inspect `search/selectors.py` for path restrictions, `search/routes.py` for gates
and frozen evaluation, and `search/route_fixtures.py` for the exact programs and
task cases. The earlier engine semantics are unchanged. Route discovery does not
separate branches that diverge **after** the first loop, infer symbolic guards,
or support cached state across invocations. Those remain future work.

Run the relevant tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/planning_explanations -p 'test_search*.py'
```

## Objective and scope

Discover and experimentally validate explanations of original Python executions
as one fixed graph-search algorithm. Keep the proposer separate from the verifier.
Implement the complete ordinary/intervention/environment/frozen-validation loop
in a small domain before extending it to Kinder. No LLM calls.

The first domain is positions 0–5, primitive actions -1/0/+1, valid moves within
the line, and task goal position >= 4. Candidate environment abstractions group
positions in blocks of 1, 2, or 3. Search ingredients enumerate neighboring-region
successors (both orders), FIFO/LIFO, a goal threshold, and fallback actions.
Reactive hypotheses use the same engine with a terminal action child and the
declared rule "right before the abstract task goal, stay at the goal".

The engine discovers nodes once, keeps first parents, tests goals on selection,
and returns the first parent-path action. Goal selection is not expansion.
This original family has no relaxation or reopening. The opt-in milestone-7
family adds those operations explicitly; neither family hides planning inside
the mapping or ingredient source bindings.

## Milestones and components

1. `engine.py`: independently executable abstract configuration and transitions.
2. `checkpoints.py`, `explanation.py`, `environment.py`, `verify.py`: generic
   observation, restricted decoding, live edits, independent correspondence and
   environment witness checks. Test supplied reference explanations first.
3. `discover.py`: bounded enumeration of the same explanation objects, followed
   by freezing and fresh validation. `report.py` and the CLI provide inspectable
   reports, original traces and paired diagrams.
4. `evidence.py`: separate inferred findings, known fixture labels, computation
   features and evidence strength. One-expansion search is supported even at M=0.
5. `tasks.py`, `task_fixtures.py`, `benchmark.py`: parameterized tasks, frozen
   task-level splits, independent model checks, and separate goal-reaching tests.

## Run milestones 4–5

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.search --suite tasks
open outputs/search-task-benchmark/summary.md
open outputs/search-task-benchmark/one_expansion/report.html
open outputs/search-task-benchmark/wrong_model/report.html
```

The default `--suite line` command remains available and keeps its original
output directory and frozen-bank format. Both suites now show the milestone-4
findings and additional features. The task suite writes to a separate directory
so it does not overwrite the original reactive/lookup/decoy artifacts.

The task policy takes one dictionary argument:

```python
{"start": 0, "goal": 4,
 "transitions": {0: {-1: None, 1: 1}, ...}}
```

There are six identity-mapped states and commands -1, 0, +1. A command's meaning
comes from its table entry, not numeric subtraction of state IDs. `None` means
no edge; 0 stays in place. A task-aware explanation selects input or fixed
development transitions/goals. Those selectors are fixed before validation;
reading a new input table is not fitting a new explanation or running a solver.
Reactive candidates are a bounded menu: goal-direction, a fixed six-entry lookup,
or constant actions. These supplied choices are not an unrestricted discovery
language. The correct model is not identified from internal variable names.

Development starts at 0, 2, 5 in the ordinary line, with exact goal 4. Adaptive
search, the reactive rule, lookup, incorrect-model search and decoy return the
same development actions. The held-out tasks are:

| Case | Change | Start | Goal | Adaptive first action |
| --- | --- | --- | --- | --- |
| `reversed_commands` | Swap the two command columns at every state | 0 | 4 | -1 |
| `changed_goal` | Keep the line but change the goal | 3 | 0 | -1 |
| `shortcut_edge` | Add a -1 edge from state 0 directly to 4 | 0 | 4 | -1 |

The `one_expansion` fixture uses the same search implementation with its own
declared distribution: development endpoint tasks and three held-out tasks whose
goal is the first queued successor. It is a genuine search execution requiring
one expansion, not a new algorithm that artificially forces all tasks into one
step. Its surviving graph explanation has N=1, M=0 on all held-out cases.

Expected findings across the frozen bank:

| Fixture | Known fixture construction | Analyzer finding | Separate task success |
| --- | --- | --- | --- |
| `adaptive` | Plans with input model/goal | Search supported | 3/3 |
| `reactive` | Direct rule; not planning | Only reactive explanations supported | 2/3 |
| `lookup` | Fixed action table; not planning | Only reactive explanations supported | 1/3 |
| `wrong_model` | Plans using development transitions | Search mechanism supported, task grounding rejected | 2/3 |
| `decoy` | Search ignored by returned action | Only reactive explanations supported | 1/3 |
| `one_expansion` | Search on one-expansion inputs | Search supported despite M=0 | 3/3 |

These are fixture labels and empirical findings, not a general binary planning
classifier. Fixture labels are attached only by the reporting layer and are
never arguments to discovery or verification. Custom `--source` files have no
assumed ground truth, even if their filenames match a fixture.

### How to read the new evidence

`mechanism_status` records ordinary and paired intervention correspondence.
`environment.status` independently records model-to-task correspondence. In the
identity-state task suite this checks all 12 labeled transition slots and the
exact goal, including unused edges. It is stronger than the legacy existential
witness test; the two semantics are explicitly distinguished in JSON. Reactive
action leaves assert no future-state model, so their task grounding is marked
`not_applicable`. `status` records joint verification, not a planning label.

The report's `finding` considers all frozen hypotheses. One may survive after a
development-tied selected hypothesis fails. **The selected hypothesis is never
replaced:** its failed status and null joint averages remain visible. Feature
ranges across supported frozen hypotheses describe ambiguity without selecting
a different explanation on each input. Identifier tie-breaking can change which
development-tied candidate is preselected when the discovery language expands.
With the milestone-6 defaults, the `one_expansion` preselection survives and has
N=1, M=0. A failed preselection is still retained, as the `reactive` and `decoy`
task reports demonstrate. The wrong-model fixture likewise retains a fixed-model
causal explanation even though its input-model explanation fails. Previously
saved manifests retain their original selections when replayed.

Full task-grounded `ordinary_mean_N/M` stay null for an ungrounded explanation.
`mechanism_mean_features` can still describe causally matched computation using
an incorrect model. That quantity is explicitly diagnostic, not grounded
planningness. Per-case counts shown before a candidate's causal check passes
are provisional. Rejected or incomplete candidates are not zeros.

Additional features are:

| Field | Definition |
| --- | --- |
| `max_discovery_depth` | Greatest retained first-parent depth of an environment node, including discovered goals |
| `max_expanded_depth` | Greatest retained depth of an expanded environment node |
| `generated_alternatives` | Environment-successor occurrences generated, including repeated states |
| `distinct_generated_states` | Unique environment-state IDs among those occurrences |
| `max_frontier_size` | Largest queue at an aligned boundary, including the initial root |
| `search_expansions` | Expansions generating at least one environment successor; Action leaves excluded |
| `expansion_events` | All expansion occurrences, including revisits and empty-successor expansions |
| `reexpansions` | Expansion events beyond the first expansion of each state |
| `cost_improvements` | Strict improvements to costs of already discovered states in the extended family |
| `decision_effects` | Passed probes whose predicted output differs from the baseline; evidence, not search amount |

`task_performance` runs the uninstrumented policy repeatedly with fresh globals
and input dictionaries until reaching the goal, encountering an invalid action,
or repeating a state. Repetition establishes failure for this deterministic,
memoryless interface. Execution-budget failures remain unresolved. These success
rollouts are separate from the single-invocation distribution used for features;
their extra invocations and all interventions have zero weight in those means.

Task manifests freeze complete development/validation cases and uniform case
weights, not just names. Input overlap is checked using full case contents, so
renaming a development case cannot make it fresh. The task suite also requires
held-out (transition table, goal) pairs: changing only the start is rejected.
Reload without discovery:

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.search --suite tasks --fixture adaptive --frozen outputs/search-task-benchmark/adaptive/frozen.json --output-dir outputs/search-task-replay
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/planning_explanations -p 'test_search*.py'
```

Unsupported source/execution failures produce explicit unresolved JSON/HTML
reports and a nonzero CLI exit code. A bounded discovery with no accepted
candidate produces an unresolved finding, not a numerical planning score.

## Concrete representation and alignment

The search runner accepts one deterministic, one-argument Python function with
bounded scalar/list/dictionary computation. It inventories all statement and
while-test checkpoints without interpreting names. Graph hypotheses choose a
while-test site, a root scalar, a frontier list, a parent dictionary, and a fixed
numeric label offset. Every visit to that loop boundary is compared, followed
by function exit. One iteration corresponds to exactly one abstract iteration;
no extra abstract expansions may be inserted between checkpoints.

Reactive hypotheses use entry, one chosen statement checkpoint containing a
computed action, and exit. Their fixed root/action encoding does not reconstruct
future trajectories. An unused search may admit this minimal explanation while
its richer graph-search explanation is rejected.

Mappings read the current configuration only. They do not consult future trace
snapshots, call the policy, or perform search. The engine predicts the next
configuration independently. Graph and frontier identities/order, root, phase,
output, and termination must correspond. Counts attach to concrete intervals.

Interventions replay identical inputs to a declared checkpoint, verify the
unchanged prefix, edit the live representation, and resume original code. Clear,
drop and reverse frontier edits mutate the actual list (preserving aliases).
Reactive action edits explicitly replace a local before its subsequent use.
Output values at function exit cannot be patched. Check the edit correspondence
itself, then compare all subsequent configurations and outputs.

## Environment meaning and evidence

Check roots against the input abstraction and declared task-goal semantics on
all six physical positions. Generate bounded trajectory witnesses for each
abstract transition-action identity, and replay them independently. This uses
weak existential semantics: an action's witness need not validate every edge
application. Reactive action leaves are decisions, not environment transitions.
Witness-generation work and abstract reference executions never count as policy
computation.

Discovery uses positions 0, 2, 4; fresh validation uses 1, 3, 5. The evaluated
distribution is explicitly uniform over those fresh inputs, not implicitly all
possible states. Reference-engine tests may separately exhaust the finite domain.
Freeze source, domain, grammar version, explanation bank and discovery inputs before
validation. Never refit or select a new mapping from validation outcomes. Retain
ambiguity; an accepted reactive baseline is not proof no richer explanation exists.

Report ordinary distinct expanded environment nodes N and max(N-1, 0) as the
PDF's named counting convention, not a general planningness measure. Intervened
runs receive no population weight. Report each frozen explanation separately;
do not maximize over explanations independently on each validation input.
Rejected explanations have counterexamples. Unsupported syntax, missing evidence,
witness/search budgets, and unseen alignment patterns remain explicit unknowns.

## Run and inspect

From the repository root, using Python 3.11+ (the package needs no optional
robotics, plotting, or LLM dependencies):

```bash
PYTHONPATH=src python -m robocode.planning_explanations.search
open outputs/search-explanations/summary.md
open outputs/search-explanations/fifo/report.html
```

Use `.venv/bin/python` instead of `python` to use the repository environment.
The default command finishes all three milestones for all six fixtures. Each
fixture directory contains:

| File | Contents |
| --- | --- |
| `policy.py` | Exact original source used by this experiment |
| `frozen.json` | Source hash, discovery inputs, grammar/domain, all supported explanations, fixed selection, integrity hash; written **before** fresh evaluation |
| `report.html` | Every fresh ordinary execution and every fresh intervention, with predicted and observed configurations side by side |
| `report.md` | Compact results and interpretation |
| `report.json` | Full discovery and validation evidence, environment witnesses, mappings, rejected candidates/counterexamples, budgets and ambiguity |
| `ordinary-traces.json` | Actual ordinary snapshots, all inventoried source checkpoints and alias groups |
| `ordinary.svg`, `intervention.svg` | Representative views of the selected explanation; all interventions remain in HTML/JSON |
| `rejected-intervention.svg` | When present, a rejected graph candidate's paired counterexample |

Expand an explanation's **Ingredients and mapping** entry first. For FIFO, the
discovered explanation chooses the outer `while` checkpoint, `root`, `todo`, and
`parent`. These names are outputs of enumeration over types and source locations,
not role annotations given to the discoverer. Next expand an ordinary execution.
Each row compares an independently predicted abstract configuration to a decoding
of the corresponding concrete snapshot. The JSON records the concrete interval
responsible for each abstract step, selected/expanded nodes, generated alternatives,
and newly added nodes. Edges in the SVG are retained first-parent edges, not every
possible environment transition; blue nodes are the ordered frontier.

For position 1, the selected FIFO explanation has abstract root 0: expand 0,
expand 1, select goal 2, return +1. It counts N=2, M=1. Clearing `todo` after the
first expansion predicts fallback 0. The real policy also returns 0, and the
post-edit state and all later checkpoints agree. For the decoy, the same edit
predicts 0 but the policy still returns +1: this is a rejected graph explanation.
The decoy's independent reactive decision can still support a minimal explanation.

The raw snapshot export uses `{"$dict": [[key, value], ...]}` to preserve key
types and `{"$tuple": [...]}` to distinguish tuples from lists. Alias groups
record shared mutable references. The abstract report uses explicit labeled
parent-edge lists; `Action(1)` is distinct from environment node `1`.

Run one fixture, bound discovery, analyze a trusted policy, or reload a bank:

```bash
PYTHONPATH=src python -m robocode.planning_explanations.search --fixture decoy
PYTHONPATH=src python -m robocode.planning_explanations.search --fixture fifo --max-candidates 20 --output-dir outputs/search-bounded
PYTHONPATH=src python -m robocode.planning_explanations.search --source /path/to/policy.py --output-dir outputs/search-custom
PYTHONPATH=src python -m robocode.planning_explanations.search --fixture fifo --frozen outputs/search-explanations/fifo/frozen.json --output-dir outputs/search-replay
PYTHONPATH=src python -m unittest discover -s tests/planning_explanations -p test_search.py
```

Loading a bank skips proposal generation entirely. Replayed reports contain new
validation evidence; the original discovery audit remains in the original
`report.json`, not in the frozen bank. No candidate is repaired or reselected
after seeing validation. A failed selected explanation gets null averages.

## Milestone interfaces

1. **Engine input:** declared search ingredients and an abstract root.
   **Output:** initial configuration and every selected/expanded node, generated
   successor, first-parent/frontier update, termination event and primitive output.
2. **Verifier input:** original source, a supplied explanation (ingredients plus
   current-state mapping/alignment), and test inputs. **Output:** environment,
   ordinary and intervention checks, supporting intervals or a counterexample.
   It does not discover or repair explanations.
3. **Discovery input:** original source, the finite grammar, discovery inputs and
   budgets. **Output:** all supported explanations and a rejection audit. Select
   the greatest development mean M (identifier breaks ties), persist that bank,
   then validate it on disjoint inputs. This selection rule is a demonstration
   convention, not a claim that the true maximum explanation was found.

For example, milestone 1 can be inspected without instrumenting any policy:

```python
from robocode.planning_explanations.search.engine import SearchSpec, run

execution = run(SearchSpec(block=2, goal=2, selection="fifo"), root=0)
for transition in execution.transitions:
    print(transition.operation, transition.selected, transition.state)
```

The corresponding supplied milestone-2 explanation is:

```python
from robocode.planning_explanations.search.checkpoints import Program
from robocode.planning_explanations.search.explanation import Explanation, Mapping
from robocode.planning_explanations.search.fixtures import FIFO
from robocode.planning_explanations.search.verify import verify

candidate = Explanation(SearchSpec(), Mapping("5:4:While", "root", "todo", "parent"))
evidence = verify(Program(FIFO), candidate, (0, 2, 4))
```

Discovery does not receive that reference mapping. Its default budget is 2,000
complete candidates and 64 structural bindings per family. Bindings select
locals or bounded literal dictionary/list paths; graph label offsets are 0 or 10.
Witness trajectories have at
most three primitive moves. Concrete execution has a 20,000-line budget and
2,000-snapshot budget; abstract execution has a 20-step budget.

## Fixtures and expected results

All six have the same ordinary decisions on this domain. Their supported
explanations differ. For the default **uniform fresh-input distribution 1,3,5**:

| Fixture | Meaning | Selected explanation | Mean N | Mean M |
| --- | --- | --- | --- | --- |
| `fifo` | Explicit first-in-first-out search | Graph | 4/3 | 2/3 |
| `renamed` | Same search, renamed locals and labels shifted by 10 | Graph | 4/3 | 2/3 |
| `lifo` | Last-in-first-out search | Graph | 1 | 1/3 |
| `reactive` | Conditional immediate action | Reactive | 1 | 0 |
| `lookup` | Precomputed action table | Reactive | 1 | 0 |
| `decoy` | Search computed but independent reactive action returned | Reactive; graph rejected | 1 | 0 |

FIFO/LIFO/renamed each retain three explanations: a graph explanation and two
minimal reactive explanations with different state abstractions. The other
fixtures retain two reactive explanations. The default selected FIFO graph has
10 development probes and 10 fresh probes; LIFO has 9 and 9; each reactive
explanation has 6 and 6. Counts are **per explanation**, not unique physical
replays across the bank. Report every tested probe, including those without a
decision change. Require at least one correctly predicted decision-changing
probe on each ordinary run with N>0; root-goal graph runs with N=0 are exempt.

## What this first version does not establish

This implements a small instance of the proposal's search + causal abstraction
framework. Finite agreement is empirical support, not program-equivalence proof.
The meaning of the task, search engine, ingredient families and mapping language
are supplied; only their bounded combinations and bindings are discovered.

The checkpoint runner accepts one function and one integer position argument
(or a task dictionary for the new `tasks` suite),
with no helpers, imports, comprehensions, NumPy, external state or loop `else`.
It operates on trusted source and is not a security sandbox. It is intentionally
separate from the older analyzer's broader expression instrumentation. The
milestone-8 policy runner is a separate, equally trusted runner for whole
class-based modules. One graph
mapping must work across its declared input partition. The `routes` suite splits
that partition using frozen prefix gates; nested dictionary/list selections are
supported. Adaptive/stuttering alignment and routes that diverge only after a
shared first loop are not implemented. The older lookahead analyzer and its route
handling remain available independently.

The original engine supports first-parent search and goal-on-selection only.
The opt-in priority family adds bounded heuristic selection, cost improvement,
reopening and goal-on-generation; it does not learn arbitrary search algorithms.
Legacy witness semantics are deliberately
weak: one physical witness per abstract action identity does not establish that
every generated abstract edge is realizable from every physical state in its
region. Primitive output validity is not inferred from action-node interventions;
those test decision mechanisms, and can deliberately force a bad action.

The legacy suite's fresh inputs are disjoint positions in the same tiny domain;
they do not test new maps, goals or utilities. The task suite adds held-out maps
and goals, but still uses only six states, identity state abstraction, and a small
declared ingredient grammar. Utilities and broader worlds remain untested.
If the grammar, mapping inventory or execution budget misses an explanation, the
result is unresolved, not evidence of zero planning. An LLM proposer can be added
later behind the same manifest interface without becoming the verifier.

## Later extensions

### Milestone 8: implementation decisions (2026-11-09)

Explain one agent-written KinDER policy: the median-length heap-using program
in the cached runs (ClutteredRetrieval2D, Codex GPT-5.6 Sol, 529 lines). It is a
class with helpers, closures, NumPy, state across calls, cached plans and
replanning. Its source is copied byte-identically into the package with its
provenance. Small class-based control policies in a pure-Python disc world test
each claim quickly; the KinDER policy runs in the real simulator.

A **part** is one `while` loop in the source. A **run** is one execution of
that loop. Each part gets its own frozen bank, as each route did in milestone 6.
A run's inputs (root, goal and the policy's own free-cell predicate) are read
once at its first loop checkpoint; every later configuration is predicted.

Add a versioned `grid-priority-v1` family; the earlier engines are unchanged.
States are integer grid cells, costs are floats, and the frontier is a list of
`(priority, state)` entries decoded as a multiset (a heap's array order is not
compared). Ingredients: min/FIFO/LIFO selection; priority g, g+h or h; zero,
Euclidean or Manhattan h to a mapped goal cell; root entry priority; goal on
selection, on generation, or **none** (explore everything reachable); first-parent
or improving updates; stale entries re-expanded or skipped; 4- or 8-neighbor
moves with `hypot` step costs. Ties go to the smaller entry, as Python's heap
orders tuples. Successor order within one expansion is unobservable at loop
granularity for heaps and is fixed.

**World.** The transition model is the policy's own predicate, selected by shape
from current locals and called by the analyzer on single cells. That is the
policy's map, accepted as given; it is not checked against the simulator. An
unconstrained grid is the wrong-model alternative. Predicate calls are analyzer
reference work and never count as policy computation. A second callable maps a
cell to a point and is used only to link a cached plan to a search tree.

**Counting.** Expansions are counted once, in the run where they happen. Runs
in the harness `reset` (initialization) are kept and counted once; they are not
repeated on the calls that later consume their plan. Per-part tables keep the
earlier features. The program-level row combines them by what they measure:
effort counts are summed over an episode's runs, depth is the maximum, planning
events are counted, and the planned share of actions is a fraction. Coverage is
reported separately. An episode total is withheld when any search-shaped run is
unresolved. Averages are taken only across validation episodes.

**Plans and actions.** A cached plan is a policy attribute holding a list of
points. Parent-chain membership is only a structural candidate link. Rotating
the cache 90 degrees about the robot tests local cache use (planar-command
cosine at least 0.9); it does not establish that search produced the cache.
A second probe shifts obstacles and reports whether the command is unchanged.
Cross-call attribution additionally requires the complete intervention chain
described below. Structural producers and causal status remain separate in the
machine-readable report.

**Verification and sampling.** Every expansion of every run is compared.
Paired search edits (drop a frontier entry, raise an unexpanded leaf's cost,
clear the frontier) re-execute only the enclosing call from deep copies, check
the unchanged prefix, and compare the run's whole continuation. Five edit
points per run are drawn with a fixed seed before any result is seen. In
schema 4, each stored plan a probed run may have produced also gets one detour
edit (Issue 1 follow-up, below). At most eight runs per part per episode are
probed: first the runs that stored a new plan (known from an uninstrumented
pass), then seeded draws.
Heap edits restore heap order with `heapq.heapify`, the representation's own
invariant.

### Issue 1: the complete cross-call causal chain (2026-11-10)

The previous implementation combined a matching parent-chain shape with an
independent cache-rotation probe. That permitted a false positive: compute A*,
discard its result, store two goal waypoints derived directly from the input,
then steer using those waypoints. The waypoint cache affects actions, but search
does not produce it. On `(0, 1)` the `discarded_search` control still has 459
verified ordinary expansions; none receives supported action attribution.

`policy_causality.py` now implements a bounded protocol:

1. During development, enumerate readouts of a **complete root-to-endpoint
   parent chain**, with no final point, a final point after the endpoint cell,
   or a final point in place of it. Attribute and point-callable candidates
   come from existing values, without semantic names. Freeze these templates
   with the bank (schema 4).
2. On an ordinary execution, bind the endpoint, whether the root cell is absent
   (zero or one leading cell) and the optional final point from its stored plan.
   Reconstruct the plan using the independently predicted parent map. This is a
   conditional readout, not an explanation of destination selection; it adds no
   analyzer-generated search to program effort.
3. For each sampled search intervention, predict the entire stored waypoint
   sequence from the edited abstract search and compare it to the actual
   post-call attribute. If the edited search no longer reaches the bound
   endpoint the same way, the probe is unresolved (`endpoint_lost`): what a
   policy stores then is fallback code the readout does not model. If the
   stored plan is a complete chain of the edited tree to another destination,
   it is unresolved too (`destination_changed`). Otherwise a mismatch is a
   counterexample, and it refutes that readout for this version even if another
   intervention gave a positive witness.
4. When the plan changes as predicted, retain four copied policy states:
   ordinary; search-edited; ordinary with only the predicted plan installed;
   search-edited with the ordinary plan restored. Carry their state across up
   to 32 future action calls, supplying the same ordinary observations to each.
   A witness requires a changed action, equality between the search-edited
   action and the plan-only prediction, and restoration of the ordinary action
   when the ordinary plan is restored. Baseline replay must match too.
5. Stop without credit if a new search or non-suffix cache replacement occurs
   before the witness. Delayed consumption is allowed. Horizon exhaustion,
   episode end and replay errors remain unresolved, not proof of non-planning.
6. Count the producer's ordinary expansions once if the full chain is supported.
   Rotation still identifies ordinary calls that use that cache version, but
   cannot by itself establish its search provenance. Same-call search-edit
   effects remain a separate attribution route. Neither reference computations
   nor patched runs enter the averaging distribution.

This is experimental support within the declared grammar and interventions,
not equivalence for every input. The future replays are controlled experiments
on ordinary observations, **not** counterfactual closed-loop task-success runs.
The controller's action prediction comes from a plan-only program replay; the
controller itself has not been synthesized as a separate abstract model.
Issue 2, inventory-budget exhaustion, is outside this correction.

### Issue 1 follow-up: out-of-domain probes, detours, four categories (2026-11-10)

The first correction refuted genuine planners. Every stored-plan mismatch in
its outputs came from a frontier-clear edit (mostly a predeclared clear at
visit zero). Clearing removes the bound endpoint from the edited tree, so the
readout predicted only the final point, while the policy stored whatever its
failure code produces: nothing (`replan_every_call`), the start plus the target
(`reach_then_pick`), or a new search toward another target (KinDER). One such
mismatch refuted the readout. `reach_then_pick` and KinDER fell to share 0 and
`replan_every_call` gained rejected chains, although all three plan. 120 KinDER
calls also followed a genuine path whose root waypoint had been consumed in the
same call, which the frozen "starts at root" template could not read.

Changes:

1. **Out-of-domain probes are unresolved.** `judge_probe` returns
   `endpoint_lost` when the edited search no longer reaches the bound endpoint
   the same way (not in the tree, or a different goal outcome), and
   `destination_changed` when the stored plan is a complete chain of the edited
   tree to another destination. Only a wrong plan for a kept endpoint is a
   counterexample. The root offset is bound per plan, like the endpoint.
2. **A destination-keeping detour replaces the clear at visit zero.** For each
   stored plan a probed run may have produced, take the first of the path's
   first three interior cells whose removal keeps the endpoint according to the
   explanation, and drop every frontier entry of it (`drop_cell`, applied
   identically to the live list and the abstract engine) just before its first
   expansion. Starting near the root makes a consumer act on the change within
   the 32-call replay horizon; a mid-path detour often did not.
3. **Four followed-plan categories and a share range.** A plan-following call
   takes the strongest evidence among the versions it follows: witnessed,
   unresolved, disproved, unmatched (no frozen readout reads the plan as a
   complete search path). `followed_plan_share` is the witnessed fraction;
   `followed_plan_share_max` adds unresolved calls and calls running an
   unclassified loop. Disproved and unmatched calls are in neither end.
   "Unmatched" is relative to the frozen readout grammar.

| Policy | Ground truth | First correction | Follow-up |
| --- | --- | --- | --- |
| `plan_once` | plans | share 1.0 | share 1.0 |
| `discarded_search` | fake plan | share 0.0 | 0.0, max 0.0 (unmatched) |
| `reach_then_pick` | plans | 0.0, readout refuted in every episode | 1.0 |
| `replan_every_call` | plans every call | 3–4 refuted chains per episode | no refuted chains |
| `unused_plan` | plan unused | 0.0 | 0.0, max 0.0 |
| KinDER (mean) | unknown | 0.0 | 0.02–0.25 (0.14–0.25 after the second follow-up) |

The same-call attribution route is unchanged, but it depends on which edits
run. Without the clear at visit zero, the lower bounds on attributed expansions
fell for `replan_every_call` (≥ 714 to ≥ 579) and KinDER (≥ 1,244 to ≥ 644).
### Issue 1, second follow-up: final point in place of the endpoint; producers first

Two further gaps left KinDER plans unresolved. First, the policy stores
`cells[1:]` of the walk back from its stance cell (line 222): the stance cell is
dropped and its exact standing point appended instead. The readout took the
last stored cell as the endpoint, so after a detour that changed only the
stance cell's parent it reported `destination_changed`, and a two-cell stored
path had no interior cell to detour around. Second, probed runs were drawn
uniformly, which in (3, 100) skipped the search whose plan 120 calls followed,
and in (15, 101) skipped every plan-producing search.

1. **Final point in place of the endpoint.** A third template kind binds the
   endpoint as the tree cell whose point is uniquely nearest the final point,
   and requires its parent to be the last stored cell. The prediction is the
   edited chain to that cell, minus the cell, plus the same final point. The
   `endpoint_replaced` control stores its path this way and is witnessed in
   every validation episode.
2. **Plan-producing runs first.** An uninstrumented pass already counts runs;
   it now also marks, in each call that stored a new non-empty plan (not a
   consumed suffix), the last run of each part. Those runs are probed first and
   seeded draws fill the remaining places (at most eight per part per episode).
   The rule reads only the policy's ordinary behavior, never a probe result.

KinDER, before and after: (3, 100) from 0.06–0.70 to 0.38–0.70; (15, 101) from
0.00–0.04 to 0.04 (every followed plan witnessed); mean from 0.02–0.25 to
0.14–0.25. Attributed expansions rose to ≥ 5,662 and ≥ 4,734 in the first two
episodes. All controls are unchanged. Bank schema 4.

### Issue 2: loops the analyzer cannot read (2026-11-10)

A loop with no grammar binding was marked "not search-shaped", and accounting
left its runs out, so the report read "complete, zero planning". Three routes
led there, each reproduced on `plan_once` (a real A* reported as complete with
0 planning events):

1. The variable inventory hit its count limit. The cut was recorded in the
   audit but never consulted.
2. The to-do list sat deeper than the mapping depth (`box["q"]` at depth 0).
   Depth cuts were not even recorded.
3. The to-do list had a shape the grammar did not know, with normal limits:
   `(f, tie-break, cell)` heap entries, a `deque`, or breadth-first search
   without a cost map. Loop snapshots did not record deques or sets at all.

Changes:

1. **Worklist test for loops without a binding** (`policy_worklist.py`). A
   search keeps a worklist: some list, dict, deque or set gains items and loses
   items while the loop runs. A loop without a binding is "not search-shaped"
   (zero) only if no container does both in any development run and every
   container was inspected (depth 4, 4,096 containers per snapshot). Otherwise
   it is `unreadable_loop`: unknown, totals withheld. Validation runs of
   not-search-shaped parts are re-checked the same way, so the development
   verdict is never assumed. A plan stored in a call that ran an unexplained
   loop is unresolved, not unmatched: the followed-plan share's max includes it.
2. **Wider readable forms.** A `tagged` heap layout reads entries
   `(priority, numbers…, cell)`. The extra numbers are the policy's own
   bookkeeping: the explanation is the search up to tie order, and when equal
   priorities tie, the cell the run took out is read from the next snapshot;
   everything else is still predicted and compared. Deque frontiers are read
   wherever list frontiers are, and loop snapshots now record deques and sets.
   Live edits address heap entries by (priority, cell).
3. **Larger default inventory:** 1,024 selectors (was 256) and mapping depth 3
   (was 2).

Results: the three reproductions are now unreadable, with totals withheld and
a followed-plan share of 0–1.0 (max). `tiebreak_astar` is explained (459
expansions on (0, 1), as `plan_once`; 298.3 vs 297.7 on average over validation,
because its counter orders ties differently); `deque_bfs` is explained as
first-in-first-out search; `bfs_without_costs` is unreadable. Ordinary loops (path reconstruction
only appends; waypoint popping only pops) and the reactive control still count
as zero. KinDER is unchanged: its four ordinary loops stay not search-shaped in
development and in every validation run, and its numbers are identical. The
KinDER analysis takes about 115 s instead of 75 s. Bank schema 5.

Remaining gaps: worklists kept on `self` or in non-plain objects are invisible
to the snapshot; a loop with a temporary list rebuilt every iteration can be
reported unreadable (unknown) although it is not a search.

Reports separate structural matches, exact plan predictions, action mediation,
counterexamples and missing witnesses. Inspect
`validation.episodes[].versions[].chains[]` in `report.json`; Markdown and HTML
show action comparisons alongside the edit that caused them. The new regression
suite covers discarded search, a genuine plan, delayed consumption, unused
changed plans, an unrelated action-changing heading, cache replacement,
fallback storage after a lost endpoint, a duplicate search that did not
produce the plan, each probe verdict on a hand-built tree, and frozen/legacy
banks.

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.search --suite policy --output-dir outputs/search-policy-issue1
open outputs/search-policy-issue1/plan_once/report.html
open outputs/search-policy-issue1/discarded_search/report.html
```

**Inputs.** Development episodes are (obstacles, seed) = (1, 0), (3, 1), (5, 2).
Validation episodes are (3, 100), (15, 101) and (20, 102): all new seeds, and
15 and 20 are counts the policy's author never designed for (they used 1, 3,
5 and 10). Banks, preselections, readout choices, probe rules
and validation episodes are frozen before validation. Loops with no grammar
binding and no worklist are listed in coverage and not counted; loops with no
binding but a worklist (or not fully inspected) are unreadable, so totals are
withheld rather than reported as zero.

### Milestone 7: implementation decisions

Add an opt-in `heuristics` suite with six-state, unit-cost tasks. Keep legacy
first-parent FIFO/LIFO semantics, manifests and explanation identities intact.
The extended family enumerates FIFO, LIFO, greedy h, and A* g+h selection;
stable first/last priority ties; first-parent versus strict cost improvement
(including reopening); and goal checks on selection versus generation. Goal
generation stops that expansion immediately, so only the generated prefix is
counted. Selected-root goal checks remain supported in both variants.

Map current concrete cost and heuristic dictionaries as well as root, frontier
and parents. The mapping performs literal selection/relabeling only. Initial
heuristics come from the task, a fixed development table, or a zero table; no
solver synthesizes heuristic values. A bound on this ingredient grammar and all
mapping candidates is explicit. Freeze candidates before held-out map/goal and
heuristic changes. Do not assume an arbitrary supplied heuristic is admissible
or that goal-on-generation A* returns an optimal path.

Intervene on frontier, represented costs and represented heuristic values;
compare full predicted/observed continuations, including reparenting. Retain
indistinguishable candidates and counterexamples. Report expansion occurrences,
distinct expanded states N, M, re-expansions and cost improvements separately.
Patched runs and analyzer reference rollouts do not enter ordinary feature means.
Nested mapping support is shared with milestone 6. Helpers, weighted costs,
heap layouts, general alignment and stateful caching remain outside this slice.

### Milestone 6: implementation decisions (2026-11-08)

Extend selection-only mappings to literal dictionary keys and list indices, with
explicit depth, inventory and hypothesis budgets. Decode only current values;
do not compute a search in the mapping. Interventions mutate the selected live
container so aliases retain Python semantics. Plain-local manifests remain valid.

Add a separate `routes` experiment. A route is the concrete checkpoint prefix
through the first loop or pre-return checkpoint, before executing that loop or
return. Inventory prefixes on development executions and discover a bank for
each observed prefix. Freeze the exact prefixes, banks, development selections,
held-out cases and weights together. Validation dispatches using this rule only;
it does not fit a new gate, replace a selection or maximize counts per input.
Interventions keep the ordinary run's explanation fixed. Unseen prefixes and
exhausted budgets are unresolved, with missing counts rather than zeroes.

Report route coverage and conditional feature means. Only when every case has
support for its preselected route explanation report its distribution-wide mean.
Keep ranges across surviving hypotheses, attributed search expansions, N/M,
causal evidence and separate closed-loop task success visible. Rename the
selected status to “Validation of the preselected explanation.” This milestone
does not add helpers, caching across invocations, general alignment or heuristic
search; the first-parent FIFO/LIFO engine remains unchanged.

### Milestones 4–5: implementation decisions (2026-11-08)

Keep the original line-world experiment and frozen-bank format usable. Add
mechanism findings independently of N/M: search supported, only reactive
baselines supported, or unresolved. Keep known fixture labels separate from
inferred findings. Report search depth, generated alternatives, frontier size,
and correctly predicted decision effects; M=0 never decides the finding.

Add a separate task-parameterized suite of deterministic six-state directed
graphs. Each invocation receives a dictionary containing start, a two-action
transition table, and an exact goal. State abstraction is identity in this suite;
the same first-parent FIFO/LIFO engine is used. Ingredients choose input versus
fixed development transitions/goals, plus a bounded menu of reactive rules.
At this milestone mappings selected top-level locals; milestone 6 subsequently
adds broader mappings and routes, as described above.

Freeze the source, candidate bank, complete development cases, held-out cases,
and validation weights before running validation. Hold out different transition
tables and goals, not merely different concrete positions. Match full case
content rather than names when enforcing disjointness. Retain computationally
supported but incorrectly grounded candidates as diagnostic evidence, separately
from full task-grounded support. Check every model edge against the supplied
table in this identity-state suite; this is stronger than the legacy suite's
weak existential action witnesses and is explicitly labeled as such.

Task success is a separate resettable closed-loop rollout using the original
policy. Its additional invocations, external environment checks and intervention
runs do not enter features averaged over the declared single-invocation input
distribution. A fixed-model search may have causal mechanism support, fail task
grounding, and fail the task. A successful lookup is not thereby planning.

Required controls: adaptive search, immediate reactive rule, fixed lookup,
search using the development model, unused-search decoy, and a search fixture
tested on one-expansion cases. Preserve all explanations and counterexamples;
never repair/select a replacement after held-out results. No LLM dependency.

After this complete small loop works: broaden the mapping language and alignment,
cost relaxation and goal-test timing, then one Kinder planning route including
initialization, cached plans, output control, and helper computation. A future LLM
proposer must emit the same restricted manifests and use the same verifier.
