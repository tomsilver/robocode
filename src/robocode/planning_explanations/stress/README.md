# Harder test programs (Step 1)

Each `.py` file here is one program with the same interface as the built-in fixtures:
`policy(s, t, u, cache)` returns action 0 or 1. Each one targets a single assumption
of the analyzer. The comment at the top of every file says what it tests and records
the prediction written **before** the first run. The `# pylint: skip-file` line is
needed because the analyzer only accepts a file containing one function, so a module
docstring isn't possible.

## Run

From the repository root:

```bash
# All programs in this folder
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations \
  --source-dir src/robocode/planning_explanations/stress \
  --output-dir outputs/planning-explanations-stress

# One program
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations \
  --source src/robocode/planning_explanations/stress/memoized.py \
  --output-dir outputs/planning-explanations-stress
```

Overview: `outputs/planning-explanations-stress/README.md`. Per-program details:
`outputs/planning-explanations-stress/<name>/report.md` and `report.json`. Programs
outside the supported Python subset are listed as "Refused" with the reason; they
get no report folder.

To add a program, drop a new `.py` file here (one function, four arguments) and
rerun the folder command. Write its prediction in the header first.

## Scoreboard

Default seeds 17/29 and default budgets. Each column adds one change to the one
before it:

1. **First run:** the original analyzer.
2. **Depth-1 roles merged:** the depth-1 model has no separate score role.
3. **Per-coordinate search:** each coordinate is matched and patch-tested on its
   own, and compatible survivors are assembled (see `docs/planning-explanations.md`,
   "Search space").
4. **Varied tables:** every input has its own transition table instead of one
   fixed table.
5. **Boundary patches:** states are replaced by every other state, and values also
   by numbers just below, at and just above the model's switching point.
6. **Model family:** five models instead of two: depth 1, 2 or 3, with a path
   valued by its leaf utility or by the sum of utilities along it.
7. **Wider Python:** `while` loops, helper functions (nested or top-level, the
   last top-level function being the policy), recursion, comprehensions,
   generator expressions, `heapq`/`math` imports, tuples, more builtins and
   container methods.
8. **Node events + cutoff:** model values are events for the nodes a search
   expands (unexpanded nodes are absent), and each model beyond depth 1 also comes
   with the sound cutoff: stop expanding the second action once one of its paths
   beats the first action's score.

| Program | First run | Depth-1 roles merged | Per-coordinate search | Varied tables | Boundary patches | Model family | Wider Python | Node events + cutoff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `while_loop` | Refused | Refused | Refused | Refused | Refused | Refused | **Supported (d2)** | Same |
| `helper_function` | Refused | Refused | Refused | Refused | Refused | Refused | **Supported (d2)** | Same |
| `comprehension` | Refused | Refused | Refused | Refused | Refused | Refused | **Unresolved: no single location holds `states_1[1]`** | Same |
| `unrolled` | Unresolved | Unresolved | **Supported (d2)** | Supported (d2) | Supported (d2) | Supported (d2) | Same | Same |
| `memoized` | Supported (d2) | Supported (d2) | Supported (d2) | **Supported (d2) on 0.83 of inputs**: one validation route never seen in discovery | Same | Same | Same | Same |
| `pruning` | Unresolved | Unresolved | Unresolved: no compatible mapping / no matching location | Same | Same | Same | Same | **Supported (d2, cutoff) on all 3 routes; 3 or 4 paths evaluated** |
| `depth_three` | Unresolved | Unresolved | Unresolved: rejected before matching | Same | Same | **Supported (d3, leaf)** | Same | Same |
| `sum_rewards` | Unresolved | Unresolved | Unresolved: rejected before matching | Same | Same | **Supported (d2, sum)** | Same | Same |
| `unused_depth` | Unresolved | **Supported (d1)**, second loop only | Supported (d1) | Supported (d1) | Supported (d1) | Supported (d1) | Same | Same |
| `unused_depth_with_variable` | Supported (d1) | Supported (d1) | Supported (d1) | Supported (d1) | Supported (d1) | Supported (d1) | Same | Same |
| `own_table` | — | — | Supported (d2) | **Unresolved: rejected before matching** | Same | Same | Same | Same |
| `margin` | — | — | Supported (d2) | Supported (d2) | **Unresolved: every value location fails a boundary patch** | Same | Same | Same |
| `sum_factored` | — | — | — | — | Unresolved: rejected before matching | Unresolved: the d2-sum model finds no per-path sums | Same | Same |
| `recursive` | — | — | — | — | — | Refused | **Supported (d2)** | Same |
| `heap_choice` | — | — | — | — | — | Refused | **Supported (d2)** | Same |
| `running_max` | — | — | — | — | — | Refused | **Unresolved**: no surviving location / no compatible mapping, depending on route | Same |
| `pruning_depth_three` | — | — | — | — | — | — | Unresolved | **Supported (d3, cutoff) on 0.83 of inputs**: 7 or 8 paths evaluated; other inputs on routes discovery never saw |
| `unsound_pruning` | — | — | — | — | — | — | Unresolved | Unresolved: rejected by every model |

"—": the program was added later; its first result is the code as it stood then.
The original six examples keep their verdicts and mappings through all eight
columns (their identifiers changed once, in the last column, because the path
value became part of each explanation). Each supported route has **one**
explanation; copies of the same value (for example `u[leaf]`, `value`, and the
element of `values`) are listed as interchangeable locations, and every one of
them also passes on fresh validation inputs. Also supported, though not in this
folder: `int(u[t[s][1]] > u[t[s][0]])` (unrolled depth 1; see the unit tests).

## Results of the first run, against the predictions

| Program | Tests | Prediction | Result | Match |
| --- | --- | --- | --- | --- |
| `while_loop` | Python subset | Refused | Refused: `While` | ✅ |
| `helper_function` | Python subset | Refused | Refused: nested function | ✅ |
| `comprehension` | Python subset | Refused | Refused: `ListComp` | ✅ |
| `unrolled` | Matching (cross-expression) | Unresolved | Unresolved: only 6 value series exist, no complete candidate | ✅ |
| `memoized` | Matching (repeated leaf) | Supported, via `memo[leaf]` | Supported on both routes, evaluations at `memo[leaf]` | ✅ |
| `pruning` | Search shape | Unresolved / rejected | Pruned inputs: unresolved (3 leaves). Full inputs: rejected, because a patch triggers the skip and leaf 4 is never read | ✅ |
| `depth_three` | Search shape | Unresolved | Unresolved: no complete candidate | ✅ |
| `sum_rewards` | How values are combined | Unresolved | Unresolved: no value matches the model's branch scores | ✅ |
| `unused_depth` | Computed vs used | Supported at depth 1 only | **Unresolved.** Depth-2 candidates were rejected as intended, but every depth-1 candidate failed too | ❌ |
| `unused_depth_with_variable` | Style sensitivity (follow-up) | Supported at depth 1 | Supported at depth 1, using the second loop only | ✅ |

Added later, each predicted before its first run:

| Program | Tests | Prediction | Result | Match |
| --- | --- | --- | --- | --- |
| `own_table` | Environment's table vs a private copy | Supported with one fixed table; unresolved once tables vary | Exactly that | ✅ |
| `margin` | Patches at the decision boundary | Supported with low/high patches; unresolved with boundary patches | Exactly that | ✅ |
| `depth_three` (model family) | Depth as a parameter | Supported at depth 3, leaf values | Exactly that | ✅ |
| `sum_rewards` (model family) | Path value as a parameter | Supported at depth 2, sum values, evaluations at `u[successor] + u[leaf]` | Exactly that | ✅ |
| `while_loop` (wider Python) | `while` loops | Supported (d2), one route | Supported (d2); the loop tests are the same on every input | ✅ |
| `helper_function` (wider Python) | Helper functions | Supported (d2); helper visits 0/1 = calls for actions 0/1 | Exactly that: depth-2 states at `t[state][0]`/`t[state][1]`, visits 0 and 1 | ✅ |
| `comprehension` (wider Python) | Comprehensions | Unresolved: no surviving location for `states_1` | Unresolved; `states_1[1]` has none. `states_1[0]` keeps two that pass the action-only screen but conflict with every depth-2 state they should feed | ◐ |
| `recursive` | Recursion, several functions, generator expression | Supported (d2) | Supported (d2): leaf utilities at the depth-0 `u[state]` reads | ✅ |
| `heap_choice` | `heapq`, tuples | Supported (d2), scores at `max(values)` | Exactly that | ✅ |
| `running_max` | Running-maximum idiom, `math.inf` | Unresolved (low confidence about which check) | Unresolved: on two routes the second leaf exists only as the comparison read, which fails its patch test; on the third no compatible mapping | ✅ |
| `pruning` (node events) | Pruning as a prediction | Supported by d2 cutoff on both routes; 3 paths where it prunes, 4 where not; full model rejected | Supported by d2 cutoff and full model rejected on every route, 3 or 4 paths. It has **three** routes, not two: its `if` can also fire after the last leaf, where breaking skips nothing | ◐ |
| `pruning_depth_three` | Cutoff at depth 3 | Supported (d3 cutoff) on routes seen in discovery; some validation inputs on unseen routes | Exactly that: 3 routes in discovery, 0.83 of validation inputs supported | ✅ |
| `unsound_pruning` | An unsound pruning rule | Rejected before matching by every model | Unresolved, but on two small routes some models made the same decisions and were rejected later: by matching, or by a patch that triggers the unsound skip | ◐ |
| `sum_factored` | Order of operations in the sum model | Unresolved: the d2-sum model finds scores but no per-path sums; every other model rejected before matching | Unresolved as predicted, but the d3-sum model got past the output check and failed at matching instead (no depth-3 states) | ◐ |

## What the runs show

1. **Syntax (3 programs)** was an engineering limit: `while`, helper functions and
   comprehensions were refused before any analysis. Fixed by the wider Python
   subset (item 9).
2. **The search model (3 programs)** was the conceptual limit. A deeper search,
   pruning, or a different way of combining values was never *wrong*; it was
   **unresolved**: "this doesn't fit my one fixed tree." The model family (item 8)
   now covers the deeper search and the summed values, and node events with the
   cutoff (item 10) cover pruning.
3. **Matching** handled a repeated leaf (`memoized`) but not code without loops
   (`unrolled`).
4. **Style sensitivity (fixed by merging the depth-1 roles).** `unused_depth` and
   `unused_depth_with_variable` behave identically, yet got different verdicts. The
   depth-1 model had separate *evaluation* and *score* roles, even though at depth 1
   they are the same number. A program had to read that number in two different
   places (for example `value = u[...]` and then `append(value)`) for both roles to
   be found. Otherwise the only later reads are at
   `return int(greedy[1] > greedy[0])`, where each read of `greedy` uses only one
   element, the same "unused copy" failure as rejected two_step candidate
   `6066afab…`. Now the depth-1 model is `states_1 → evaluations → decision`
   (`ROLES` and `action_values` in `task.py`). Depth 2 still has `scores`, because
   there the max over two leaves is a separate computation.
5. **Unrolled code (fixed by the per-coordinate search).** A role used to need all
   its values from one expression, so code without loops (`unrolled.py`,
   `int(u[t[s][1]] > u[t[s][0]])`) could never be matched. Now each coordinate is
   matched and patch-tested on its own, and survivors are combined only if every
   pair is compatible. In `unrolled.py` the report also shows *why* some locations
   fail: each separate read of `left` feeds only one leaf, so patching it can't
   move both leaves the way changing the first state should; `t[s][0]` can.
6. **A private copy of the table (fixed by varied tables).** With one fixed table,
   `own_table` was supported: its copy equals `t` on every input. With a table per
   input, its decisions differ from the model's wherever the copy is wrong, so it is
   rejected before matching. It *does* plan, with its own model; telling "plans
   with a wrong model" apart from "doesn't plan" is a later step. Cost: more inputs
   take branch-specific code paths (in `memoized`, which leaf repeats now varies),
   so 12 discovery inputs cover fewer routes, and one validation route stayed
   unresolved.
7. **Decision rule (fixed by boundary patches).** `margin` picks action 1 only with
   a lead above 0.5. All utilities end in .25, so ordinary runs and the ±100 patches
   only produce whole-number differences, where the two rules agree. A patch 0.125
   past the switching point separates them: for example, scores[0] 21.25 → −10.875
   (just below scores[1] = −10.75) makes the model choose 1, and the program keeps 0.
   Trying every other state also removed two after-the-fact copies in two_step
   (`t[s]` and `t[successor]` read in the *second* iteration, element 0) that had
   passed their own patch test only because neither of the two old state
   replacements changed the predicted action.
8. **Depth and path value (fixed by the model family).** Each model is a depth
   (1, 2, 3) and a path value (leaf utility, or sum along the path), so the
   analyzer can now say *how deep* and *how valued*: `depth_three` is supported at
   depth 3 and `sum_rewards` at depth 2 with sums, each with exactly one complete
   mapping checked. Two limits showed up. First, `sum_factored` makes the same
   decisions as `sum_rewards` but adds `u[middle]` after the maximum, so no
   per-path sum exists and it stays unresolved: the model fixes the order of
   operations, not just the result. Second, summed models often decide alike
   (depth-2 and depth-3 sums agree on about 90% of inputs, because the first
   step's utility is added to every path under an action), so the output check
   alone rarely separates them; for `sum_rewards` and `sum_factored` the depth-3
   sum model passed it and was rejected only at matching.
9. **Wider Python.** `while_loop`, `helper_function`, `recursive` and
   `heap_choice` are supported at depth 2, each with one complete mapping
   checked. No call context was needed for helpers: within one route, calls
   happen in the same order on every input, so "visit 1 of this helper line" is
   always the call for action 1. Two programs show where the *single-location*
   requirement bites, and both are common in agent code. `comprehension`
   recomputes `t[s][a]` for every leaf, so no read of the first state moves both
   of its leaves. `running_max` (`best = -inf; if v > best: best = v`) reads
   each leaf twice, in the comparison and in the assignment, and neither read
   alone behaves like the model's evaluation; patches that flip a comparison
   also change the route. Both would need a coordinate to map to a *set* of
   reads patched together, or patches that follow control flow (fix 5).
10. **Pruning (fixed by node events with a cutoff).** A model's values are now
    events for the nodes its search expands, so "the last leaf was never looked
    at" is something a model can predict. `pruning` is supported by the depth-2
    cutoff model on every route, and the report now says how many paths each
    route's search evaluated (3 where it prunes, 4 where it does not), which is the
    first planning-amount feature that varies with the input. Patches test the
    cutoff in both directions: lowering the first leaf of action 1 makes the model
    expand the last leaf again, and the program's action-1 score then equals that
    leaf's value; raising it on a full route makes the model skip the last leaf, and
    the program never reaches it. `pruning_depth_three` shows the same at depth 3,
    with the coverage cost of many routes (5 possible stopping points, 12 inputs).
    `unsound_pruning` shows that the cutoff is not "any pruning": its rule changes
    decisions, and where it happens not to, a patch exposes it. Not covered:
    best-first search, where which node is expanded next depends on values with no
    recorded branch, so one route mixes different expansion patterns.
