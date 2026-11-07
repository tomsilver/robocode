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

| Program | First run | Depth-1 roles merged | Per-coordinate search | Varied tables | Boundary patches |
| --- | --- | --- | --- | --- | --- |
| `while_loop`, `helper_function`, `comprehension` | Refused | Refused | Refused | Refused | Refused |
| `unrolled` | Unresolved | Unresolved | **Supported (d2)** | Supported (d2) | Supported (d2) |
| `memoized` | Supported (d2) | Supported (d2) | Supported (d2) | **Supported (d2) on 0.83 of inputs**: one validation route never seen in discovery | Same |
| `pruning` | Unresolved | Unresolved | Unresolved: no compatible mapping / no matching location | Same | Same |
| `depth_three` | Unresolved | Unresolved | Unresolved: rejected before matching | Same | Same |
| `sum_rewards` | Unresolved | Unresolved | Unresolved: rejected before matching | Same | Same |
| `unused_depth` | Unresolved | **Supported (d1)**, second loop only | Supported (d1) | Supported (d1) | Supported (d1) |
| `unused_depth_with_variable` | Supported (d1) | Supported (d1) | Supported (d1) | Supported (d1) | Supported (d1) |
| `own_table` | — | — | Supported (d2) | **Unresolved: rejected before matching** | Same |
| `margin` | — | — | Supported (d2) | Supported (d2) | **Unresolved: every value location fails a boundary patch** |

"—": the program was added later; its first result is the code as it stood then.
The original six examples keep their verdicts and their exact frozen explanations
through all five columns. Each supported route has **one** explanation; copies of
the same value (for example `u[leaf]`, `value`, and the element of `values`) are
listed as interchangeable locations, and every one of them also passes on fresh
validation inputs. Also supported, though not in this folder:
`int(u[t[s][1]] > u[t[s][0]])` (unrolled depth 1; see the unit tests).

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

## What the runs show

1. **Syntax (3 programs)** is an engineering limit: `while`, helper functions and
   comprehensions are refused before any analysis.
2. **The search model (3 programs)** is the conceptual limit. A deeper search,
   pruning, or a different way of combining values is never *wrong*; it is
   **unresolved**. The analyzer can't say "this plans more deeply"; it only says
   "this doesn't fit my one fixed tree."
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
