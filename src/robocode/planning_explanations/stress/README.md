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

Default seeds 17/29 and default budgets. "First run" is the original analyzer.
"Fix A" merges the depth-1 evaluation and score roles. "Fix 2" matches and
patch-tests each coordinate on its own and assembles compatible survivors (see
`docs/planning-explanations.md`, "Search space").

| Program | First run | After fix A | After fix 2 |
| --- | --- | --- | --- |
| `while_loop`, `helper_function`, `comprehension` | Refused | Refused | Refused |
| `unrolled` | Unresolved | Unresolved | **Supported (depth 2)** |
| `memoized` | Supported (depth 2) | Supported (depth 2) | Supported (depth 2) |
| `pruning` | Unresolved / rejected | Unresolved / rejected | Unresolved: no compatible mapping (all 4 leaves read) or no matching location (3 leaves read) |
| `depth_three` | Unresolved | Unresolved | Unresolved: output differs from the depth-2 model, so rejected before matching |
| `sum_rewards` | Unresolved | Unresolved | Unresolved: output differs from the depth-2 model, so rejected before matching |
| `unused_depth` | **Unresolved** | **Supported (depth 1), second loop only** | Supported (depth 1), second loop only |
| `unused_depth_with_variable` | Supported (depth 1) | Supported (depth 1) | Supported (depth 1) |

The original six examples keep their verdicts after both fixes. After fix 2 each
supported route has **one** explanation; copies of the same value (for example
`u[leaf]`, `value`, and the element of `values`) are listed as interchangeable
locations instead of as separate explanations, and every one of them also passes on
fresh validation inputs. Also supported now, though not in this folder:
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

## What the run shows

1. **Syntax (3 programs)** is an engineering limit: `while`, helper functions and
   comprehensions are refused before any analysis.
2. **The search model (3 programs)** is the conceptual limit. A deeper search,
   pruning, or a different way of combining values is never *wrong*; it is
   **unresolved**. The analyzer can't say "this plans more deeply"; it only says
   "this doesn't fit my one fixed tree."
3. **Matching** handled a repeated leaf (`memoized`) but not code without loops
   (`unrolled`).
4. **Style sensitivity (found here, fixed by fix A).** `unused_depth` and
   `unused_depth_with_variable` behave identically, yet get different verdicts. The
   depth-1 model has separate *evaluation* and *score* roles, even though at depth 1
   they are the same number. A program has to read that number in two different
   places (for example `value = u[...]` and then `append(value)`) for both roles to
   be found. Otherwise the only later reads are at
   `return int(greedy[1] > greedy[0])`, where each read of `greedy` uses only one
   element, the same "unused copy" failure as rejected two_step candidate
   `6066afab…`.

   **Fix A:** the depth-1 model is now `states_1 → evaluations → decision`, with no
   separate `scores` role (`ROLES` and `action_values` in `task.py`). Depth 2
   still has `scores`, because there the max over two leaves is a separate
   computation.

5. **Unrolled code (fixed by fix 2).** A role used to need all its values from one
   expression, so code without loops (`unrolled.py`,
   `int(u[t[s][1]] > u[t[s][0]])`) could never be matched. Now each coordinate is
   matched and patch-tested on its own, and survivors are combined only if every
   pair is compatible. In `unrolled.py` the report also shows *why* some locations
   fail: each separate read of `left` feeds only one leaf, so patching it can't
   move both leaves the way changing the first state should; `t[s][0]` can.
