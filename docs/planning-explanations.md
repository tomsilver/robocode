# Planning explanations: prototype specification

Recorded 2026-11-05. This document preserves the agreed scope and decisions for
subsequent work. The implementation and its limitations are documented below.

## Goal

Discover and experimentally validate planning explanations for **program
executions**, and average a planningness measure over an explicitly chosen
distribution of task states. Decided 2026-11-08: planningness is the effective
lookahead depth, reported as an interval that keeps unknown outcomes unknown (see
"Lower bound and planningness" below).

## Agreed initial scope

- Restricted, deterministic Python; scalars, lists, and dictionaries.
- A tiny resettable navigation environment: approximately six states, two actions,
  and supplied terminal utilities.
- One-step and two-step lookahead.
- Bounded enumeration of computational hypotheses and mappings, with no initial
  LLM dependency.

## Method

1. Instrument execution generically, recording intermediate values and checkpoints
   and allowing targeted interventions. Verify that observation preserves behavior.
2. Automatically inventory candidate locations, without meaningful names or
   hand-annotated internal roles.
3. Jointly enumerate hypotheses and mappings to existing program values. Mappings
   may select and rearrange values; they may not run a solver.
4. Use the supplied task transitions and utilities to test future-state and
   evaluation roles.
5. Filter with ordinary executions, then paired interventions comparing predicted
   and observed downstream effects.
6. Retain multiple supported explanations and rejection counterexamples.
7. Freeze explanations before validation on fresh inputs.
8. Evaluate ordinary executions under the chosen state distribution, permitting
   input-dependent planning and reactive routes.

## Invariants

- Matching outputs alone is insufficient.
- Unknown or unresolved is **not** zero planning.
- Interventions validate explanations. Patched runs are not the distribution over
  which planning is averaged.
- Count the original program's computation, not analyzer-generated rollouts.
- Keep evidence strength separate from amount of planning.

## Fixtures and first milestone

Fixtures: explicit planner; renamed/dictionary variant; precomputed action lookup;
conditional reactive/planning program; decoy that computes search but ignores it.
Include both one-step and two-step planners.

The first milestone must recover the simple planner's explanation, reject that
explanation for the decoy, and distinguish the conditional program's execution
routes. Deliver a runnable demonstration, meaningful tests, and readable plus
machine-readable reports with mappings, checkpoints, interventions,
predicted/observed effects, rejected candidates, supported horizon, evaluated
alternatives, and ambiguity.

## Implementation decisions

The prototype uses generic AST expression checkpoints and bounded expression
occurrence selectors. The task input boundary may specify state, transition table,
and utility vector; internal identifiers carry no semantic labels. The initial
hypothesis family is exhaustive deterministic lookahead with first-action tie
breaking and a maximum backup, parameterized by depth (1, 2 or 3), by how a
path is valued (its last state's utility, or the sum of utilities along it), and
by whether the search uses the sound cutoff (stop expanding the second action
once one of its paths beats the first action's score). Successors come either
from the environment's table or from a fixed table of the program's own,
inferred from its traces.
Restrictions, enumeration limits, validation splits,
and unsupported cases must be explicit in reports. A failed or bounded search
does not establish absence of planning.

## Run and inspect

From the repository root, with Python 3.11 or 3.12 (the existing `.venv` works):

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/planning_explanations -v
```

The prototype and unittest suite require only the standard library. There is no
LLM call, simulator dependency, or package installation. The tests also work with
the repository's pytest runner when development dependencies are installed.

Start at `outputs/planning-explanations/README.md`. Each fixture directory contains:

| File | Inspect it for |
| --- | --- |
| `source.py` | Exact original program, without instrumentation hooks |
| `frozen.json` | Immutable hypothesis/mapping manifest, saved before fresh inputs are generated |
| `report.md` | Ordinary routes, features, mappings, evidence and rejection witnesses |
| `report.json` | All complete candidate rejections, supported candidates, paired effects and input data |
| `ordinary-traces.json` | Raw expression snapshots for every ordinary validation execution |

`summary.json` contains the same overview in machine-readable form. Generated
reports live under the repository's ignored `outputs/` directory; regenerate
them rather than treating them as source files.

Example variations:

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations --fixture conditional
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations \
  --fixture conditional --state-weights 0.75 0.25 0 0 0 0 \
  --output-dir outputs/planning-explanations-weighted
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations \
  --source /absolute/path/to/program.py --output-dir outputs/my-explanation
```

Custom source is a sequence of function definitions, optionally preceded by
`import heapq` / `import math` (or `from` imports of their public names). The
**last** top-level function is the policy and takes four positional arguments:
current state, transition table, terminal utility vector, and externally cached
action table. Every other function, top-level or nested, is a helper; recursion
is allowed. Bodies may use `for` and `while` loops, list and dictionary
comprehensions, generator expressions, slices, a fixed set of builtins (`range`,
`len`, `min`, `max`, `abs`, `sum`, `sorted`, `reversed`, `enumerate`, `zip`,
`any`, `all`, `int`, `float`, `list`, `dict`, `tuple`, `set`) and container
methods (`append`, `extend`, `insert`, `pop`, `get`, `items`, `keys`, `values`,
`add`). Classes, lambdas, `try`/`with` and `global`/`nonlocal` are refused, as
are other imports and top-level statements. A program that raises on an
ordinary input, or loops more than 10,000 times in one run, is reported as
failed rather than crashing a folder run. Parameter and local names are
arbitrary. The extra action table is control
data: it lets the lookup and decoy fixtures have exactly the same ordinary outputs
as the two-step planner for every generated case. All fixture programs receive
it; planners ignore it. Its preparation happens **outside measured execution**.

## Plot the ordinary tree and interventions

After generating reports, run this from the repository root. Plotting uses
Matplotlib (already a project dependency); the analyzer itself stays stdlib-only.

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.plot
```

The default is the two-step fixture's first supported validation case and mapping.
The command prints the paths of two PNGs in the fixture's
`plots/<case>/<mapping-id>/` directory:

- **Ordinary tree:** states, leaf utilities, branch scores, and the returned action,
  read from the unpatched trace using that frozen mapping. Nodes are action paths:
  two paths can end in the same state.
- **Intervention comparison:** by default, the low and high replacements for the
  first leaf evaluation. Each column is a separate patch to the same ordinary
  execution. The top row shows the model prediction; the bottom row shows the
  recorded program response. The orange outline marks the target; Δ marks a
  change relative to the ordinary execution. Patched runs do not receive any
  population weight.

To choose other experiments, list the case-local intervention numbers first:

```bash
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.plot \
  outputs/planning-explanations/two_step --case validation-0-0 --list
PYTHONPATH=src .venv/bin/python -m robocode.planning_explanations.plot \
  outputs/planning-explanations/two_step --case validation-0-0 \
  --interventions 1 14 21 --format svg
```

Use `--explanation ID` to choose among supported mappings, or
`--all-interventions` to export all patches in groups of up to three per figure.
PNG, SVG and PDF are supported. The command reads existing `report.json` and
`ordinary-traces.json`; it never executes the fixture or solves the task. A
supported case with no recorded intervention gets an ordinary plot only.
Unresolved cases (including the lookup and decoy) do not get a fabricated tree.

## Small architecture

Source is in `src/robocode/planning_explanations/`:

- `task.py`: resettable six-state, two-action navigation and the candidate
  models (`Model`: depth, path value, pruning and an optional own transition
  table, with its roles and its patched depth-first rollout of node events).
  Every state has a supplied utility. States are not absorbing.
- `fixtures.py`: six ordinary programs, including one- and two-step planners.
- `instrument.py`: validates the Python subset and inserts expression hooks
  using Python's AST. A checkpoint is identified by source span and expression
  kind, visit number, and optional literal list/tuple/dictionary path. Inside a
  helper, the visit number already identifies the caller: within one route,
  calls happen in the same order on every input. Sets run but are not recorded,
  and infinities are written to JSON as the strings `"inf"`/`"-inf"`. Observing a value returns the original object; snapshots
  are deep copies. Interventions substitute a single expression result, or a
  selected element of a copied expression result. They do not globally mutate
  every alias. Execution resets inputs and globals on every run.
- `discovery.py`: factorized search (per-coordinate matching and patch tests,
  then assembly from pairwise-compatible survivors), the full ordinary and paired
  tests, and validation of a frozen bank with its ordinary state-weighted
  population.
- `report.py`: JSON serialization, frozen-bank loading, and Markdown reports.
- `__main__.py`: reproducible demonstration and CLI.

To inspect a single explanation, open the first mapping table in the two-step
report. It should select two observed first successors, four observed second
successors, four utility values, and two backed-up action scores. Source names
appear only to make the result readable; discovery tests values and execution
order rather than interpreting those names. Copies of the same value (for
example `u[leaf]`, the variable `value`, and the element of `values`) are listed
under the explanation as interchangeable locations, each re-tested on fresh
inputs as a single-coordinate swap, rather than as separate explanations.

For a concrete negative witness, find that explanation's ID in the decoy report.
The initial outputs match. A changed successor predicts changed evaluations,
backups, and sometimes a changed action. The decoy still returns its cached
action, so the explanation is rejected. The tests also cover cases where the
final action agrees but intermediate predictions disagree.

## Search space and validation protocol

- Candidate models (`task.MODELS`) are depth-first two-action search to depth
  1, 2 or 3, with a maximum backup and action-zero tie breaking. A path is valued
  either by its last state's utility (`leaf`) or by the sum of the utilities of
  every state on it, the start state excluded (`sum`). Beyond depth 1 a model may
  use the sound **cutoff**: once a path under the second action is worth strictly
  more than the first action's score, the rest of the second action is not
  expanded. The decision cannot change, so the cutoff never changes outputs; it
  changes which nodes are expanded, depending on the values seen. At depth 1 the
  path values coincide and there is nothing to cut, so there are nine models.
- Model values are **node events**. `states_k[i]` is the state reached by the
  i-th path of k actions, `evaluations[i]` the value of the i-th complete path
  and, beyond depth 1, `scores[a]` the best evaluated path value under first
  action `a`. A node the search does not expand is `null`. A route must expand
  the same nodes on all its discovery inputs (otherwise the search ends as
  `shape_varies`); only expanded nodes are coordinates, and the mapping holds
  `null` for the others. Under a patch the model predicts which nodes are
  expanded, and a mapped location the patched program never reaches reads as
  `null`, so a cutoff triggered by a patch is a prediction that can pass.
  "Evaluated alternatives" in the report counts the paths a route's search
  evaluates: fewer where it prunes.
- **Own transition table.** Every model is also tried with the program's own
  fixed table instead of the environment's. The table is inferred per route and
  depth from the trace, not the source. Candidate successors are recorded
  integers 0-5 present in every run; constants and re-reads of the parent do not
  count. A first state must be a fixed function of the start state and a deeper
  state a fixed function of its parent's value, all forming one table that
  differs from the environment somewhere (otherwise reading the environment
  explains it). Every entry must be seen on two inputs, so a route needs each of
  its start states twice. Each candidate table is then tested like any model;
  most are rejected before matching. Fresh inputs that need a row the table
  lacks are not covered, and state patches into such rows are counted as
  untestable. Reports give the table and how often it agrees with each input's
  environment.
  Depth 1 deliberately has no `scores` role: it would be a copy of the same
  numbers, and demanding a separate program location for that copy made support
  depend on code style (`stress/unused_depth.py`). Intermediate maxima (per
  depth-2 node at depth 3) are not roles either: a program may compute them or
  not. Every model does need one value per complete path, so a program that adds
  a shared term after the maximum is not covered (`stress/sum_factored.py`).
- Sum-valued models often make the same decisions: depth-2 and depth-3 sums agree
  on about 90% of generated inputs, and depth-1 with depth-2 sum on about 85%,
  because the first step's utility is added to every path under an action.
  Outputs alone then rarely separate them; matching and patches do.
- The search is factorized by model coordinate (one state, evaluation or score):
  1. The program's ordinary outputs must equal the model's decisions on every
     discovery input of the route. This needs no mapping, so a wrong model is
     usually rejected before matching.
  2. Each coordinate is matched to every recorded numeric value, at any
     expression, visit or container element, that equals the model's value in
     all of the route's discovery runs. Different coordinates of one role may
     come from different expressions, so unrolled code is covered
     (`stress/unrolled.py`).
  3. Each matching location is patched on its own. The action must change
     exactly as the model predicts, or the location is rejected with that
     counterexample.
  4. Survivors are assembled into one complete mapping by backtracking. Every
     pair must respect the model's temporal order and be compatible: under each
     patch of one, the other holds the model's predicted value. The full paired
     test is a conjunction of exactly these single-location and pairwise checks,
     so cost grows with the number of locations, not their product. For
     two_step, 49 matching locations would form about 18.4 million complete
     combinations; the first assembled mapping is supported.
  5. Other survivors that fit in place of one coordinate are recorded as
     interchangeable locations.
  6. The assembled mapping is re-checked with the full ordinary and paired
     tests, which also produce the reported evidence. A disagreement would be
     reported as `verification_failed`, never hidden.
- Mappings select recorded values only; they never call the model or combine
  values. Model rollouts create expected values only.
- Paired tests substitute every other state ID at each state coordinate, so a
  decision-changing state is tried whenever one exists. At each evaluation and
  action-score coordinate they substitute a number far below and one far above
  every utility, plus values just below, exactly at, and just above the
  coordinate's switching point: the value at which the model's decision flips
  (found from the model, 0.125 away). Generated utilities all end in .25, so
  ordinary runs and the far patches only ever produce whole-number differences;
  the switching-point patches are the only ones that test where the decision
  actually flips. Predicted and observed downstream mapped values and final
  actions must agree. At least one output-changing intervention per tested input
  is required. Each report counts its *informative* patches, whose predicted
  action differs from the ordinary run; the others are tested only through
  internal values.
- Default budgets per route and model: 64 matching locations per coordinate,
  128 complete mappings, 100,000 assembly steps, and two intervention cases.
  Every rejected location has a counterexample; pairwise conflicts have counts
  plus a sample of 16. Truncation is explicit.
- Defaults: discovery seed 17, validation seed 29, two independently sampled
  contexts for each initial state in each split (`--inputs-per-state` changes
  this; more inputs let discovery see more routes). Each context has its own
  utilities and its own transition table (two different successors per state).
  With one fixed table, the states a program visits depend only on the start
  state, and a program carrying a private copy of the table is indistinguishable
  from one reading `t`. The default initial-state distribution is uniform;
  `--state-weights` overrides it. Contexts within each state have equal
  conditional weight.
- Branch routes are the generic executed `if` and `while` decisions and
  comprehension filters, so a while loop's iteration count is part of its route.
  For-loop lengths are not; a data-dependent `for` loop can therefore shift visit
  numbers between inputs of one route, which shows up as unresolved, not as false
  support. For each frozen mapping,
  validation tests all fresh ordinary executions on its route and paired
  interventions on the first two fresh cases. A fresh input whose route has no
  supported explanation (a route discovery never saw, or one whose explanations
  failed) is then tested against every frozen explanation from another route that
  was not falsified, with the same full test. No mapping repair or discovery
  occurs on validation data. Reused discovery inputs are rejected even if renamed.
- `load_frozen(path)` restores a bank for separate validation without discovery;
  source and manifest digests detect a mismatched program or modified mapping.

The ordinary-population report contains **only unpatched runs**. Supported
horizon, root alternatives, and evaluated leaf alternatives are properties of a
supported explanation grounded in recorded original values. These are not counts
of analyzer rollouts or unique machine operations. Counts across ambiguous
explanations are never added. Evidence counts and validation status are reported
separately.

## Lower bound and planningness

Every fresh input ends in one of three states:

- **Supported planning:** a frozen explanation passes the full test there. Its
  amount is known (below).
- **No lookahead:** the action never changes when only the input's *future*
  changes. The test keeps the current state, its own utility and the cached
  actions, and changes everything else, one at a time: each other state's
  utility pushed far up and far down (10 changes), each transition redirected to
  every other state (60), then 16 random futures. If none of these 86 changes
  moves the action, the input does not look ahead, and its amount is 0. This is
  evidence from interventions, not a proof. The definition is behavioral: a
  program that searches a fixed internal model with fixed internal values never
  responds to the environment and counts as no lookahead
  (`stress/fixed_internal_planner.py`); so does a planner whose own model makes
  every option look the same from some state (`stress/partly_wrong.py`, which
  believes state 0 is absorbing).
- **Unresolved:** some change to the future moved the action (the report shows
  the first such change), so the input looks ahead, but no model was verified.
  Each carries a reason: `unseen_route` (discovery never saw the route and no
  other route's explanation passed), `not_covered` (an own table lacks a row the
  input needs), `rejected` (the route's explanations failed on fresh inputs),
  `budget` (a discovery search ran out of budget there) or `no_model_fits`
  (it plans in a way the family does not cover).

**Supported mass** is a lower bound on the share of states where the program
plans, and every verified planner also passes the no-lookahead test as looking
ahead.

**Planningness** is the weighted **effective lookahead depth**: for an
explained input, log2 of the complete paths its verified search evaluates (a full
depth-d search scores d, greedy scores 1, pruning lowers it); no lookahead scores
0. It is an interval. Unresolved inputs look ahead by an unknown amount: they
count their **behavioral lower bound** at the lower end and the family **max**
(currently 3, the deepest model the analyzer can verify, not a true bound) at the
upper end. A program planning deeper than the family
would sit above the interval, so the max must be revisited whenever the model
family or domain changes. When several explanations disagree on an input's
amount, the smaller counts at the lower end and the larger at the upper end. The
report also gives the average over inputs with a verified amount and their share.
Examples: two-step planner [2, 2]; conditional [1, 1] (odd states depth 2, even
states no lookahead); lookup [0, 0]; comprehension [2, max 3] (looks ahead at
depth 2, no model fits).

The **behavioral lower bound** needs no model. It builds tree-shaped futures: the
start state has no incoming moves, and each first successor is entered only from
it, so the move for action a is used only by paths starting with a, and the next
move only by paths starting with (a, b). Redirecting one such move and seeing the
action change shows that the decision compares a path through it; different
starts are different paths, so at most 4 paths (a bound of 2) can be shown. Only
futures in which the program takes the same route as on the input count, so the
evidence is about the same computation (this keeps it valid for pruning). Across
the stress programs it never exceeded a verified amount on any of the 210
supported inputs and equalled it on 150; it is 0 for programs that ignore the
environment's moves (their own transition table).

The average over verified inputs is an estimate from the sampled contexts: how
often a pruning program prunes varies between samples, and more inputs per state
narrow it.

## First milestone results

With default seeds and budgets:

| Fixture | Supported explanations | Interchangeable locations | Supported horizon | Ordinary routes |
| --- | ---: | ---: | --- | --- |
| One-step planner | 1 | 10 | 1 | Supported throughout |
| Two-step planner | 1 | 22 | 2 | Supported throughout |
| Renamed dictionary planner | 1 | 22 | 2 | Supported throughout |
| Precomputed action lookup | 0 | — | Unknown | Unresolved |
| Conditional | 1 on planning route | 22 | 2 on odd states | Odd states supported; even states unresolved |
| Decoy search | 0 | — | Unknown | Every matching location rejected by its own patch test |

All six default searches finish without exhausting their budgets. Every frozen
interchangeable location also passes on fresh validation inputs. Per-input
transition tables and switching-point patches left all six verdicts and frozen
mappings unchanged. Two-step validation now runs 111 paired interventions
instead of 48, and 32 of them predict a different action than the ordinary run.
The five-model family also leaves every verdict and mapping unchanged; the
explanation identifiers changed once, because the model's path value is now part
of each explanation. Of the harder programs, `depth_three` is supported at depth
3 and `sum_rewards` at depth 2 with summed path values. The wider Python subset
left every earlier route, mapping and identifier unchanged, and adds supported
`while`, helper-function, recursive and `heapq` versions of the two-step planner.
Node events with an optional cutoff also leave all of these unchanged; the
pruning program is supported by the depth-2 cutoff model on each of its routes,
evaluating 3 paths where it prunes and 4 where it does not. Own transition tables
also leave all of these unchanged (no program that reads `t` gets one); the
private-copy and formula programs are supported at depth 2 with their exact own
tables, which agree with the random environment on about 1 in 6 transitions.

The even-state conditional route is reactive by fixture construction. The
analyzer conservatively reports that it has no supported lookahead explanation,
rather than claiming proof of zero planning. Both conditional routes and their
checkpoints remain inspectable.

Harder test programs live in `src/robocode/planning_explanations/stress/`, with
predictions and results in that folder's README. Run them all with
`--source-dir src/robocode/planning_explanations/stress`.

## Deliberate limits and next steps

This is empirical support for a constrained explanation family, not an algorithmic
equivalence proof or a general detector of planning. The Python subset excludes
classes, lambdas, exceptions, imports beyond `heapq` and `math`, methods beyond
the listed container methods, and cyclic containers. Best-first search, where
which node is expanded at step k depends on values without any recorded branch,
does not fit node events grouped by route; it needs coordinates per step. An
own model that is only partly the program's own ("the environment, except state
0 is absorbing") is neither a fixed table nor the environment, and stays
unresolved. Programs are trusted and
tiny; syntax checking and expression and loop budgets are not a
resource-isolation sandbox. Only ordinary output and input
mutation are checked for observation preservation, not timing or resource usage.

Future work can expand the mapping grammar and algorithm family, add joint or
interchange interventions, and evaluate independently specified task-state
distributions. Planningness here measures lookahead amount only; other amounts
(alternatives considered per decision, expansions, model accuracy) can be added
beside it, keeping unknown outcomes unknown.
