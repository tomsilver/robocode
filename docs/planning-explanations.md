# Planning explanations: prototype specification

Recorded 2026-11-05. This document preserves the agreed scope and decisions for
subsequent work. The implementation and its limitations are documented below.

## Goal

Discover and experimentally validate planning explanations for **program
executions**. Eventually average a planningness metric over an explicitly chosen
distribution of task states. For now, report features and evidence; do not invent
a final scalar metric.

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
hypothesis family is exhaustive deterministic lookahead with a maximum backup and
first-action tie breaking. Restrictions, enumeration limits, validation splits,
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

Custom source must contain one function with four positional arguments: current
state, transition table, terminal utility vector, and externally cached action
table. Parameter and local names are arbitrary. The extra action table is control
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

- `task.py`: resettable six-state, two-action navigation and the candidate model.
  Every state has a supplied utility at the lookahead cutoff. States are not
  absorbing; this is a finite-horizon task with no intermediate reward.
- `fixtures.py`: six ordinary programs, including one- and two-step planners.
- `instrument.py`: inserts expression hooks using Python's AST. A checkpoint is
  identified by source span and expression kind, visit number, and optional literal
  list/dictionary path. Observing a value returns the original object; snapshots
  are deep copies. Interventions substitute a single expression result, or a
  selected element of a copied expression result. They do not globally mutate
  every alias. Execution resets inputs and globals on every run.
- `discovery.py`: inventories series of existing values, enumerates models and
  rearrangements, filters ordinary runs, and tests paired interventions. It also
  validates a frozen bank and describes the ordinary state-weighted population.
- `report.py`: JSON serialization, frozen-bank loading, and Markdown reports.
- `__main__.py`: reproducible demonstration and CLI.

To inspect a single explanation, open the first mapping table in the two-step
report. It should select two observed first successors, four observed second
successors, four utility values, and two backed-up action scores. Source names
appear only to make the result readable; discovery tests values and execution
order rather than interpreting those names. Alternative mappings through copies
can also survive; they remain separate explanations.

For a concrete negative witness, find that explanation's ID in the decoy report.
The initial outputs match. A changed successor predicts changed evaluations,
backups, and sometimes a changed action. The decoy still returns its cached
action, so the explanation is rejected. The tests also cover cases where the
final action agrees but intermediate predictions disagree.

## Search space and validation protocol

- Candidate models have horizon 1 or 2, exhaustive two-action expansion, terminal
  utility evaluation, maximum backup, and action-zero tie breaking.
- A mapping selects a complete series of 2 or 4 numeric values at one expression
  site/path across its visits, or the 2 or 4 numeric leaves of one container
  checkpoint. All permutations are considered within the budget. This does not
  enumerate arbitrary combinations of independent scalar assignment sites.
- Mappings must match task-generated future states, terminal evaluations and
  backups across the route's discovery cases, and respect their temporal order.
  Model rollouts create expected values only; mappings never call that model.
- Paired tests substitute two alternative state IDs at each state coordinate,
  and low/high numbers at each evaluation and action-score coordinate. Predicted
  and observed downstream mapped values and final actions must agree. At least
  one output-changing intervention per tested input is required.
- Default budgets per route: 256 inventoried series, 4,096 permutations per role
  and horizon, 128 complete candidates, and two intervention cases. Every tested
  complete-candidate rejection has a counterexample. Failed role permutations
  have counts plus a labeled sample of 16 counterexamples. Truncation is explicit.
- Defaults: discovery seed 17, validation seed 29, two independently sampled
  utility contexts for each initial state in each split. The default initial-state
  distribution is uniform; `--state-weights` overrides it. Contexts within each
  state have equal conditional weight. The transition table stays fixed in the
  default demo, though `make_case` accepts other valid tables.
- Branch routes are generic executed `if` decisions. For each frozen mapping,
  validation tests all fresh ordinary executions on its route and paired
  interventions on the first two fresh cases. No mapping repair or discovery
  occurs on validation data. Reused discovery inputs are rejected even if renamed.
- `load_frozen(path)` restores a bank for separate validation without discovery;
  source and manifest digests detect a mismatched program or modified mapping.

The ordinary-population report contains **only unpatched runs**. Supported
horizon, root alternatives, and evaluated leaf alternatives are properties of a
supported explanation grounded in recorded original values. These are not counts
of analyzer rollouts or unique machine operations. Counts across ambiguous
explanations are never added. Evidence counts and validation status are reported
separately. Supported/unresolved population mass measures evidence coverage, not
amount of planning; there is deliberately no planningness score.

## First milestone results

With default seeds and budgets:

| Fixture | Supported explanations | Supported horizon | Ordinary routes |
| --- | ---: | --- | --- |
| One-step planner | 2 | 1 | Supported throughout |
| Two-step planner | 4 | 2 | Supported throughout |
| Renamed dictionary planner | 2 | 2 | Supported throughout |
| Precomputed action lookup | 0 | Unknown | Unresolved |
| Conditional | 4 on planning route | 2 on odd states | Odd states supported; even states unresolved |
| Decoy search | 0 | Unknown | Search explanations rejected by interventions |

All six default searches finish without exhausting their enumeration budgets.
The even-state conditional route is reactive by fixture construction. The
analyzer conservatively reports that it has no supported lookahead explanation,
rather than claiming proof of zero planning. Both conditional routes and their
checkpoints remain inspectable.

## Deliberate limits and next steps

This is empirical support for a constrained explanation family, not an algorithmic
equivalence proof or a general detector of planning. The Python subset excludes
imports, helpers, recursion, comprehensions, arbitrary methods, and cyclic
containers. Programs are trusted and tiny; syntax checking and an expression
budget are not a resource-isolation sandbox. Only ordinary output and input
mutation are checked for observation preservation, not timing or resource usage.

Future work can expand the mapping grammar and algorithm family, add joint or
interchange interventions, vary transition graphs, and evaluate independently
specified task-state distributions. A planningness metric remains a separate
research decision. Preserve unknown outcomes and distinguish amount from evidence
strength when adding it.
