# Absolute actions by run-level success

This analysis compares the four main methods shown in the paper: Planner,
GenPlan, Codex (black-box), and Claude Code (black-box). The Claude Code
source-access condition is not part of the four-method comparison.

The five PDF pages show all runs, runs below 100% success, runs with exactly 100%
success, runs below 80% success, and runs with at least 80% success. Thresholds
are applied independently to each synthesis seed. Within a qualifying seed,
actions are averaged over solved episodes only because failed episodes are
censored by the step limit. Bars then average those seed-level means; error bars
are normal-approximation 95% confidence intervals across seeds. Lower is better.

Each page includes every environment with at least one qualifying seed that
solved an episode. A method is omitted only when it has no qualifying observation
for that environment; methods do not need to qualify on the same environments.

The analysis selected the newest non-`outdated` completed result for every
method/environment/replicate tuple in the final Taildrop archives received on
2026-09-15, supplemented by newer completed local results for Dynamo and
BaseMotion that are reflected in the paper table but postdate those archives.
Codex Rovers contains four synthesis seeds; the missing seed is not replaced
with an older result. The BaseMotion Codex bar cannot be reconstructed because
its per-episode results are not present in either source.

## Input archive checksums

- `Baselines.zip`: `109cb25dc8b5be0bc96d896dee8f37a95dc684aa8b61152dd004be82aa7bd5f8`
- `Blackbox.zip`: `e9acca13102acb47614d7ba4108f2af34645b818e1c7275d23822b2eaee2b3d8`
- `Whitebox.zip`: `a9ee8e78f185a0b65ba4d275a4e886f852dfb4102518c2ee2c41f69607759a06`

`Whitebox.zip` is checksummed for provenance but is not used in this four-method
black-box comparison.

## LLM policy-structure judgment

`experiments/judge_policy_structure.py` reproducibly classifies final policies
from individually perfect synthesis seeds into one of three mutually exclusive
categories: `planning`, `stateful`, or `direct`. It invokes the Claude Code CLI
without tools, requires JSON-schema-constrained output, treats policy source as
untrusted text, and records a rationale and concrete code evidence for audit.
Judgments are cached by the policy code hash, model, and rubric version.

For example, given the final-timing manifest and its extracted policy caches:

```bash
uv run python experiments/judge_policy_structure.py \
  --manifest final_timing_3ep/manifest.json \
  --policy-root .final-timing-cache \
  --policy-root .final-timing-cache-v2 \
  --cache-dir analysis/policy-structure/cache \
  --output-csv analysis/policy-structure/judgments.csv \
  --output-summary analysis/policy-structure/summary.json \
  --model opus
```

The manifest must provide `method`, `environment`, `seed`, `solve_rate`, and
`source` for each run. By default, only rows with `solve_rate == 1.0` are judged.
The CSV is the auditable result; the summary JSON contains the per-method counts
used in a compact paper table. Cached model outputs should be retained when
reporting the result so the classifications do not silently change between runs.

Pass `--success-group below-perfect` to classify the complementary set of seeds,
or `--success-group all` to classify both groups together. The solve rate is used
only to select policies and is not included in the judging prompt, preventing it
from biasing the structural classification.
