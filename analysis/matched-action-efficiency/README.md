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
