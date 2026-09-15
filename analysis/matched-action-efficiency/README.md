# Absolute actions in fully successful runs

This analysis compares the four main methods shown in the paper: Planner,
GenPlan, Codex (black-box), and Claude Code (black-box). The Claude Code
source-access condition is not part of the four-method comparison.

Each method/environment bar uses only synthesis replicate seeds whose final policy
solved all 100 held-out episodes. A method is omitted in environments where none
of its seeds achieved 100%. First, action counts are averaged across the 100
episodes within each successful seed. The bar is then the mean across successful
seeds, and its error bar is a normal-approximation 95% confidence interval across
those seed-level means. Lower is better.

The one-page PDF includes every environment in which at least one method has a
100%-successful seed; it does not require all methods to succeed in an environment.

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
