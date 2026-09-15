# Absolute actions on jointly solved episodes

This analysis compares the four main methods shown in the paper: Planner,
GenPlan, Codex (black-box), and Claude Code (black-box). The Claude Code
source-access condition is not part of the four-method comparison.

An observation is keyed by environment, synthesis replicate seed, and held-out
episode index. It is retained only when all four methods have a completed result
for that key and all four solve the episode. The plot reports each method's mean
absolute action count in each environment; error bars are normal-approximation
95% confidence intervals over those matched episodes. Lower is better.

The one-page PDF includes each environment with at least one jointly solved
episode. Environments without a four-way jointly solved episode are omitted.

The analysis selected the newest non-`outdated` completed result for every
method/environment/replicate tuple in the final Taildrop archives received on
2026-09-15. Codex Rovers contains four synthesis seeds; the missing seed is not
replaced with an older result.

## Input archive checksums

- `Baselines.zip`: `109cb25dc8b5be0bc96d896dee8f37a95dc684aa8b61152dd004be82aa7bd5f8`
- `Blackbox.zip`: `e9acca13102acb47614d7ba4108f2af34645b818e1c7275d23822b2eaee2b3d8`
- `Whitebox.zip`: `a9ee8e78f185a0b65ba4d275a4e886f852dfb4102518c2ee2c41f69607759a06`

`Whitebox.zip` is checksummed for provenance but is not used in this four-method
black-box comparison.
