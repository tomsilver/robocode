# Matched-instance action efficiency

This analysis compares the four main methods shown in the paper: Planner,
GenPlan, Codex (black-box), and Claude Code (black-box). The Claude Code
source-access condition is not part of the four-method comparison.

An observation is keyed by environment, synthesis replicate seed, and held-out
episode index. It is retained only when all four methods have a completed result
for that key and all four solve the episode. Each method's action count is divided
by the planner's action count on the same key. A value below 1 therefore means
fewer actions than the planner.

The first PDF page is an equal-weight average of the environment-level empirical
cumulative distribution functions (ECDFs), so environments with more matched
episodes do not dominate it. The remaining pages cover every environment in the
two main tables. Pages without a curve distinguish missing four-method coverage
from an empty intersection of solved episodes.

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
